#!/bin/bash
# Pressure-level 3D fields
# (u, t, v, optionally zg) from echam6 native spectral output.
#
# Echam6 stores 3D atmosphere as spectral state (st temperature, svo
# vorticity, sd divergence, lsp log surface pressure). The afterburner
# derives u/v from svo/sd, t from st, and geopotential height (geopoth)
# from t + lsp via hydrostatic integration, then interpolates to the
# requested pressure levels.
#
# Output layout matches what part11/part12 expect:
#   atm_remapped_1m_pl_<var>_1m_pl_YYYY-YYYY.nc  (12 monthly steps, 16 plevels)
#
# Usage:
#   ./preprocess_AWI-ESM2_echam_3d.sh <expname> <echam_indir> <outdir> <starty> <endy> [target_grid]
#
# After it runs the outdir is the same as for phase 2a, so the existing
# oifs symlink already points reval at it.

set -euo pipefail

CDO=${CDO:-/work/ab0246/a270092/software/cdo_build/cdo-1.9.10/src/cdo}
if [ ! -x "$CDO" ]; then
    echo "ERROR: cdo not executable at $CDO" >&2
    exit 1
fi

if [ "$#" -lt 5 ]; then
    sed -n '3,18p' "$0"
    exit 1
fi

expname=$1
indir=$2
outdir=$3
starty=$4
endy=$5
target_grid=${6:-r360x180}

mkdir -p "$outdir"
tmpdir="$outdir/tmp_pl"
mkdir -p "$tmpdir"

# Standard AMIP/CMIP pressure levels in Pa (1000 hPa down to 10 hPa).
# Comma-separated for the afterburner namelist LEVEL slot.
PLEVELS_PA="100000,92500,85000,70000,50000,40000,30000,25000,20000,15000,10000,7000,5000,3000,2000,1000"

# ECHAM grib codes for the derived/grid output:
#   130 = t  (temperature)
#   131 = u
#   132 = v
#   156 = geopoth (geopotential height)
CODES="130,131,132,156"

# After-burner namelist template. TYPE=30 = pressure-level grid point,
# MEAN=1 = monthly mean (echam already outputs monthly so this is a no-op
# for averaging but ensures the right step semantics).
read -r -d '' NAMELIST <<EOF || true
&SELECT
  TYPE=30,
  CODE=${CODES},
  LEVEL=${PLEVELS_PA},
  MEAN=1,
&END
EOF

# After-burner emits these names; rename for the part scripts' filenames.
declare -A varmap=(
    [u]=u
    [t]=t
    [v]=v
    [zg]=geopoth
)

echo "Preprocessing $expname (3D pl) years $starty-$endy -> $outdir (grid $target_grid)"
echo "Pressure levels (Pa): $PLEVELS_PA"
echo "Variables: ${!varmap[@]}"

for y in $(seq "$starty" "$endy"); do
    yyyy=$(printf "%04d" "$y")

    if [ ! -f "${indir}/${expname}_${yyyy}01.01_echam" ]; then
        echo "  WARN year $yyyy missing, skipping" >&2
        continue
    fi

    # Quick skip if all per-variable outputs already exist.
    all_done=true
    for rname in "${!varmap[@]}"; do
        out="$outdir/atm_remapped_1m_pl_${rname}_1m_pl_${yyyy}-${yyyy}.nc"
        [ -f "$out" ] || { all_done=false; break; }
    done
    if $all_done; then
        echo "  year $yyyy already done, skipping"
        continue
    fi

    # Step 1: afterburner per month -> t/u/v/geopoth on plevels.
    for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
        monthly_tmp="$tmpdir/pl_${yyyy}${m}.nc"
        [ -f "$monthly_tmp" ] && continue
        echo "$NAMELIST" | "$CDO" -s -t echam6 -f nc after \
            "${indir}/${expname}_${yyyy}${m}.01_echam" \
            "$monthly_tmp"
    done

    # Step 2: mergetime monthly tmps, remap, split per variable.
    yearly_merged="$tmpdir/pl_${yyyy}_merged.nc"
    "$CDO" -O -s -f nc \
        -remapbil,"$target_grid" \
        -mergetime \
        "$tmpdir/pl_${yyyy}"??.nc \
        "$yearly_merged"

    for rname in "${!varmap[@]}"; do
        src=${varmap[$rname]}
        out="$outdir/atm_remapped_1m_pl_${rname}_1m_pl_${yyyy}-${yyyy}.nc"
        if [ -f "$out" ]; then
            continue
        fi
        "$CDO" -O -s -f nc \
            -chname,"$src","$rname" \
            -selname,"$src" \
            "$yearly_merged" \
            "$out"
    done

    rm -f "$tmpdir/pl_${yyyy}"??.nc "$yearly_merged"
    echo "  year $yyyy done"
done

rmdir "$tmpdir" 2>/dev/null || true

echo
echo "Done."
