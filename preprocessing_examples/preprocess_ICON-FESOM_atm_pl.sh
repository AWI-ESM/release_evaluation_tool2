#!/bin/bash
# Preprocess ICON 3D atmosphere output to pressure levels for
# release_evaluation_tool2.
#
# Takes ICON's bundled atm_3d_ml multi-variable files (daily instantaneous
# values on the unstructured R02B04 grid, 90 generalized-height levels),
# vertically interpolates to a fixed pressure-level set via `cdo ap2pl`
# (which uses the 3D `pres` field, CF standard_name = air_pressure),
# monmeans them, remaps to a regular lat/lon grid, and splits into
# per-variable per-year files using the IFS short-name layout the part
# scripts already expect, with the `_pl` infix that
# part11_zonal_plots.py and part12_qbo.py look for:
#   atm_remapped_1m_pl_<var>_1m_pl_<YYYY>-<YYYY>.nc
#
# Usage:
#   ./preprocess_ICON-FESOM_atm_pl.sh <expname> <icon_indir> <outdir> <starty> <endy> [target_grid]
#
# Example (last 3 years of the asd0094 spinup):
#   ./preprocess_ICON-FESOM_atm_pl.sh asd0094 \
#     /work/bk1415/a270044/YET_ON_YAC/icon-fesom/exp009/icon \
#     /work/ab0246/a270092/runtime/ICON_FESOM_asd0094/outdata/oifs \
#     2105 2107 r360x180
#
# Notes:
#   - cdo `ml2pl` requires ECMWF-style hybrid sigma pressure coords which
#     ICON does NOT use (it stores air_pressure as a 3D diagnostic on its
#     generalized_height vertical axis). `ap2pl` is the right operator
#     here: it ingests the 3D `pres` field directly.
#   - The "Surface pressure not found" warning from ap2pl is expected and
#     harmless for our target levels (all of them are above the model
#     surface at this grid).
#   - We interpolate on the native unstructured grid BEFORE remapping, so
#     the pressure surfaces aren't smeared horizontally by conservative
#     remap.

set -euo pipefail

if [[ $# -lt 5 ]]; then
    head -36 "$0" | tail -31
    exit 1
fi

expname=$1
icon_indir=$2
outdir=$3
starty=$4
endy=$5
target_grid=${6:-r360x180}

mkdir -p "$outdir"
tmpdir="$outdir/tmp_pl"
mkdir -p "$tmpdir"

# Use the szip-capable cdo build (one binary across all preprocessors).
CDO=/work/ab0246/a270092/software/cdo_build/cdo-1.9.10/src/cdo

# Target pressure levels in Pa (19-level ERA5 standard set):
#   1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100,
#   70, 50, 30, 20, 10, 5, 1 hPa
# Matches the ERA5 reanalysis plev count so part11_zonal_plots.py rmsd
# arrays broadcast cleanly (8-vs-19 mismatch was a hard error before),
# and the upper-strato levels (down to 1 hPa) still cover QBO needs.
plevs_pa="100000,92500,85000,70000,60000,50000,40000,30000,25000,20000,15000,10000,7000,5000,3000,2000,1000,500,100"

# ICON atm_3d_ml short name -> reval/IFS short name mapping.
# part11_zonal_plots wants u, t; part12_qbo wants u.  Include v and q as
# well since the user spec listed them and they cost little extra here.
declare -A ren=(
    [u]=u
    [v]=v
    [temp]=t
    [qv]=q
)

# Source variables required for ap2pl: every wanted var PLUS pres
# (otherwise cdo can't identify the 3D pressure field).
src_vars="pres,u,v,temp,qv"

echo "Preprocessing $expname years $starty-$endy -> $outdir (grid $target_grid)"
echo "  pressure levels [Pa]: $plevs_pa"

for ((yyyy=starty; yyyy<=endy; yyyy++)); do
    src="$icon_indir/${expname}_atm_3d_ml_${yyyy}0101T000000Z.nc"
    if [[ ! -r $src ]]; then
        echo "  WARN year $yyyy missing ($src), skipping" >&2
        continue
    fi
    echo "  year $yyyy"

    # Step 1: select needed vars, vertically interpolate to plev on the
    # native unstructured grid, monthly-mean, and remap. This single
    # chained cdo invocation is the expensive bit.
    #
    # Order rationale:
    #   selname,<vars>         drop unused fields (clc, rh, geopot, ...)
    #   ap2pl,<plevs>          interp to pressure surfaces using `pres`
    #   monmean                collapse 365 daily steps to 12 months
    #   seltimestep,1/12       drop the trailing partial-month boundary
    #                          fragment that monmean emits when the source
    #                          twice-daily file spans into the next year
    #                          (part12_qbo.py assumes exactly 12/yr)
    #   remapcon,<target>      conservative remap to regular lat/lon
    # ap2pl runs before remap so pressure surfaces aren't smeared
    # horizontally; monmean before remap shrinks the data feeding remap.
    year_remapped="$tmpdir/${yyyy}_pl_monmean_remapped.nc"
    if [[ ! -f $year_remapped ]]; then
        "$CDO" -O -f nc \
            -remapcon,"$target_grid" \
            -seltimestep,1/12 \
            -monmean \
            -ap2pl,"$plevs_pa" \
            -selname,"$src_vars" \
            "$src" "$year_remapped"
    fi

    # Step 2: per-variable rename + split.
    for src_name in "${!ren[@]}"; do
        rname=${ren[$src_name]}
        out="$outdir/atm_remapped_1m_pl_${rname}_1m_pl_${yyyy}-${yyyy}.nc"
        [[ -f $out ]] && continue
        "$CDO" -O -f nc -chname,"${src_name},${rname}" -selname,"$src_name" \
            "$year_remapped" "$out"
    done
done

echo "Done. Files in $outdir/."
echo "Sample: $(ls "$outdir"/atm_remapped_1m_pl_u_1m_pl_*.nc 2>/dev/null | wc -l) years of u on pressure levels"
