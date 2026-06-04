#!/bin/bash
# Preprocess AWI-ESM2 echam6 native monthly output into the
# atm_remapped_1m_<var>_1m_<YYYY>-<YYYY>.nc layout the reval part
# scripts expect (the same layout AWI-ESM3 XIOS emits directly).
#
# Handles surface fields (echam stream). Pressure-level 3D fields
# (3D winds + zg) needs cdo afterburner for spectral->grid plus
# pressure-level interp and is handled in a separate script.
#
# Usage:
#   ./preprocess_AWI-ESM2_echam.sh <expname> <echam_indir> <outdir> <starty> <endy> [target_grid]
# Example:
#   ./preprocess_AWI-ESM2_echam.sh PI_wisofix_c \
#     /work/ba1066/a270107/esm_tools/EXP/PI_wisofix_c/outdata/echam \
#     /work/ba1066/a270107/esm_tools/EXP/PI_wisofix_c/outdata/echam_remapped \
#     5840 6004
#
# After it runs, symlink the output dir so the reval scripts find it
# under the hardcoded /oifs/ path:
#   ln -s echam_remapped $(dirname "$outdir")/oifs
#
# Units (echam6 PI_wisofix_c codes table):
#   - All radiation/heat-flux variables are W/m**2, not accumulated.
#   - Precipitation (aprl/aprc/aprs) is kg/m**2/s.
#   - temp2 in K, aclcov dimensionless [0..1].
# No divide-by-seconds-per-month is needed for this configuration.

set -euo pipefail

# Use the dedicated cdo 1.9.10 build; the conda env cdo can hang or
# differ on echam6 native I/O.
CDO=${CDO:-/work/ab0246/a270092/software/cdo_build/cdo-1.9.10/src/cdo}
if [ ! -x "$CDO" ]; then
    echo "ERROR: cdo not executable at $CDO" >&2
    exit 1
fi

if [ "$#" -lt 5 ]; then
    sed -n '3,28p' "$0"
    exit 1
fi

expname=$1
indir=$2
outdir=$3
starty=$4
endy=$5
target_grid=${6:-r360x180}

mkdir -p "$outdir"
tmpdir="$outdir/tmp"
mkdir -p "$tmpdir"

# Use cdo's built-in echam6 codetable; the run-emitted .codes file is in
# a different format that setpartabn rejects.

# reval/IFS-style name -> echam6 native name (in the _echam stream)
declare -A varmap=(
    [2t]=temp2
    [tcc]=aclcov
    [lsp]=aprl
    [cp]=aprc
    [sf]=aprs
    [sshf]=ahfs
    [slhf]=ahfl
    [ssr]=srads
    [str]=trads
    [tsr]=srad0
    [ttr]=trad0
    [tsrc]=sraf0
    [ttrc]=traf0
)

# Comma-separated list of source (echam) names for one selname call.
src_list=$(IFS=,; echo "${varmap[*]}")

echo "Preprocessing $expname years $starty-$endy -> $outdir (grid $target_grid)"
echo "Variables: ${!varmap[@]}"

for y in $(seq "$starty" "$endy"); do
    yyyy=$(printf "%04d" "$y")

    # Quick existence check on Jan; assume rest of year is present if Jan is.
    if [ ! -f "${indir}/${expname}_${yyyy}01.01_echam" ]; then
        echo "  WARN year $yyyy missing, skipping" >&2
        continue
    fi

    # Step 1: per-month -> single tmp netcdf with all required vars, named.
    for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
        monthly_tmp="$tmpdir/all_${yyyy}${m}.nc"
        [ -f "$monthly_tmp" ] && continue
        "$CDO" -t echam6 -O -s -f nc \
            -selname,"$src_list" \
            "${indir}/${expname}_${yyyy}${m}.01_echam" \
            "$monthly_tmp"
    done

    # Step 2: per-variable -> mergetime 12 monthly tmps, remap, rename.
    for rname in "${!varmap[@]}"; do
        src=${varmap[$rname]}
        out="$outdir/atm_remapped_1m_${rname}_1m_${yyyy}-${yyyy}.nc"
        if [ -f "$out" ]; then
            continue
        fi
        # -r writes CF-compliant relative time ("days since ..."). Without
        # it, cdo emits time:units = "day as %Y%m%d.%f", which xarray
        # cannot decode as cftime and breaks .dt.year accessors downstream.
        "$CDO" -r -O -s -f nc \
            -remapbil,"$target_grid" \
            -chname,"$src","$rname" \
            -selname,"$src" \
            -mergetime \
            "$tmpdir/all_${yyyy}"??.nc \
            "$out"
    done

    # Year done -> drop its monthly tmps.
    rm -f "$tmpdir/all_${yyyy}"??.nc

    echo "  year $yyyy done"
done

rmdir "$tmpdir" 2>/dev/null || true

echo
echo "Done. To make reval find these:"
echo "  ln -s $(basename "$outdir") $(dirname "$outdir")/oifs"
