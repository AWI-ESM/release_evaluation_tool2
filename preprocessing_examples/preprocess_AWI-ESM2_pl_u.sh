#!/bin/bash
# Spectral echam6 -> pressure-level u for QBO (part12).
#
# Chain: sp2gp -dv2uv on (svo, sd) gives gridded u/v at the model's
# native T63 gaussian; merge with aps (surface pressure) so ml2pl can
# do the hybrid-sigma -> pressure-level interpolation; select u and
# rename so the output file matches the reval part12 layout
# (`atm_remapped_1m_pl_u_1m_<YYYY>-<YYYY>.nc`).
#
# Usage:
#   ./preprocess_AWI-ESM2_pl_u.sh <expname> <echam_indir> <outdir> <starty> <endy> [target_grid]

set -euo pipefail

CDO=${CDO:-/work/ab0246/a270092/software/cdo_build/cdo-1.9.10/src/cdo}

if [ "$#" -lt 5 ]; then
    sed -n '3,15p' "$0"
    exit 1
fi

expname=$1
indir=$2
outdir=$3
starty=$4
endy=$5
target_grid=${6:-r360x180}

# 19 pressure levels chosen to match part12_qbo.py's hardcoded x-axis
# (100, 92.5, 85, 70, 60, 50, 40, 30, 25, 20, 15, 10, 7, 5, 3, 2, 1, 0.5,
# 0.1 hPa), converted to Pa. Order is identical so the contour plot's
# y-axis aligns with the data's pressure dimension.
QBO_PLEV_PA="10000,9250,8500,7000,6000,5000,4000,3000,2500,2000,1500,1000,700,500,300,200,100,50,10"

mkdir -p "$outdir"
tmpdir="$outdir/tmp_pl_u"
mkdir -p "$tmpdir"

for y in $(seq "$starty" "$endy"); do
    yyyy=$(printf "%04d" "$y")
    out="$outdir/atm_remapped_1m_pl_u_1m_pl_${yyyy}-${yyyy}.nc"
    [ -f "$out" ] && continue

    # Skip incomplete years.
    if [ ! -f "${indir}/${expname}_${yyyy}01.01_echam" ]; then
        echo "  WARN ${yyyy}: missing Jan, skipping"
        continue
    fi

    monthly_pl=()
    for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
        f_in="${indir}/${expname}_${yyyy}${m}.01_echam"
        [ -f "$f_in" ] || { echo "  WARN ${yyyy}-${m} missing"; continue; }
        f_out="$tmpdir/u_${yyyy}${m}.nc"
        # uv on gaussian + aps -> merge -> ml2pl -> select u, rename to 'u'.
        uv="$tmpdir/uv_${yyyy}${m}.nc"
        aps="$tmpdir/aps_${yyyy}${m}.nc"
        "$CDO" -r -O -f nc -t echam6 -sp2gp -dv2uv -selname,svo,sd "$f_in" "$uv" >/dev/null 2>&1
        "$CDO" -r -O -f nc -t echam6 -selname,aps "$f_in" "$aps" >/dev/null 2>&1
        "$CDO" -r -O -f nc -t echam6 \
            -remapbil,"$target_grid" \
            -chname,u,u \
            -selname,u \
            -ml2pl,$QBO_PLEV_PA \
            -merge "$uv" "$aps" \
            "$f_out" >/dev/null 2>&1
        rm -f "$uv" "$aps"
        monthly_pl+=("$f_out")
    done

    if [ "${#monthly_pl[@]}" -eq 0 ]; then
        echo "  WARN ${yyyy}: no monthly p-level data produced, skipping"
        continue
    fi

    "$CDO" -r -O -f nc -mergetime "${monthly_pl[@]}" "$out"
    rm -f "${monthly_pl[@]}"
    echo "  year $yyyy done"
done

rmdir "$tmpdir" 2>/dev/null || true
echo "pressure-level u preproc complete: $outdir"
