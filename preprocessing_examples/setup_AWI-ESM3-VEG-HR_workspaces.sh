#!/bin/bash
# Build a unified workspace tree for the AWI-ESM3-VEG-HR run so reval can
# consume it. The source data lives across four separate run dirs:
#   - Spinup_cont2  (1350-1649)   awiesm3-v3.4.1, OLD oifs naming
#                                  atm_remapped_1m_<var>_1m_<YYYY>-<YYYY>.nc
#   - Spinup_cont3  (1650-1679)   awiesm3-v3.4.2, NEW oifs naming
#                                  atm_remapped_1m_<var>_<YYYY>-<YYYY>.nc
#                                  (plus _original and _reduced_gaussian_grid
#                                   intermediate variants we ignore)
#   - piControl     (1850-1865)   awiesm3-v3.4.2, NEW oifs naming
#   - historical    (1850-1879)   awiesm3-v3.4.2, NEW oifs naming
#
# Reval scripts look for `atm_remapped_1m_<var>_1m_<YYYY>-<YYYY>.nc` for OIFS
# and `<var>.fesom.<YYYY>.nc` for FESOM. This script:
#   - symlinks Spinup_cont2 OIFS files directly (already correct name)
#   - symlinks Spinup_cont3 / piControl / historical OIFS files with the
#     missing `_1m_` infix inserted before the year range
#   - skips `_original` and `_reduced_gaussian_grid` cdo intermediates
#   - symlinks FESOM / lpj_guess / oasis3mct / rnfmap files directly (same
#     name in all sources)
#
# Output layout under DST_ROOT:
#   AWI-ESM3-VEG-HR-Spinup/outdata/{oifs,fesom,lpj_guess,oasis3mct,rnfmap,xios}/
#   AWI-ESM3-VEG-HR-piControl/outdata/...
#   AWI-ESM3-VEG-HR-historical/outdata/...

set -euo pipefail

SRC_CONT2=/work/bb1469/a270089/runtime/awiesm3-v3.4.1/AWI-ESM3-VEG-HR-CMIP7-Spinup_cont2
SRC_CONT3=/work/bb1469/a270089/runtime/awiesm3-v3.4.2/AWI-ESM3-VEG-HR-CMIP7-Spinup_cont3
SRC_PI=/work/bb1469/a270089/runtime/awiesm3-v3.4.2/AWI-ESM3-VEG-HR-CMIP7-piControl
SRC_HIST=/work/bb1469/a270089/runtime/awiesm3-v3.4.2/AWI-ESM3-VEG-HR-CMIP7-historical
DST_ROOT=${1:-/work/bb1469/a270092/runtime/awiesm3-v3.4.2}

DST_SPINUP=$DST_ROOT/AWI-ESM3-VEG-HR-Spinup
DST_PI=$DST_ROOT/AWI-ESM3-VEG-HR-piControl
DST_HIST=$DST_ROOT/AWI-ESM3-VEG-HR-historical

STREAMS=(oifs fesom lpj_guess oasis3mct rnfmap xios)
for d in "$DST_SPINUP" "$DST_PI" "$DST_HIST"; do
    for s in "${STREAMS[@]}"; do
        mkdir -p "$d/outdata/$s"
    done
done

# Spinup_cont2: OLD naming already matches what reval expects. Symlink as-is.
echo "=== Spinup_cont2 1350-1649 (OLD naming -> as-is) ==="
for s in "${STREAMS[@]}"; do
    [[ -d $SRC_CONT2/outdata/$s ]] || continue
    n=0
    for f in $SRC_CONT2/outdata/$s/*; do
        [[ -f $f ]] || continue
        base=$(basename "$f")
        # In OIFS, skip cdo intermediates if any leak in
        if [[ $s == oifs && $base == *_original.nc ]]; then continue; fi
        if [[ $s == oifs && $base == *_reduced_gaussian_grid.nc ]]; then continue; fi
        dst="$DST_SPINUP/outdata/$s/$base"
        [[ -e $dst ]] && continue
        ln -s "$f" "$dst"
        n=$((n+1))
    done
    echo "  $s : $n symlinks"
done

# Spinup_cont3: NEW naming -> insert "_1m_" before year in OIFS atm_remapped files.
echo "=== Spinup_cont3 1650-1679 (NEW naming -> renamed) ==="
for s in "${STREAMS[@]}"; do
    [[ -d $SRC_CONT3/outdata/$s ]] || continue
    n=0
    for f in $SRC_CONT3/outdata/$s/*; do
        [[ -f $f ]] || continue
        base=$(basename "$f")
        if [[ $s == oifs ]]; then
            # Skip intermediates
            [[ $base == *_original.nc ]] && continue
            [[ $base == *_reduced_gaussian_grid.nc ]] && continue
            # Rewrite to the reval-expected layout:
            #   atm_remapped_1m_pl_<var>_<YYYY>-<YYYY>.nc
            #     -> atm_remapped_1m_pl_<var>_1m_pl_<YYYY>-<YYYY>.nc
            #   atm_remapped_1m_<var>_<YYYY>-<YYYY>.nc       (non-pl)
            #     -> atm_remapped_1m_<var>_1m_<YYYY>-<YYYY>.nc
            # (part11_zonal_plots.py and part12_qbo.py look for the
            #  double _1m_pl_ infix on pressure-level files;
            #  everything else uses the single _1m_ infix.)
            if [[ $base =~ ^atm_remapped_1m_pl_(.+)_([0-9]{4}-[0-9]{4})\.nc$ ]]; then
                var=${BASH_REMATCH[1]}
                yrs=${BASH_REMATCH[2]}
                dst="$DST_SPINUP/outdata/$s/atm_remapped_1m_pl_${var}_1m_pl_${yrs}.nc"
                [[ -e $dst ]] && continue
                ln -s "$f" "$dst"
                n=$((n+1))
                continue
            fi
            if [[ $base =~ ^atm_remapped_1m_(.+)_([0-9]{4}-[0-9]{4})\.nc$ ]]; then
                var=${BASH_REMATCH[1]}
                yrs=${BASH_REMATCH[2]}
                dst="$DST_SPINUP/outdata/$s/atm_remapped_1m_${var}_1m_${yrs}.nc"
                [[ -e $dst ]] && continue
                ln -s "$f" "$dst"
                n=$((n+1))
                continue
            fi
            # Any other 1d / atmos_1h pattern: keep as-is (reval doesn't
            # consume it, but leaving the symlink makes inspection easier).
            dst="$DST_SPINUP/outdata/$s/$base"
            [[ -e $dst ]] && continue
            ln -s "$f" "$dst"
            n=$((n+1))
        else
            # Non-OIFS streams: just symlink under the same name.
            dst="$DST_SPINUP/outdata/$s/$base"
            [[ -e $dst ]] && continue
            ln -s "$f" "$dst"
            n=$((n+1))
        fi
    done
    echo "  $s : $n symlinks"
done

# piControl and historical: NEW naming, single source each.
link_run() {
    local src=$1 dst=$2 label=$3
    echo "=== $label (NEW naming -> renamed) ==="
    for s in "${STREAMS[@]}"; do
        [[ -d $src/outdata/$s ]] || continue
        local n=0
        for f in $src/outdata/$s/*; do
            [[ -f $f ]] || continue
            local base=$(basename "$f")
            if [[ $s == oifs ]]; then
                [[ $base == *_original.nc ]] && continue
                [[ $base == *_reduced_gaussian_grid.nc ]] && continue
                # pl files get _1m_pl_ infix; others get _1m_ infix.
                if [[ $base =~ ^atm_remapped_1m_pl_(.+)_([0-9]{4}-[0-9]{4})\.nc$ ]]; then
                    local var=${BASH_REMATCH[1]}
                    local yrs=${BASH_REMATCH[2]}
                    local out="$dst/outdata/$s/atm_remapped_1m_pl_${var}_1m_pl_${yrs}.nc"
                    [[ -e $out ]] && continue
                    ln -s "$f" "$out"
                    n=$((n+1))
                    continue
                fi
                if [[ $base =~ ^atm_remapped_1m_(.+)_([0-9]{4}-[0-9]{4})\.nc$ ]]; then
                    local var=${BASH_REMATCH[1]}
                    local yrs=${BASH_REMATCH[2]}
                    local out="$dst/outdata/$s/atm_remapped_1m_${var}_1m_${yrs}.nc"
                    [[ -e $out ]] && continue
                    ln -s "$f" "$out"
                    n=$((n+1))
                    continue
                fi
            fi
            local out="$dst/outdata/$s/$base"
            [[ -e $out ]] && continue
            ln -s "$f" "$out"
            n=$((n+1))
        done
        echo "  $s : $n symlinks"
    done
}

link_run "$SRC_PI"   "$DST_PI"   "piControl 1850-1865"
link_run "$SRC_HIST" "$DST_HIST" "historical 1850-1879"

echo ""
echo "=== sanity check coverage ==="
for d in "$DST_SPINUP" "$DST_PI" "$DST_HIST"; do
    printf "%s : oifs %d files, fesom %d files\n" "${d##*/}" \
        $(ls "$d/outdata/oifs"/atm_remapped_1m_2t_1m_*.nc 2>/dev/null | wc -l) \
        $(ls "$d/outdata/fesom"/a_ice.fesom.*.nc 2>/dev/null | wc -l)
done

echo ""
echo "Done. Workspace at $DST_ROOT/"
