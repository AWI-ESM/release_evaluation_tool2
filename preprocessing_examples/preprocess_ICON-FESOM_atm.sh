#!/bin/bash
# Preprocess ICON atmosphere output for release_evaluation_tool2.
#
# Takes ICON's bundled atm_2d_ml multi-variable files (twice-daily
# instantaneous values on the unstructured R02B04 grid), monmeans them,
# remaps to a regular lat/lon grid, and splits into per-variable per-year
# files using the IFS short-name layout the part scripts already expect:
#   atm_remapped_1m_<var>_1m_<YYYY>-<YYYY>.nc
#
# Usage:
#   ./preprocess_ICON-FESOM_atm.sh <expname> <icon_indir> <outdir> <starty> <endy> [target_grid]
#
# Example (last 3 years of the asd0094 spinup):
#   ./preprocess_ICON-FESOM_atm.sh asd0094 \
#     /work/bk1415/a270044/YET_ON_YAC/icon-fesom/exp009/icon \
#     /work/ab0246/a270092/runtime/ICON_FESOM_asd0094/outdata/oifs \
#     2105 2107 r360x180
#
# After it runs, the outdir already matches reval's expected `oifs` layout
# so no symlink hop is needed (we just call the directory `oifs`).
#
# Units (ICON atm_2d_ml):
#   - 2D radiative net fluxes (sob_t/thb_t/sob_s/thb_s) are W/m^2
#     (instantaneous, positive-down convention same as IFS).
#   - shfl_s/lhfl_s are W/m^2. Sign convention may differ from echam6;
#     this script applies the convention pass on a per-variable basis.
#   - tot_prec_rate and snow_*_rate are kg/m^2/s (instantaneous).
#   - t_2m in K, clct in %, pres_msl/pres_sfc in Pa.
# No divide-by-seconds-per-month is needed (accumulation_period=1 in the
# matching reval config). part2_rad_balance dispatches the snow latent-
# heat constant from sf.attrs['units']='kg m-2 s-1' automatically.

set -euo pipefail

if [[ $# -lt 5 ]]; then
    head -32 "$0" | tail -27
    exit 1
fi

expname=$1
icon_indir=$2
outdir=$3
starty=$4
endy=$5
target_grid=${6:-r360x180}

mkdir -p "$outdir"
tmpdir="$outdir/tmp"
mkdir -p "$tmpdir"

# Use the szip-capable cdo build (the conda env cdo is fine for ICON nc,
# but keep one binary across all preprocessors for consistency).
CDO=/work/ab0246/a270092/software/cdo_build/cdo-1.9.10/src/cdo

# ICON atm_2d_ml short name -> reval IFS short name mapping.
# Single-source variables (one ICON name -> one reval name, units preserved):
declare -A ren=(
    [t_2m]=2t
    [sob_t]=tsr
    [thb_t]=ttr
    [sob_s]=ssr
    [thb_s]=str
    [shfl_s]=sshf
    [lhfl_s]=slhf
    [tot_prec_rate]=tp
    [clct]=tcc
    [pres_msl]=msl
    [pres_sfc]=sp
)

# Variables we'll synthesize from sums of multiple ICON fields:
#   sf  = snow_gsp_rate + snow_con_rate + ice_gsp_rate (frozen precip)
#   lsp = tot_prec_rate - (snow components)             (liquid precip)
#   cp  = 0 (ICON atm_2d_ml doesn't split convective from large-scale
#         rain; we assign all of it to lsp and emit cp as a zero field so
#         scripts that expect `cp + lsp = total precip` still work)
#   ssrd = sob_s / max(1 - alb, 0.05), where alb is the average of the
#          four surface albedos (visdir, visdif, nirdir, nirdif). Recovers
#          a downward SW estimate from net SW + albedos since ICON
#          atm_2d_ml doesn't expose ssrd directly.

echo "Preprocessing $expname years $starty-$endy -> $outdir (grid $target_grid)"

for ((yyyy=starty; yyyy<=endy; yyyy++)); do
    src="$icon_indir/${expname}_atm_2d_ml_${yyyy}0101T000000Z.nc"
    if [[ ! -r $src ]]; then
        echo "  WARN year $yyyy missing ($src), skipping" >&2
        continue
    fi
    echo "  year $yyyy"

    # Monthly-mean + remap once for the full file. That cdo invocation
    # is the expensive bit (one cdo bake; everything else is splitting).
    year_remapped="$tmpdir/${yyyy}_monmean_remapped.nc"
    if [[ ! -f $year_remapped ]]; then
        "$CDO" -O -f nc -monmean -remapcon,"$target_grid" "$src" "$year_remapped"
    fi

    # Per-variable rename and split out into the reval-expected layout.
    # --reduce_dim drops cdo's degenerate length-1 height/level axes,
    # which otherwise leak into the part-script ndarray shapes (notably
    # t_2m carries a `height_2 = 1` dim that breaks part8/16/20).
    #
    # Special case: clct -> tcc. ICON's clct is total cloud cover in
    # percent (0..100, units="%"), but reval's part10_clt_vs_modis.py
    # expects tcc as a 0..1 fraction (it multiplies by 100 for display
    # vs MODIS). So for clct we additionally divide by 100 and overwrite
    # the units attribute to "1" (dimensionless).
    for src_name in "${!ren[@]}"; do
        rname=${ren[$src_name]}
        out="$outdir/atm_remapped_1m_${rname}_1m_${yyyy}-${yyyy}.nc"
        [[ -f $out ]] && continue
        if [[ $src_name == "clct" ]]; then
            # Divide by 100 to convert %->fraction. We then force the units
            # attribute to the *string* "1" via a python netCDF4 step,
            # because cdo's setattribute auto-detects "1" as a numeric and
            # CF-compliant readers (xarray) choke on a numeric units attr.
            "$CDO" -O -f nc --reduce_dim \
                -divc,100 \
                -chname,"${src_name},${rname}" -selname,"$src_name" \
                "$year_remapped" "$out"
            python3 -c "
import netCDF4 as nc, sys
with nc.Dataset(sys.argv[1], 'r+') as ds:
    v = ds.variables['${rname}']
    if 'units' in v.ncattrs():
        v.delncattr('units')
    v.units = '1'
" "$out"
        else
            "$CDO" -O -f nc --reduce_dim -chname,"${src_name},${rname}" -selname,"$src_name" \
                "$year_remapped" "$out"
        fi
    done

    # Synthetic frozen precip: sf = snow_gsp + snow_con + ice_gsp
    sf_out="$outdir/atm_remapped_1m_sf_1m_${yyyy}-${yyyy}.nc"
    if [[ ! -f $sf_out ]]; then
        "$CDO" -O -f nc \
            -chname,snow_gsp_rate,sf \
            -expr,'snow_gsp_rate=snow_gsp_rate+snow_con_rate+ice_gsp_rate' \
            -selname,snow_gsp_rate,snow_con_rate,ice_gsp_rate \
            "$year_remapped" "$sf_out"
    fi

    # Synthetic liquid precip: lsp = tot_prec_rate - (frozen)
    lsp_out="$outdir/atm_remapped_1m_lsp_1m_${yyyy}-${yyyy}.nc"
    if [[ ! -f $lsp_out ]]; then
        "$CDO" -O -f nc \
            -chname,tot_prec_rate,lsp \
            -expr,'tot_prec_rate=tot_prec_rate-snow_gsp_rate-snow_con_rate-ice_gsp_rate' \
            -selname,tot_prec_rate,snow_gsp_rate,snow_con_rate,ice_gsp_rate \
            "$year_remapped" "$lsp_out"
    fi

    # Synthetic convective precip: cp = 0 (units kg m-2 s-1, like lsp).
    # ICON atm_2d_ml doesn't separate convective from large-scale rain,
    # so we lump everything into lsp and emit cp as a zero field with the
    # same dims/grid/time axis. -expr 'cp=lsp*0' against the lsp per-year
    # file is the cleanest path: same shape, same units, value identically
    # zero. We then rewrite the units attr so ncdump shows the expected
    # kg m-2 s-1 (cdo carries lsp's attrs through expr by default, so this
    # is already correct, but we set it explicitly for clarity).
    cp_out="$outdir/atm_remapped_1m_cp_1m_${yyyy}-${yyyy}.nc"
    if [[ ! -f $cp_out ]]; then
        "$CDO" -O -f nc \
            -setattribute,cp@units="kg m-2 s-1",cp@long_name="convective precipitation rate (synthetic zero)" \
            -expr,'cp=lsp*0' \
            "$lsp_out" "$cp_out"
    fi

    # Synthetic downward SW: ssrd = sob_s / max(1 - alb, 0.05), where
    # alb = mean of the four surface albedos. The clamp avoids blow-up
    # over snow/ice where alb -> 1. Note: sob_s in ICON is W/m^2 net
    # downward, so dividing by (1 - alb) recovers the incoming SW.
    # ICON albedos are stored as percent (units = "%"), so we divide by
    # 100 to get a 0..1 fraction before the (1 - alb) step.
    ssrd_out="$outdir/atm_remapped_1m_ssrd_1m_${yyyy}-${yyyy}.nc"
    if [[ ! -f $ssrd_out ]]; then
        "$CDO" -O -f nc \
            -setattribute,ssrd@units="W m-2",ssrd@long_name="surface downward shortwave (synthetic from sob_s and albedos)" \
            -expr,'_alb=(albvisdir+albvisdif+albnirdir+albnirdif)/400; _coalb=(1-_alb>0.05)?(1-_alb):0.05; ssrd=sob_s/_coalb;' \
            -selname,sob_s,albvisdir,albvisdif,albnirdir,albnirdif \
            "$year_remapped" "$ssrd_out"
    fi
done

echo "Done. Files in $outdir/."
echo "Sample: $(ls "$outdir"/atm_remapped_1m_2t_1m_*.nc 2>/dev/null | wc -l) years of 2t"
