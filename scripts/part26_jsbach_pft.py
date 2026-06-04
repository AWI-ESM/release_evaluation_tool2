"""Dominant PFT map from JSBACH `act_fpc` (code 31 in the `_veg` stream).

`act_fpc` is the actually-realised Foliar Projective Cover per PFT
produced by the DYNVEG dynamic-vegetation module — it reflects what
plants grew given competition, climate and disturbance, rather than
the prescribed/allocated cover fraction. 11-level (PFT) field on the
T63 Gaussian grid. The dominant PFT per gridbox is the argmax over
the PFT axis; cells whose total `act_fpc` is below `BARREN_THRESHOLD`
are classified as "barren".
"""
import sys
import os
import glob
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bg_routines.config_loader import *

SCRIPT_NAME = os.path.basename(__file__)
print(SCRIPT_NAME)
update_status(SCRIPT_NAME, " Started")

# JSBACH 11-PFT lookup. Order matches the level dim from selcode,12.
# Names follow MPI-ESM jsbach defaults; if the run uses a different PFT
# selection these labels should be revisited.
# JSBACH3 11-tile PFT selection — verified by reading the per-tile
# `cover_type` field in this run's jsbach.nc input (each tile carries a
# specific LCT 21 ID).  LCT IDs in parentheses.
JSBACH_PFTS = [
    'TrBE',     # 1.  Tropical broadleaf evergreen tree (LCT 2)
                #     (also covers Glacier on ice-sheet cells, LCT 1)
    'TrBD',     # 2.  Tropical broadleaf deciduous tree (LCT 3)
    'ExBE',     # 3.  Extra-tropical evergreen tree     (LCT 4)
    'ExBD',     # 4.  Extra-tropical deciduous tree     (LCT 5)
    'RaShrub',  # 5.  Raingreen shrub                   (LCT 10)
    'CdShrub',  # 6.  Cold (deciduous) shrub            (LCT 11)
    'C3Gr',     # 7.  C3 grass                          (LCT 12)
    'C4Gr',     # 8.  C4 grass                          (LCT 13)
    'C3Past',   # 9.  C3 pasture                        (LCT 15)
    'C4Past',   # 10. C4 pasture                        (LCT 16)
    'Crops',    # 11. Crops (C3 + C4 mixed)             (LCT 20, 21)
]
# Colours match part26_lpjg_pft.py's palette (same project convention):
# TrBE/TrBD->tropical greens, TeBE/TeBD->purples (extra-trop trees),
# shrubs->purples, C3/C4 grass->yellow/orange, pastures->browns from
# LPJG's C3G/C4G shifted, tundra->grey.
PFT_COLORS = [
    '#1b5e20',  # TrBE  - tropical evergreen (LPJG TrBE)
    '#388e3c',  # TrBD  - tropical deciduous (LPJG TrIBE green)
    '#0d47a1',  # TeBE  - extra-trop evergreen (LPJG BNE blue)
    '#1976d2',  # TeBD  - extra-trop deciduous (LPJG BINE)
    '#7b1fa2',  # RaShrub - raingreen shrub (LPJG TeBS purple)
    '#ba68c8',  # CdShrub - cold shrub (LPJG IBS lighter purple)
    '#fdd835',  # C3Gr  - C3 grass (LPJG C3G)
    '#ff8f00',  # C4Gr  - C4 grass (LPJG C4G)
    '#8d6e63',  # C3Past - C3 pasture (brown variant)
    '#a1887f',  # C4Past - C4 pasture (lighter brown)
    '#bdbdbd',  # Tundra - grey (LPJG Barren)
]
# Barren threshold operates on (desert_fpc + bare_fpc) directly — a
# cell is rendered as Barren if combined desert + bare-soil cover
# exceeds this fraction of the grid box.
BARREN_THRESHOLD = 0.5

jsbach_dir = os.path.join(historic_path, 'jsbach')
years = range(historic_last25y_start, historic_last25y_end + 1)

candidates = sorted(glob.glob(os.path.join(jsbach_dir, f'*_{years[0]:04d}01.01_veg')))
if not candidates:
    raise FileNotFoundError(f"No _veg file for year {years[0]} under {jsbach_dir}")
expname = re.sub(r'_\d{6}\.01_veg$', '', os.path.basename(candidates[0]))
print(f"  expname: {expname}")

files = []
for y in years:
    for m in range(1, 13):
        f = os.path.join(jsbach_dir, f'{expname}_{y:04d}{m:02d}.01_veg')
        if os.path.isfile(f):
            files.append(f)
print(f"  {len(files)} monthly _veg files")

# Use act_fpc (code 31, _veg stream) — the realised Foliar Projective
# Cover per PFT from the dynamic-vegetation module DYNVEG. Earlier
# attempts used cover_fract (sums to 1 over PFTs in any vegetated
# gridbox → wrong PFT in deserts) or box_veg_ratio (allocated, not
# realised cover); act_fpc directly reflects what plants actually grew.
def _read_veg_field(code, name):
    arg = (f"-remapcon,r360x180 -chname,var{code},{name} -selcode,{code} -timmean "
           f"-mergetime [ {' '.join(files)} ]")
    return np.squeeze(np.asarray(cdo.copy(input=arg, returnArray=name)))

print("  Running cdo (act_fpc)...")
arr = _read_veg_field(31, 'act_fpc')           # (11 PFTs, lat, lon)
print(f"  act_fpc shape: {arr.shape}")
n_pfts = arr.shape[0]

print("  Running cdo (desert_fpc + bare_fpc)...")
desert_fpc = _read_veg_field(34, 'desert_fpc') # (lat, lon)
bare_fpc   = _read_veg_field(35, 'bare_fpc')   # (lat, lon)
bare_total = desert_fpc + bare_fpc
print(f"  bare+desert max={np.nanmax(bare_total):.3f}, mean={np.nanmean(bare_total):.3f}")

# Trim labels/colors to actual PFT count if config differs.
if n_pfts != len(JSBACH_PFTS):
    print(f"  WARN: file has {n_pfts} PFTs, defaults assume {len(JSBACH_PFTS)}; using PFT_<i> labels")
    pft_labels = [f'PFT_{i+1}' for i in range(n_pfts)]
    pft_colors = (PFT_COLORS * ((n_pfts // len(PFT_COLORS)) + 1))[:n_pfts]
else:
    pft_labels = JSBACH_PFTS
    pft_colors = PFT_COLORS

# Dominant PFT is the argmax over the PFT axis of act_fpc; cells are
# tagged Barren when the TOTAL realised vegetated cover (sum of act_fpc
# across PFTs) is below threshold — a cell with 60% vegetated cover
# stays in its dominant-PFT class even if (desert_fpc + bare_fpc) is
# also non-trivial. Ocean cells (where all per-PFT FPCs are 0 or NaN)
# get their own mask so argmax doesn't paint them as TrBE.
total_veg = np.where(np.isfinite(arr), arr, 0).sum(axis=0)
is_ocean = ~np.isfinite(arr).any(axis=0) | (total_veg <= 0)
dominant = arr.argmax(axis=0).astype(np.float32)
# BARREN_THRESHOLD reinterpreted: cells with less than this fraction of
# the gridbox covered by vegetation are Barren.
dominant[total_veg < BARREN_THRESHOLD] = n_pfts  # barren bin
n_ocean_bin = n_pfts + 1
dominant[is_ocean] = n_ocean_bin

labels = pft_labels + ['Barren', 'Ocean']
colors_list = pft_colors + ['#bdbdbd', '#cfe6f5']
cmap = mpl.colors.ListedColormap(colors_list)
bounds = np.arange(len(labels) + 1) - 0.5
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

lons = np.arange(0.5, 360, 1.0)
lats = np.arange(-89.5, 90, 1.0)

fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())
ax.set_global()
ax.coastlines(resolution='110m', linewidth=0.5)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
cf = ax.pcolormesh(lons, lats, dominant, cmap=cmap, norm=norm,
                   transform=ccrs.PlateCarree(), shading='auto')

# Legend
patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors_list, labels)]
ax.legend(handles=patches, loc='lower left', fontsize=8, ncol=2, frameon=True)
ax.set_title(f'{model_version} — dominant JSBACH PFT '
             f'({years.start}-{years.stop - 1} mean)')

ofile = os.path.join(out_path, 'jsbach_dominant_PFT.png')
plt.savefig(ofile, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"  saved {ofile}")

update_status(SCRIPT_NAME, " Completed")
