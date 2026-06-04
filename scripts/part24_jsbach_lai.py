"""Annual-mean LAI from JSBACH (replaces part24_lpjg_lai.py for echam6/jsbach).

JSBACH writes LAI as code 107 in the `_jsbach` stream, with 11 PFT levels
on the T63 Gaussian grid (192x96).
"""
import sys
import os
import glob
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bg_routines.config_loader import *
from bg_routines.ipcc_cmaps import get_abs_cmap

SCRIPT_NAME = os.path.basename(__file__)
print(SCRIPT_NAME)
update_status(SCRIPT_NAME, " Started")

jsbach_dir = os.path.join(historic_path, 'jsbach')
years = range(historic_last25y_start, historic_last25y_end + 1)

# Detect expname from filenames in the stream dir. Pattern:
#   <expname>_YYYYMM.01_jsbach
candidates = sorted(glob.glob(os.path.join(jsbach_dir, f'*_{years[0]:04d}01.01_jsbach')))
if not candidates:
    raise FileNotFoundError(f"No _jsbach file for year {years[0]} under {jsbach_dir}")
expname = re.sub(r'_\d{6}\.01_jsbach$', '', os.path.basename(candidates[0]))
print(f"  expname: {expname}")

# Build list of monthly files for all years x months
files = []
for y in years:
    for m in range(1, 13):
        f = os.path.join(jsbach_dir, f'{expname}_{y:04d}{m:02d}.01_jsbach')
        if os.path.isfile(f):
            files.append(f)
if not files:
    raise FileNotFoundError("No _jsbach monthly files in target year range")
print(f"  {len(files)} monthly _jsbach files spanning {years.start}-{years.stop - 1}")

# Climatological mean of LAI (code 107) and box_veg_ratio (code 24,
# vegetated fraction of the grid box per PFT) remapped to r360x180.
# LAI in `_jsbach` is the per-PFT leaf area within the PFT tile (m^2
# leaves per m^2 of that tile), NOT the grid-box mean. Summing raw
# per-PFT LAIs gives ~30-60 (sum of 11 tile values), which is what
# produced the saturated 8+-everywhere plot.
#
# Grid-box LAI = sum_pft(box_veg_ratio_pft * lai_pft)
# where box_veg_ratio already encodes both cover_fract and
# veg_ratio_max, so this collapses to the proper area-weighted total.
def _load_pft_field(code, name):
    arg = (f"-remapcon,r360x180 -chname,var{code},{name} -selcode,{code} -timmean "
           f"-mergetime [ {' '.join(files)} ]")
    field = cdo.copy(input=arg, returnArray=name)
    return np.squeeze(np.asarray(field))

print("  Running cdo (lai)...")
lai_pft = _load_pft_field(107, 'lai')
print(f"  lai_pft shape: {lai_pft.shape}")
print("  Running cdo (box_veg_ratio)...")
veg_pft = _load_pft_field(24, 'box_veg_ratio')
print(f"  box_veg_ratio shape: {veg_pft.shape}")

total_lai = (lai_pft * veg_pft).sum(axis=0)
total_lai = np.ma.masked_where(total_lai <= 0, total_lai)

lons = np.arange(0.5, 360, 1.0)
lats = np.arange(-89.5, 90, 1.0)

fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())
ax.set_global()
ax.coastlines(resolution='110m', linewidth=0.5)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
cf = ax.pcolormesh(lons, lats, total_lai, cmap=get_abs_cmap('lai'), vmin=0, vmax=8,
                   transform=ccrs.PlateCarree(), shading='auto')
cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
cbar.set_label('LAI [m²/m²]')
ax.set_title(f'{model_version} — total LAI '
             f'({years.start}-{years.stop - 1} mean)')

ofile = os.path.join(out_path, 'jsbach_LAI_total.png')
plt.savefig(ofile, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"  saved {ofile}")

update_status(SCRIPT_NAME, " Completed")
