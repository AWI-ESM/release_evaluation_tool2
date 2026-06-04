"""Total terrestrial carbon stock from JSBACH yasso pools.

The `_yasso` stream carries 18 box-mean carbon pools (kg-mol C per
gridbox): {acid,water,ethanol,nonsoluble}_{ag,bg}{1,2} plus humus_{1,2}.
Total terrestrial C is the sum across all 18 pools.
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

candidates = sorted(glob.glob(os.path.join(jsbach_dir, f'*_{years[0]:04d}01.01_yasso')))
if not candidates:
    raise FileNotFoundError(f"No _yasso file for year {years[0]} under {jsbach_dir}")
expname = re.sub(r'_\d{6}\.01_yasso$', '', os.path.basename(candidates[0]))
print(f"  expname: {expname}")

# Codes 31-39 are the _1 (woody) series, 41-49 the _2 (non-woody) series.
YASSO_CODES = list(range(31, 40)) + list(range(41, 50))
print(f"  summing {len(YASSO_CODES)} yasso pool codes")

files = []
for y in years:
    for m in range(1, 13):
        f = os.path.join(jsbach_dir, f'{expname}_{y:04d}{m:02d}.01_yasso')
        if os.path.isfile(f):
            files.append(f)
if not files:
    raise FileNotFoundError("No _yasso monthly files in target year range")
print(f"  {len(files)} monthly _yasso files")

codes_arg = ','.join(str(c) for c in YASSO_CODES)
# Pull all 18 yasso pool variables, timmean'd and remapped, into one
# NetCDF file. We then sum across variables (18 pool types) and the PFT
# level axis (11 PFTs per variable) in numpy. (cdo's `enssum` operates
# across *files*, not across variables within a file, so applying it
# here was a no-op — the output ended up with 18 separate variables.)
input_arg = (
    f"-remapcon,r360x180 -timmean -selcode,{codes_arg} "
    f"-mergetime [ {' '.join(files)} ]"
)
print("  Running cdo...")
# Explicit output path: cdo.copy with only `input=` returns a
# CdoTempfileStore-managed filename whose finalizer can delete the file
# before we read it. `-f nc` forces NetCDF (otherwise we'd get GRIB,
# jsbach's native format).
tmp_nc = os.path.join(out_path, '_part25_total_C.nc')
cdo.copy(input=input_arg, output=tmp_nc, options='-O -f nc')
with Dataset(tmp_nc) as ds:
    _meta = {'lon', 'lat', 'time', 'time_bnds', 'lat_bnds', 'lon_bnds',
             'mlev', 'lev', 'plev', 'belowsurface'}
    pool_vars = [v for v in ds.variables if v not in _meta]
    print(f"  summing {len(pool_vars)} variables x 11 PFTs to total C")
    arr = None
    for v in pool_vars:
        # Each var has shape (time=1, level=11, lat, lon)
        a = np.squeeze(np.asarray(ds.variables[v][:], dtype=np.float64))
        # Reduce PFT axis (level=11) to per-gridbox sum
        if a.ndim == 3:
            a = a.sum(axis=0)
        arr = a if arr is None else (arr + a)
print(f"  total C shape: {arr.shape}")

# Units: mol(C) m-2(grid box). Convert to kg(C) m-2 for plotting.
arr_kgm2 = arr * 12.011 / 1000.0
arr_kgm2 = np.ma.masked_where(arr_kgm2 <= 0, arr_kgm2)

lons = np.arange(0.5, 360, 1.0)
lats = np.arange(-89.5, 90, 1.0)

fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())
ax.set_global()
ax.coastlines(resolution='110m', linewidth=0.5)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
cf = ax.pcolormesh(lons, lats, arr_kgm2, cmap=get_abs_cmap('carbon'), vmin=0,
                   vmax=np.nanpercentile(arr_kgm2.compressed(), 99),
                   transform=ccrs.PlateCarree(), shading='auto')
cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
cbar.set_label('Total soil C [kg(C) m⁻²]')
ax.set_title(f'{model_version} — yasso total terrestrial carbon '
             f'({years.start}-{years.stop - 1} mean)')

ofile = os.path.join(out_path, 'jsbach_carbon_total.png')
plt.savefig(ofile, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"  saved {ofile}")

# Global total in PgC for the print summary.
# Assumes 1x1 degree grid cell area = cos(lat) * (1 deg^2 of earth surface).
# Use a coarse approximation: cell_area_km2 = 111.32**2 * cos(lat_rad)
lat_rad = np.deg2rad(lats)
cell_area_m2 = (111320.0 ** 2) * np.cos(lat_rad)[:, None] * np.ones((1, len(lons)))
total_kg = np.sum(np.where(arr_kgm2.mask, 0, arr_kgm2) * cell_area_m2)
total_pgc = total_kg / 1e12
print(f"  global terrestrial C ~= {total_pgc:.1f} PgC")

update_status(SCRIPT_NAME, " Completed")
