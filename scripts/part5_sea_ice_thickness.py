# Add the parent directory to sys.path and load config
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bg_routines.config_loader import *
from bg_routines.ipcc_cmaps import get_abs_cmap
import matplotlib.tri as mtri
from pyproj import Proj, Transformer

SCRIPT_NAME = os.path.basename(__file__)
print(SCRIPT_NAME)
update_status(SCRIPT_NAME, " Started")

# ── settings ─────────────────────────────────────────────────────────────────
variable = 'm_ice'
input_paths = [historic_path, pi_ctrl_path]
input_names = [historic_name, pi_ctrl_name]

_clim_window = globals().get('clim_window_years', 25)
years_per_path = {
    historic_name: range(historic_end - (_clim_window - 1), historic_end + 1),
    pi_ctrl_name:  range(pi_ctrl_end  - (_clim_window - 1), pi_ctrl_end  + 1),
}

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    return colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))

new_cmap = truncate_colormap(get_abs_cmap('m_ice'), 0.15, 1)

# ── BENCHMARK: native-mesh approach ──────────────────────────────────────────
t_total = time.time()

# ── 1. Build triangulation once from the already-loaded mesh ─────────────────
t0 = time.time()
lon_nodes = mesh.x2
lat_nodes = mesh.y2
elem = mesh.elem  # (n_tri, 3), 0-based

# Mask triangles that span the dateline (>180° lon range)
tri_lons = lon_nodes[elem]
dateline_mask = (tri_lons.max(axis=1) - tri_lons.min(axis=1)) > 180.0

# Latitude of the map edge per hemisphere (set_extent below uses the same values)
edge_lat = {'NH': 50.0, 'SH': -55.0}

def _build_triang(lon, lat, elem, bad_base, cartopy_proj, lat_edge):
    """Pre-project nodes via pyproj (vectorised C) for fast draw-time rendering.
    Uses proj4_init string directly to avoid slow PROJ authority DB lookup.

    Only triangles with a node inside the square map frame are kept. The frame is
    the box around the latitude circle lat_edge; handing matplotlib the whole globe
    (most of it projected far outside the axes) cost about 90 s per figure against
    about 10 s for the visible part."""
    src = Proj("epsg:4326")
    tgt = Proj(cartopy_proj.proj4_init)
    t = Transformer.from_proj(src, tgt, always_xy=True)
    x, y = t.transform(lon, lat)
    finite = np.isfinite(x) & np.isfinite(y)
    half = 1.02 * abs(t.transform(0.0, lat_edge)[1])
    inside = finite & (np.abs(x) <= half) & (np.abs(y) <= half)
    keep = ~bad_base & finite[elem].all(axis=1) & inside[elem].any(axis=1)
    x = np.where(finite, x, 0.0)
    y = np.where(finite, y, 0.0)
    return mtri.Triangulation(x, y, triangles=elem[keep]), keep

proj_nh = ccrs.NorthPolarStereo()
proj_sh = ccrs.SouthPolarStereo()
triang, tri_keep = {}, {}
for _h, _p in (('NH', proj_nh), ('SH', proj_sh)):
    triang[_h], tri_keep[_h] = _build_triang(lon_nodes, lat_nodes, elem, dateline_mask, _p, edge_lat[_h])
print(f"[BENCH] Triangulation built: {time.time()-t0:.1f}s  "
      f"({elem.shape[0]:,} triangles, {dateline_mask.sum():,} dateline-masked, "
      f"drawn NH {tri_keep['NH'].sum():,} / SH {tri_keep['SH'].sum():,})")

# ── 2. Load monthly data on native mesh (no remap) ───────────────────────────
def load_month_native(variable, exp_path, years, month, meshpath, mesh_file):
    """CDO selmon on native grid — no remap, just month extraction."""
    files = [f"{exp_path}/fesom/{variable}.fesom.{y}.nc" for y in years]
    existing = [p for p in files if os.path.exists(p)]
    if not existing:
        return None
    raw = cdo.timmean(
        input=(
            f"-selmon,{month} "
            f"-setgrid,{meshpath}/{mesh_file} "
            f"-cat [ {' '.join(existing)} ]"
        ),
        returnArray=variable,
    )
    return np.squeeze(raw)  # (n_nodes,)

data_mean = OrderedDict()
t0 = time.time()
for exp_path, exp_name in zip(input_paths, input_names):
    t1 = time.time()
    yrs = list(years_per_path[exp_name])
    mar = load_month_native(variable, exp_path, yrs, 3, meshpath, mesh_file)
    sep = load_month_native(variable, exp_path, yrs, 9, meshpath, mesh_file)
    data_mean[exp_name] = {
        'March':     np.nan_to_num(mar, 0.0) if mar is not None else np.zeros(len(lon_nodes)),
        'September': np.nan_to_num(sep, 0.0) if sep is not None else np.zeros(len(lon_nodes)),
    }
    print(f"[BENCH] {exp_name}: loaded {len(yrs)} years in {time.time()-t1:.1f}s")

print(f"[BENCH] All data loaded in {time.time()-t0:.1f}s")

# ── 3. Plot model data on native mesh ────────────────────────────────────────
t0 = time.time()
for seas in ['March', 'September']:
    for hemi in ['NH', 'SH']:
        for exp_name in input_names:
            values = data_mean[exp_name][seas]

            fig = plt.figure(figsize=(6, 6))
            if hemi == 'SH':
                levels = [0.1,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]
                proj   = proj_sh
                ax     = plt.axes(projection=proj)
                ax.set_extent([-180, 180, edge_lat['SH'], -90], ccrs.PlateCarree())
            else:
                levels = [0.1,0.5,1,1.5,2,2.5,3,3.5,4]
                proj   = proj_nh
                ax     = plt.axes(projection=proj)
                ax.set_extent([-180, 180, edge_lat['NH'], 90], ccrs.PlateCarree())

            # tripcolor on pre-projected native mesh — transform=proj means
            # "data coords are already in proj space, don't re-project"
            tc = ax.tripcolor(triang[hemi], values, cmap=new_cmap,
                              vmin=levels[0], vmax=levels[-1], shading='flat',
                              transform=proj, zorder=1)

            ax.set_title(f"{exp_name}\n{seas} {hemi} sea ice thickness",
                         fontsize=13, fontweight='bold')
            cb = plt.colorbar(tc, ax=ax, orientation='horizontal',
                              fraction=0.046, pad=0.04)
            cb.set_label('m', size=12)
            cb.set_ticks(levels)
            cb.ax.tick_params(labelsize=11)

            ax.add_feature(cfeature.NaturalEarthFeature(
                'physical', 'land', '50m',
                edgecolor='face', facecolor='lightgrey'), zorder=3)
            ax.coastlines(resolution='50m', color='black', linewidth=1, zorder=4)
            ax.gridlines(linewidth=0.5, color='gray', alpha=0.3, linestyle='--')
            plt.tight_layout()

            tag = 'historic' if exp_name == historic_name else 'pi-control'
            plt.savefig(out_path + f"{tag}_{seas}_{hemi}_sea_ice_thickness.png",
                        dpi=300, bbox_inches='tight')
            plt.close()

print(f"[BENCH] Model plots rendered in {time.time()-t0:.1f}s")

# ── 4. GIOMAS reference (kept on regular grid — data is regular) ──────────────
import cmocean as cmo
new_cmap = truncate_colormap(get_abs_cmap('m_ice'), 0.15, 1)

path = observation_path + '/GIOMAS/GIOMAS_heff_miss_time_mon.nc'
if os.path.exists(path):
    t0 = time.time()
    intermediate = xr.open_mfdataset(path, combine='by_coords',
                                     engine='netcdf4', use_cftime=True)
    giomas = intermediate.compute()
    x_g = np.asarray(giomas.lon_scaler).flatten()
    y_g = np.asarray(giomas.lat_scaler).flatten()

    res = [180, 180]
    lon_g  = np.linspace(0, 360, res[0])
    lat_g  = np.linspace(-90, 90, res[1])
    lon2, lat2 = np.meshgrid(lon_g, lat_g)
    points = np.vstack((x_g, y_g)).T

    sit = []
    for t in tqdm(range(giomas['heff'].shape[0])):
        nn = NearestNDInterpolator(points,
             np.nan_to_num(np.asarray(giomas['heff'][t]).flatten(), 0))
        sit.append(nn((lon2, lat2)))
    sit = np.asarray(sit)
    print(f"[BENCH] GIOMAS loaded+interpolated in {time.time()-t0:.1f}s")

    for seas in ['March', 'September']:
        nseas = 2 if seas == 'March' else 8
        for hemi in ['NH', 'SH']:
            data_nonan = np.nan_to_num(sit[nseas], 0)
            fig = plt.figure(figsize=(6, 6))
            if hemi == 'SH':
                levels = [0.1,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]
                ax = plt.axes(projection=ccrs.SouthPolarStereo())
                ax.set_extent([-180, 180, -55, -90], ccrs.PlateCarree())
            else:
                levels = [0.1,0.5,1,1.5,2,2.5,3,3.5,4]
                ax = plt.axes(projection=ccrs.NorthPolarStereo())
                ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())

            imf = ax.contourf(lon2, lat2, data_nonan, cmap=new_cmap,
                              levels=levels, extend='both',
                              transform=ccrs.PlateCarree(), zorder=1)
            ax.contour(lon2, lat2, data_nonan, levels=levels,
                       colors='black', linewidths=0.5,
                       transform=ccrs.PlateCarree(), zorder=2)
            ax.set_title(f"GIOMAS {seas} {hemi} sea ice thickness",
                         fontsize=13, fontweight='bold')
            cb = plt.colorbar(imf, orientation='horizontal', ticks=levels,
                              fraction=0.046, pad=0.04)
            cb.set_label('m', size=12)
            cb.ax.tick_params(labelsize=11)
            ax.add_feature(cfeature.NaturalEarthFeature(
                'physical', 'land', '50m',
                edgecolor='face', facecolor='lightgrey'), zorder=3)
            ax.coastlines(resolution='50m', color='black', linewidth=1, zorder=4)
            ax.gridlines(linewidth=0.5, color='gray', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(out_path + f"GIOMAS_{seas}_{hemi}_sea_ice_thickness.png",
                        dpi=300, bbox_inches='tight')
            plt.close()
else:
    print(f"GIOMAS file not found, skipping: {path}")

print(f"[BENCH] Total wall time: {time.time()-t_total:.1f}s")
update_status(SCRIPT_NAME, " Completed")
