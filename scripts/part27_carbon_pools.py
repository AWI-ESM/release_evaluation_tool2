# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bg_routines.config_loader import *

SCRIPT_NAME = os.path.basename(__file__)  # Get the current script name

print(SCRIPT_NAME)

# Mark as started
update_status(SCRIPT_NAME, " Started")

"""Carbon pools of the coupled system: ocean (REcoM), land (LPJ-GUESS), atmosphere (CO2 tracer).

For every pool whose output exists, over spinup_start..spinup_end:
  ocean       dissolved inorganic carbon and, where written, dissolved and particulate organic
              carbon, integrated over the ocean volume [PgC]
  land        vegetation, litter and soil carbon of LPJ-GUESS, integrated over the land [PgC]
  atmosphere  mass of the prognostic CO2 tracer, integrated over the pressure levels [PgC]; in
              concentration-driven runs, where the tracer is not maintained, the prescribed CO2
              that the ocean sees (xCO2atm of FESOM-REcoM)

Writes
  carbon_pools_timeseries.png  each pool as the change since the first year, the pools side by
                               side, and their sum, with the drift of the last 30 years
  carbon_pools_maps.png        per pool the inventory per square metre in the climatology window
                               (pi_ctrl_start..pi_ctrl_end) and its change since the first five years

A pool without output is left out; the script fails only if none of the three is there.
"""

import glob
import re
from joblib import Parallel, delayed

PGC_PER_MMOL = 12.011e-3 / 1.0e15          # mmol C -> Pg C
KGC_PER_MMOL = 12.011e-6                   # mmol C -> kg C
M_AIR, M_CO2, M_C = 28.9647, 44.0095, 12.011
RE, GRAV = 6371229.0, 9.80665
PGC_PER_PPM = 5.1352e18 / (M_AIR * 1e-3) * 1e-6 * M_C * 1e-3 / 1e12   # dry air mass of the atmosphere -> 2.13 PgC per ppm
OCEAN_ORGANIC = ['DOC', 'PhyC', 'DiaC', 'HetC', 'DetC', 'Zoo2C', 'Zoo3C', 'idetz2c']
N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
YEARS = list(range(spinup_start, spinup_end + 1))
CLIM = (pi_ctrl_start, pi_ctrl_end)
FIRST = (spinup_start, min(spinup_start + 4, spinup_end))
CACHE = os.path.join(os.path.dirname(out_path.rstrip('/')), 'spinup_cache', model_version)
os.makedirs(CACHE, exist_ok=True)


# --------------------------------------------------------------------------------------------
# Ocean
# --------------------------------------------------------------------------------------------
def fesom_file(var, year):
    return os.path.join(spinup_path, 'fesom', f'{var}.fesom.{year}.nc')


def ocean_geometry():
    """Volume of every (node, layer) cell and the node coordinates, from the mesh diagnostics."""
    md = xr.open_dataset(os.path.join(meshpath, 'fesom.mesh.diag.nc'))
    zi = np.abs(md['nz'].values)
    dz = np.diff(zi)
    vol = (md['nod_area'].values[:len(dz)] * dz[:, None]).T        # (nod2, nz1)
    lon = ((md['lon'].values + 180.0) % 360.0) - 180.0
    return vol, dz, lon, md['lat'].values


def ocean_year(year, vol, dz):
    """Global inventories [PgC] and the DIC column inventory [kgC m-2] of one year (cached)."""
    fn = os.path.join(CACHE, f'carbon_ocean_{year}.npz')
    if os.path.isfile(fn):
        c = np.load(fn)
        return float(c['dic']), float(c['org']), c['col']
    with xr.open_dataset(fesom_file('DIC', year)) as ds:
        dic = ds['DIC'].mean('time').values.astype(np.float64)       # mmol m-3, (nod2, nz1)
    wet = np.isfinite(dic) & (dic > 0)
    dic = np.where(wet, dic, 0.0)
    total = float((dic * vol).sum()) * PGC_PER_MMOL
    col = (dic * dz[None, :]).sum(1) * KGC_PER_MMOL                  # kgC m-2
    org = 0.0
    for v in OCEAN_ORGANIC:
        f = fesom_file(v, year)
        if not os.path.isfile(f):
            continue
        with xr.open_dataset(f) as ds:
            x = ds[v].mean('time').values.astype(np.float64)
        if x.ndim != 2 or x.shape != dic.shape:                      # vertically integrated diagnostics are skipped
            continue
        org += float((np.where(wet & np.isfinite(x), x, 0.0) * vol).sum()) * PGC_PER_MMOL
    np.savez(fn, dic=total, org=org, col=col.astype(np.float32))
    return total, org, col


def ocean_pool():
    ys = [y for y in YEARS if os.path.isfile(fesom_file('DIC', y))]
    if not ys:
        print('ocean: no DIC output, pool left out')
        return None
    vol, dz, lon, lat = ocean_geometry()
    res = Parallel(n_jobs=N_JOBS)(delayed(ocean_year)(y, vol, dz) for y in tqdm(ys, desc='ocean'))
    ts = pd.DataFrame({'DIC': [r[0] for r in res], 'organic': [r[1] for r in res]}, index=ys)
    col = {y: r[2] for y, r in zip(ys, res)}
    mean = lambda a, b: np.mean([col[y] for y in ys if a <= y <= b], axis=0)
    # nearest ocean node on a 1 degree grid, blank where the nearest node is far (land)
    glon, glat = np.meshgrid(np.arange(-179.5, 180, 1.0), np.arange(-89.5, 90, 1.0))
    xyz = lambda lo, la: np.column_stack([np.cos(np.deg2rad(la)) * np.cos(np.deg2rad(lo)),
                                          np.cos(np.deg2rad(la)) * np.sin(np.deg2rad(lo)), np.sin(np.deg2rad(la))])
    dist, idx = cKDTree(xyz(lon, lat)).query(xyz(glon.ravel(), glat.ravel()))
    far = (dist > 2.0 * np.deg2rad(1.0)).reshape(glon.shape)
    grid = lambda f: np.where(far, np.nan, f[idx].reshape(glon.shape))
    return dict(ts=ts, lon=glon, lat=glat, clim=grid(mean(*CLIM)), change=grid(mean(*CLIM) - mean(*FIRST)),
                unit='kgC m$^{-2}$', label='ocean: DIC column inventory')


# --------------------------------------------------------------------------------------------
# Land
# --------------------------------------------------------------------------------------------
def lpjg_dir():
    base = lpjg_path.rstrip('/')
    for d in (os.path.join(base, 'lpj_guess'), base):
        if os.path.isdir(d):
            return d
    return None


def octahedral_cell_area(lat_deg):
    """Cell area [m2] of points of an octahedral reduced Gaussian grid (TCO), from their latitudes.

    The number of latitude rows per hemisphere follows from the row spacing; row j counted from
    the pole has 20 + 4 (j - 1) points and shares its Gaussian weight equally among them.
    """
    rows_ = np.unique(np.round(lat_deg, 4))
    n = int(round(90.0 / np.min(np.diff(rows_))))                    # rows per hemisphere
    xg, wg = np.polynomial.legendre.leggauss(2 * n)
    glat_ = np.rad2deg(np.arcsin(xg))
    j = np.abs(glat_[None, :] - np.asarray(lat_deg)[:, None]).argmin(1)
    from_pole = np.minimum(j, 2 * n - 1 - j) + 1
    nlon = 20 + 4 * (from_pole - 1)
    return 4.0 * np.pi * RE ** 2 * (wg[j] / 2.0) / nlon


def land_pool():
    """LPJ-GUESS carbon pools from the NetCDF files of lpjg2nc (cpool*.nc per leg) or, for runs
    that keep the text tables, from cpool.out."""
    d = lpjg_dir()
    if d is None:
        print('land: no LPJ-GUESS directory, pool left out')
        return None
    legs = sorted(x for x in glob.glob(os.path.join(d, '*')) if os.path.isdir(x) and re.match(r'^\d{8}-\d{8}$', os.path.basename(x)))
    glon, glat = np.meshgrid(np.arange(-179.5, 180, 1.0), np.arange(-89.5, 90, 1.0))
    la = np.deg2rad(glat[:, 0])
    cell = (RE ** 2 * np.deg2rad(1.0) * (np.sin(la + np.deg2rad(0.5)) - np.sin(la - np.deg2rad(0.5))))[:, None] + 0 * glon
    rows, maps = {}, {}
    remapped_area = {}       # year -> area of the 1 degree cells that carry a value
    native_area = None       # land area of the model's own points, once a leg provides it

    def from_points(lo, la_, val, y):
        """Totals on the model's own points with their cell areas; map by nearest point."""
        pa = octahedral_cell_area(la_)
        rows[y] = dict(vegetation=float((val['VegC'] * pa).sum()) / 1e12,
                       soil_litter=float(((val['SoilC'] + val['LitterC']) * pa).sum()) / 1e12,
                       total=float((val['Total'] * pa).sum()) / 1e12)
        dist, idx = cKDTree(np.column_stack([lo, la_])).query(np.column_stack([glon.ravel(), glat.ravel()]))
        maps[y] = np.where((dist < 1.0).reshape(glon.shape), val['Total'][idx].reshape(glon.shape), np.nan)
        return float(pa.sum())

    for leg in legs:
        y0 = int(os.path.basename(leg)[:4])
        native = os.path.join(leg, 'cpool.nc')                      # lpjg2nc --nc_keep_unstruc
        nc = sorted(glob.glob(os.path.join(leg, 'cpool*remap*.nc')))
        if os.path.isfile(native):
            with xr.open_dataset(native, decode_times=False) as ds:
                ds = ds.load()
            lo = ((np.ravel(ds['lon'].values) + 180.0) % 360.0) - 180.0
            la_ = np.ravel(ds['lat'].values)
            for i in range(ds.sizes['time']):
                val = {v: np.ravel(ds[v].values[i]) for v in ('VegC', 'SoilC', 'LitterC', 'Total')}
                ok = np.isfinite(val['Total'])
                native_area = from_points(lo[ok], la_[ok], {v: x[ok] for v, x in val.items()}, y0 + i)
        elif nc:
            with xr.open_dataset(nc[0], decode_times=False) as ds:
                ds = ds.load()
            lon = ((ds['lon'].values + 180.0) % 360.0) - 180.0
            order = np.argsort(lon)
            flip = ds['lat'].values[0] > ds['lat'].values[-1]
            def get(v, i):
                x = ds[v].values[i][:, order]
                return x[::-1] if flip else x
            if get('Total', 0).shape != glon.shape:
                print(f'land: {nc[0]} is not on the 1 degree grid, leg skipped')
                continue
            for i in range(ds.sizes['time']):
                y = y0 + i
                tot = get('Total', i)
                rows[y] = dict(vegetation=np.nansum(get('VegC', i) * cell) / 1e12,
                               soil_litter=np.nansum((get('SoilC', i) + get('LitterC', i)) * cell) / 1e12,
                               total=np.nansum(tot * cell) / 1e12)
                maps[y] = tot
                remapped_area[y] = float(cell[np.isfinite(tot)].sum())
        elif os.path.isfile(os.path.join(leg, 'cpool.out')):
            t = pd.read_csv(os.path.join(leg, 'cpool.out'), sep=r'\s+')
            for y, q in t.groupby('Year'):
                lo = ((q['Lon'].values + 180.0) % 360.0) - 180.0
                native_area = from_points(lo, q['Lat'].values, {v: q[v].values for v in ('VegC', 'SoilC', 'LitterC', 'Total')}, int(y))
    # The conservative remap spreads land values into neighbouring coast and ice-edge cells, so the
    # 1 degree files cover more area than the model's land (about 12 % at TCO95) and their totals
    # are too large by that factor. Where a leg on the model's own points gives the true land
    # area, the remapped totals are scaled to it; otherwise they are left and flagged.
    if remapped_area:
        if native_area:
            for y, a_ in remapped_area.items():
                for k in rows[y]:
                    rows[y][k] *= native_area / a_
            print(f'land: totals of {len(remapped_area)} remapped years scaled to the land area of the native points ({native_area:.4e} m2)')
        else:
            print('land: WARNING only remapped files, totals include the remap spread into coast cells (too large by about 10 %)')
    ys = [y for y in YEARS if y in rows]
    if not ys:
        print('land: no cpool output in the requested years, pool left out')
        return None
    ts = pd.DataFrame([rows[y] for y in ys], index=ys)
    mean = lambda a, b: np.nanmean([maps[y] for y in ys if a <= y <= b], axis=0)
    return dict(ts=ts, lon=glon, lat=glat, clim=mean(*CLIM), change=mean(*CLIM) - mean(*FIRST),
                unit='kgC m$^{-2}$', label='land: vegetation, litter and soil carbon')


# --------------------------------------------------------------------------------------------
# Atmosphere
# --------------------------------------------------------------------------------------------
def co2_file(year):
    for pat in (f'atm_remapped_1m_pl_co2_1m_pl_{year}-{year}.nc', f'atm_remapped_1m_pl_co2_{year}-{year}.nc'):
        f = os.path.join(spinup_path, 'oifs', pat)
        if os.path.isfile(f):
            return f
    return None


def atmosphere_year(year):
    """Tracer mass [PgC], mass-weighted mean mixing ratio [ppm] and the column-mean map [ppm]."""
    with xr.open_dataset(co2_file(year)) as ds:
        x = ds['co2'].mean('time_counter').load()                    # kg/kg, (level, lat, lon)
    lev = [d for d in x.dims if d not in ('lat', 'lon')][0]
    p = x[lev].values.astype(np.float64)
    order = np.argsort(p)
    p, q = p[order], x.values[order].astype(np.float64)
    edges = np.concatenate([[0.0], 0.5 * (p[1:] + p[:-1]), [101325.0]])
    dp = np.diff(edges)                                               # Pa per level, surface at standard pressure
    colmass = (q * dp[:, None, None]).sum(0) / GRAV                   # kg CO2 m-2
    w = np.cos(np.deg2rad(x['lat'].values))[:, None] + 0 * colmass
    area = 4.0 * np.pi * RE ** 2
    mass = float((colmass * w).sum() / w.sum()) * area * M_C / M_CO2 / 1e12
    ppm_map = colmass / (dp.sum() / GRAV) * M_AIR / M_CO2 * 1e6
    ppm = float((ppm_map * w).sum() / w.sum())
    return mass, ppm, ppm_map, x['lon'].values, x['lat'].values


def atmosphere_pool():
    ys = [y for y in YEARS if co2_file(y)]
    if not ys:
        print('atmosphere: no CO2 tracer output, pool left out')
        return None
    res = Parallel(n_jobs=N_JOBS)(delayed(atmosphere_year)(y) for y in tqdm(ys, desc='atmosphere'))
    ts = pd.DataFrame({'CO2': [r[0] for r in res], 'ppm': [r[1] for r in res]}, index=ys)
    lon, lat = res[0][3], res[0][4]
    lon = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    maps = {y: r[2][:, order] for y, r in zip(ys, res)}
    mean = lambda a, b: np.mean([maps[y] for y in ys if a <= y <= b], axis=0)
    glon, glat = np.meshgrid(lon[order], lat)
    # A tracer that leaves 100-2000 ppm is not a carbon pool (in concentration-driven runs nothing
    # maintains it): it is reported, but kept out of the sum and the maps.
    valid = bool(ts['ppm'].min() > 100.0 and ts['ppm'].max() < 2000.0)
    if not valid:
        print(f"atmosphere: tracer between {ts['ppm'].min():.3g} and {ts['ppm'].max():.3g} ppm, not treated as a pool")
    return dict(ts=ts, lon=glon, lat=glat, clim=mean(*CLIM), change=mean(*CLIM) - mean(*FIRST),
                unit='ppm', label='atmosphere: column-mean CO2 tracer', valid=valid)


def prescribed_co2():
    """CO2 mixing ratio the ocean is given [ppm], area mean of xCO2atm of FESOM-REcoM, per year."""
    ys = [y for y in YEARS if os.path.isfile(fesom_file('xCO2atm', y))]
    if not ys:
        return None
    md = xr.open_dataset(os.path.join(meshpath, 'fesom.mesh.diag.nc'))
    area = md['nod_area'].values[0]
    def one(y):
        with xr.open_dataset(fesom_file('xCO2atm', y)) as ds:
            x = ds['xCO2atm'].mean('time').values.astype(np.float64)
        ok = np.isfinite(x) & (x > 0)
        return float((x[ok] * area[ok]).sum() / area[ok].sum())
    ppm = Parallel(n_jobs=N_JOBS)(delayed(one)(y) for y in tqdm(ys, desc='prescribed CO2'))
    return pd.Series(ppm, index=ys)


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------
def drift(series, n=30):
    """Linear trend of the last n years [per year]."""
    s = series.dropna().iloc[-n:]
    if len(s) < 5:
        return np.nan
    return float(np.polyfit(s.index.values, s.values, 1)[0])


def plot_timeseries(pools):
    names = [k for k in ('ocean', 'land', 'atmosphere') if pools.get(k)]
    fig, axes = plt.subplots(len(names) + 1, 1, figsize=(10, 3.0 * (len(names) + 1)), sharex=True)
    total = None
    colors_ = {'ocean': '#1f5fa8', 'land': '#2c7d3a', 'atmosphere': '#b04a2f'}
    for ax, k in zip(axes, names):
        ts = pools[k]['ts']
        if k == 'ocean':
            parts = {'DIC': ts['DIC']}
            if ts['organic'].abs().max() > 0:
                parts['organic carbon'] = ts['organic']
            main = ts['DIC'] + ts['organic']
        elif k == 'land':
            parts = {'vegetation': ts['vegetation'], 'soil and litter': ts['soil_litter']}
            main = ts['total']
        else:
            parts = {}
            main = ts['CO2']
            if not pools[k]['valid']:
                presc = pools[k].get('prescribed')
                ax.set_title('atmosphere: prescribed CO2 (the tracer is not maintained in this run)', fontsize=11, fontweight='bold', loc='left')
                if presc is None:
                    ppm = ts['ppm']
                    ax.text(0.5, 0.5, f'The CO2 tracer is not a physical field in this run\n'
                            f'({ppm.iloc[0]:.3g} ppm in {ppm.index[0]}, {ppm.iloc[-1]:.3g} ppm in {ppm.index[-1]}),\n'
                            'and no prescribed CO2 (xCO2atm) was found.', transform=ax.transAxes, ha='center', va='center', fontsize=10, color='0.25')
                    ax.set_yticks([])
                    continue
                mass = presc * PGC_PER_PPM
                ax.plot(mass.index, mass - mass.iloc[0], color=colors_[k], lw=2.2, label='prescribed CO2 (seen by the ocean)')
                # what the drift of the other pools would do to the atmosphere if it were free
                other = None
                for kk in ('ocean', 'land'):
                    if pools.get(kk):
                        t2 = pools[kk]['ts']
                        m2 = t2['DIC'] + t2['organic'] if kk == 'ocean' else t2['total']
                        other = (m2 - m2.iloc[0]) if other is None else other.add(m2 - m2.iloc[0], fill_value=np.nan)
                if other is not None:
                    implied = -other.dropna()
                    ax.plot(implied.index, implied.values, color=colors_[k], lw=1.2, ls='--', label='carbon released by ocean and land (not added to the atmosphere)')
                    ax2 = ax.twinx()
                    lo, hi = ax.get_ylim()
                    ax2.set_ylim(presc.iloc[0] + lo / PGC_PER_PPM, presc.iloc[0] + hi / PGC_PER_PPM)
                    ax2.set_ylabel('ppm')
                    note = (f'{presc.iloc[0]:.2f} ppm = {mass.iloc[0]:.0f} PgC, held fixed;  ocean and land released {implied.iloc[-1]:+.0f} PgC by {implied.index[-1]}, '
                            f'which would be {implied.iloc[-1] / PGC_PER_PPM:+.0f} ppm')
                else:
                    note = f'{presc.iloc[0]:.2f} ppm = {mass.iloc[0]:.0f} PgC, held fixed'
                ax.axhline(0, color='0.4', lw=0.8)
                ax.text(0.99, 0.10, note, transform=ax.transAxes, fontsize=8.5, color='0.25', ha='right')
                ax.set_ylabel('PgC')
                ax.legend(fontsize=8, loc='upper left', frameon=False)
                ax.grid(alpha=0.3)
                continue
        ax.plot(main.index, main - main.iloc[0], color=colors_[k], lw=2.2, label=f'{k} total')
        for (lab, s), ls in zip(parts.items(), ('--', ':')):
            ax.plot(s.index, s - s.iloc[0], color=colors_[k], lw=1.2, ls=ls, label=lab)
        ax.axhline(0, color='0.4', lw=0.8)
        note = f'{main.iloc[0]:.0f} PgC in {main.index[0]}, {main.iloc[-1]:.0f} PgC in {main.index[-1]};  last 30 years {drift(main):+.2f} PgC/yr'
        if k == 'atmosphere':
            ppm = ts['ppm']
            note += f';  {ppm.iloc[-1]:.1f} ppm'
            if ppm.min() < 100 or ppm.max() > 2000:
                note += '  (tracer outside 100-2000 ppm: not a maintained field)'
        ax.set_title(f'{k}: change since {main.index[0]}', fontsize=11, fontweight='bold', loc='left')
        ax.text(0.01, 0.04, note, transform=ax.transAxes, fontsize=8.5, color='0.25')
        ax.set_ylabel('PgC')
        ax.legend(fontsize=8, loc='upper right', frameon=False)
        ax.grid(alpha=0.3)
        d = (main - main.iloc[0])
        total = d if total is None else total.add(d, fill_value=np.nan)
    ax = axes[-1]
    common = total.dropna()
    for k in names:
        if not pools[k].get('valid', True):
            continue
        ts = pools[k]['ts']
        main = ts['DIC'] + ts['organic'] if k == 'ocean' else ts['total'] if k == 'land' else ts['CO2']
        ax.plot(main.index, main - main.iloc[0], color=colors_[k], lw=1.2, label=k)
    ax.plot(common.index, common.values, color='k', lw=2.2, label='sum of the pools shown')
    ax.axhline(0, color='0.4', lw=0.8)
    ax.set_title('all pools: change since the first year', fontsize=11, fontweight='bold', loc='left')
    ax.text(0.01, 0.04, f'sum, last 30 years {drift(common):+.2f} PgC/yr (years present in every pool; a prescribed atmosphere is not part of the sum)', transform=ax.transAxes, fontsize=8.5, color='0.25')
    ax.set_ylabel('PgC'); ax.set_xlabel('Year')
    ax.legend(fontsize=8, loc='upper right', frameon=False, ncol=4)
    ax.grid(alpha=0.3)
    fig.suptitle(f'{model_version}: carbon pools {YEARS[0]}-{YEARS[-1]}', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path + 'carbon_pools_timeseries.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', out_path + 'carbon_pools_timeseries.png')


def plot_maps(pools):
    names = [k for k in ('ocean', 'land', 'atmosphere') if pools.get(k) and pools[k].get('valid', True)]
    fig = plt.figure(figsize=(6.2 * len(names), 7.6))
    cmaps = {'ocean': 'viridis', 'land': 'YlGn', 'atmosphere': 'magma_r'}
    for j, k in enumerate(names):
        p = pools[k]
        for i, (field, title, cmap) in enumerate(((p['clim'], f'{CLIM[0]}-{CLIM[1]}', cmaps[k]),
                                                  (p['change'], f'{CLIM[0]}-{CLIM[1]} minus {FIRST[0]}-{FIRST[1]}', 'RdBu_r'))):
            ax = fig.add_subplot(2, len(names), i * len(names) + j + 1, projection=ccrs.Robinson())
            ax.set_global()
            finite = field[np.isfinite(field)]
            if i == 0:
                vmin, vmax = np.percentile(finite, [2, 98])
            else:
                vmax = np.percentile(np.abs(finite), 98); vmin = -vmax
            if vmax <= vmin:
                vmax = vmin + 1e-6
            m = ax.pcolormesh(p['lon'], p['lat'], field, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), shading='auto')
            ax.coastlines(resolution='110m', linewidth=0.4)
            ax.set_title(f"{p['label']}\n{title}" if i == 0 else f'change: {title}', fontsize=9.5, fontweight='bold')
            cb = plt.colorbar(m, ax=ax, orientation='horizontal', shrink=0.8, pad=0.04, extend='both')
            cb.set_label(p['unit'], fontsize=9)
    fig.suptitle(f'{model_version}: carbon pools per square metre', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path + 'carbon_pools_maps.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', out_path + 'carbon_pools_maps.png')


try:
    pools = {'ocean': ocean_pool(), 'land': land_pool(), 'atmosphere': atmosphere_pool()}
    if not any(pools.values()):
        raise ValueError('no carbon pool output found (no REcoM DIC, no LPJ-GUESS cpool, no CO2 tracer)')
    # Concentration-driven runs: the tracer is absent or not maintained, the atmosphere is the
    # prescribed CO2.
    atm = pools['atmosphere']
    if atm is None or not atm['valid']:
        presc = prescribed_co2()
        if presc is not None:
            if atm is None:
                atm = dict(ts=pd.DataFrame({'CO2': np.nan, 'ppm': np.nan}, index=presc.index), valid=False)
                pools['atmosphere'] = atm
            atm['prescribed'] = presc
            atm['ts'] = atm['ts'].assign(prescribed_ppm=presc, prescribed_PgC=presc * PGC_PER_PPM)
            print(f'atmosphere: prescribed CO2 {presc.iloc[0]:.2f} to {presc.iloc[-1]:.2f} ppm')
    table = pd.concat({k: p['ts'] for k, p in pools.items() if p}, axis=1)
    table.to_csv(out_path + 'carbon_pools_timeseries.csv', float_format='%.4f')
    print(table.iloc[[0, -1]].T)
    plot_timeseries(pools)
    plot_maps(pools)

    # Mark script as completed
    update_status(SCRIPT_NAME, "Completed")
    print(f"\n{SCRIPT_NAME} completed successfully!")

except Exception as e:
    print(f"Error in {SCRIPT_NAME}: {str(e)}")
    import traceback
    traceback.print_exc()
    update_status(SCRIPT_NAME, f"Failed: {str(e)}")
    raise
