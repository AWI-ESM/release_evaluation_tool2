"""Tools for long spin-up records that are stitched from several experiments.

Two things the spin-up plots need once the record is not one experiment on one machine:

* a per-year CACHE for the diagnostics that read 3D ocean fields (the global-mean
  temperature profile of the Hovmoeller diagram, the AMOC index). A year that is in
  the cache is never recomputed, so years whose 3D output lives on another machine
  can be precomputed there (scripts/precompute_spinup_cache.py) and only the cache
  shipped. Each entry is a small .npy named <kind>_<year>.npy.

* ANNOTATIONS: a config list ``spinup_annotations = [(year, 'label'), ...]`` drawn as
  dotted vertical lines with rotated labels, and an optional shaded evaluation window.

This module deliberately does not import the reval config, so the standalone
precompute script can use it with nothing but a mesh directory and an output folder.
"""
import os
import numpy as np


# ----------------------------------------------------------------------------
# cache
# ----------------------------------------------------------------------------
def cache_file(cache_dir, kind, year):
    return os.path.join(cache_dir, f"{kind}_{int(year)}.npy")


def cache_get(cache_dir, kind, year):
    """Cached array for one year, or None."""
    if not cache_dir:
        return None
    fn = cache_file(cache_dir, kind, year)
    if os.path.exists(fn):
        return np.load(fn)
    return None


def cache_put(cache_dir, kind, year, arr):
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    tmp = cache_file(cache_dir, kind, year) + f".tmp{os.getpid()}.npy"
    np.save(tmp, np.asarray(arr))
    os.replace(tmp, cache_file(cache_dir, kind, year))


# ----------------------------------------------------------------------------
# global-mean profile of a 3D FESOM tracer (Hovmoeller input)
# ----------------------------------------------------------------------------
def hovm_profile(cdo, fesom_dir, variable, year, meshpath, mesh_file):
    """Annual global-mean profile, exactly the operator chain part3 has always used:
    yearmean -fldmean -setctomiss,0 -setgrid,<mesh>. Returns a 1D array (depth)."""
    path = f"{fesom_dir}/{variable}.fesom.{year}.nc"
    if not os.path.exists(path):
        return None
    raw = cdo.yearmean(
        input=f"-fldmean -setctomiss,0 -setgrid,{meshpath}/{mesh_file} {path}",
        returnArray=variable,
    )
    a = np.squeeze(np.asarray(raw))
    while a.ndim > 1:
        a = np.nanmean(a, axis=0)
    return a.astype(np.float32)


def hovm_profiles(cdo, fesom_dir, variable, years, meshpath, mesh_file, cache_dir=None):
    """Profiles for all years that are cached or computable. Returns (years, array)."""
    kind = f"hovm_{variable}"
    got_years, rows = [], []
    for y in years:
        a = cache_get(cache_dir, kind, y)
        if a is None:
            a = hovm_profile(cdo, fesom_dir, variable, y, meshpath, mesh_file)
            if a is None:
                continue
            cache_put(cache_dir, kind, y, a)
        got_years.append(int(y))
        rows.append(np.asarray(a, dtype=np.float32))
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)
    n = min(len(r) for r in rows)
    return got_years, np.array([r[:n] for r in rows], dtype=np.float32)


# ----------------------------------------------------------------------------
# AMOC index
# ----------------------------------------------------------------------------
AMOC_FIELDS = ("amoc_26n", "amoc_max_20_60n", "lat_of_max", "depth_of_max_26n")


def amoc_index(pf, mesh, fesom_dir, year, min_depth=500.0, nlats=200):
    """Atlantic overturning for one year from the annual-mean vertical velocity, with
    the routine part17 uses for the section plot (pyfesom2 xmoc_data, 'Atlantic_MOC').

    Returns [maximum at 26.5N below min_depth, maximum over 20-60N below min_depth,
    latitude of that maximum, depth of the 26.5N maximum], in Sv / degrees / metres.
    """
    if not os.path.exists(f"{fesom_dir}/w.fesom.{year}.nc"):
        return None
    w = pf.get_data(fesom_dir, 'w', [year], mesh, how='mean', compute=True, silent=True)
    lats, moc = pf.xmoc_data(mesh, w, nlats=nlats, mask='Atlantic_MOC')
    moc = np.asarray(moc)
    z = np.abs(np.asarray(mesh.zlev))[: moc.shape[1]]
    deep = z >= min_depth
    j26 = int(np.argmin(np.abs(lats - 26.5)))
    col = moc[j26, deep]
    band = (lats >= 20) & (lats <= 60)
    sub = moc[band][:, deep]
    jj = np.unravel_index(int(np.nanargmax(sub)), sub.shape)
    return np.array([np.nanmax(col), np.nanmax(sub), lats[band][jj[0]],
                     z[deep][int(np.nanargmax(col))]], dtype=np.float64)


def amoc_series(pf, mesh, fesom_dir, years, cache_dir=None):
    got_years, rows = [], []
    for y in years:
        a = cache_get(cache_dir, "amoc", y)
        if a is None:
            a = amoc_index(pf, mesh, fesom_dir, y)
            if a is None:
                continue
            cache_put(cache_dir, "amoc", y, a)
        got_years.append(int(y))
        rows.append(np.asarray(a, dtype=np.float64))
    return got_years, (np.array(rows) if rows else np.empty((0, len(AMOC_FIELDS))))


# ----------------------------------------------------------------------------
# annotations
# ----------------------------------------------------------------------------
def annotate_spinup(ax, annotations=None, eval_window=None, eval_label='HIST & PICT',
                    fontsize=6.5, color='0.2', label_y=0.985):
    """Mark what changed where along a stitched spin-up.

    annotations : list of (year, label). A dotted line at each year, the label rotated
                  and tucked against the top of the axes.
    eval_window : (start, end) years shaded as the evaluation window.
    """
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    x0, x1 = ax.get_xlim()
    if eval_window:
        a, b = eval_window
        ax.axvspan(a - 0.5, b + 0.5, color='0.5', alpha=0.13, lw=0, zorder=0)
        # the span disappears under filled contours (Hovmoeller); the edge line does not
        ax.axvline(a - 0.5, color=color, lw=1.0, ls='-', zorder=5)
        ax.text(0.5 * (a + b), 0.02, eval_label, transform=trans, ha='center', va='bottom',
                fontsize=fontsize, color=color, rotation=90, zorder=6,
                bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none', alpha=0.6))
    for year, label in (annotations or []):
        if not (min(x0, x1) <= year <= max(x0, x1)):
            continue
        ax.axvline(year, color=color, lw=0.6, ls=':', zorder=5)
        ax.text(year, label_y, f" {label}", transform=trans, rotation=90, ha='right', va='top',
                fontsize=fontsize, color=color, zorder=6,
                bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none', alpha=0.6))
    ax.set_xlim(x0, x1)
