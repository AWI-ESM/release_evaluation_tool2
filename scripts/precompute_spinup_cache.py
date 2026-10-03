#!/usr/bin/env python3
"""Fill the spin-up cache (bg_routines/spinup_tools.py) for a range of years, without a
reval config. Run it on the machine that holds the 3D ocean output and copy the cache
directory to where reval runs; part3 (Hovmoeller) and part27 (AMOC) then read those
years from the cache instead of from temp.fesom.<year>.nc / w.fesom.<year>.nc.

    python scripts/precompute_spinup_cache.py --fesom-dir <exp>/outdata/fesom \\
        --meshpath <mesh dir> --mesh-file mesh.nc --years 1850 1919 \\
        --cache-dir <dir> [--what hovm amoc] [--variable temp]

Years already in the cache are skipped, so the call can be repeated or split over
several processes with disjoint year ranges.
"""
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bg_routines import spinup_tools as st  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--fesom-dir', required=True)
p.add_argument('--meshpath', required=True)
p.add_argument('--mesh-file', default='mesh.nc')
p.add_argument('--years', nargs=2, type=int, required=True, metavar=('FIRST', 'LAST'))
p.add_argument('--cache-dir', required=True)
p.add_argument('--what', nargs='+', default=['hovm', 'amoc'], choices=['hovm', 'amoc'])
p.add_argument('--variable', default='temp')
a = p.parse_args()
years = range(a.years[0], a.years[1] + 1)

if 'hovm' in a.what:
    from cdo import Cdo
    # the cdo next to this interpreter (conda env), else whatever is on PATH; sys.prefix is
    # not reliable for an env that was moved after creation
    _cands = [os.path.join(os.path.dirname(sys.executable), 'cdo'), os.path.join(sys.prefix, 'bin', 'cdo')]
    _cdo_bin = next((c for c in _cands if os.path.exists(c)), None)
    if _cdo_bin:
        # python-cdo re-instantiates itself with a bare 'cdo' on operator calls, so the
        # binary has to be on PATH as well as passed in
        os.environ['PATH'] = os.path.dirname(_cdo_bin) + os.pathsep + os.environ.get('PATH', '')
    cdo = Cdo(cdo=_cdo_bin) if _cdo_bin else Cdo()
    for y in years:
        if st.cache_get(a.cache_dir, f'hovm_{a.variable}', y) is not None:
            continue
        prof = st.hovm_profile(cdo, a.fesom_dir, a.variable, y, a.meshpath, a.mesh_file)
        if prof is None:
            print(f'hovm {y}: no {a.variable}.fesom.{y}.nc', flush=True)
            continue
        st.cache_put(a.cache_dir, f'hovm_{a.variable}', y, prof)
        print(f'hovm {y}: {len(prof)} levels, surface {prof[0]:.3f}', flush=True)

if 'amoc' in a.what:
    import pyfesom2 as pf
    mesh = pf.load_mesh(a.meshpath)
    for y in years:
        if st.cache_get(a.cache_dir, 'amoc', y) is not None:
            continue
        idx = st.amoc_index(pf, mesh, a.fesom_dir, y)
        if idx is None:
            print(f'amoc {y}: no w.fesom.{y}.nc', flush=True)
            continue
        st.cache_put(a.cache_dir, 'amoc', y, idx)
        print(f'amoc {y}: 26.5N {idx[0]:.2f} Sv, max 20-60N {idx[1]:.2f} Sv at {idx[2]:.1f}N', flush=True)
