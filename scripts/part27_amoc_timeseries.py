# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bg_routines.config_loader import *
from bg_routines import spinup_tools as st

SCRIPT_NAME = os.path.basename(__file__)
print(SCRIPT_NAME)
update_status(SCRIPT_NAME, " Started")

# AMOC time series along the spin-up
# ----------------------------------
# One value per year: the maximum of the Atlantic overturning streamfunction at 26.5N
# below 500 m (the RAPID definition), and the maximum over 20-60N. Both come from the
# annual-mean vertical velocity with the routine part17 uses for the section plot
# (pyfesom2 xmoc_data, mask 'Atlantic_MOC'). Reading a year of w costs seconds and a
# gigabyte, so each year goes through the spin-up cache (spinup_cache_path); years whose
# w.fesom.<year>.nc is on another machine are supplied by precompute_spinup_cache.py.

_ts_end = globals().get('spinup_timeseries_end', spinup_end)
_annotations = globals().get('spinup_annotations', None)
_cache_dir = globals().get('spinup_cache_path', None) or os.path.join(out_path, 'spinup_cache')
fesom_path = spinup_path + '/fesom/'
years_req = range(spinup_start, _ts_end + 1)

print(f"AMOC index for {spinup_start}-{_ts_end} from {fesom_path}, cache {_cache_dir}")
years, idx = st.amoc_series(pf, mesh, fesom_path, years_req, _cache_dir)
if len(years) < 2:
    print("WARNING: fewer than 2 years with w or a cache entry. Skipping plot.")
    update_status(SCRIPT_NAME, " Completed")
    sys.exit(0)
if len(years) < len(years_req):
    print(f"  {len(years_req) - len(years)} years have neither w.fesom.<year>.nc nor a cache entry: "
          f"{sorted(set(years_req) - set(years))[:10]} ...")

years = np.asarray(years)
a26, amax = idx[:, 0], idx[:, 1]

def running_mean(x, n=11):
    """Centred running mean, NaN where the window does not fit."""
    if len(x) < n:
        return np.full(len(x), np.nan)
    out = np.full(len(x), np.nan)
    out[n // 2: len(x) - n // 2] = np.convolve(x, np.ones(n) / n, mode='valid')
    return out

fig, ax = plt.subplots(figsize=(13, 4.8) if _annotations else (7.2, 3.8))
c26, cmax = 'tab:blue', '0.35'
ax.plot(years, amax, color=cmax, lw=0.5, alpha=0.35)
ax.plot(years, a26, color=c26, lw=0.5, alpha=0.35)
ax.plot(years, running_mean(amax), color=cmax, lw=1.6, label='maximum 20-60N')
ax.plot(years, running_mean(a26), color=c26, lw=1.8, label='26.5N')
# RAPID array at 26.5N: about 17 Sv since 2004 with an interannual standard deviation of
# about 1.5 Sv. A present-day reference; drawn as a band, not as a target line.
ax.axhspan(17.0 - 1.5, 17.0 + 1.5, color='tab:blue', alpha=0.10, lw=0, label='RAPID since 2004, 26.5N (~17 ± 1.5 Sv)')
ax.set_ylabel('Atlantic overturning [Sv]', size=13)
ax.set_xlabel('Year', size=13)
ax.set_title('AMOC, annual maximum below 500 m (thick: 11-year running mean)', fontweight='bold')
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.grid(axis='y', color='0.85', lw=0.5)
ax.legend(fontsize=10, loc='lower left', ncol=3)
st.annotate_spinup(ax, _annotations,
                   eval_window=(pi_ctrl_start, pi_ctrl_end) if _ts_end > spinup_end else None)
fig.tight_layout()
plt.savefig(out_path + 'amoc_timeseries.png', dpi=dpi, bbox_inches='tight')

_n = min(15, len(years))
print(f"AMOC 26.5N: first {_n} years {np.mean(a26[:_n]):.2f} Sv, last {_n} years {np.mean(a26[-_n:]):.2f} Sv; "
      f"maximum 20-60N: {np.mean(amax[:_n]):.2f} -> {np.mean(amax[-_n:]):.2f} Sv")
update_status(SCRIPT_NAME, " Completed")
