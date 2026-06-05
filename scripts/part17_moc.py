# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bg_routines.config_loader import *
from bg_routines.ipcc_cmaps import get_bias_cmap

SCRIPT_NAME = os.path.basename(__file__)
print(SCRIPT_NAME)

update_status(SCRIPT_NAME, " Started")

# Meridional overturning circulation (GMOC + AMOC) from vertical velocities
# over the last 25 historic years of the run.

years = range(historic_last25y_start, historic_last25y_end + 1)
fesom_path = historic_path + '/fesom/'

print(f"Loading w from {fesom_path} for years {years.start}-{years.stop - 1}")
data = pf.get_data(fesom_path, 'w', years, mesh, how='mean', compute=True)
print(f"  loaded shape: {data.shape}")

levels = np.linspace(-30, 30, 51)

for which, mask, label in [
    ('gmoc', None, 'Global MOC'),
    ('amoc', 'Atlantic_MOC', 'Atlantic MOC'),
]:
    print(f"Computing {label}...")
    if mask is None:
        lats, moc = pf.xmoc_data(mesh, data, nlats=200)
    else:
        lats, moc = pf.xmoc_data(mesh, data, nlats=200, mask=mask)

    plt.figure(figsize=(10, 3))
    pf.plot_xyz(mesh, moc, xvals=lats, maxdepth=7000,
                cmap=get_bias_cmap('moc'), levels=levels, facecolor='gray',
                label='Sv')
    plt.title(f'{label} — {historic_name} ({years.start}-{years.stop - 1})')

    ofile = os.path.join(out_path, f'{which}.png')
    plt.savefig(ofile, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  saved {ofile}")

print("=== MOC plots complete ===")
update_status(SCRIPT_NAME, " Completed")
