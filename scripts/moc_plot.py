import pyfesom2 as pf
import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np

mesh = pf.load_mesh('/work/ab0246/a270092/input/fesom2/dars2/')

data = pf.get_data('/work/bb1469/a270089/runtime/awicm3-develop/awicm3-dev-TCO319L137-DARS2_branchoff_360s_1850_TUNE10_GGAUSS_001/outdata/fesom/', 'w', range(1860,1867), mesh, how="mean", compute=True )

print(data.shape)

# Transpose data from (levels, nodes) to (nodes, levels) as expected by xmoc_data
data = data.T

lats, moc = pf.xmoc_data(mesh, data, nlats=200)

plt.figure(figsize=(10, 3))
pf.plot_xyz(mesh, moc, xvals=lats, maxdepth=7000, cmap=cm.seismic, levels = np.linspace(-30, 30, 51), 
             facecolor='gray')

plt.savefig("gmoc.png", dpi=300, bbox_inches='tight')



lats, moc = pf.xmoc_data(mesh, data, nlats=200, mask="Atlantic_MOC")

plt.figure(figsize=(10, 3))
pf.plot_xyz(mesh, moc, xvals=lats, maxdepth=7000, cmap=cm.seismic, levels = np.linspace(-30, 30, 51), 
             facecolor='gray')

plt.savefig("amoc.png", dpi=300, bbox_inches='tight')


