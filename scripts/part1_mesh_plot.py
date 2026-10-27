# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bg_routines.config_loader import *
from bg_routines.ipcc_cmaps import get_abs_cmap

SCRIPT_NAME = os.path.basename(__file__)  # Get the current script name

print(SCRIPT_NAME)

# Mark as started
update_status(SCRIPT_NAME, " Started")



#levels = np.linspace(5, 30, 11)
figsize=(9, 5)

mesh = pf.load_mesh(meshpath)
data = xr.open_dataset(meshpath+'/fesom.mesh.diag.nc')
# Resolution as triangle height h (Danilov 2022, doi:10.1029/2022MS003177): for equilateral triangles the
# dual (node) area is A = 2*h^2/sqrt(3), so h = (sqrt(3)/2)^(1/2)*sqrt(A). The diameter of an equal-area
# circle, 2*sqrt(A/pi) as in fesom2 mesh_resolution, is 21% coarser. nod_area is taken at the surface
# (zero under ice shelves) or, where that is zero, at the top of the water column.
nod_area = data['nod_area'].values
nod_area = nod_area[np.argmax(nod_area > 0, axis=0), np.arange(nod_area.shape[1])]
nod_area = np.sqrt(np.sqrt(3)/2*nod_area)/1e3

data=nod_area,
cmap=get_abs_cmap('mesh_res'),
influence=1600000,
interp="nn",
mapproj="pc",
ptype="cf",
units=None,
titles=None,
distances_path=None,
inds_path=None,
qhull_path=None,
basepath=meshpath,
interpolated_data=None,
lonreg=None,
latreg=None,
no_pi_mask=False,

box=[-180, 180, -90, 90]
res=[1440, 720]

if not isinstance(data, list):
    data = [data]
if titles:
    if not isinstance(titles, list):
        titles = [titles]
    if len(titles) != len(data):
        raise ValueError(
            "The number of titles do not match the number of data fields, please adjust titles (or put to None)"
        )


radius_of_influence = influence

left, right, down, up = box
lonNumber, latNumber = res

lonreg = np.linspace(left, right, lonNumber)
latreg = np.linspace(down, up, latNumber)

lonreg2, latreg2 = np.meshgrid(lonreg, latreg)


interpolated = pf.interpolate_for_plot(
    data[0],
    mesh,
    lonreg2,
    latreg2,
    interp=interp[0],
    distances_path=distances_path[0],
    inds_path=inds_path[0],
    radius_of_influence=radius_of_influence[0],
    basepath=basepath[0],
    qhull_path=qhull_path[0],
    )

data_model_mean = OrderedDict()

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figsize, subplot_kw={'projection': ccrs.EqualEarth()})

ax.set_global()
ax.add_feature(cfeature.COASTLINE,zorder=3)

#imf=plt.contourf(lonreg, latreg, np.squeeze(interpolated), cmap=get_abs_cmap('mesh_res'), 
#                 levels=levels, extend='both',
#                 transform=ccrs.PlateCarree(),zorder=1)
imf=plt.contourf(lonreg, latreg, np.squeeze(interpolated), cmap=get_abs_cmap('mesh_res'), 
                 extend='both',
                 transform=ccrs.PlateCarree(),zorder=1)

ax.set_ylabel('K')
ax.set_title(mesh_name+" mesh resolution",fontweight="bold")
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='lightgrey'))

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
              linewidth=1, color='gray', alpha=0.2, linestyle='-')
gl.bottom_labels = False

cbar_ax_abs = fig.add_axes([0.15, 0.06, 0.7, 0.04])
cbar_ax_abs.tick_params(labelsize=12)

#every_other_tick = levels[::2]  # Select every other level for labeling

#cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal',ticks=every_other_tick)
cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal')
cb.set_label(label="Horizontal Resolution [km]", size='12')
cb.ax.tick_params(labelsize='11')

ofile=out_path+'mesh_resolution'
    
if ofile is not None:
    plt.savefig(ofile, dpi=dpi,bbox_inches='tight')




# Mark as completed
update_status(SCRIPT_NAME, " Completed")

