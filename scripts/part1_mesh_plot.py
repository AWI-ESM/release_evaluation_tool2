# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

SCRIPT_NAME = os.path.basename(__file__)  # Get the current script name

print(SCRIPT_NAME)

# Mark as started
update_status(SCRIPT_NAME, " Started")


levels = np.linspace(5, 30, 11)
figsize=(6,4.5)

print('does data arrive here')
print(data)

nod_area = data['nod_area'][0,:].values
nod_area = (np.sqrt(nod_area/np.pi)/1e3)*2

data=nod_area,
cmap=cmo.cm.thermal_r,
influence=1600000,
interp="nn",
mapproj="pc",
ptype="cf",
units=None,
titles=None,
distances_path=None,
inds_path=None,
qhull_path=None,
basepath=None,
interpolated_data=None,
lonreg=None,
latreg=None,
no_pi_mask=False,

box=[-180, 180, -90, 90]
res=[3600, 1800]

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

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figsize, subplot_kw={'projection': ccrs.Robinson(central_longitude=-160)})
ax.add_feature(cfeature.COASTLINE,zorder=3)

imf=plt.contourf(lonreg, latreg, np.squeeze(interpolated), cmap=cmo.cm.thermal_r, 
                 levels=levels, extend='both',
                 transform=ccrs.PlateCarree(),zorder=1)

ax.set_ylabel('K')
ax.set_title(mesh_name+" mesh resolution",fontweight="bold")
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='lightgrey'))

plt.tight_layout() 

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
              linewidth=1, color='gray', alpha=0.2, linestyle='-')
gl.xlabels_bottom = False

cbar_ax_abs = fig.add_axes([0.15, 0.11, 0.7, 0.05])
cbar_ax_abs.tick_params(labelsize=12)
cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal',ticks=levels)
cb.set_label(label="Horizontal Resolution [km]", size='12')
cb.ax.tick_params(labelsize='11')

ofile=out_path+'mesh_resolution'
    
if ofile is not None:
    plt.savefig(ofile, dpi=dpi,bbox_inches='tight')


# Mark as completed
update_status(SCRIPT_NAME, " Completed")

