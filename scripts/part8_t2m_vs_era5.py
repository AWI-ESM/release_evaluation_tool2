# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

SCRIPT_NAME = os.path.basename(__file__)  # Get the current script name

print(SCRIPT_NAME)

# Mark as started
update_status(SCRIPT_NAME, " Started")


mesh = pf.load_mesh(meshpath)
data = xr.open_dataset(meshpath+'/fesom.mesh.diag.nc')

# parameters cell
input_paths = [historic_path]
input_names = [historic_name]

if reanalysis=='ERA5':
    clim='ERA5'
    clim_var='t2m'
    climatology_files = ['T2M_yearmean.nc']
    title='Near surface (2m) air tempereature vs. ERA5'
    climatology_path = observation_path+'/era5/netcdf/'
elif reanalysis=='NCEP2':
    clim='NCEP2'
    clim_var='air'
    climatology_files = ['air.2m.timemean.nc']
    title='Near surface (2m) air tempereature vs. NCEP2'
    climatology_path =  observation_path+'/NCEP2/'

exps=[]
for year in range(historic_last25y_start, historic_last25y_end + 1):
    exps.append(year)
        
figsize=(6, 4.5)
ofile = None
res = [360, 180]
var = ['2t']
levels = [-8.0,-5.0,-3.0,-2.0,-1.0,-.6,-.2,.2,.6,1.0,2.0,3.0,5.0,8.0]
contour_outline_thickness = 0

# Set number of columns, in case of multiple variables
def define_rowscol(input_paths, columns=len(input_paths), reduce=0):
    number_paths = len(input_paths) - reduce
#     columns = columns
    if number_paths < columns:
        ncol = number_paths
    else:
        ncol = columns
    nrows = math.ceil(number_paths / columns)
    return [nrows, ncol]

# Mean Deviation weighted
def md(predictions, targets, wgts):
    output_errors = np.average((predictions - targets), axis=0, weights=wgts)
    return (output_errors).mean()

# Load reanalysis data

reanalysis_path = climatology_path+climatology_files[0]
data_reanalysis_mean = np.squeeze(cdo.timmean(input="-remapcon,r"+str(res[0])+"x"+str(res[1])+" "+str(reanalysis_path),returnArray=clim_var))

# Load model Data
def load_parallel(variable,path):
    data1 = cdo.timmean(input="-remapcon,r"+str(res[0])+"x"+str(res[1])+" "+str(path),returnArray=variable)
    return data1




chunk_size = 10  # Process in chunks of 20

data = OrderedDict()
for exp_path, exp_name in zip(input_paths, input_names):
    data[exp_name] = {}
    for v in var:
        datat = []
        t = []

        # Process in chunks
        for i in range(0, len(exps), chunk_size):
            chunk = exps[i:i + chunk_size]
            chunk_t = []
            
            for exp in chunk:
                path = f"{exp_path}/oifs/atm_remapped_1m_{v}_1m_{exp:04d}-{exp:04d}.nc"
                temporary = dask.delayed(load_parallel)(v, path)
                chunk_t.append(temporary)

            with ProgressBar():
                datat_chunk = dask.compute(*chunk_t, scheduler='threads')
            
            datat.extend(datat_chunk)

        data[exp_name][v] = np.squeeze(datat)

data_model = OrderedDict()
data_model_mean = OrderedDict()



for exp_name in input_names:
    data_model[exp_name] = np.squeeze(data[exp_name][v]) 
    data_model_mean[exp_name] = data_model[exp_name]
    if len(np.shape(data_model_mean[exp_name])) > 2:
        data_model_mean[exp_name] = np.mean(data_model_mean[exp_name],axis=0)
  

print(np.shape(data_model_mean[exp_name]))
print(np.shape(data_reanalysis_mean))

lon = np.arange(0, 360, 1)
lat = np.arange(-90, 90, 1)
data_model_mean[exp_name], lon = add_cyclic_point(data_model_mean[exp_name], coord=lon)


lon = np.arange(0, 360, 1)
lat = np.arange(-90, 90, 1)
data_reanalysis_mean, lon = add_cyclic_point(data_reanalysis_mean, coord=lon)

print(np.shape(data_model_mean[exp_name]))
print(np.shape(data_reanalysis_mean))


coslat = np.cos(np.deg2rad(lat))
wgts = np.squeeze(np.sqrt(coslat)[..., np.newaxis])
rmsdval = sqrt(mean_squared_error(data_model_mean[exp_name],data_reanalysis_mean,sample_weight=wgts))
mdval = md(data_model_mean[exp_name],data_reanalysis_mean,wgts)


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define figure layout
nrows, ncol = define_rowscol(input_paths)
fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=figsize,
                         subplot_kw={'projection': ccrs.PlateCarree()})  # Use PlateCarree globally

if isinstance(axes, np.ndarray):
    axes = axes.flatten()
else:
    axes = [axes]

# Loop through input names and plot data
for i, exp_name in enumerate(input_names):
    print(exp_name)
    
    ax = axes[i]
    ax.add_feature(cfeature.COASTLINE, zorder=3)

    # Contour plot
    imf = ax.contourf(lon, lat, data_model_mean[exp_name] - data_reanalysis_mean, 
                       cmap=plt.cm.PuOr_r, levels=levels, extend='both',
                       transform=ccrs.PlateCarree(), zorder=1)
    
    line_colors = ['black' for _ in imf.levels]
    imc = ax.contour(lon, lat, data_model_mean[exp_name] - data_reanalysis_mean, 
                     colors=line_colors, levels=levels, linewidths=contour_outline_thickness,
                     transform=ccrs.PlateCarree(), zorder=1)

    ax.set_ylabel('K')
    ax.set_xlabel('Simulation Year')
    ax.set_title(title, fontweight="bold")

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.2, linestyle='-')
    gl.xlabels_bottom = False

    # Bias & RMSD Text
    textrsmd = f'rmsd={round(rmsdval, 3)}'
    textbias = f'bias={round(mdval, 3)}'
    props = dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.5)

    ax.text(0.02, 0.4, textrsmd, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', bbox=props, zorder=4)
    ax.text(0.02, 0.3, textbias, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', bbox=props, zorder=4)

# Colorbar
cbar_ax_abs = fig.add_axes([0.15, 0.11, 0.7, 0.05])
cbar_ax_abs.tick_params(labelsize=12)
cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal', ticks=levels)
cb.set_label(label="°C", size=14)
cb.ax.tick_params(labelsize=12)

# Hide every other tick label
for label in cb.ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)

# Save figure
ofile = f't2m_vs_{clim}'
if ofile is not None:
    plt.savefig(out_path + ofile, dpi=dpi, bbox_inches='tight')

# Mark script as completed
update_status(SCRIPT_NAME, "Completed")

