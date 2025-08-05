# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

SCRIPT_NAME = os.path.basename(__file__)  # Get the current script name

print(SCRIPT_NAME)

# Mark as started
update_status(SCRIPT_NAME, " Started")


# parameters cell
input_paths = [historic_path]
input_names = [historic_name]

climatology_files = ['precip.mon.mean_timmean.nc']
climatology_path =  observation_path+'/gpcp/'

exps=[]

for year in range(historic_last25y_start, historic_last25y_end + 1):
    exps.append(year)
        
figsize=(6, 4.5)
res = [180, 90]
var = ['cp', 'lsp']
variable_clim = 'pr'
levels = [-4,-3,-2,-1,-.5,-0.2,0.2,.5,1,2,3,4]
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

# Calculate Root Mean Square Deviation (RMSD)
def rmsd(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Mean Deviation weighted
def md(predictions, targets, wgts):
    output_errors = np.average((predictions - targets), axis=0, weights=wgts)
    return (output_errors).mean()

# Load GPCP reanalysis data

GPCP_path = climatology_path+climatology_files[0]
GPCP_data = cdo.yearmean(input="-remapcon,r"+str(res[0])+"x"+str(res[1])+" "+str(GPCP_path),returnArray=variable_clim)*86400


# Load model Data
def load_parallel(variable,path):
    data1 = (cdo.timmean(input="-remapcon,r"+str(res[0])+"x"+str(res[1])+" "+str(path),returnArray=variable)/accumulation_period)*1000*86400
    return data1

data = OrderedDict()
for exp_path, exp_name  in zip(input_paths, input_names):
    data[exp_name] = {}
    for v in var:
        datat = []
        t = []
        temporary = []
        for exp in tqdm(exps):

            path = exp_path+'/oifs/atm_remapped_1m_'+v+'_1m_'+f'{exp:04d}-{exp:04d}.nc'
            temporary = dask.delayed(load_parallel)(v,path)
            t.append(temporary)

        with ProgressBar():
            datat = dask.compute(t)
        data[exp_name][v] = np.squeeze(datat)
        
data_model_mean = np.mean(data[historic_name]['cp'] + \
                          data[historic_name]['lsp'],axis=0)
        
data_reanalysis_mean = np.mean(GPCP_data,axis=0)

print(np.shape(data_model_mean))
print(np.shape(data_reanalysis_mean))

lon = np.arange(0, 360, 2)
lat = np.arange(-90, 90, 2)
data_model_mean, lon = add_cyclic_point(data_model_mean, coord=lon)

lon = np.arange(0, 360, 2)
lat = np.arange(-90, 90, 2)
data_reanalysis_mean, lon = add_cyclic_point(data_reanalysis_mean, coord=lon)

print(np.shape(data_model_mean))
print(np.shape(data_reanalysis_mean))

coslat = np.cos(np.deg2rad(lat))
wgts = np.squeeze(np.sqrt(coslat)[..., np.newaxis])
rmsdval = sqrt(mean_squared_error(data_model_mean,data_reanalysis_mean,sample_weight=wgts))
mdval = md(data_model_mean,data_reanalysis_mean,wgts)


nrows, ncol = define_rowscol(input_paths)
fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=figsize, subplot_kw={'projection': ccrs.Robinson(central_longitude=200)})
if isinstance(axes, np.ndarray):
    axes = axes.flatten()
else:
    axes = [axes]
i = 0


    
for key in input_names:

    axes[i]=plt.subplot(nrows,ncol,i+1,projection=ccrs.Robinson(central_longitude=200))
    axes[i].add_feature(cfeature.COASTLINE,zorder=3)
    
    
    imf=plt.contourf(lon, lat, data_model_mean-
                    data_reanalysis_mean, cmap=plt.cm.PuOr, 
                     levels=levels, extend='both',
                     transform=ccrs.PlateCarree(),zorder=1)
    line_colors = ['black' for l in imf.levels]
    imc=plt.contour(lon, lat, data_model_mean-
                    data_reanalysis_mean, colors=line_colors, 
                    levels=levels, linewidths=contour_outline_thickness,
                    transform=ccrs.PlateCarree(),zorder=1)

    axes[i].set_xlabel('Simulation Year')
    
    axes[i].set_title("Precipitation vs. GPCP",fontweight="bold")
    plt.tight_layout() 
    gl = axes[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='-')

    gl.xlabels_bottom = False
    
    textrsmd='rmsd='+str(round(rmsdval,3))
    textbias='bias='+str(round(mdval,3))
    props = dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.5)
    axes[i].text(0.08, 0.33, textrsmd, transform=axes[i].transAxes, fontsize=11,
        verticalalignment='top', bbox=props, zorder=4)
    axes[i].text(0.08, 0.25, textbias, transform=axes[i].transAxes, fontsize=11,
        verticalalignment='top', bbox=props, zorder=4)
    
    
    i = i+1
    
    cbar_ax_abs = fig.add_axes([0.15, 0.11, 0.7, 0.05])
    cbar_ax_abs.tick_params(labelsize=12)
    cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal',ticks=levels)
    cb.set_label(label="mm/day", size='14')
    cb.ax.tick_params(labelsize='12')

    
for label in cb.ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)

    
ofile='precip_vs_GPCP'
    
if ofile is not None:
    plt.savefig(out_path + ofile, dpi=dpi, bbox_inches='tight')

# Mark script as completed
update_status(SCRIPT_NAME, "Completed")

