# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

SCRIPT_NAME = os.path.basename(__file__)  # Get the current script name

print(SCRIPT_NAME)

# Mark as started
update_status(SCRIPT_NAME, " Started")
# Load MODIS climatology data

# parameters cell
input_paths = [historic_path]
input_names = [historic_name]
exps = list(range(historic_last25y_start, historic_last25y_end+1))


climatology_files = ['clt_MODIS_yearmean.nc']
climatology_path =  observation_path+'/MODIS/'

figsize=(6, 4.5)
dpi = 300
ofile = None
res = [180, 91]
variable = ['tcc', 'lcc', 'hcc']
variable_clim = 'clt'
title='Cloud area fraction vs. MODIS'
mapticks = [-50,-30,-20,-10,-6,-2,2,6,10,20,30,50]

contour_outline_thickness = 0
levels = np.linspace(-5, 5, 21)

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

# Load CERES satobs data (https://doi.org/10.1175/JCLI-D-17-0208.1)


CERES_path = climatology_path + climatology_files[0]
CERES_Dataset = Dataset(CERES_path)
CERES_Data = OrderedDict()
CERES_CRF = CERES_Dataset.variables[variable_clim][:]

# Load model data in chunks of 5
def load_parallel(variable, path):
    data1 = cdo.timmean(input="-remapcon,r" + str(res[0]) + "x" + str(res[1]) + " " + str(path), returnArray=variable) * 100
    return data1

data = OrderedDict()
chunk_size = 5  # Process in chunks of 5

for exp_path, exp_name in zip(input_paths, input_names):
    data[exp_name] = {}
    for v in variable:
        datat = []
        for i in range(0, len(exps), chunk_size):
            chunk = exps[i:i + chunk_size]
            t = []
            for exp in chunk:
                path = f"{exp_path}/oifs/atm_remapped_1m_{v}_1m_{exp:04d}-{exp:04d}.nc"
                t.append(dask.delayed(load_parallel)(v, path))
            with ProgressBar():
                datat_chunk = dask.compute(*t, scheduler='threads')
            datat.extend(datat_chunk)
        data[exp_name][v] = np.squeeze(datat)

# Compute Model Mean
crf_sw_model = OrderedDict()
crf_sw_model_mean = OrderedDict()

for exp_name in input_names:
    crf_sw_model[exp_name] = np.squeeze(data[exp_name]['tcc']) 
    crf_sw_model_mean[exp_name] = np.mean(crf_sw_model[exp_name], axis=0)
    if len(np.shape(crf_sw_model_mean[exp_name])) > 2:
        crf_sw_model_mean[exp_name] = np.mean(crf_sw_model_mean[exp_name], axis=0)

crf_sw_satobs_mean = np.mean(CERES_CRF, axis=0)

# Longitude & Latitude Handling
lon = np.arange(0, 360, 2)
lat = np.linspace(-90, 90, 91)  # Ensures correct shape for latitude

crf_sw_model_mean[exp_name], lon_cyclic = add_cyclic_point(crf_sw_model_mean[exp_name], coord=lon)
crf_sw_satobs_mean, _ = add_cyclic_point(crf_sw_satobs_mean, coord=lon)

# Compute Weighted RMSD & MD
coslat = np.cos(np.deg2rad(lat))
wgts = np.sqrt(coslat)[:, np.newaxis]  # Expand to (lat, 1) and match dimensions
wgts = np.broadcast_to(wgts, crf_sw_model_mean[exp_name].shape)  # Broadcast to (lat, lon)

rmsdval = sqrt(mean_squared_error(crf_sw_model_mean[exp_name], crf_sw_satobs_mean, sample_weight=wgts))
mdval = md(crf_sw_model_mean[exp_name], crf_sw_satobs_mean, wgts)

# Define figure layout
nrows, ncol = define_rowscol(input_paths)
fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()}, dpi=dpi)

if isinstance(axes, np.ndarray):
    axes = axes.flatten()
else:
    axes = [axes]

# Loop through input names and plot
for i, exp_name in enumerate(input_names):
    ax = axes[i]
    ax.add_feature(cfeature.COASTLINE, zorder=3)

    # Contour plot
    imf = ax.contourf(lon_cyclic, lat, crf_sw_model_mean[exp_name] - crf_sw_satobs_mean,
                      cmap='PuOr_r', levels=mapticks, extend='both',
                      transform=ccrs.PlateCarree(), zorder=1)

    line_colors = ['black' for _ in imf.levels]
    imc = ax.contour(lon_cyclic, lat, crf_sw_model_mean[exp_name] - crf_sw_satobs_mean,
                     colors=line_colors, levels=mapticks,
                     linewidths=contour_outline_thickness,
                     transform=ccrs.PlateCarree(), zorder=1)

    ax.set_ylabel('W/m²')
    ax.set_xlabel('Simulation Year')
    ax.set_title(title, fontweight="bold")

    # Gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.2, linestyle='-')
    gl.xlabels_bottom = False

    # Bias & RMSD Text
    textrsmd = f'rmsd={round(rmsdval, 3)}'
    textbias = f'bias={round(mdval, 3)}'
    props = dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.5)

    ax.text(0.02, 0.35, textrsmd, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', bbox=props, zorder=4)
    ax.text(0.02, 0.25, textbias, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', bbox=props, zorder=4)

# Add Colorbar (Positioned Correctly)
cbar_ax_abs = fig.add_axes([0.15, 0.05, 0.7, 0.03])  # Adjusted position to remove box effect
cbar_ax_abs.tick_params(labelsize=12)
cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal', ticks=mapticks)
cb.set_label(label="%", size=14)
cb.ax.tick_params(labelsize=12)

# Remove extra subplot spacing
plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Fine-tuned spacing
plt.tight_layout(rect=[0, 0.07, 1, 1])  # Prevents extra box while preserving layout

# Save figure
ofile = variable[0] + '_vs_MODIS'
if ofile is not None:
    plt.savefig(out_path + ofile, dpi=dpi, bbox_inches='tight')

# Create 5° latitude bins for zonal analysis
lat_bins = np.arange(-90, 91, 5)
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

# Function to compute zonal means in latitude bins
def compute_zonal_mean_by_bin(data, lat_grid, lat_bins):
    result = np.zeros(len(lat_bins) - 1)
    
    for i in range(len(lat_bins) - 1):
        lat_min, lat_max = lat_bins[i], lat_bins[i+1]
        mask = (lat_grid >= lat_min) & (lat_grid < lat_max)
        if np.any(mask):
            # For 2D data (lat, lon), compute mean across longitudes for each latitude band
            if len(data.shape) == 2:
                # Get weighted mean across longitudes for each latitude in this band
                lat_means = np.mean(data[mask, :], axis=1)  # Mean across longitudes for each lat
                # Get weighted mean across latitudes in this band
                weights = np.cos(np.deg2rad(lat_grid[mask]))
                weights = weights / np.sum(weights)  # Normalize weights
                result[i] = np.sum(lat_means * weights)
            else:
                # For 1D data (already aggregated by latitude)
                result[i] = data[mask].mean()
    
    return result

# ================ PLOT 1: TCC Bias by Latitude Bin ================
print('Creating TCC bias zonal plot...')
plt.figure(figsize=(10, 6), dpi=dpi)

# Compute zonal bias for each latitude bin
# First calculate bias for each lat/lon point
bias_data = crf_sw_model_mean[exp_name] - crf_sw_satobs_mean

# Now compute zonal means by latitude band
bias_by_lat = np.zeros(len(lat_bins) - 1)
for i in range(len(lat_bins) - 1):
    lat_min, lat_max = lat_bins[i], lat_bins[i+1]
    lat_indices = np.where((lat >= lat_min) & (lat < lat_max))[0]
    
    if len(lat_indices) > 0:
        # Calculate weighted mean across all longitudes for each latitude in this band
        lat_weights = np.cos(np.deg2rad(lat[lat_indices]))
        band_bias = np.zeros(len(lat_indices))
        
        for j, lat_idx in enumerate(lat_indices):
            band_bias[j] = np.mean(bias_data[lat_idx, :])
        
        # Weighted average across the latitudes in this band
        bias_by_lat[i] = np.average(band_bias, weights=lat_weights)

plt.plot(lat_centers, bias_by_lat, 'b-', linewidth=2, label='TCC Bias')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.xlabel('Latitude (°)')
plt.ylabel('Cloud Fraction Bias (%)')
plt.title('TCC Bias vs. MODIS by 5° Latitude Bands')
plt.xticks(np.arange(-90, 91, 15))
plt.legend()

# Save the TCC bias zonal plot
ofile_zonal_bias = variable[0] + '_bias_zonal'
plt.savefig(out_path + ofile_zonal_bias, dpi=dpi, bbox_inches='tight')

# ================ PLOT 2: Absolute LCC, HCC, TCC by Latitude Bin ================
print('Creating absolute cloud cover zonal plot...')
plt.figure(figsize=(10, 6), dpi=dpi)

# Prepare data for absolute cloud fractions
cloud_data = {}
cloud_labels = {'tcc': 'Total Cloud Cover', 'lcc': 'Low Cloud Cover', 'hcc': 'High Cloud Cover'}
cloud_colors = {'tcc': 'k', 'lcc': 'b', 'hcc': 'r'}

# Compute zonal means for each cloud type
for v_index, v in enumerate(['tcc', 'lcc', 'hcc']):
    if v in data[exp_name]:
        # Calculate mean across all years
        cloud_mean = np.mean(data[exp_name][v], axis=0)
        if len(np.shape(cloud_mean)) > 2:
            cloud_mean = np.mean(cloud_mean, axis=0)
        
        # Add cyclic point
        cloud_mean, _ = add_cyclic_point(cloud_mean, coord=lon)
        
        # Calculate zonal mean by latitude band
        cloud_data[v] = np.zeros(len(lat_bins) - 1)
        for i in range(len(lat_bins) - 1):
            lat_min, lat_max = lat_bins[i], lat_bins[i+1]
            lat_indices = np.where((lat >= lat_min) & (lat < lat_max))[0]
            
            if len(lat_indices) > 0:
                # Calculate weighted mean across all longitudes for each latitude in this band
                lat_weights = np.cos(np.deg2rad(lat[lat_indices]))
                band_means = np.zeros(len(lat_indices))
                
                for j, lat_idx in enumerate(lat_indices):
                    band_means[j] = np.mean(cloud_mean[lat_idx, :])
                
                # Weighted average across the latitudes in this band
                cloud_data[v][i] = np.average(band_means, weights=lat_weights)
        
        # Plot the zonal mean
        plt.plot(lat_centers, cloud_data[v], color=cloud_colors[v], linewidth=2, label=cloud_labels[v])

plt.grid(True, alpha=0.3)
plt.xlabel('Latitude (°)')
plt.ylabel('Cloud Fraction (%)')
plt.title('Cloud Cover by 5° Latitude Bands')
plt.xticks(np.arange(-90, 91, 15))
plt.ylim(0, 100)
plt.legend()

# Save the absolute cloud cover zonal plot
ofile_zonal_abs = 'cloud_cover_zonal'
plt.savefig(out_path + ofile_zonal_abs, dpi=dpi, bbox_inches='tight')

# Mark script as completed
update_status(SCRIPT_NAME, "Completed")

