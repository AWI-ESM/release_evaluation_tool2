# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bg_routines.config_loader import *

SCRIPT_NAME = os.path.basename(__file__)  # Get the current script name

print(SCRIPT_NAME)

# Mark as started
update_status(SCRIPT_NAME, " Started")



# interpolate data onto regular grid
# select Nino index region
box = 'Nino34'


if box == 'Nino12':
    lon_min=270 #(-90+360) 
    lon_max=280 #(-80+360) 
    lat_min=-10
    lat_max= 0
elif box == 'Nino3':
    lon_min=210 #(-150+360)
    lon_max=270 #(-90+360)
    lat_min=-5
    lat_max= 5
elif box == 'Nino34':
    lon_min=190 #(-170+360)
    lon_max=240 #(-120+360)
    lat_min=-5
    lat_max= 5
elif box == 'Nino4':
    lon_min=160 
    lon_max=210 #(-150+360)
    lat_min=-5
    lat_max= 5


print(f"Box Coords: lon({lon_min} to {lon_max}), lat({lat_min} to {lat_max})")



# parameters cell
variable = 'sst'
input_paths = [historic_path+'/fesom/']
years = range(historic_start, historic_end+1)
figsize=(10, 5)


# Import utils for dynamic batch sizing
try:
    from utils import get_optimal_batch_size
except ImportError:
    # If running from different directory
    sys.path.append(os.path.dirname(__file__))
    from utils import get_optimal_batch_size

# load mesh (only needed for cdo setgrid path, not loading full data)
# mesh = pf.load_mesh(meshpath, ...) # functionality moved to CDO

if len(years) < 3:
    print("WARNING: ENSO analysis requires at least 3 years of data. Skipping.")
    update_status(SCRIPT_NAME, " Completed")
    sys.exit(0)

def _cdo_remap_one(variable, path, meshpath, mesh_file):
    """CDO monmean + remapnn to global 1deg grid."""
    # remapnn to r360x180 (1 degree global)
    # Use -monmean to ensure monthly data
    data = cdo.remapnn('r360x180',
        input=f'-monmean -setgrid,{meshpath}/{mesh_file} {path}',
        returnArray=variable
    )
    return np.squeeze(data)

t1 = time.time()
print(f"Loading and remapping {variable} data for years {years[0]}-{years[-1]} — CDO + Dask parallel...")

file_paths = [f"{input_paths[0]}/{variable}.fesom.{year}.nc" for year in years]
existing = [f for f in file_paths if os.path.exists(f)]

if not existing:
    print("No data found!")
    sys.exit(1)

# Dynamic batch size
chunk_size = get_optimal_batch_size(existing[0], safety_factor=2.0, max_procs=16)

# Process in parallel
annual_chunks = []
for i in range(0, len(existing), chunk_size):
    chunk = existing[i:i + chunk_size]
    tasks = [dask.delayed(_cdo_remap_one)(variable, f, meshpath, mesh_file) for f in chunk]
    with ProgressBar():
        results = dask.compute(*tasks, scheduler='threads')
    annual_chunks.extend(results)
    print(f"  Batch {i//chunk_size + 1}/{math.ceil(len(existing)/chunk_size)} done")

# Stack: (Time, Lat, Lon)
# r360x180: Lon 0..360 (360), Lat -90..90 (180)
data_raw = np.concatenate(annual_chunks, axis=0)
t2 = time.time()
print(f"Data load + remap time: {t2 - t1:.2f} seconds. Shape: {data_raw.shape}")

# Define grid coordinates for r360x180
# CDO default r360x180:
# Lon: 0 to 359? Or 0.5 to 359.5?
# cdo griddes r360x180 says xfirst=0, xinc=1. yfirst=-89.5, yinc=1?
# Usually standard is x: 0 to 360 (exclusive), y: -90 to 90.
# Let's construct it to match data shape (180, 360)
model_lon = np.linspace(0, 359, 360) # Approx
model_lat = np.linspace(-89.5, 89.5, 180) # Approx
lon2_global, lat2_global = np.meshgrid(model_lon, model_lat)

# Proceed with analysis using gridded data
# Original script detrending
# data_raw shape: (Time, Lat, Lon)
# Detrend along time axis (axis 0)
data_raw = signal.detrend(data_raw, axis=0)

# Seasonal Cycle Removal
# Reshape: (Years, 12, Lat, Lon)
n_years = data_raw.shape[0] // 12
data_raw_reshape = data_raw.reshape(n_years, 12, data_raw.shape[1], data_raw.shape[2])

# Mean seasonal cycle
data_season_cycle = np.mean(data_raw_reshape, axis=0)

# Anomaly
data = data_raw - np.tile(data_season_cycle, (n_years, 1, 1)).reshape(data_raw.shape)

# Select ENSO region for EOF
# Lon: 110 to 290, Lat: -46 to 46
# Find indices
lon_idx = (model_lon >= 110) & (model_lon <= 290)
lat_idx = (model_lat >= -46) & (model_lat <= 46)

# Slice data
sst_eof_region = data[:, :, lon_idx][:, lat_idx, :]
# Need to construct the sliced coordinate arrays for plotting
lon2 = lon2_global[lat_idx, :][:, lon_idx]
lat2 = lat2_global[lat_idx, :][:, lon_idx]

# Perform EOF on sliced data
sst = sst_eof_region
# ... EOF solver ... (sst is now Time x Lat x Lon, solver expects Time x Space?)
# Eof solver expects (Time, Lat, Lon) or (Time, Space).
# We have (Time, Lat, Lon).
# Solver weights need to match (Lat, Lon) or (Lat,).

# ... Existing logic continues ...


# ... EOF solver ...
# Fix weights for EOF
lat_slice = model_lat[lat_idx] # 1D array of latitudes in the slice
coslat = np.cos(np.deg2rad(lat_slice))
wgts = np.sqrt(coslat)[..., np.newaxis] # (N_lat, 1) matches (Time, N_lat, N_lon)?
# Eof package: if weights is (N_lat, 1), it broadcasts to (N_lat, N_lon). Correct.
solver = Eof(sst, weights=wgts)

# Retrieve the leading EOF, expressed as the correlation between the leading
# PC time series and the input SST anomalies at each grid point, and the
# leading PC time series itself.
eof1_corr = solver.eofsAsCorrelation(neofs=1)
eof1 = solver.eofs(neofs=1, eofscaling=0)

eof_abs = solver.eofs(neofs=1)
pc1 = -np.squeeze(solver.pcs(npcs=1, pcscaling=1))

# Sign of correlation is arbitrary, but plot should be positive
if np.mean(eof1_corr) < 0:
    eof1_corr = -eof1_corr
if np.mean(eof1) < 0:
    eof1 = -eof1
    
title='EOF1 as correlation between PC1 time series and the input data'
    
# Debug: Creating figure
print("Creating figure...")
fig = plt.figure(figsize=(9, 5.56))

# Debug: Checking colormap
print("Checking colormap...")
try:
    colormap = plt.colormaps.get_cmap('PuOr_r')  # Updated for modern Matplotlib
except AttributeError:
    colormap = plt.cm.PuOr_r  # Fallback for older versions
print(f"Colormap Loaded: {colormap}")

# Debug: Ensure proper data types
print(f"Shape of eof1_corr: {eof1_corr.shape}, Type: {type(eof1_corr)}")
print(f"Shape of lon2: {lon2.shape}, Type: {type(lon2)}")
print(f"Shape of lat2: {lat2.shape}, Type: {type(lat2)}")

eof1_corr = np.nan_to_num(eof1_corr.astype(np.float64))  # Ensure proper numeric type

# Debug: Checking coordinate ranges
print(f"Longitude range: {lon2.min()} to {lon2.max()}")
print(f"Latitude range: {lat2.min()} to {lat2.max()}")

# Create projection
print("Creating GeoAxes with PlateCarree projection...")
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=190))

# Contour levels
clevs = np.linspace(-1, 1, 21)
print(f"Contour levels: {clevs}")

# Debug: Creating filled contour plot
print("Creating filled contour plot...")

# Define mask for the region
region_mask = (lon2 >= 190) & (lon2 <= 240) & (lat2 >= -5) & (lat2 <= 5)

# Compute mean in the specified region
region_mean = np.nanmean(eof1_corr.squeeze()[region_mask])

# If mean is negative, flip the sign of the entire array
if region_mean < 0:
    eof1_corr = -eof1_corr

fill = ax.contourf(lon2, lat2, eof1_corr.squeeze(), clevs,
                   transform=ccrs.PlateCarree(), cmap=colormap, zorder=-1)
print(f"Fill contour levels: {fill.levels}")

# Debug: Creating contour lines
line_colors = ['black' for _ in fill.levels]
print(f"Line contour colors: {line_colors}")
print("Creating line contour plot...")
con = ax.contour(lon2, lat2, eof1_corr.squeeze(), clevs, colors=line_colors, linewidths=0.3,
                 transform=ccrs.PlateCarree(), zorder=-1)

# Adding features
print("Adding land and coastline features...")
ax.add_feature(cfeature.LAND, color='lightgrey')
ax.add_feature(cfeature.COASTLINE)


# Add bounding box
ax.add_patch(mpatches.Rectangle(xy=[lon_min, lat_min], width=lon_max-lon_min, height=lat_max-lat_min,
                                facecolor='none', edgecolor='Black', lw=2,
                                transform=ccrs.PlateCarree(), zorder=6))

# Debug: Text annotations
print("Adding text annotations for box corners...")
plt.text(lon_min-202, lat_min-2, f"{lon_min}/{lat_min}°")
plt.text(lon_min-202, lat_max-2, f"{lon_min}/{lat_max}°")
plt.text(lon_max-189, lat_max-2, f"{lon_max}/{lat_max}°")
plt.text(lon_max-189, lat_min-2, f"{lon_max}/{lat_min}°")

# Debug: Adding gridlines
print("Adding gridlines...")
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='-')
gl.xlabels_bottom = False

# Colorbar setup
print("Creating colorbar...")
cbar_ax_abs = fig.add_axes([0.15, 0.1, 0.7, 0.05])
cb = fig.colorbar(fill, cax=cbar_ax_abs, orientation='horizontal')
cb.ax.tick_params(labelsize=12)
cb.set_label('Correlation coefficient', fontsize=12)

# Fixing title type
title = str("EOF1 as correlation between PC1 time series and the input data")  # Ensure it's a string
print(f"Setting figure title: {title}")
ax.set_title(title, fontweight="bold")

# Debug: Save figure
ofile = 'HIST'
print(f"Output filename: {ofile}")

if ofile:
    ofile_long = f"{ofile}_enso_eof_corr.png"
    total_path = os.path.join(out_path, ofile_long)

    # Ensure output directory exists
    if not os.path.exists(out_path):
        print(f"Creating output directory: {out_path}")
        os.makedirs(out_path)

    print(f"Saving figure to: {total_path}")

    try:
        # Ensure valid metadata
        fig.canvas.manager.set_window_title("EOF Plot")  # Avoid non-numeric titles

        # Check the figure before saving
        print(f"Figures open: {plt.get_fignums()}")
        print(f"Active figure type: {type(fig)}")
        print(f"Axes in figure: {fig.axes}")

        # Check if all axes have numeric data
        for ax in fig.axes:
            print(f"Checking axis: {ax}")
            print(f"Title: {ax.get_title()}, Type: {type(ax.get_title())}")
            print(f"X-label: {ax.get_xlabel()}, Type: {type(ax.get_xlabel())}")
            print(f"Y-label: {ax.get_ylabel()}, Type: {type(ax.get_ylabel())}")

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                print(f"Tick label: '{label.get_text()}', Type: {type(label.get_text())}")

            for text in ax.texts:
                print(f"Text: '{text.get_text()}', Type: {type(text.get_text())}")

        plt.savefig(total_path, dpi=300)
        print("Figure saved successfully!")

    except Exception as e:
        print(f"Error saving figure: {e}")

# plt.show() # Disabled for batch run

    
    
# Nino Index Calculation
# Select Nino region using slicing instead of interpolation
nino_lon_idx = (model_lon >= lon_min) & (model_lon <= lon_max)
nino_lat_idx = (model_lat >= lat_min) & (model_lat <= lat_max)

# data is already monthly anomaly (Time, Lat, Lon)
sst_nino_region = data[:, :, nino_lon_idx][:, nino_lat_idx, :]

# Area mean (Lat, Lon axes are 1, 2)
sst_area_mean = np.nanmean(sst_nino_region, axis=(1, 2))

# Reshape for analysis
# sst_area_mean is (N_months,)
# Original script expects sst_nino to be (Years, 12)
sst_nino = sst_area_mean.reshape(len(sst_area_mean)//12, 12)
sst_nino_ano = sst_nino - np.mean(sst_nino)

obs_path = observation_path+'/hadisst2/box'



# Process Observations
# --------------------
# Re-import CDO not needed, used from config
# cdo = Cdo() # Already in config
obs_raw = cdo.copy(input=str(obs_path),returnArray='sst')

# Ensure obs_raw is 3D (Time, Lat, Lon)
if obs_raw.ndim == 1:
    obs_raw = obs_raw[:, np.newaxis, np.newaxis]

# Truncate to match some length? Original script had obs_raw = obs_raw[0:1812]
# 1812 months = 151 years.
if obs_raw.shape[0] > 1812:
    obs_raw = obs_raw[0:1812]

# Detrend observations
# obs_raw = signal.detrend(obs_raw, axis=0) # Optional, original script commented it out

# Reshape to add monthly time axis
# obs_raw is (Time, Lat, Lon)
n_obs_years = obs_raw.shape[0] // 12
obs_raw_reshape = obs_raw.reshape(n_obs_years, obs_raw.shape[1], obs_raw.shape[2], 12)

# Calculate seasonal cycle
obs_season_cycle = np.mean(obs_raw_reshape, axis=0)

# Repeat seasonal cycle
obs_season_cycle_repeat = np.repeat(obs_season_cycle[np.newaxis,...], n_obs_years, axis=0)

# Reshape into original format
obs_season_cycle_repeat_reshape = obs_season_cycle_repeat.reshape(obs_raw.shape)

# Remove seasonal cycle from obs
obs = obs_raw - obs_season_cycle_repeat_reshape


def smooth3(x,beta):
    """ kaiser window smoothing """
    window_len=3
    beta=2
    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w = np.kaiser(window_len,beta)
    y = np.convolve(w/w.sum(),s,mode='valid')
    return y[1:len(y)-1]

obs_nino = obs.reshape(len(obs)//12, 12)
obs_nino_ano = obs_nino - np.mean(obs_nino)

# Seasonal smoothing
sst_nino_ano_smooth=smooth3(sst_nino_ano.flatten(),len(sst_nino_ano.flatten()))
obs_nino_ano_smooth=smooth3(obs_nino_ano.flatten(),len(obs_nino_ano.flatten()))

# Plot the leading PC time series.

plt.figure(figsize=figsize)
plt.plot(sst_nino_ano_smooth, color='black', linewidth=1) 
plt.axhline(0, color='k')
plt.title(historic_name+' '+box+' Index Time Series',fontweight="bold")
plt.xlabel('Month',fontsize=13)
plt.ylabel('°C',fontsize=13)
ax.tick_params(labelsize=13)

plt.ylim(-2.5, 2.5)
plt.axhline(y=1, color='grey', linestyle='--')
plt.axhline(y=-1, color='grey', linestyle='--')

months = np.arange(len(sst_nino_ano.flatten()))

plt.fill_between(months, sst_nino_ano_smooth, 0, where = (sst_nino_ano_smooth > 0), color='Orange',alpha=0.25)
plt.fill_between(months, sst_nino_ano_smooth, -0, where = (sst_nino_ano_smooth < -0), color='darkblue',alpha=0.25)

plt.fill_between(months, sst_nino_ano_smooth, 1, where = (sst_nino_ano_smooth > 1), color='Orange')
plt.fill_between(months, sst_nino_ano_smooth, -1, where = (sst_nino_ano_smooth < -1), color='darkblue')


if ofile is not None:
    ofile_long = f"{ofile}_"+box+"_enso_box_index.png"
    plt.savefig(f"{out_path+ofile_long}", dpi=dpi,bbox_inches='tight')

    
    
# Plot the leading PC time series.

plt.figure(figsize=figsize)
plt.plot(obs_nino_ano_smooth, color='black', linewidth=1) 
plt.axhline(0, color='k')
plt.title('HadISST '+box+' Index Time Series',fontweight="bold")
plt.xlabel('Month',fontsize=13)
plt.ylabel('°C',fontsize=13)
ax.tick_params(labelsize=13)

plt.ylim(-2.5, 2.5)
plt.axhline(y=1, color='grey', linestyle='--')
plt.axhline(y=-1, color='grey', linestyle='--')

months = np.arange(len(obs_nino_ano.flatten()))

plt.fill_between(months, obs_nino_ano_smooth, 0, where = (obs_nino_ano_smooth > 0), color='Orange',alpha=0.25)
plt.fill_between(months, obs_nino_ano_smooth, -0, where = (obs_nino_ano_smooth < -0), color='darkblue',alpha=0.25)
plt.fill_between(months, obs_nino_ano_smooth, 1, where = (obs_nino_ano_smooth > 1), color='Orange')
plt.fill_between(months, obs_nino_ano_smooth, -1, where = (obs_nino_ano_smooth < -1), color='darkblue')


if ofile is not None:
    ofile_long = f"HadISST_"+box+"_enso_box_index.png"
    plt.savefig(f"{out_path+ofile_long}", dpi=dpi,bbox_inches='tight')

    
    
# Obtain data
Ntotal = len(sst_nino_ano_smooth)
data = sst_nino_ano_smooth

# This is  the colormap I'd like to use.
try:
    cm = plt.colormaps['PuOr_r']
except AttributeError:
    cm = plt.cm.get_cmap('PuOr_r')

# Get the histogramp
nbins = 13
minbin = -3
maxbin = 3
bins = np.linspace(minbin,maxbin,nbins)

Y,X = np.histogram(data, bins=bins)
Y = (Y*100)/np.sum(Y)
x_span = X.max()-X.min()
corr=(x_span/nbins)/2
C = [cm(((x-X.min()+corr)/x_span)) for x in X]

fig, ax = plt.subplots(figsize=figsize)

plt.bar(X[:-1],Y,color=C,width=X[1]-X[0],edgecolor='black',align='edge')
plt.xlim((minbin, maxbin))
plt.title(historic_name+' '+box+' temperature anomaly distribution',fontweight="bold")
plt.ylabel("Occurance [%]",fontsize=13)
plt.xlabel("Temperature anomaly [°C]",fontsize=13)
ax.tick_params(labelsize=13)


if ofile is not None:
    ofile_long = f"{ofile}_"+box+"_enso_temperature_distribution.png"
    plt.savefig(f"{out_path+ofile_long}", dpi=dpi,bbox_inches='tight')
    
    
    
# Obtain data
Ntotal = len(obs_nino_ano_smooth)
data = obs_nino_ano_smooth

# This is  the colormap I'd like to use.
try:
    cm = plt.colormaps['PuOr_r']
except AttributeError:
    cm = plt.cm.get_cmap('PuOr_r')

# Get the histogramp
nbins = 13
minbin = -3
maxbin = 3
bins = np.linspace(minbin,maxbin,nbins)

Y,X = np.histogram(data, bins=bins)
Y = (Y*100)/np.sum(Y)
x_span = X.max()-X.min()
corr=(x_span/nbins)/2
C = [cm(((x-X.min()+corr)/x_span)) for x in X]

fig, ax = plt.subplots(figsize=figsize)

plt.bar(X[:-1],Y,color=C,width=X[1]-X[0],edgecolor='black',align='edge')
plt.xlim((minbin, maxbin))
plt.title('HadISST '+box+' temperature anomaly distribution',fontweight="bold")
plt.ylabel("Occurance [%]",fontsize=13)
plt.xlabel("Temperature anomaly [°C]",fontsize=13)

if ofile is not None:
    ofile_long = f"HadISST_"+box+"_enso_temperature_distribution.png"
    plt.savefig(f"{out_path+ofile_long}", dpi=dpi,bbox_inches='tight')

    
    

def smooth(x,beta):
    """ kaiser window smoothing """
    window_len=201
    beta=200
    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w = np.kaiser(window_len,beta)
    y = np.convolve(w/w.sum(),s,mode='valid')
    return y[100:len(y)-100]




f, Pxx_den = signal.periodogram(sst_nino_ano.flatten(),nfft=8000)
f_obs, Pxx_den_obs = signal.periodogram(obs_nino_ano.flatten(),nfft=8000)
#f_obsn, Pxx_den_obsn = signal.periodogram(obs_nino_ano.flatten(),nfft=250)

fig, ax = plt.subplots(figsize=figsize)

#ax.plot(f_obsn,Pxx_den_obsn/np.mean(Pxx_den_obsn),linewidth=1,color='orange',label='HadISST')

ax.semilogx(f_obs,smooth(Pxx_den_obs/np.mean(Pxx_den_obs),len(Pxx_den)),linewidth=2,color='darkblue',label='HadISST')
ax.semilogx(f,smooth(Pxx_den/np.mean(Pxx_den),len(Pxx_den)),linewidth=2,color='orange',label='AWI-CM3 HIST')


ax.set_xlim([0.0015, 0.1])
#ax.set_ylim([0.01, 25])
plt.xlabel('Frequency [Cycles/Month]',fontsize=13)
plt.ylabel('Normalized PSD',fontsize=13)
plt.legend(loc='upper left',fontsize=13)
#ax.set_xscale('log')
ax.tick_params(labelsize=13)

def twelve_over(x):
    """Vectorized 12/x, treating x==0 manually"""
    x = np.array(x*12).astype(float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = np.inf
    x[~near_zero] = 1 / x[~near_zero]
    return x

# the function "12/x" is its own inverse
inverse = twelve_over

#secax.set_xticks([50,12,8,6,5,4,3,2,1])
secax = ax.secondary_xaxis('top', functions=(twelve_over, inverse))
secax.set_xlabel('Period [Years]',fontsize=13)
secax.set_xlabel('Period [Years]',fontsize=13)
secax.xaxis.set_major_formatter(FormatStrFormatter("%1.f"))
secax.xaxis.set_minor_formatter(FormatStrFormatter("%1.f"))
secax.tick_params(axis='x', which='major', labelsize=11)
secax.tick_params(axis='x', which='minor', labelsize=11)

if ofile is not None:
    ofile_long = f"{ofile}_enso_"+box+"_box_norm_psd.png"
    plt.savefig(f"{out_path+ofile_long}", dpi=300,bbox_inches='tight')

# Mark script as completed
update_status(SCRIPT_NAME, "Completed")
