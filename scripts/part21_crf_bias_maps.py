# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

SCRIPT_NAME = os.path.basename(__file__)  # Get the current script name

print(SCRIPT_NAME)

# Mark as started
update_status(SCRIPT_NAME, " Started")

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from sklearn.metrics import mean_squared_error
from math import sqrt, ceil
from tqdm import tqdm

# Parameters
input_paths = [historic_path]
input_names = [historic_name]
exps = list(range(historic_last25y_start, historic_last25y_end+1))

climatology_file = 'CERES_EBAF_Ed4.1_Subset_CLIM01-CLIM12.nc'
climatology_path = observation_path + '/CERES/'
figsize = (6, 4.5)
contour_outline_thickness = 0

# Map plotting parameters - using same levels as part9_rad_vs_ceres.py
mapticks = [-50, -30, -20, -10, -6, -2, 2, 6, 10, 20, 30, 50]

def define_rowscol(input_paths, columns=len(input_paths), reduce=0):
    """Define subplot layout"""
    number_paths = len(input_paths) - reduce
    if number_paths < columns:
        ncol = number_paths
    else:
        ncol = columns
    nrows = ceil(number_paths / columns)
    return [nrows, ncol]

# Mean Deviation weighted
def md(predictions, targets, wgts):
    output_errors = np.average((predictions - targets), axis=0, weights=wgts)
    return (output_errors).mean()

def global_area_mean(da):
    """Calculate proper area-weighted global mean."""
    # Find latitude coordinate
    lat_coord = None
    for coord in ['lat', 'latitude', 'y']:
        if coord in da.coords:
            lat_coord = coord
            break
    
    if lat_coord is None:
        raise ValueError(f"No latitude coordinate found. Available coords: {list(da.coords.keys())}")
    
    # Calculate cosine latitude weights (proper area weighting for regular lat-lon grid)
    weights = np.cos(np.deg2rad(da[lat_coord]))
    
    # Compute weighted mean over spatial dimensions
    spatial_dims = [lat_coord]
    if 'lon' in da.dims:
        spatial_dims.append('lon')
    elif 'longitude' in da.dims:
        spatial_dims.append('longitude')
    
    da_global = da.weighted(weights).mean(dim=spatial_dims)
    
    return da_global

def load_model_crf_data(exp_path, exp_name, years):
    """Load model cloud radiative forcing data using xarray"""
    print(f"Loading model CRF data for {exp_name}...")
    
    # Variables needed for CRF calculation
    variables = ['tsr', 'tsrc', 'ttr', 'ttrc']  # TOA SW/LW all-sky and clear-sky
    data = {}
    
    for var in variables:
        print(f"  Loading {var}...")
        files = []
        for year in years:
            filepath = f"{exp_path}/oifs/atm_remapped_1m_{var}_1m_{year:04d}-{year:04d}.nc"
            if os.path.exists(filepath):
                files.append(filepath)
        
        if not files:
            print(f"ERROR: No files found for {var}")
            return None
            
        # Load and process files
        ds = xr.open_mfdataset(files, combine="by_coords", parallel=False,
                             chunks={'time_counter': 12}, 
                             decode_times=True, use_cftime=True,
                             combine_attrs='drop_conflicts')
        
        var_data = ds[var] / accumulation_period  # Normalize flux
        
        # Take time mean to get climatology
        var_clim = var_data.mean(dim='time_counter')
        data[var] = var_clim.compute()
    
    # Calculate cloud radiative forcing
    # SWCRF = All-sky - Clear-sky (negative means cooling)
    swcrf = data['tsr'] - data['tsrc']
    
    # LWCRF = All-sky - Clear-sky (positive means warming)  
    lwcrf = data['ttr'] - data['ttrc']
    
    # Net CRF = SWCRF + LWCRF
    net_crf = swcrf + lwcrf
    
    return {
        'swcrf': swcrf,
        'lwcrf': lwcrf, 
        'net_crf': net_crf
    }

def load_ceres_data():
    """Load CERES observational cloud radiative forcing data"""
    print("Loading CERES observational data...")
    
    ceres_file = os.path.join(climatology_path, climatology_file)
    if not os.path.exists(ceres_file):
        print(f"ERROR: CERES file not found: {ceres_file}")
        return None
        
    ds = xr.open_dataset(ceres_file)
    
    # Take annual mean of monthly climatology
    ceres_data = {
        'swcrf': ds['toa_cre_sw_clim'].mean(dim='ctime'),
        'lwcrf': ds['toa_cre_lw_clim'].mean(dim='ctime'),
        'net_crf': ds['toa_cre_net_clim'].mean(dim='ctime')
    }
    
    return ceres_data

def interpolate_ceres_to_model(ceres_data, model_lat, model_lon):
    """Interpolate CERES data to model grid"""
    print("Interpolating CERES data to model grid...")
    
    interpolated_data = {}
    for var_name, ceres_var in ceres_data.items():
        # Interpolate to model grid
        interpolated = ceres_var.interp(lat=model_lat, lon=model_lon, method='linear')
        interpolated_data[var_name] = interpolated
        
    return interpolated_data

def calculate_statistics(model_data, obs_data, lat):
    """Calculate area-weighted RMSD and mean bias"""
    # Create area weights
    coslat = np.cos(np.deg2rad(lat))
    weights = np.broadcast_to(coslat[:, np.newaxis], model_data.shape)
    
    # Calculate bias
    bias = model_data - obs_data
    
    # Create mask for valid data (not NaN)
    valid_mask = ~(np.isnan(model_data) | np.isnan(obs_data))
    
    if not np.any(valid_mask):
        print("  Warning: No valid data points for statistics")
        return np.nan, np.nan
    
    # Apply mask to data and weights
    model_valid = model_data[valid_mask]
    obs_valid = obs_data[valid_mask]
    weights_valid = weights[valid_mask]
    bias_valid = bias[valid_mask]
    
    # Area-weighted statistics
    rmsd = np.sqrt(np.average((model_valid - obs_valid)**2, weights=weights_valid))
    mean_bias = np.average(bias_valid, weights=weights_valid)
    
    return rmsd, mean_bias

# Main processing
print("=== Cloud Radiative Forcing Bias Maps ===")

# Load observational data
ceres_data = load_ceres_data()
if ceres_data is None:
    print("ERROR: Could not load CERES data")
    sys.exit(1)

# Create plots for each CRF component
crf_components = [
    ('swcrf', mapticks, 'Shortwave Cloud Radiative Forcing', 'PuOr_r'),
    ('lwcrf', mapticks, 'Longwave Cloud Radiative Forcing', 'PuOr_r'), 
    ('net_crf', mapticks, 'Net Cloud Radiative Forcing', 'PuOr_r')
]

for crf_type, levels, title_base, cmap in crf_components:
    print(f"\n=== Processing {title_base} ===")
    
    # Define figure layout
    nrows, ncol = define_rowscol(input_paths)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=figsize, 
                           subplot_kw={'projection': ccrs.PlateCarree()}, dpi=dpi)
    
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Process each experiment
    for i, (exp_path, exp_name) in enumerate(zip(input_paths, input_names)):
        print(f"  Processing {exp_name}...")
        ax = axes[i]
        
        # Load model data
        model_data = load_model_crf_data(exp_path, exp_name, exps)
        if model_data is None:
            print(f"  ERROR: Could not load model data for {exp_name}")
            continue
        
        # Get coordinates
        lat = model_data[crf_type].lat.values
        lon = model_data[crf_type].lon.values
        
        # Interpolate CERES data to model grid
        ceres_interp = interpolate_ceres_to_model(ceres_data, model_data[crf_type].lat, model_data[crf_type].lon)
        
        # Calculate bias
        model_vals = model_data[crf_type].values
        ceres_vals = ceres_interp[crf_type].values
        bias = model_vals - ceres_vals
        
        # Calculate statistics
        rmsd, mean_bias = calculate_statistics(model_vals, ceres_vals, lat)
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, zorder=3)
        
        # Contour plot
        imf = ax.contourf(lon, lat, bias, 
                         cmap=cmap, levels=levels, extend='both',
                         transform=ccrs.PlateCarree(), zorder=1)
        
        # Add contour lines
        line_colors = ['black' for _ in imf.levels]
        imc = ax.contour(lon, lat, bias, 
                        colors=line_colors, levels=levels, 
                        linewidths=contour_outline_thickness,
                        transform=ccrs.PlateCarree(), zorder=1)
        
        # Set title
        ax.set_title(f'{title_base} Bias\n{exp_name} - CERES EBAF', fontweight="bold")
        
        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.2, linestyle='-')
        gl.xlabels_bottom = False
        
        # Statistics text boxes
        textrmsd = f'rmsd={rmsd:.2f}'
        textbias = f'bias={mean_bias:.2f}'
        props = dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.5)
        
        ax.text(0.02, 0.35, textrmsd, transform=ax.transAxes, fontsize=13,
                verticalalignment='top', bbox=props, zorder=4)
        ax.text(0.02, 0.25, textbias, transform=ax.transAxes, fontsize=13,
                verticalalignment='top', bbox=props, zorder=4)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.15, 0.11, 0.7, 0.05])
    cbar_ax.tick_params(labelsize=12)
    cb = fig.colorbar(imf, cax=cbar_ax, orientation='horizontal', ticks=mapticks)
    cb.set_label(label="W/m²", size=14)
    cb.ax.tick_params(labelsize=12)
    
    # Save figure
    ofile = f'{crf_type}_bias_vs_ceres'
    plt.savefig(out_path + ofile, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {ofile}.png")

print(f"\n=== Cloud Radiative Forcing Bias Maps Completed ===")
print(f"Maps saved to: {out_path}")

# Mark as completed
update_status(SCRIPT_NAME, " Completed")
