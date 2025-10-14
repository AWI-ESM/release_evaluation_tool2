import sys
import os
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Ensure safe multiprocessing
multiprocessing.set_start_method("fork", force=True)

# Add the parent directory to sys.path and load config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

SCRIPT_NAME = os.path.basename(__file__)
print(SCRIPT_NAME)
update_status(SCRIPT_NAME, "Started")

# Config
figsize = (7.2, 3.8)
var = ['ssr', 'str', 'tsr', 'ttr', 'tsrc', 'ttrc', 'sf', 'slhf', 'sshf']
exps = list(range(spinup_start, spinup_end + 1))
ofile = "radiation_budget.png"

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
    # This accounts for the fact that grid cells get smaller towards the poles
    weights = np.cos(np.deg2rad(da[lat_coord]))
    
    # For regular grids, this is equivalent to proper area weighting since:
    # - All longitude bands have the same Δlon
    # - Area ∝ cos(lat) * Δlat * Δlon, and Δlat is constant
    # - So relative weights are just cos(lat)
    
    # Compute weighted mean over spatial dimensions
    spatial_dims = [lat_coord]
    if 'lon' in da.dims:
        spatial_dims.append('lon')
    elif 'longitude' in da.dims:
        spatial_dims.append('longitude')
    
    da_global = da.weighted(weights).mean(dim=spatial_dims)
    
    return da_global

def load_yearly_data_simple(path, var, years, pattern, freq):
    """Load and process data to yearly means - optimized xarray approach"""
    print(f"Processing variable: {var}")
    
    files = []
    for year in years:
        filepath = os.path.join(path, pattern.format(year=year))
        if os.path.exists(filepath):
            files.append(filepath)
        else:
            print(f"WARNING: Missing file {filepath}")
    
    if not files:
        print(f"ERROR: No files found for variable '{var}'")
        return None
    
    try:
        print(f"Loading {len(files)} files for {var}...")
        
        # Load files with explicit time decoding to handle mixed calendar types
        ds = xr.open_mfdataset(files, combine="by_coords", parallel=False, 
                             chunks={'time_counter': 12}, 
                             decode_times=True, use_cftime=True,
                             combine_attrs='drop_conflicts')
        
        # Get the variable data
        var_data = ds[var]
        
        # Normalize by accumulation period ONLY for flux variables (not temperature)
        if var != '2t':  # Don't normalize temperature
            var_data = var_data / accumulation_period
        
        # Calculate global area mean
        global_mean = global_area_mean(var_data)
        
        # Convert to yearly means using groupby
        yearly_data = global_mean.groupby('time_counter.year').mean()
        
        # Force computation to avoid lazy evaluation
        yearly_data = yearly_data.compute()
        
        print(f"Loaded {var}: {len(files)} files -> {len(yearly_data)} yearly values")
        return yearly_data
        
    except Exception as e:
        print(f"ERROR loading {var}: {e}")
        return None

def detect_file_pattern(path, var, years):
    """Detect file pattern and frequency for a variable"""
    # Try 1m files first
    pattern_1m = f"atm_remapped_1m_{var}_1m_{{year:04d}}-{{year:04d}}.nc"
    test_file = os.path.join(path, pattern_1m.format(year=years[0]))
    if os.path.exists(test_file):
        return pattern_1m, "1m"
    
    # Try 6h files
    pattern_6h = f"atm_remapped_6h_{var}_6h_{{year:04d}}-{{year:04d}}.nc"
    test_file = os.path.join(path, pattern_6h.format(year=years[0]))
    if os.path.exists(test_file):
        return pattern_6h, "6h"
    
    return None, None

def load_spatial_data(variable, path):
    """Load spatial data for mapping using xarray"""
    try:
        if not os.path.exists(path):
            print(f"WARNING: File does not exist: {path}")
            return None
            
        # Load single file
        ds = xr.open_dataset(path, decode_times=True, use_cftime=True)
        
        # Get the variable data
        var_data = ds[variable]
        
        # Normalize by accumulation period ONLY for flux variables (not temperature)
        if variable != '2t':  # Don't normalize temperature
            var_data = var_data / accumulation_period
        
        # Take time mean if multiple time steps
        if 'time_counter' in var_data.dims:
            var_data = var_data.mean('time_counter')
            
        return var_data.values
            
    except Exception as e:
        print(f"ERROR processing spatial data {path}: {e}")
        return None

if __name__ == "__main__":
    print("Loading radiation balance data using proper xarray approach...")
    
    # Define years and path
    years = list(range(spinup_start, spinup_end + 1))
    
    # Detect file patterns and load data
    patterns = {}
    frequencies = {}
    data = {}
    
    required_vars = ['ssr', 'str', 'tsr', 'ttr', 'tsrc', 'ttrc', 'sf', 'slhf', 'sshf']
    
    print("Detecting file patterns...")
    for variable in required_vars:
        pattern, freq = detect_file_pattern(spinup_path + "/oifs", variable, years)
        if pattern is None:
            print(f"ERROR: No files found for {variable}")
            continue
        patterns[variable] = pattern
        frequencies[variable] = freq
        print(f"Found {variable} files: {freq} frequency")
    
    print("\nLoading data with proper area weighting...")
    for variable in required_vars:
        if variable in patterns:
            yearly_data = load_yearly_data_simple(spinup_path + "/oifs", variable, years, patterns[variable], frequencies[variable])
            if yearly_data is not None:
                data[variable] = yearly_data.values
            else:
                print(f"Failed to load {variable}")
                data[variable] = None

    print("\n=== DEBUG: Radiation Balance Calculation ===")
    
    # Check if all required variables are available
    required_vars = ['ssr', 'str', 'sshf', 'slhf', 'sf', 'tsr', 'ttr']
    cloud_forcing_vars = ['tsrc', 'ttrc']  # Optional for cloud forcing
    missing_vars = [v for v in required_vars if data.get(v) is None]
    missing_cf_vars = [v for v in cloud_forcing_vars if data.get(v) is None]
    
    if missing_vars:
        print(f"ERROR: Missing variables: {missing_vars}")
        print("Cannot calculate radiation balance without all required variables!")
        sys.exit(1)
    
    if missing_cf_vars:
        print(f"WARNING: Missing cloud forcing variables: {missing_cf_vars}")
        print("Cloud forcing calculations will be skipped.")
        calculate_cloud_forcing = False
    else:
        calculate_cloud_forcing = True
    
    # Debug individual components before calculation
    print("\n--- Individual Variable Statistics ---")
    all_vars = required_vars + cloud_forcing_vars
    for v in all_vars:
        if data.get(v) is not None:
            vals = np.squeeze(data[v]).flatten()
            print(f"{v:>6}: shape={vals.shape}, min={np.min(vals):>10.3f}, max={np.max(vals):>10.3f}, mean={np.mean(vals):>10.3f}, std={np.std(vals):>10.3f}")
        else:
            print(f"{v:>6}: None")

    #Calculate budget:
    # Data is now already global means for each year
    print(f"Using data from {len(data['ssr'])} years")
    
    # Data is already global means - no need for additional averaging
    ssr_vals = data['ssr']
    str_vals = data['str'] 
    sshf_vals = data['sshf']
    slhf_vals = data['slhf']
    sf_vals = data['sf']
    
    print(f"\n--- Surface Budget Components ---")
    print(f"SSR (shortwave down):     min={np.min(ssr_vals):>10.3f}, max={np.max(ssr_vals):>10.3f}, mean={np.mean(ssr_vals):>10.3f}")
    print(f"STR (longwave up):       min={np.min(str_vals):>10.3f}, max={np.max(str_vals):>10.3f}, mean={np.mean(str_vals):>10.3f}")
    print(f"SSHF (sensible heat):    min={np.min(sshf_vals):>10.3f}, max={np.max(sshf_vals):>10.3f}, mean={np.mean(sshf_vals):>10.3f}")
    print(f"SLHF (latent heat):      min={np.min(slhf_vals):>10.3f}, max={np.max(slhf_vals):>10.3f}, mean={np.mean(slhf_vals):>10.3f}")
    print(f"SF (snowfall):           min={np.min(sf_vals):>10.3f}, max={np.max(sf_vals):>10.3f}, mean={np.mean(sf_vals):>10.3f}")
    
    sf_heat_flux = sf_vals * 333550000  # Heat of fusion conversion
    print(f"SF heat flux (W/m²):     min={np.min(sf_heat_flux):>10.3f}, max={np.max(sf_heat_flux):>10.3f}, mean={np.mean(sf_heat_flux):>10.3f}")
    
    surface = ssr_vals + str_vals + sshf_vals + slhf_vals - sf_heat_flux
    print(f"Surface total:           min={np.min(surface):>10.3f}, max={np.max(surface):>10.3f}, mean={np.mean(surface):>10.3f}")
    
    #multiply by heat of fusion: 333550000 mJ/kg - then we get the flux in W/m2
    tsr_vals = data['tsr']
    ttr_vals = data['ttr']
    
    print(f"\n--- TOA Budget Components ---")
    print(f"TSR (TOA shortwave):     min={np.min(tsr_vals):>10.3f}, max={np.max(tsr_vals):>10.3f}, mean={np.mean(tsr_vals):>10.3f}")
    print(f"TTR (TOA longwave):      min={np.min(ttr_vals):>10.3f}, max={np.max(ttr_vals):>10.3f}, mean={np.mean(ttr_vals):>10.3f}")
    
    toa = tsr_vals + ttr_vals
    print(f"TOA total:               min={np.min(toa):>10.3f}, max={np.max(toa):>10.3f}, mean={np.mean(toa):>10.3f}")
    
    rad_balance = toa - surface
    print(f"\n--- Radiation Balance ---")
    print(f"Balance (TOA - Surface): min={np.min(rad_balance):>10.3f}, max={np.max(rad_balance):>10.3f}, mean={np.mean(rad_balance):>10.3f}")
    
    # Cloud forcing calculations
    if calculate_cloud_forcing:
        tsrc_vals = data['tsrc']
        ttrc_vals = data['ttrc']
        
        # LWCF = ttr - ttrc (longwave cloud forcing)
        # Positive LWCF means clouds reduce OLR (trap longwave)
        lwcf_vals = ttr_vals - ttrc_vals
        
        # SWCF = tsr - tsrc (shortwave cloud forcing)
        swcf_vals = tsr_vals - tsrc_vals
        
        
        print(f"\n--- Cloud Forcing ---")
        print(f"LWCF (LW cloud forcing): min={np.min(lwcf_vals):>10.3f}, max={np.max(lwcf_vals):>10.3f}, mean={np.mean(lwcf_vals):>10.3f}")
        print(f"SWCF (SW cloud forcing): min={np.min(swcf_vals):>10.3f}, max={np.max(swcf_vals):>10.3f}, mean={np.mean(swcf_vals):>10.3f}")
        
        # Net cloud forcing
        net_cf_vals = lwcf_vals + swcf_vals
        print(f"Net cloud forcing:       min={np.min(net_cf_vals):>10.3f}, max={np.max(net_cf_vals):>10.3f}, mean={np.mean(net_cf_vals):>10.3f}")
        
        print(f"\nInterpretation:")
        print(f"- Positive LWCF ({np.mean(lwcf_vals):>6.3f} W/m²) means clouds trap longwave radiation")
        print(f"- {'Positive' if np.mean(swcf_vals) > 0 else 'Negative'} SWCF ({np.mean(swcf_vals):>6.3f} W/m²) means clouds {'reduce' if np.mean(swcf_vals) < 0 else 'increase'} reflected shortwave")
        print(f"- Net cloud effect: {np.mean(net_cf_vals):>6.3f} W/m² ({'warming' if np.mean(net_cf_vals) > 0 else 'cooling'})")
    
    # Check for suspicious values
    if np.all(np.abs(surface) < 1e-6):
        print("WARNING: Surface fluxes are essentially zero - this suggests a data loading problem!")
    if np.all(np.abs(toa) < 1e-6):
        print("WARNING: TOA fluxes are essentially zero - this suggests a data loading problem!")
    if np.all(np.abs(rad_balance) < 1e-6):
        print("WARNING: Radiation balance is essentially zero - this is highly suspicious for a climate model!")

    #Plot:
    def smooth(x,beta):
        """ kaiser window smoothing """
        # Ensure window length is no longer than the timeseries
        window_len = min(11, len(x))
        if window_len < 3:
            return x  # No smoothing for very short series
        # Make window_len odd for proper centering
        if window_len % 2 == 0:
            window_len -= 1
        beta=10
        # extending the data at beginning and at the end
        # to apply the window at the borders
        s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
        w = np.kaiser(window_len,beta)
        y = np.convolve(w/w.sum(),s,mode='valid')
        # Adjust trimming based on window size
        trim = window_len//2
        return y[trim:len(y)-trim] if len(y) > 2*trim else x

    fig, axes = plt.subplots(figsize=figsize)
    years = range(spinup_start, spinup_start + len(data['ssr']))

    plt.plot(years,surface,linewidth=1,color='darkblue', label='_nolegend_')
    plt.plot(years,toa,linewidth=1,color='orange', label='_nolegend_')
    plt.plot(years,(toa-surface),linewidth=1,color='grey', label='_nolegend_')

    # Only plot smoothed lines if we have enough data points
    if len(surface) >= 3:
        surface_smooth = smooth(surface,len(surface))
        toa_smooth = smooth(toa,len(toa))
        balance_smooth = smooth((toa-surface),len(toa-surface))
        
        # Ensure smoothed arrays match the years array length
        if len(surface_smooth) == len(years):
            plt.plot(years,surface_smooth,color='darkblue')
            plt.plot(years,toa_smooth,color='orange')
            plt.plot(years,balance_smooth,color='grey')

    axes.set_title('Radiative balance',fontweight="bold")

    plt.axhline(y=0, color='black', linestyle='-')
    plt.ylabel('W/m²',size='13')
    plt.xlabel('Year',size='13')

    #plt.axvline(x=1650,color='grey',alpha=0.6)

    plt.axhline(y=0,color='grey',alpha=0.6)

    axes2 = axes.twinx()
    axes2.set_ylim(axes.get_ylim())

    axes.xaxis.set_minor_locator(MultipleLocator(10))
    # Fix matplotlib tick locator issue by using reasonable intervals
    y_range = abs(axes.get_ylim()[1] - axes.get_ylim()[0])
    minor_tick_interval = max(1.0, y_range / 50)  # Reasonable number of minor ticks
    axes.yaxis.set_minor_locator(MultipleLocator(minor_tick_interval))
    axes2.yaxis.set_minor_locator(MultipleLocator(minor_tick_interval))

    axes.tick_params(labelsize='12')
    axes2.tick_params(labelsize='12')

    axes.legend(['Net SFC', 'Net TOA', '\u0394(SFC - TOA)'],fontsize=11)
    plt.tight_layout()

    if ofile is not None:
        plt.savefig(out_path+ofile, dpi=dpi,bbox_inches='tight')

    # Load spatial data for map plots
    print("\n=== Loading spatial data for map plots ===")
    
    # Load all surface energy balance components for spatial maps
    ssr_spatial_list = []
    str_spatial_list = []
    sshf_spatial_list = []
    slhf_spatial_list = []
    sf_spatial_list = []
    
    # Load data from all available years using sequential processing
    years_to_process = list(range(spinup_start, spinup_end + 1))
    print(f"Loading spatial data for {len(years_to_process)} years sequentially...")
    
    # Process in chunks to avoid memory issues
    batch_size = 20  # Process in chunks to avoid memory issues
    data_batches = []
    
    for i in range(0, len(years_to_process), batch_size):
        batch = years_to_process[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(years_to_process) + batch_size - 1)//batch_size}")
        
        batch_data = []
        for year in tqdm(batch):
            ssr_path = f"{spinup_path}/oifs/atm_remapped_1m_ssr_1m_{year:04d}-{year:04d}.nc"
            str_path = f"{spinup_path}/oifs/atm_remapped_1m_str_1m_{year:04d}-{year:04d}.nc"
            sshf_path = f"{spinup_path}/oifs/atm_remapped_1m_sshf_1m_{year:04d}-{year:04d}.nc"
            slhf_path = f"{spinup_path}/oifs/atm_remapped_1m_slhf_1m_{year:04d}-{year:04d}.nc"
            sf_path = f"{spinup_path}/oifs/atm_remapped_1m_sf_1m_{year:04d}-{year:04d}.nc"
            
            # Load data for each variable sequentially
            ssr_data = load_spatial_data('ssr', ssr_path)
            str_data = load_spatial_data('str', str_path)
            sshf_data = load_spatial_data('sshf', sshf_path)
            slhf_data = load_spatial_data('slhf', slhf_path)
            sf_data = load_spatial_data('sf', sf_path)
            
            if all(data is not None for data in [ssr_data, str_data, sshf_data, slhf_data, sf_data]):
                # Take mean over time dimension if 3D
                if ssr_data.ndim == 3:
                    ssr_data = np.mean(ssr_data, axis=0)
                if str_data.ndim == 3:
                    str_data = np.mean(str_data, axis=0)
                if sshf_data.ndim == 3:
                    sshf_data = np.mean(sshf_data, axis=0)
                if slhf_data.ndim == 3:
                    slhf_data = np.mean(slhf_data, axis=0)
                if sf_data.ndim == 3:
                    sf_data = np.mean(sf_data, axis=0)
                
                ssr_spatial_list.append(ssr_data)
                str_spatial_list.append(str_data)
                sshf_spatial_list.append(sshf_data)
                slhf_spatial_list.append(slhf_data)
                sf_spatial_list.append(sf_data)
    
    # Calculate multi-year mean
    if ssr_spatial_list and str_spatial_list and sshf_spatial_list and slhf_spatial_list and sf_spatial_list:
        ssr_spatial = np.mean(ssr_spatial_list, axis=0)
        str_spatial = np.mean(str_spatial_list, axis=0)
        sshf_spatial = np.mean(sshf_spatial_list, axis=0)
        slhf_spatial = np.mean(slhf_spatial_list, axis=0)
        sf_spatial = np.mean(sf_spatial_list, axis=0)
    else:
        ssr_spatial = None
        str_spatial = None
        sshf_spatial = None
        slhf_spatial = None
        sf_spatial = None
    
    if all(data is not None for data in [ssr_spatial, str_spatial, sshf_spatial, slhf_spatial, sf_spatial]):
            
        # Calculate complete surface energy balance (same as global mean calculation)
        sf_heat_flux_spatial = sf_spatial * 333550000  # Heat of fusion conversion
        net_surface_balance = ssr_spatial + str_spatial + sshf_spatial + slhf_spatial - sf_heat_flux_spatial
        
        # Create coordinate arrays for plotting
        lon = np.linspace(0, 359, net_surface_balance.shape[1])
        lat = np.linspace(-90, 90, net_surface_balance.shape[0])
        
        # Calculate zonal mean
        zonal_mean = np.mean(net_surface_balance, axis=1, keepdims=True)
        
        # Calculate anomaly from zonal mean
        net_surface_anomaly = net_surface_balance - zonal_mean
        
        # Plot 1: Net Surface Radiation Map
        fig1, ax1 = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Create irregular spaced levels for better detail at different ranges
        vmax = max(abs(np.nanmin(net_surface_balance)), abs(np.nanmax(net_surface_balance)))
        
        # Define irregular levels with more detail near zero
        base_levels = np.array([-300, -100, -30, -10, -3, -1, 1, 3, 10, 30, 100, 300])
        
        # Use the full base levels - extend='both' will handle values beyond ±300
        levels = base_levels.copy()
        
        # Create colors for each level interval - direct mapping
        colors = []
        
        for i in range(len(levels) - 1):
            level_low = levels[i]
            level_high = levels[i+1]
            
            # Direct mapping based on exact level boundaries
            if level_low == -300 and level_high == -100:
                colors.append('#053061')  # Darkest blue
            elif level_low == -100 and level_high == -30:
                colors.append('#2166ac')  # Dark blue
            elif level_low == -30 and level_high == -10:
                colors.append('#4393c3')  # Medium blue
            elif level_low == -10 and level_high == -3:
                colors.append('#92c5de')  # Light blue
            elif level_low == -3 and level_high == -1:
                colors.append('#d1e5f0')  # Very light blue
            elif level_low == -1 and level_high == 1:
                colors.append('#ffffff')   # White
            elif level_low == 1 and level_high == 3:
                colors.append('#fde0dd')  # Very light red
            elif level_low == 3 and level_high == 10:
                colors.append('#f4a582')  # Light red
            elif level_low == 10 and level_high == 30:
                colors.append('#d6604d')  # Medium red
            elif level_low == 30 and level_high == 100:
                colors.append('#b2182b')  # Dark red
            elif level_low == 100 and level_high == 300:
                colors.append('#67001f')  # Darkest red
        
        # Create colormap and normalization with extend colors for values beyond ±300
        custom_cmap = ListedColormap(colors)
        custom_cmap.set_under('#000033')  # Ultra dark blue for < -300
        custom_cmap.set_over('#330000')   # Ultra dark red for > +300
        norm = BoundaryNorm(levels, ncolors=len(colors))
        
        # Plot the data
        im1 = ax1.contourf(lon, lat, net_surface_balance, 
                          levels=levels, cmap=custom_cmap, norm=norm, transform=ccrs.PlateCarree(), extend='both')
        
        # Add map features
        ax1.add_feature(cfeature.COASTLINE)
        ax1.add_feature(cfeature.BORDERS)
        ax1.set_global()
        ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.1, shrink=0.8)
        cbar1.set_label('Net Surface Energy Balance (W/m²)', fontsize=12)
        
        ax1.set_title(f'Net Surface Energy Balance - Multi-year Mean ({spinup_start}-{spinup_end})', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(out_path+'net_surface_radiation_map.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Net Surface Radiation Anomaly Map
        fig2, ax2 = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Use same levels for anomaly plot
        levels_anom = base_levels.copy()
        
        # Create colors for anomaly (same direct mapping)
        colors_anom = []
        
        for i in range(len(levels_anom) - 1):
            level_low = levels_anom[i]
            level_high = levels_anom[i+1]
            
            # Direct mapping based on exact level boundaries
            if level_low == -300 and level_high == -100:
                colors_anom.append('#053061')  # Darkest blue
            elif level_low == -100 and level_high == -30:
                colors_anom.append('#2166ac')  # Dark blue
            elif level_low == -30 and level_high == -10:
                colors_anom.append('#4393c3')  # Medium blue
            elif level_low == -10 and level_high == -3:
                colors_anom.append('#92c5de')  # Light blue
            elif level_low == -3 and level_high == -1:
                colors_anom.append('#d1e5f0')  # Very light blue
            elif level_low == -1 and level_high == 1:
                colors_anom.append('#ffffff')   # White
            elif level_low == 1 and level_high == 3:
                colors_anom.append('#fde0dd')  # Very light red
            elif level_low == 3 and level_high == 10:
                colors_anom.append('#f4a582')  # Light red
            elif level_low == 10 and level_high == 30:
                colors_anom.append('#d6604d')  # Medium red
            elif level_low == 30 and level_high == 100:
                colors_anom.append('#b2182b')  # Dark red
            elif level_low == 100 and level_high == 300:
                colors_anom.append('#67001f')  # Darkest red
        
        custom_cmap_anom = ListedColormap(colors_anom)
        custom_cmap_anom.set_under('#000033')  # Ultra dark blue for < -300
        custom_cmap_anom.set_over('#330000')   # Ultra dark red for > +300
        norm_anom = BoundaryNorm(levels_anom, ncolors=len(colors_anom))
        
        im2 = ax2.contourf(lon, lat, net_surface_anomaly, 
                          levels=levels_anom, cmap=custom_cmap_anom, norm=norm_anom, transform=ccrs.PlateCarree(), extend='both')
        
        # Add map features
        ax2.add_feature(cfeature.COASTLINE)
        ax2.add_feature(cfeature.BORDERS)
        ax2.set_global()
        ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.1, shrink=0.8)
        cbar2.set_label('Net Surface Energy Balance Anomaly from Zonal Mean (W/m²)', fontsize=12)
        
        ax2.set_title(f'Net Surface Energy Balance - Zonal Mean Anomaly - Multi-year Mean ({spinup_start}-{spinup_end})', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(out_path+'net_surface_radiation_anomaly.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Map plots saved:")
        print(f"  - Net surface energy balance map: {out_path}net_surface_radiation_map.png")
        print(f"  - Net surface energy balance anomaly: {out_path}net_surface_radiation_anomaly.png")
    else:
        print("Warning: Could not load spatial data for map plots")

    # Mark as completed
    update_status(SCRIPT_NAME, " Completed")
