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
from bg_routines.config_loader import *
from bg_routines.ipcc_cmaps import IPCC_LINE

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
        
        # Load files with explicit time decoding to handle mixed calendar types.
        # AWI-CM3 XIOS uses `time_counter`; the AWI-ESM2 echam preprocessor
        # uses `time`.
        ds = xr.open_mfdataset(files, combine="by_coords", parallel=False,
                             decode_times=True, use_cftime=True,
                             combine_attrs='drop_conflicts')
        time_dim = 'time_counter' if 'time_counter' in ds.dims else 'time'
        ds = ds.chunk({time_dim: 12})

        # Get the variable data
        var_data = ds[var]

        # Normalize by accumulation period ONLY for flux variables (not temperature)
        if var != '2t':  # Don't normalize temperature
            var_data = var_data / accumulation_period

        # Calculate global area mean
        global_mean = global_area_mean(var_data)

        # Convert to yearly means using groupby
        yearly_data = global_mean.groupby(f'{time_dim}.year').mean()
        
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

if __name__ == "__main__":
    print("Loading radiation balance data using proper xarray approach...")
    
    # Radiative drift over the spinup is the diagnostic of interest;
    # historic period is too short to show convergence behaviour.
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
    
    # Choose sf-to-heat-flux conversion factor based on the raw sf units.
    #   AWI-ESM3 XIOS:           sf in 'm' (water-equivalent depth, accumulated)
    #                            -> after /accumulation_period it's m/s,
    #                               so multiply by rho_water * Lf = 1000 * 334_000.
    #   AWI-ESM2 echam preproc:  sf in 'kg m-2 s-1' (mass flux, accumulation_period=1)
    #                            -> multiply by Lf = 334_000.
    sf_first_file = os.path.join(spinup_path + "/oifs", patterns['sf'].format(year=years[0]))
    try:
        _sf_units = xr.open_dataset(sf_first_file)['sf'].attrs.get('units', '').strip()
    except Exception:
        _sf_units = ''
    if _sf_units == 'kg m-2 s-1':
        sf_conv = 334_000.0           # heat of fusion of ice
    elif _sf_units == 'm':
        sf_conv = 333_550_000.0       # rho_water * heat of fusion (water-equivalent depth)
    else:
        print(f"WARN: unrecognized sf units '{_sf_units}', defaulting to AWI-ESM3 (rho*Lf=3.34e8)")
        sf_conv = 333_550_000.0
    print(f"sf units = '{_sf_units}' -> sf_conv = {sf_conv:g}")
    sf_heat_flux = sf_vals * sf_conv
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
    def smooth(x, window_len=None):
        """Kaiser-window low-pass smoothing.

        With a 3830-year spinup an 11-year window produced a near-raw
        noise carpet; scale the window to ~5% of series length (with
        odd 31..301 cap) so multidecadal drift is the dominant signal.
        """
        if window_len is None:
            window_len = max(31, min(301, (len(x) // 20) | 1))
        if len(x) < window_len or window_len < 3:
            return np.asarray(x)
        if window_len % 2 == 0:
            window_len -= 1
        beta = 10
        s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
        w = np.kaiser(window_len, beta)
        y = np.convolve(w / w.sum(), s, mode='valid')
        trim = window_len // 2
        return y[trim:len(y) - trim]

    fig, axes = plt.subplots(figsize=figsize)
    years = range(spinup_start, spinup_start + len(data['ssr']))

    # Plot the smoothed running mean only — annual values were too noisy
    # to read across a multi-millennium spinup. Light raw lines kept at
    # very low alpha to hint at variability without dominating.
    plt.plot(years, surface, linewidth=0.4, color=IPCC_LINE['cool'], alpha=0.15, label='_nolegend_')
    plt.plot(years, toa,     linewidth=0.4, color=IPCC_LINE['warm'], alpha=0.15, label='_nolegend_')
    plt.plot(years, (toa - surface), linewidth=0.4, color='grey',   alpha=0.15, label='_nolegend_')

    if len(surface) >= 31:
        surface_smooth = smooth(surface)
        toa_smooth     = smooth(toa)
        balance_smooth = smooth(toa - surface)
        if len(surface_smooth) == len(years):
            plt.plot(years, surface_smooth, color=IPCC_LINE['cool'], linewidth=1.6)
            plt.plot(years, toa_smooth,     color=IPCC_LINE['warm'], linewidth=1.6)
            plt.plot(years, balance_smooth, color='grey',            linewidth=1.6)

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

    # Mark as completed
    update_status(SCRIPT_NAME, " Completed")
