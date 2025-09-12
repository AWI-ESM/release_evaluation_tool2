import sys
import os
import xarray as xr
import pandas as pd
import numpy as np
import dask
from dask.distributed import Client, as_completed, progress
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

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

# Generate correct area weights for the remapped grid using CDO
import subprocess
import tempfile

def get_area_weights():
    """Generate area weights that match the data grid"""
    sample_file = f"{spinup_path}/oifs/atm_remapped_1m_ssr_1m_{spinup_start}-{spinup_start}.nc"
    weights_file = "/tmp/radiation_area_weights.nc"
    
    # Generate area weights using CDO
    cmd = f"cdo gridarea {sample_file} {weights_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate area weights: {result.stderr}")
    
    # Load the generated weights
    area_ds = xr.open_dataset(weights_file, decode_times=False, engine="netcdf4")
    area_weights = area_ds['cell_area']  # CDO gridarea creates 'cell_area' variable
    area_weights_data = area_weights.load()  # Load into memory
    area_ds.close()
    
    return area_weights_data

try:
    area_weights = get_area_weights()
    print(f"Generated area weights: shape={area_weights.shape}, sum={area_weights.sum().values:.2e}")
except Exception as e:
    print(f"Error generating area weights: {e}")
    area_weights = None

if __name__ == "__main__":
    # Set up Dask client with a user-defined number of workers
    NUM_WORKERS = 3  # Adjust as needed
    client = Client(n_workers=NUM_WORKERS, threads_per_worker=1)
    print(client)

    def load_parallel(variable, path, area_weights_data):
        try:
            ds = xr.open_dataset(path, decode_times=False, engine="netcdf4")
            
            if variable not in ds:
                print(f"Warning: {variable} not found in {path}")
                return None
            
            da = ds[variable]
            
            time_var = "time_counter" if "time_counter" in ds else "time_centered" if "time_centered" in ds else None
            if not time_var:
                print(f"Error: No valid time variable found in {path}")
                return None
            
            da = da.rename({time_var: "time"})
            da["time"] = pd.to_datetime(ds[time_var].values, origin="1775-01-01", unit="s")
            
            # Recreate area weights DataArray from passed data
            area_weights_matched = xr.DataArray(
                area_weights_data,
                dims=['lat', 'lon'],
                coords={'lat': da.lat, 'lon': da.lon}
            )
            
            # Check if weights are valid (non-zero)
            if area_weights_matched.sum().values <= 0:
                raise ValueError("Area weights sum to zero or negative!")
            
            # Apply area weighting
            weighted_sum = (da * area_weights_matched).sum(dim=["lat", "lon"])
            weight_sum = area_weights_matched.sum()
            da = weighted_sum / weight_sum

            # Compute weighted yearly mean (to match CDO yearmean)
            da = da.resample(time="YE").mean(skipna=True)
            da = da / accumulation_period
            
            ds.close()
            return da.values
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return None

    data = {}
    
    # Convert area weights to numpy array for passing to workers
    if area_weights is None:
        raise RuntimeError("Area weights are required but not loaded!")
    
    area_weights_data = area_weights.values

    for v in var:
        print(f"Processing variable: {v}")
        futures = {}
        for exp in exps:
            path = f"{spinup_path}/oifs/atm_remapped_1m_{v}_1m_{exp:04d}-{exp:04d}.nc"
            futures[exp] = client.submit(load_parallel, v, path, area_weights_data)
        
        results = []
        for future in tqdm(as_completed(futures.values()), total=len(futures)):
            result = future.result()
            if result is not None:
                results.append(result)
        
        if results:
            data[v] = np.squeeze(np.array(results))
        else:
            data[v] = None
            print(f"Warning: No valid results for variable {v}")

    client.close()  # Shutdown Dask cluster when done

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
    ssr_vals = np.squeeze(data['ssr']).flatten()
    str_vals = np.squeeze(data['str']).flatten() 
    sshf_vals = np.squeeze(data['sshf']).flatten()
    slhf_vals = np.squeeze(data['slhf']).flatten()
    sf_vals = np.squeeze(data['sf']).flatten()
    
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
    tsr_vals = np.squeeze(data['tsr']).flatten()
    ttr_vals = np.squeeze(data['ttr']).flatten()
    
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
        tsrc_vals = np.squeeze(data['tsrc']).flatten()
        ttrc_vals = np.squeeze(data['ttrc']).flatten()
        
        # LWCF = ttr - ttrc (longwave cloud forcing)
        # Positive LWCF means clouds reduce OLR (trap longwave)
        lwcf_vals = ttr_vals - ttrc_vals
        
        # SWCF = tsr - tsrc (shortwave cloud forcing)
        swcf_vals = tsr_vals - tsrc_vals
        
        print(f"\n--- Cloud Forcing Components ---")
        print(f"TSRC (clear-sky TOA SW): min={np.min(tsrc_vals):>10.3f}, max={np.max(tsrc_vals):>10.3f}, mean={np.mean(tsrc_vals):>10.3f}")
        print(f"TTRC (clear-sky TOA LW): min={np.min(ttrc_vals):>10.3f}, max={np.max(ttrc_vals):>10.3f}, mean={np.mean(ttrc_vals):>10.3f}")
        
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
        window_len=11
        beta=10
        # extending the data at beginning and at the end
        # to apply the window at the borders
        s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
        w = np.kaiser(window_len,beta)
        y = np.convolve(w/w.sum(),s,mode='valid')
        return y[5:len(y)-5]

    fig, axes = plt.subplots(figsize=figsize)
    years = range(spinup_start, spinup_end+1)

    plt.plot(years,surface,linewidth=1,color='darkblue', label='_nolegend_')
    plt.plot(years,toa,linewidth=1,color='orange', label='_nolegend_')
    plt.plot(years,(toa-surface),linewidth=1,color='grey', label='_nolegend_')

    plt.plot(years,smooth(surface,len(surface)),color='darkblue')
    plt.plot(years,smooth(toa,len(toa)),color='orange')
    plt.plot(years,smooth((toa-surface),len(toa-surface)),color='grey')

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
