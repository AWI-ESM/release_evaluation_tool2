import sys
import os
import xarray as xr
import pandas as pd
import numpy as np
import dask
from dask.distributed import Client, as_completed, progress
from tqdm import tqdm
import multiprocessing

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
var = ['ssr', 'str', 'tsr', 'ttr', 'sf', 'slhf', 'sshf']
exps = list(range(spinup_start, spinup_end + 1))
ofile = "radiation_budget.png"

# Load cell area weights from external file (lazy loading)
area_file = f"{spinup_path}/../restart/oasis3mct/areas.nc"
area_variable = f"{oasis_oifs_grid_name}.srf"
try:
    area_ds = xr.open_dataset(area_file, decode_times=False, engine="netcdf4", chunks={})
    area_weights = area_ds[area_variable].load()  # Load into memory once, it's small

    # Rename dimensions to match expected ones
    if f"y_{oasis_oifs_grid_name}" in area_weights.dims and f"x_{oasis_oifs_grid_name}" in area_weights.dims:
        area_weights = area_weights.rename({f"y_{oasis_oifs_grid_name}": "lat", f"x_{oasis_oifs_grid_name}": "lon"})

    area_ds.close()
except Exception as e:
    print(f"Error loading area file: {e}")
    area_weights = None

if __name__ == "__main__":
    # Set up Dask client with a user-defined number of workers
    NUM_WORKERS = 3  # Adjust as needed
    client = Client(n_workers=NUM_WORKERS, threads_per_worker=1)
    print(client)

    def load_parallel(variable, path):
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
            
            # Apply preloaded area weights if available
            if area_weights is not None:
                try:
                    # Interpolate area weights to match da's grid
                    area_weights_matched = area_weights.interp_like(da, method="nearest")
                    da = (da * area_weights_matched).sum(dim=["lat", "lon"]) / area_weights_matched.sum()
                except Exception as e:
                    print(f"Error applying area weights: {e}")
                    da = da.mean(dim=["lat", "lon"])  # Fallback
            else:
                da = da.mean(dim=["lat", "lon"])  # No weights available

            # Compute weighted yearly mean (to match CDO yearmean)
            da = da.resample(time="YE").mean(skipna=True)
            da = da / accumulation_period
            ds.close()
            return da.values
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return None

    data = {}

    for v in var:
        futures = {}
        for exp in exps:
            path = f"{spinup_path}/oifs/atm_remapped_1m_{v}_1m_{exp:04d}-{exp:04d}.nc"
            futures[exp] = client.submit(load_parallel, v, path)
        
        results = []
        for future in tqdm(as_completed(futures.values()), total=len(futures)):
            result = future.result()
            if result is not None:
                results.append(result)
        
        data[v] = np.squeeze(np.array(results)) if results else None

    client.close()  # Shutdown Dask cluster when done

    #Calculate budget:
    surface =   np.squeeze(data['ssr']).flatten() + \
                np.squeeze(data['str']).flatten() + \
                np.squeeze(data['sshf']).flatten() + \
                np.squeeze(data['slhf']).flatten() - \
                np.squeeze(data['sf']).flatten()*333550000 
    #multiply by heat of fusion: 333550000 mJ/kg - then we get the flux in W/m2
    toa = np.squeeze(data['tsr']).flatten() + \
          np.squeeze(data['ttr']).flatten()

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
    axes.yaxis.set_minor_locator(MultipleLocator(0.2))
    axes2.yaxis.set_minor_locator(MultipleLocator(0.2))

    axes.tick_params(labelsize='12')
    axes2.tick_params(labelsize='12')

    axes.legend(['Net SFC', 'Net TOA', '\u0394(SFC - TOA)'],fontsize=11)
    plt.tight_layout()

    if ofile is not None:
        plt.savefig(out_path+ofile, dpi=dpi,bbox_inches='tight')


    # Mark as completed
    update_status(SCRIPT_NAME, " Completed")
