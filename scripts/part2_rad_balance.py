# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

SCRIPT_NAME = os.path.basename(__file__)  # Get the current script name

print(SCRIPT_NAME)

# Mark as started
update_status(SCRIPT_NAME, " Started")


#Config
figsize=(7.2, 3.8)
var = ['ssr', 'str', 'tsr', 'ttr', 'sf', 'slhf', 'sshf'] 
exps = list(range(spinup_start, spinup_end+1))
ofile = "radiation_budget.png"
#var must have order:  
#1. Surface net solar radiation
#2. Surface net thermal radiation
#3. Top net solar radiation
#4. Top net thermal radiation

# Load model data using cdo
def load_parallel(variable,path):
    data1 = cdo.yearmean(input='-fldmean '+str(path),returnArray=variable)/accumulation_period
    return data1



import xarray as xr
import pandas as pd

def load_parallel_xr(variable, path):
    try:
        # Open dataset with correct backend
        ds = xr.open_dataset(path, decode_times=False, engine="netcdf4")

        # Ensure variable exists in dataset
        if variable not in ds:
            print(f"Warning: {variable} not found in {path}")
            return None

        da = ds[variable]  # Extract only the DataArray

        # Identify correct time variable (choose `time_counter` or `time_centered`)
        if "time_counter" in ds:
            time_var = "time_counter"
        elif "time_centered" in ds:
            time_var = "time_centered"
        else:
            print(f"Error: No valid time variable found in {path}")
            return None

        # Rename time dimension if needed
        if time_var in da.dims:
            da = da.rename({time_var: "time"})

        # Convert time from seconds since 1775-01-01 to datetime
        time_units = ds[time_var].attrs.get("units", "seconds since 1775-01-01 00:00:00")
        da["time"] = pd.to_datetime(ds[time_var].values, origin="1775-01-01", unit="s")

        # Compute spatial mean (latitude, longitude)
        da = da.mean(dim=["lat", "lon"])

        # Resample to yearly mean (equivalent to `cdo yearmean`)
        da = da.resample(time="YE").mean()  # 'YE' is the correct yearly frequency

        # Normalize by accumulation period
        da = da / accumulation_period

        ds.close()  # Close dataset to prevent locks

        return da.values  # Return NumPy array

    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

data = {}

for v in var:
    tasks = []
    for exp in tqdm(exps):
        path = f"{spinup_path}/oifs/atm_remapped_1m_{v}_1m_{exp:04d}-{exp:04d}.nc"
        tasks.append(dask.delayed(load_parallel)(v, path))  # Use delayed execution

    try:
        with ProgressBar():
            datat = dask.compute(*tasks, scheduler='threads')  # Use threaded execution

        # Ensure only valid (non-None) results are stored
        datat = [d for d in datat if d is not None]

    except Exception as e:
        print(f"Parallel loading failed for {v}, falling back to serial execution. Error: {e}")
        datat = []
        for exp in tqdm(exps):
            path = f"{spinup_path}/oifs/atm_remapped_1m_{v}_1m_{exp:04d}-{exp:04d}.nc"
            result = load_parallel(v, path)  # Serial execution
            if result is not None:
                datat.append(result)

    data[v] = np.squeeze(np.array(datat)) if datat else None  # Handle missing variables


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
