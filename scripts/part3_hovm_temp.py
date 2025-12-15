# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

SCRIPT_NAME = os.path.basename(__file__)  # Get the current script name

print(SCRIPT_NAME)

# Mark as started
update_status(SCRIPT_NAME, " Started")

# # Hovmöller diagram Temperature
figsize=(7.2, 3.8)

# Load model Data
data = OrderedDict()

variable='temp'
ofile = 'Hovmoeller_'+variable+'.png'

input_paths = [spinup_path+'/fesom/']
input_names = [spinup_name]
years = range(spinup_start, spinup_end+1)

maxdepth = 10000

levels = [-1.5, 1.5, 11]
mapticks = np.arange(levels[0],levels[1],0.1)


# Import weight file utility
from utils import ensure_weight_file

# Ensure weight file exists before processing
weight_file = ensure_weight_file(remap_resolution, meshpath, mesh_file)

# Load reference data
path=reference_path+'/'+variable+'.fesom.'+str(reference_years)+'.nc'
data_ref = cdo.yearmean(input='-fldmean -setctomiss,0 -remap,r'+remap_resolution+','+weight_file+' -setgrid,'+meshpath+'/'+mesh_file+' '+str(path),returnArray=variable)
data_ref = np.squeeze(data_ref)

import xarray as xr
import pyfesom2 as pf

def compute_global_mean_by_depth(var_data):
    """Compute global mean for each depth level and year using vectorized operations"""
    
    # Replace 0 with NaN (same as CDO's setctomiss,0)
    var_data = var_data.where(var_data != 0)
    
    # Compute mean across spatial dimension (nodes) - matches CDO's fldmean
    # This is a simple mean, not area-weighted, matching the original CDO behavior
    global_mean = var_data.mean(dim='nod2')
    
    # Compute and return as numpy array
    print("Computing global means (this may take a minute)...")
    result = global_mean.compute()
    
    return result.values.astype(np.float32)

for exp_path, exp_name in zip(input_paths, input_names):
    print(f"Processing {exp_name} using PyFESOM2/xarray approach...")
    
    # Build file paths
    file_paths = [f"{exp_path}/{variable}.fesom.{year}.nc" for year in years]
    print(f"Opening {len(file_paths)} files with xarray.open_mfdataset...")
    
    # Open all files at once with xarray (lazy loading)
    # parallel=False to avoid NetCDF thread-safety issues
    dataset = xr.open_mfdataset(
        file_paths, 
        combine='by_coords', 
        parallel=False,
        decode_times=True,
        use_cftime=True,
        chunks={'time': 12}  # Chunk by month for efficiency
    )
    
    # Get variable and compute yearly means (lazy)
    print("Grouping by year and computing temporal means...")
    var_data = dataset[variable]
    yearly_data = var_data.groupby('time.year').mean('time')
    
    # Compute global means for each depth level
    print("Computing global means for each depth level...")
    data[exp_name] = compute_global_mean_by_depth(yearly_data)
    
    print(f"Final shape: {np.shape(data[exp_name])}")
    print(f"Completed processing for {exp_name}")
    
    # Close dataset
    dataset.close()


    
# Read depths from 3D file, since mesh.zlevs is empty..
depths = pd.read_csv(meshpath+'/aux3d.out',  nrows=mesh.nlev) 

# Reshape data and expand reference climatology to length for data in preparation for Hovmöller diagram
data_ref_expand = OrderedDict()



# Data is already in shape (years, depths), just verify and use it
print(f"Data shape: {np.shape(data[exp_name])}")
print(f"Expected: ({len(years)}, {len(depths)-1})")

# Data is already in the correct format - no reshaping needed!
# Shape is (years, depths) which is what we need for the Hovmöller plot

# Expand reference data
data_ref_expand = np.tile(data_ref, (np.shape(data[exp_name])[0], 1))

# Debugging print to verify no NoneType issues
print(f"Final data shape: {np.shape(data[exp_name])}, dtype: {data[exp_name].dtype}")




# Flip data for contourf plot
data_diff = OrderedDict()
for exp_name in input_names:
    data_diff[exp_name]=data[exp_name]-data_ref_expand
    
# Prepare coordianates for contourf plot
X,Y = np.meshgrid(years,depths[:len(depths)-1])

# Calculate number of rows and columns for plot
def define_rowscol(input_paths, columns=len(input_paths), reduce=0):
    number_paths = len(input_paths) - reduce
#     columns = columns
    if number_paths < columns:
        ncol = number_paths
    else:
        ncol = columns
    nrows = math.ceil(number_paths / columns)
    return [nrows, ncol]

class MinorSymLogLocator(mticker.Locator):
    """
    Dynamically find minor tick positions based on the positions of major ticks for a symlog scaling.
    
    Attributes
    ----------
    linthresh : float
        The same linthresh value used when setting the symlog scale.
        
    """
    
    def __init__(self, linthresh):
        #super().__init__()
        self.linthresh = linthresh

    def __call__(self):
        majorlocs = self.axis.get_majorticklocs()
        # iterate through minor locs
        minorlocs = []
        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)
        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a {0} type.'.format(type(self)))

        
# Plot data and save it.
mapticks = np.arange(levels[0],levels[1],0.1)

nrows, ncols = define_rowscol(input_paths)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols,figsize[1]*nrows))


if isinstance(axes, np.ndarray):
    axes = axes.flatten()
else:
    axes = [axes]


i = 0
for exp_name in data_diff:
    data_diff[exp_name] = np.array(data_diff[exp_name], dtype=np.float64)
    im = axes[i].contourf(X,Y,data_diff[exp_name].T,levels=mapticks, cmap=cm.PuOr_r, extend='both')
    axes[i].set_title('Global ocean temperature bias',fontweight="bold")
    axes[i].set_ylabel('Depth [m]',size=13)
    axes[i].set_xlabel('Year',size=13)
    axes[i].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axes[i].set_ylim(-maxdepth)
    axes[i].set_yscale('symlog')
    axes[i].xaxis.set_minor_locator(MultipleLocator(10))

    axes[i].yaxis.set_minor_locator(MinorSymLogLocator(-10e4))
    # turn minor ticks off
    #axes[i].yaxis.set_minor_locator(NullLocator())



    i = i+1
fig.tight_layout()

if variable == "temp":
    label='°C'
elif variable == "salt":
    label='PSU'
    
try:
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),location='bottom', label=label, shrink=0.8, aspect=30, pad=0.2)
except:
    cbar = fig.colorbar(im, ax=axes,location='bottom', label=label, shrink=0.8, aspect=25, pad=0.16)

cbar.ax.tick_params(labelsize=12) 
cbar.ax.tick_params(labelsize=12) 



if out_path+ofile is not None:
    plt.savefig(out_path+ofile, dpi=dpi,bbox_inches='tight')

# Mark as completed
update_status(SCRIPT_NAME, " Completed")
