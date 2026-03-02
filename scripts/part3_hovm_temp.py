# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bg_routines.config_loader import *

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


# Import utils for dynamic batch sizing
try:
    from utils import get_optimal_batch_size
except ImportError:
    # If running from different directory
    sys.path.append(os.path.dirname(__file__))
    from utils import get_optimal_batch_size

# CDO helper: compute yearly global-mean depth profile for one file
def load_parallel_fldmean(variable, path, meshpath, mesh_file):
    data1 = cdo.yearmean(
        input=f'-fldmean -setctomiss,0 -setgrid,{meshpath}/{mesh_file} {path}',
        returnArray=variable
    )
    return np.squeeze(data1)

# Load reference data
path=reference_path+'/'+variable+'.fesom.'+str(reference_years)+'.nc'
print(f"Loading reference data from {path}...")
data_ref = load_parallel_fldmean(variable, path, meshpath, mesh_file)
# Average over time if multi-timestep
if data_ref.ndim > 1:
    data_ref = np.nanmean(data_ref, axis=0)
n_ref_depths = len(data_ref)
print(f"Reference data shape: {data_ref.shape} ({n_ref_depths} depth levels)")

# Calculate optimal batch size based on first file
sample_file = f"{input_paths[0]}/{variable}.fesom.{years[0]}.nc"
chunk_size = get_optimal_batch_size(sample_file, safety_factor=2.0, max_procs=16)

for exp_path, exp_name in zip(input_paths, input_names):
    print(f"Processing {exp_name} — CDO fldmean + Dask parallel...")
    
    file_paths = [f"{exp_path}/{variable}.fesom.{year}.nc" for year in years]
    
    datat = []
    for i in range(0, len(file_paths), chunk_size):
        chunk = file_paths[i:i + chunk_size]
        chunk_t = [dask.delayed(load_parallel_fldmean)(variable, f, meshpath, mesh_file) for f in chunk]
        with ProgressBar():
            datat_chunk = dask.compute(*chunk_t, scheduler='threads')
        datat.extend(datat_chunk)
        print(f"  Batch {i//chunk_size + 1}/{math.ceil(len(file_paths)/chunk_size)} done")
    
    data_full = np.array(datat, dtype=np.float32)
    n_common = min(n_ref_depths, data_full.shape[1])
    data[exp_name] = data_full[:, :n_common]
    
    print(f"Final shape: {np.shape(data[exp_name])} (aligned to {n_common} common levels)")
    print(f"Completed processing for {exp_name}")


    
# Read depths from 3D file, since mesh.zlevs is empty..
depths = pd.read_csv(meshpath+'/aux3d.out',  nrows=mesh.nlev) 

# Reshape data and expand reference climatology to length for data in preparation for Hovmöller diagram
data_ref_expand = OrderedDict()



# Data is already in shape (years, depths), just verify and use it
print(f"Data shape: {np.shape(data[exp_name])}")
print(f"Expected: ({len(years)}, {len(depths)-1})")

# Data is already in the correct format - no reshaping needed!
# Shape is (years, depths) which is what we need for the Hovmöller plot

# Align reference to common depth levels and expand
n_common = np.shape(data[exp_name])[1]
data_ref_expand = np.tile(data_ref[:n_common], (np.shape(data[exp_name])[0], 1))

# Debugging print to verify no NoneType issues
print(f"Final data shape: {np.shape(data[exp_name])}, dtype: {data[exp_name].dtype}")




# Flip data for contourf plot
data_diff = OrderedDict()
for exp_name in input_names:
    data_diff[exp_name]=data[exp_name]-data_ref_expand
    
# Prepare coordianates for contourf plot
if len(years) < 2:
    print("WARNING: Hovmoller diagram requires at least 2 years of data. Skipping plot.")
    update_status(SCRIPT_NAME, " Completed")
    sys.exit(0)

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
