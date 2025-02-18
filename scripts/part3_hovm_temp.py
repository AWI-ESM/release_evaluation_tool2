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


# Load reference data
path=reference_path+'/'+variable+'.fesom.'+str(reference_years)+'.nc'
data_ref = cdo.yearmean(input='-fldmean -setctomiss,0 -remap,r'+remap_resolution+','+meshpath+'/weights_unstr_2_r'+remap_resolution+'.nc -setgrid,'+meshpath+'/'+mesh_file+' '+str(path),returnArray=variable)
data_ref = np.squeeze(data_ref)

def load_parallel(variable,path,remap_resolution,meshpath,mesh_file):
    data1 = cdo.yearmean(input='-fldmean -setctomiss,0 -remap,r'+remap_resolution+','+meshpath+'/weights_unstr_2_r'+remap_resolution+'.nc -setgrid,'+meshpath+'/'+mesh_file+' '+str(path),returnArray=variable)
    return data1


batch_size = 50  # Limit to 50 files at a time

for exp_path, exp_name in zip(input_paths, input_names):
    data[exp_name] = []
    temp_results = []

    # Process in batches
    for i in range(0, len(years), batch_size):
        batch_years = years[i : i + batch_size]  # Get a batch of 50 years
        t = []

        for year in batch_years:
            path = f"{exp_path}/{variable}.fesom.{year}.nc"
            temp = dask.delayed(load_parallel)(variable, path, remap_resolution, meshpath, mesh_file)
            t.append(temp)

        with ProgressBar():
            batch_data = dask.compute(*t)  # Compute only the current batch
            temp_results.extend(batch_data)
    data[exp_name] = np.array(data[exp_name], dtype=np.float64)
    data[exp_name] = np.squeeze(np.array(temp_results, dtype=object))
    print(np.shape(data[exp_name]))
    print(data[exp_name])
    #data[exp_name] = np.squeeze(temp_results)


    
# Read depths from 3D file, since mesh.zlevs is empty..
depths = pd.read_csv(meshpath+'/aux3d.out',  nrows=mesh.nlev) 

# Reshape data and expand reference climatology to length for data in preparation for Hovmöller diagram
data_ref_expand = OrderedDict()



# Get the shape of the data
data_shape = np.shape(data[exp_name][0])
print(f"Data shape for first entry: {data_shape}")
print(f"Expected depth count: {len(depths)-1}")
print(f"Number of years in data: {len(data[exp_name])}")

# Initialize newdata array
newdata = np.empty((len(depths)-1, len(data[exp_name])), dtype=np.float64)  # Avoid NoneType issues
print(f"Newdata shape: {np.shape(newdata)}")

# Iterate through the data and populate newdata
for i in range(len(data[exp_name])):
    try:
        reshaped_data = np.array(data[exp_name][i]).squeeze()  # Remove unnecessary dimensions
        
        # Handle different cases
        if reshaped_data.shape == (len(depths)-1,):
            newdata[:, i] = reshaped_data  # Correct shape
        elif len(reshaped_data.shape) > 1 and reshaped_data.shape[0] > 1:
            print(f"Warning: Multiple entries at year index {i}, taking the first one.")
            newdata[:, i] = reshaped_data[0]  # Take the first entry if multiple exist
        else:
            raise ValueError(f"Unexpected shape {reshaped_data.shape} at year index {i}")  # Catch any unhandled cases

    except Exception as e:
        print(f"Skipping index {i} due to error: {e}")
        newdata[:, i] = np.nan  # Assign NaN to prevent NoneType issues

# Transpose and update data
data[exp_name] = newdata.T

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
