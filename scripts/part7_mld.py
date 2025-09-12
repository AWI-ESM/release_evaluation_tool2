# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

SCRIPT_NAME = os.path.basename(__file__)  # Get the current script name

print(SCRIPT_NAME)

# Mark as started
update_status(SCRIPT_NAME, " Started")

mesh = pf.load_mesh(meshpath)



# -------------------------------
# Function Definitions
# -------------------------------

def define_rowscol(input_paths, columns=2, reduce=0):
    """Calculate the number of rows and columns for subplots."""
    number_paths = len(input_paths) - reduce
    ncol = min(number_paths, columns)
    nrows = math.ceil(number_paths / columns)
    return [nrows, ncol]

# Import weight file utility
from utils import ensure_weight_file

def load_parallel(variable, path, remap_resolution, meshpath, mesh_file):
    """Load data in parallel using CDO."""
    weight_file = ensure_weight_file(remap_resolution, meshpath, mesh_file)
    data1 = cdo.yseasmean(
        input=f'-setmissval,nan -setctomiss,0 -remap,r{remap_resolution},{weight_file} -setgrid,{meshpath}/{mesh_file} {path}',
        returnArray=variable
    )
    return data1

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Truncate a colormap."""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

def plot_data(variable, hemisphere, projection, extent, filename, levels, factor, new_cmap, extend):
    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(projection=projection)
    ax.add_feature(cfeature.COASTLINE, zorder=3)
    ax.set_extent(extent, ccrs.PlateCarree())
    
    depth_index = 3  # Choosing a consistent depth level
    data_2d = factor * data_model_mean[exp_name][depth_index, :, :]
    
    imf = ax.contourf(lon, lat, data_2d, cmap=new_cmap, levels=levels, extend=extend,
                      transform=ccrs.PlateCarree(), zorder=1)
    ax.contour(lon, lat, data_2d, levels=levels, colors='black', linewidths=0.2,
               transform=ccrs.PlateCarree(), zorder=1)
    
    ax.set_title(f"{variable} - {hemisphere}", fontweight="bold")
    cb = plt.colorbar(imf, orientation='horizontal', fraction=0.046, pad=0.04)
    cb.set_label(f"{variable} value")
    
    plt.savefig(os.path.join(out_path, filename), dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")



# -------------------------------
# Initialization
# -------------------------------

variables = ['MLD2', 'a_ice']
input_paths = [historic_path+'/fesom/']
input_names = [historic_name]
years = range(historic_last25y_start, historic_last25y_end+1)
batch_size = 20  # Process files in batches


# -------------------------------
# Data Loading
# -------------------------------

data = OrderedDict()
for variable in variables:
    for exp_path, exp_name in zip(input_paths, input_names):
        data[exp_name] = []
        for i in range(0, len(years), batch_size):
            batch_years = years[i : i + batch_size]
            t = [dask.delayed(load_parallel)(variable, f"{exp_path}/{variable}.fesom.{year}.nc", remap_resolution, meshpath, mesh_file) for year in tqdm(batch_years)]
            with ProgressBar():
                batch_data = dask.compute(*t, scheduler='threads')
            data[exp_name].extend(batch_data)
        data[exp_name] = np.squeeze(np.array(data[exp_name], dtype=object))


    # -------------------------------
    # Data Processing
    # -------------------------------

    data_model_mean = OrderedDict()
    for exp_name in input_names:
        data_model_mean[exp_name] = np.array(data[exp_name], dtype=np.float64)
        print(f"Checking {exp_name}: shape={np.shape(data_model_mean[exp_name])}")
        if np.size(data_model_mean[exp_name]) == 0:
            print(f"Error: {exp_name} contains no valid data!")
            continue  # Skip this experiment if it's empty
        if len(np.shape(data_model_mean[exp_name])) > 2:
            if np.isnan(data_model_mean[exp_name]).all():
                print(f"Warning: {exp_name} contains only NaNs. Filling with zeros.")
                data_model_mean[exp_name] = np.zeros_like(data_model_mean[exp_name][0])
            else:
                data_model_mean[exp_name] = np.nanmean(data_model_mean[exp_name], axis=0)
        print(f"Final shape after nanmean: {np.shape(data_model_mean[exp_name])}")

    lon_size = data_model_mean[historic_name].shape[-1]
    lat_size = data_model_mean[exp_name].shape[-2]
    lon = np.linspace(0, 360, lon_size, endpoint=False)
    lat = np.linspace(-90, 90, lat_size)
    data_model_mean[historic_name], lon = add_cyclic_point(data_model_mean[historic_name], coord=lon)

    # -------------------------------
    # Plotting Configuration
    # -------------------------------

    # Generate plots
    for exp_name in input_names:
        if variable == 'a_ice':
            levels = [1,10,20,30,40,50,60,70,80,90,100]
            factor = 100
            new_cmap = truncate_colormap(cmo.cm.ice, 0.15, 1)
            extend = 'min'
        else:
            levels = [0, 0.2, 0.5,  1,  2, 2.5,  3, 3.5, 4]
            factor = -0.001
            new_cmap = truncate_colormap(plt.cm.PuOr, 0.5, 1)
            extend = 'both'
        
        plot_data(variable, 'Southern Hemisphere', ccrs.SouthPolarStereo(), [-180, 180, -55, -90], f"{variable}_SH.png", levels, factor, new_cmap, extend)
        plot_data(variable, 'Northern Hemisphere', ccrs.NorthPolarStereo(), [-180, 180, 50, 90], f"{variable}_NH.png", levels, factor, new_cmap, extend)


# -------------------------------
# Completion
# -------------------------------
update_status(SCRIPT_NAME, " Completed")


