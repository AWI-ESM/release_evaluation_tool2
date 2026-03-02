# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bg_routines.config_loader import *

SCRIPT_NAME = os.path.basename(__file__)
print(SCRIPT_NAME)

# Mark as started
update_status(SCRIPT_NAME, " Started")

# Ocean Temperature Cross-Sections along Standard Multi-Segment Lines
figsize = (12, 6)
variable = 'temp'
ofile_atlantic = 'ocean_temp_section_atlantic.png'
ofile_pacific = 'ocean_temp_section_pacific.png'

# Standard WOCE-like transect coordinates (simplified to single segments)
# Atlantic A16-like transect (approximately 20°W meridional section)
atlantic_transect = {
    'name': 'Atlantic Meridional (20°W)',
    'lon_start': -20.0, 'lat_start': 65.0, 
    'lon_end': -10.0, 'lat_end': -60.0,
    'npoints': 100
}

# Pacific P16-like transect (approximately 150°W meridional section)  
pacific_transect = {
    'name': 'Pacific Meridional (150°W)',
    'lon_start': -150.0, 'lat_start': 60.0,
    'lon_end': -150.0, 'lat_end': -60.0, 
    'npoints': 100
}

# Years to average over
years = range(historic_last25y_start, historic_last25y_end + 1)
input_path = historic_path + '/fesom/'

# Import utils for dynamic batch sizing
try:
    from utils import get_optimal_batch_size
except ImportError:
    # If running from different directory
    sys.path.append(os.path.dirname(__file__))
    from utils import get_optimal_batch_size

def _cdo_timmean_one(variable, path, meshpath, mesh_file):
    """CDO timmean on one file → annual-mean 3D snapshot in memory."""
    data = cdo.timmean(
        input=f'-setgrid,{meshpath}/{mesh_file} {path}',
        returnArray=variable
    )
    # CDO timmean returns (1, nlev, nodes) or (1, nodes)
    # Safely remove time dimension
    if data.shape[0] == 1:
        return data[0, ...]
    return np.squeeze(data)

def load_and_average_temperature_data():
    """Load and average temperature data over specified years.
    Uses CDO timmean per file in parallel via Dask, then averages in numpy."""
    print(f"Loading temperature data for years {years[0]}-{years[-1]} — CDO + Dask parallel...")
    
    file_paths = [f"{input_path}/{variable}.fesom.{year}.nc" for year in years]
    existing = [f for f in file_paths if os.path.exists(f)]
    if not existing:
        raise ValueError(f"No temperature files found in {input_path}")
    
    # Calculate optimal batch size
    # 3D temp files are large (~4.4GB), so use safety factor 4.0
    chunk_size = get_optimal_batch_size(existing[0], safety_factor=4.0, max_procs=16)
    
    print(f"  Found {len(existing)} files, processing in parallel batches (batch_size={chunk_size})...")
    
    annual_means = []
    for i in range(0, len(existing), chunk_size):
        chunk = existing[i:i + chunk_size]
        tasks = [dask.delayed(_cdo_timmean_one)(variable, f, meshpath, mesh_file) for f in chunk]
        with ProgressBar():
            results = dask.compute(*tasks, scheduler='threads')
        
        # Check shapes
        for r in results:
            if annual_means and r.shape != annual_means[0].shape:
                 print(f"Warning: Shape mismatch! {r.shape} vs {annual_means[0].shape}")
        
        annual_means.extend(results)
        print(f"  Batch {i//chunk_size + 1}/{math.ceil(len(existing)/chunk_size)} done")
    
    # Ensure all arrays have same shape
    first_shape = annual_means[0].shape
    valid_means = [a for a in annual_means if a.shape == first_shape]
    if len(valid_means) < len(annual_means):
        print(f"Warning: Dropped {len(annual_means) - len(valid_means)} arrays due to shape mismatch")
        
    temp_avg = np.mean(valid_means, axis=0).astype(np.float32)
    
    # CDO returns (Level, Node), but PyFESOM functions expect (Node, Level)
    # Heuristic: Nodes (3M) > Levels (56)
    if temp_avg.shape[0] < temp_avg.shape[1]:
        print(f"Transposing data from {temp_avg.shape} to (Node, Level) format for PyFESOM...")
        temp_avg = temp_avg.T
        
    print(f"Temperature data shape: {temp_avg.shape}, averaged over {len(valid_means)} years")
    return temp_avg

def create_and_plot_transect(transect_info, temp_data, output_file):
    """Create transect using PyFESOM2 and plot it manually (plot_transect is deprecated)"""
    
    print(f"Creating transect: {transect_info['name']}")
    
    # Create transect coordinates using PyFESOM2
    lonlat = pf.transect_get_lonlat(
        transect_info['lon_start'], transect_info['lat_start'],
        transect_info['lon_end'], transect_info['lat_end'], 
        transect_info['npoints']
    )
    
    print(f"Transect coordinates created with {len(lonlat[0])} points")
    
    # Get transect data using get_transect
    # Returns: dist (1D array of distances), transect_data (2D array [npoints, nlevels])
    dist, transect_data = pf.get_transect(temp_data, mesh, lonlat)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare depth axis (using full depth levels or midpoints?)
    # transect_data shape is (npoints, nlevels). mesh.zlev has nlevels or nlevels+1?
    # Usually FESOM data is on layers. Let's assume mesh.zlev matches or we slice it.
    depths = mesh.zlev
    if len(depths) != transect_data.shape[1]:
         # Adjust depths if size mismatch (e.g. interfaces vs layers)
         depths = depths[:transect_data.shape[1]]
    
    # Plot using contourf
    # X=dist, Y=depths, Z=transect_data.T
    levels = np.arange(-2, 30, 0.5)
    cmap = cm.RdYlBu_r
    
    # Fill contours
    cs = ax.contourf(dist, depths, transect_data.T, levels=levels, cmap=cmap, extend='both')
    
    # Add contour lines
    major_isotherms = [0, 5, 10, 15, 20, 25]
    ax.contour(dist, depths, transect_data.T, levels=major_isotherms, colors='black', linewidths=0.5, alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label("Temperature (°C)")
    
    # Formatting
    ax.set_title(f"{transect_info['name']} - Multi-year Mean ({historic_last25y_start}-{historic_last25y_end})")
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_xlabel('Distance along transect (km)', fontsize=12)
    ax.invert_yaxis() # Depth goes down
    ax.set_ylim(5000, 0) # Limit to 5000m depth
    
    # Save plot
    plt.tight_layout()
    plt.savefig(out_path + output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Temperature section plot saved: {out_path + output_file}")

if __name__ == "__main__":
    try:
        # Load and average temperature data using PyFESOM2
        temp_avg = load_and_average_temperature_data()
        
        # Process Atlantic transect
        print(f"\nProcessing {atlantic_transect['name']} transect...")
        create_and_plot_transect(atlantic_transect, temp_avg, ofile_atlantic)
        
        # Process Pacific transect  
        print(f"\nProcessing {pacific_transect['name']} transect...")
        create_and_plot_transect(pacific_transect, temp_avg, ofile_pacific)
        
        print(f"\n=== Ocean Temperature Cross-Sections Complete ===")
        print(f"Atlantic section: {out_path + ofile_atlantic}")
        print(f"Pacific section: {out_path + ofile_pacific}")
        
    except Exception as e:
        print(f"Error in ocean temperature sections: {e}")
        import traceback
        traceback.print_exc()

# Mark as completed
update_status(SCRIPT_NAME, " Completed")
