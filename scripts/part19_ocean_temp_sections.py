# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

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

def load_and_average_temperature_data():
    """Load and average temperature data over specified years using PyFESOM2"""
    print(f"Loading temperature data for years {years[0]}-{years[-1]}...")
    
    # Use PyFESOM2 to load data
    temp_data = pf.get_data(input_path, variable, years, mesh, depth=None)
    
    print(f"Temperature data shape: {temp_data.shape}")
    print(f"Successfully loaded {len(years)} years of data")
    
    return temp_data

def create_and_plot_transect(transect_info, temp_data, output_file):
    """Create transect using PyFESOM2 and plot it"""
    
    print(f"Creating transect: {transect_info['name']}")
    
    # Create transect coordinates using PyFESOM2
    lonlat = pf.transect_get_lonlat(
        transect_info['lon_start'], transect_info['lat_start'],
        transect_info['lon_end'], transect_info['lat_end'], 
        transect_info['npoints']
    )
    
    print(f"Transect coordinates created with {len(lonlat[0])} points")
    
    # Create the plot using PyFESOM2's plot_transect function
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot transect with PyFESOM2
    cs = pf.plot_transect(
        temp_data, mesh, lonlat,
        maxdepth=5000,
        levels=np.arange(-2, 30, 0.5),
        cmap=cm.RdYlBu_r,
        label="Temperature (°C)",
        title=f"{transect_info['name']} - Multi-year Mean ({historic_last25y_start}-{historic_last25y_end})"
    )
    
    # Add contour lines for major isotherms
    major_isotherms = [0, 5, 10, 15, 20, 25]
    cs_lines = pf.plot_transect(
        temp_data, mesh, lonlat,
        maxdepth=5000,
        levels=major_isotherms,
        colors='black',
        linewidths=0.5,
        alpha=0.7
    )
    
    # Improve formatting
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_xlabel('Distance along transect (km)', fontsize=12)
    
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
