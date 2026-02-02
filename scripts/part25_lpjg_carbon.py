# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

SCRIPT_NAME = os.path.basename(__file__)  # Get the current script name

print(SCRIPT_NAME)

# Mark as started
update_status(SCRIPT_NAME, " Started")

# Import LPJ-GUESS helper functions
from lpjg_helpers import read_lpjg_output, interpolate_to_grid

# LPJ-GUESS analysis settings
GRID_RES = 1.0
lpjg_analysis_year = historic_last25y_start  # From config


############################
# Main Analysis            #
############################

def setup_global_map(ax, title):
    """Setup global map with coastlines and gridlines."""
    ax.set_global()
    ax.coastlines(resolution='110m', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    ax.set_title(title, fontsize=12, fontweight='bold')


def plot_variable(ax, df, var, title, vmin, vmax, cmap, units, year):
    """Plot a variable on a global map."""
    setup_global_map(ax, title)
    lon_grid, lat_grid, grid_values, data = interpolate_to_grid(df, var, year, GRID_RES)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    cf = ax.pcolormesh(lon_mesh, lat_mesh, grid_values, cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(), shading='auto', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3, zorder=2)
    ax.coastlines(resolution='110m', linewidth=0.5, zorder=3)
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
    cbar.set_label(units, fontsize=10)


try:
    # Read carbon mass output data
    print(f"Reading carbon mass data from {spinup_path} for year {lpjg_analysis_year}...")
    cmass_data = read_lpjg_output(spinup_path, "cmass.out", lpjg_analysis_year)
    
    if cmass_data is None or cmass_data.empty:
        raise ValueError(f"No carbon mass data found for year {lpjg_analysis_year}")
    
    print(f"Loaded carbon mass data with {len(cmass_data)} grid points")
    print(f"Available variables: {list(cmass_data.columns)}")
    
    # Create figure for total carbon
    print("Creating total carbon mass plot...")
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    plot_variable(ax, cmass_data, 'Total', 
                  f"{model_version} - Year {lpjg_analysis_year}\nVegetation Carbon Mass (Total)", 
                  0, 20, 'Greens', 'Carbon [kg C/m²]', lpjg_analysis_year)
    plt.tight_layout()
    output_file = out_path + f'lpjg_Carbon_total_year{lpjg_analysis_year}.png'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    
    # Create figure for individual PFTs (if available)
    pfts = ['BNE', 'BINE', 'BNS', 'TeNE', 'TeBS', 'IBS', 'TeBE', 'TrBE', 'TrIBE', 'TrBR', 'C3G', 'C4G']
    available_pfts = [p for p in pfts if p in cmass_data.columns]
    if len(available_pfts) > 0:
        print(f"Creating PFT-specific carbon mass plots for {len(available_pfts)} PFTs...")
        n_pfts = len(available_pfts)
        n_cols = 3
        n_rows = (n_pfts + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(18, 5 * n_rows))
        for i, pft in enumerate(available_pfts):
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection=ccrs.Robinson())
            plot_variable(ax, cmass_data, pft, 
                          f"{pft} Carbon Mass", 
                          0, 15, 'Greens', 'Carbon [kg C/m²]', lpjg_analysis_year)
        
        plt.tight_layout()
        output_file = out_path + f'lpjg_Carbon_by_PFT_year{lpjg_analysis_year}.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_file}")
    
    # Try to read soil carbon if available
    try:
        print(f"Checking for soil carbon data...")
        cpool_data = read_lpjg_output(spinup_path, "cpool.out", lpjg_analysis_year)
        
        if cpool_data is not None and not cpool_data.empty and 'Total' in cpool_data.columns:
            print("Creating soil carbon pool plot...")
            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
            plot_variable(ax, cpool_data, 'Total', 
                          f"{model_version} - Year {lpjg_analysis_year}\nSoil Carbon Pool (Total)", 
                          0, 30, 'YlOrBr', 'Carbon [kg C/m²]', lpjg_analysis_year)
            plt.tight_layout()
            output_file = out_path + f'lpjg_SoilCarbon_total_year{lpjg_analysis_year}.png'
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_file}")
            
            # Calculate total ecosystem carbon (vegetation + soil)
            print("Creating total ecosystem carbon plot...")
            # Merge the two dataframes
            total_carbon = cmass_data.copy()
            total_carbon['Ecosystem_Total'] = total_carbon['Total'] + cpool_data['Total']
            
            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
            plot_variable(ax, total_carbon, 'Ecosystem_Total', 
                          f"{model_version} - Year {lpjg_analysis_year}\nTotal Ecosystem Carbon (Vegetation + Soil)", 
                          0, 50, 'BrBG', 'Carbon [kg C/m²]', lpjg_analysis_year)
            plt.tight_layout()
            output_file = out_path + f'lpjg_EcosystemCarbon_total_year{lpjg_analysis_year}.png'
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_file}")
    except Exception as e:
        print(f"Could not process soil carbon data: {str(e)}")
    
    # Calculate and print statistics
    print("\n=== Carbon Statistics ===")
    total_carbon = cmass_data[cmass_data['Year'] == lpjg_analysis_year]['Total']
    print(f"Vegetation Carbon - Mean: {total_carbon.mean():.3f}, Min: {total_carbon.min():.3f}, Max: {total_carbon.max():.3f}")
    
    # Mark script as completed
    update_status(SCRIPT_NAME, "Completed")
    print(f"\n{SCRIPT_NAME} completed successfully!")

except Exception as e:
    print(f"Error in {SCRIPT_NAME}: {str(e)}")
    import traceback
    traceback.print_exc()
    update_status(SCRIPT_NAME, f"Failed: {str(e)}")
    raise
