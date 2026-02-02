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
    # Read LAI output data
    print(f"Reading LAI data from {spinup_path} for year {lpjg_analysis_year}...")
    lai_data = read_lpjg_output(spinup_path, "lai.out", lpjg_analysis_year)
    
    if lai_data is None or lai_data.empty:
        raise ValueError(f"No LAI data found for year {lpjg_analysis_year}")
    
    print(f"Loaded LAI data with {len(lai_data)} grid points")
    print(f"Available variables: {list(lai_data.columns)}")
    
    # Create figure for total LAI
    print("Creating total LAI plot...")
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    plot_variable(ax, lai_data, 'Total', 
                  f"{model_version} - Year {lpjg_analysis_year}\nLeaf Area Index (Total)", 
                  0, 8, 'YlGn', 'LAI [m²/m²]', lpjg_analysis_year)
    plt.tight_layout()
    output_file = out_path + f'lpjg_LAI_total_year{lpjg_analysis_year}.png'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    
    # Create figure for individual PFTs (if available)
    # Note: PFT definitions are in part26_lpjg_pft.py
    pfts = ['BNE', 'BINE', 'BNS', 'TeNE', 'TeBS', 'IBS', 'TeBE', 'TrBE', 'TrIBE', 'TrBR', 'C3G', 'C4G']
    available_pfts = [p for p in pfts if p in lai_data.columns]
    if len(available_pfts) > 0:
        print(f"Creating PFT-specific LAI plots for {len(available_pfts)} PFTs...")
        n_pfts = len(available_pfts)
        n_cols = 3
        n_rows = (n_pfts + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(18, 5 * n_rows))
        for i, pft in enumerate(available_pfts):
            print(f"  Plotting {pft} ({i+1}/{len(available_pfts)})...")
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection=ccrs.Robinson())
            plot_variable(ax, lai_data, pft, 
                          f"{pft} LAI", 
                          0, 6, 'YlGn', 'LAI [m²/m²]', lpjg_analysis_year)
        
        plt.tight_layout()
        output_file = out_path + f'lpjg_LAI_by_PFT_year{lpjg_analysis_year}.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_file}")
    
    # Calculate and print statistics
    print("\n=== LAI Statistics ===")
    total_lai = lai_data[lai_data['Year'] == lpjg_analysis_year]['Total']
    print(f"Total LAI - Mean: {total_lai.mean():.3f}, Min: {total_lai.min():.3f}, Max: {total_lai.max():.3f}")
    
    # Mark script as completed
    update_status(SCRIPT_NAME, "Completed")
    print(f"\n{SCRIPT_NAME} completed successfully!")

except Exception as e:
    print(f"Error in {SCRIPT_NAME}: {str(e)}")
    import traceback
    traceback.print_exc()
    update_status(SCRIPT_NAME, f"Failed: {str(e)}")
    raise
