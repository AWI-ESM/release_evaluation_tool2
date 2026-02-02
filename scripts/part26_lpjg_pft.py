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
from lpjg_helpers import read_lpjg_output, interpolate_to_grid, interpolate_dominant_pft
import matplotlib.colors as mcolors

############################
# PFT Definitions          #
############################

PFTS = ['BNE', 'BINE', 'BNS', 'TeNE', 'TeBS', 'IBS', 'TeBE', 'TrBE', 'TrIBE', 'TrBR', 'C3G', 'C4G']
PFT_COLORS = {'BNE': '#0d47a1', 'BINE': '#1976d2', 'BNS': '#64b5f6', 'TeNE': '#4a148c', 'TeBS': '#7b1fa2',
              'IBS': '#ba68c8', 'TeBE': '#6a1b9a', 'TrBE': '#1b5e20', 'TrIBE': '#388e3c', 'TrBR': '#81c784',
              'C3G': '#fdd835', 'C4G': '#ff8f00', 'Barren': '#bdbdbd'}
BARREN_THRESHOLD = 0.2
MAX_DISTANCE = 5.0
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


def plot_dominant_pft_map(ax, df, title, year):
    """Plot dominant PFT on a global map."""
    setup_global_map(ax, title)
    lon_grid, lat_grid, grid_values, data, available_pfts = interpolate_dominant_pft(
        df, year, PFTS, GRID_RES, BARREN_THRESHOLD, MAX_DISTANCE)
    colors = [PFT_COLORS.get(pft, '#808080') for pft in available_pfts]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(available_pfts), 1), cmap.N)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    ax.pcolormesh(lon_mesh, lat_mesh, grid_values, cmap=cmap, norm=norm,
                  transform=ccrs.PlateCarree(), shading='auto', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3, zorder=2)
    ax.coastlines(resolution='110m', linewidth=0.5, zorder=3)
    return available_pfts


try:
    # Read LAI output data for PFT analysis
    print(f"Reading LAI data from {spinup_path} for year {lpjg_analysis_year}...")
    lai_data = read_lpjg_output(spinup_path, "lai.out", lpjg_analysis_year)
    
    if lai_data is None or lai_data.empty:
        raise ValueError(f"No LAI data found for year {lpjg_analysis_year}")
    
    print(f"Loaded LAI data with {len(lai_data)} grid points")
    print(f"Available variables: {list(lai_data.columns)}")
    
    # Create figure for dominant PFT
    print("Creating dominant PFT plot...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    available_pfts = plot_dominant_pft_map(ax, lai_data, 
                                           f"{model_version} - Year {lpjg_analysis_year}\nDominant Plant Functional Type (by LAI)", 
                                           lpjg_analysis_year)
    
    # Create legend
    patches = [mpatches.Patch(color=PFT_COLORS.get(pft, '#808080'), label=pft) for pft in available_pfts]
    ax.legend(handles=patches, loc='lower center', ncol=7, fontsize=9, 
              bbox_to_anchor=(0.5, -0.15), framealpha=0.9)
    
    plt.tight_layout()
    output_file = out_path + f'lpjg_dominant_PFT_year{lpjg_analysis_year}.png'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    
    # Calculate PFT distribution statistics
    print("\n=== PFT Distribution Statistics ===")
    data_subset = lai_data[lai_data['Year'] == lpjg_analysis_year]
    available_pfts_cols = [p for p in PFTS if p in data_subset.columns]
    
    for pft in available_pfts_cols:
        pft_values = data_subset[pft]
        print(f"{pft:8s} - Mean: {pft_values.mean():6.3f}, "
              f"Min: {pft_values.min():6.3f}, "
              f"Max: {pft_values.max():6.3f}, "
              f"Non-zero: {(pft_values > 0).sum():6d} points")
    
    # Create PFT coverage map showing number of PFTs per grid cell
    print("Creating PFT richness (diversity) plot...")
    pft_cols = [col for col in available_pfts_cols if col in data_subset.columns]
    data_subset['pft_count'] = (data_subset[pft_cols] > BARREN_THRESHOLD).sum(axis=1)
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    setup_global_map(ax, f"{model_version} - Year {lpjg_analysis_year}\nPlant Functional Type Richness")
    
    lon_grid, lat_grid, grid_values, _ = interpolate_to_grid(data_subset, 'pft_count', lpjg_analysis_year, GRID_RES)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    cf = ax.pcolormesh(lon_mesh, lat_mesh, grid_values, cmap='viridis', vmin=0, vmax=len(pft_cols),
                       transform=ccrs.PlateCarree(), shading='auto', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3, zorder=2)
    ax.coastlines(resolution='110m', linewidth=0.5, zorder=3)
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
    cbar.set_label('Number of PFTs', fontsize=10)
    
    plt.tight_layout()
    output_file = out_path + f'lpjg_PFT_richness_year{lpjg_analysis_year}.png'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    
    # Create fractional coverage maps for major PFT categories
    print("Creating fractional coverage maps by PFT category...")
    fig = plt.figure(figsize=(18, 12))
    
    # Define PFT categories
    pft_categories = {
        'Trees (Total)': ['BNE', 'BINE', 'BNS', 'TeNE', 'TeBS', 'IBS', 'TeBE', 'TrBE', 'TrIBE', 'TrBR'],
        'Boreal Trees': ['BNE', 'BINE', 'BNS'],
        'Temperate Trees': ['TeNE', 'TeBS', 'IBS', 'TeBE'],
        'Tropical Trees': ['TrBE', 'TrIBE', 'TrBR'],
        'Grasses': ['C3G', 'C4G'],
        'C3 Grasses': ['C3G'],
        'C4 Grasses': ['C4G'],
    }
    
    for i, (category, pfts) in enumerate(pft_categories.items()):
        available_cat_pfts = [p for p in pfts if p in data_subset.columns]
        if len(available_cat_pfts) == 0:
            continue
            
        data_subset[category] = data_subset[available_cat_pfts].sum(axis=1)
        
        ax = fig.add_subplot(3, 3, i + 1, projection=ccrs.Robinson())
        setup_global_map(ax, f"{category}")
        
        lon_grid, lat_grid, grid_values, _ = interpolate_to_grid(data_subset, category, lpjg_analysis_year, GRID_RES)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        cf = ax.pcolormesh(lon_mesh, lat_mesh, grid_values, cmap='YlGn', vmin=0, vmax=6,
                           transform=ccrs.PlateCarree(), shading='auto', zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3, zorder=2)
        ax.coastlines(resolution='110m', linewidth=0.5, zorder=3)
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', shrink=0.7, pad=0.02)
        cbar.set_label('LAI [m²/m²]', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    output_file = out_path + f'lpjg_PFT_categories_year{lpjg_analysis_year}.png'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    
    # Mark script as completed
    update_status(SCRIPT_NAME, "Completed")
    print(f"\n{SCRIPT_NAME} completed successfully!")

except Exception as e:
    print(f"Error in {SCRIPT_NAME}: {str(e)}")
    import traceback
    traceback.print_exc()
    update_status(SCRIPT_NAME, f"Failed: {str(e)}")
    raise
