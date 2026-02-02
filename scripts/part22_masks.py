# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

SCRIPT_NAME = os.path.basename(__file__)
print(SCRIPT_NAME)

# Mark as started
update_status(SCRIPT_NAME, " Started")

# Plot all masks from masks.nc file
masks_file = '/work/bb1469/a270092/runtime/awicm3-develop/CORE3_SPIN_900s/run_20400101-20491231/work/masks.nc'

print(f"Loading masks from: {masks_file}")
ds = xr.open_dataset(masks_file)

# Identify all mask variables (variables ending with .msk)
mask_vars = [var for var in ds.data_vars if '.msk' in var]
print(f"Found {len(mask_vars)} mask variables: {mask_vars}")

# Define grid configurations for different mask types
grid_configs = {
    'A096': {'x_dim': 'x_A096', 'y_dim': 'y_A096', 'lon': 'A096.lon', 'lat': 'A096.lat', 'title': 'Atmosphere Grid (A096)'},
    'L096': {'x_dim': 'x_L096', 'y_dim': 'y_L096', 'lon': 'L096.lon', 'lat': 'L096.lat', 'title': 'Land Grid (L096)'},
    'R096': {'x_dim': 'x_R096', 'y_dim': 'y_R096', 'lon': 'R096.lon', 'lat': 'R096.lat', 'title': 'Runoff Grid (R096)'},
    'TCO95-land': {'x_dim': 'x_TCO95-land', 'y_dim': 'y_TCO95-land', 'lon': 'TCO95-land.lon', 'lat': 'TCO95-land.lat', 'title': 'TCO95 Land Grid'},
    'RnfA': {'x_dim': 'x', 'y_dim': 'y', 'lon': None, 'lat': None, 'title': 'Runoff to Atmosphere'},
    'RnfO': {'x_dim': 'x', 'y_dim': 'y', 'lon': None, 'lat': None, 'title': 'Runoff to Ocean'},
    'feom': {'x_dim': 'x_feom', 'y_dim': 'y_feom', 'lon': None, 'lat': None, 'title': 'FESOM Ocean Mesh'},
}

# Create figure with subplots for all masks
n_masks = len(mask_vars)
n_cols = 3
n_rows = int(np.ceil(n_masks / n_cols))

fig = plt.figure(figsize=(16, 4 * n_rows))

for idx, mask_var in enumerate(mask_vars, 1):
    # Extract grid prefix from mask variable name
    prefix = mask_var.replace('.msk', '')
    config = grid_configs.get(prefix, None)
    
    ax = fig.add_subplot(n_rows, n_cols, idx)
    
    mask_data = ds[mask_var].values.squeeze()
    
    if config and config['lon'] is not None:
        # Plot with geographic coordinates
        lon = ds[config['lon']].values.squeeze()
        lat = ds[config['lat']].values.squeeze()
        
        scatter = ax.scatter(lon, lat, c=mask_data, cmap='RdYlBu', s=0.5, alpha=0.7)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
    else:
        # Plot as 1D or 2D array
        if mask_data.ndim == 1:
            im = ax.scatter(np.arange(len(mask_data)), np.zeros_like(mask_data), 
                           c=mask_data, cmap='RdYlBu', s=0.1)
            ax.set_xlabel('Index')
            ax.set_ylabel('')
            ax.set_ylim(-0.5, 0.5)
        else:
            im = ax.imshow(mask_data, cmap='RdYlBu', aspect='auto', origin='lower')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
    
    title = config['title'] if config else mask_var
    ax.set_title(f'{title}\n({mask_var})', fontsize=10, fontweight='bold')
    
    # Add colorbar
    if config and config['lon'] is not None:
        plt.colorbar(scatter, ax=ax, label='Mask Value', shrink=0.8)
    elif mask_data.ndim > 1:
        plt.colorbar(im, ax=ax, label='Mask Value', shrink=0.8)

plt.suptitle(f'OASIS Coupling Masks\n{os.path.basename(masks_file)}', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

ofile = out_path + 'masks_overview.png'
plt.savefig(ofile, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"Overview plot saved: {ofile}")

# Create individual detailed plots for each mask with geographic projection
for mask_var in mask_vars:
    prefix = mask_var.replace('.msk', '')
    config = grid_configs.get(prefix, None)
    
    if config and config['lon'] is not None:
        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.Robinson()})
        
        lon = ds[config['lon']].values.squeeze()
        lat = ds[config['lat']].values.squeeze()
        mask_data = ds[mask_var].values.squeeze()
        
        # Use different colors for mask values
        colors = ['navy', 'crimson']
        cmap = mpl.colors.ListedColormap(colors)
        bounds = [-0.5, 0.5, 1.5]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        
        scatter = ax.scatter(lon, lat, c=mask_data, cmap=cmap, norm=norm, 
                            s=0.3, alpha=0.8, transform=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        
        ax.set_title(f'{config["title"]}\n{mask_var}', fontsize=12, fontweight='bold')
        
        # Custom legend
        legend_elements = [mpatches.Patch(facecolor='navy', label='0 (Inactive)'),
                         mpatches.Patch(facecolor='crimson', label='1 (Active)')]
        ax.legend(handles=legend_elements, loc='lower left')
        
        ofile = out_path + f'mask_{prefix}.png'
        plt.savefig(ofile, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Individual mask plot saved: {ofile}")

# Also create RnfA and RnfO plots as 2D regular grids
for mask_var in ['RnfA.msk', 'RnfO.msk']:
    if mask_var in ds.data_vars:
        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.Robinson()})
        
        mask_data = ds[mask_var].values
        
        # These are on a regular 512x256 grid
        lon = np.linspace(-180, 180, 512)
        lat = np.linspace(-90, 90, 256)
        lon2d, lat2d = np.meshgrid(lon, lat)
        
        colors = ['navy', 'crimson']
        cmap = mpl.colors.ListedColormap(colors)
        
        im = ax.pcolormesh(lon2d, lat2d, mask_data, cmap=cmap, 
                          transform=ccrs.PlateCarree(), shading='auto')
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        
        prefix = mask_var.replace('.msk', '')
        config = grid_configs.get(prefix, {})
        title = config.get('title', mask_var)
        ax.set_title(f'{title}\n{mask_var}', fontsize=12, fontweight='bold')
        
        legend_elements = [mpatches.Patch(facecolor='navy', label='0 (Inactive)'),
                         mpatches.Patch(facecolor='crimson', label='1 (Active)')]
        ax.legend(handles=legend_elements, loc='lower left')
        
        ofile = out_path + f'mask_{prefix}.png'
        plt.savefig(ofile, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Individual mask plot saved: {ofile}")

ds.close()
print(f"\n=== Mask Plotting Complete ===")
print(f"Output directory: {out_path}")

# Mark as completed
update_status(SCRIPT_NAME, " Completed")
