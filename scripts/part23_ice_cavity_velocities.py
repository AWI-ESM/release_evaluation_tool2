# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *
from scipy.interpolate import griddata

SCRIPT_NAME = os.path.basename(__file__)
print(SCRIPT_NAME)

# Mark as started
update_status(SCRIPT_NAME, " Started")

# Ice Shelf Cavity Velocity Visualization
# Following standard oceanographic visualization practices:
# - Vertically-averaged velocity with magnitude (color) and direction (arrows)
# - Vertical cross-sections along cavity centerlines
# - Temperature/salinity context panels

# Define ice cavity configurations with transect lines for cross-sections
ice_cavities = {
    'Ross': {
        'lon_min': 160.0, 'lon_max': -150.0,
        'lat_min': -85.0, 'lat_max': -75.0,
        'title': 'Ross Ice Shelf',
        'cross_dateline': True,
        'transect_lat': -80.0,  # Latitude for E-W section
        'transect_lon': 180.0,  # Longitude for N-S section
        'central_lon': 180.0
    },
    'Filchner-Ronne': {
        'lon_min': -80.0, 'lon_max': -20.0,
        'lat_min': -84.0, 'lat_max': -74.0,
        'title': 'Filchner-Ronne Ice Shelf',
        'cross_dateline': False,
        'transect_lat': -79.0,
        'transect_lon': -50.0,
        'central_lon': -50.0
    },
    'Pine_Island': {
        'lon_min': -110.0, 'lon_max': -95.0,
        'lat_min': -76.0, 'lat_max': -73.0,
        'title': 'Pine Island Glacier',
        'cross_dateline': False,
        'transect_lat': -74.5,
        'transect_lon': -102.0,
        'central_lon': -102.0
    },
    'Amery': {
        'lon_min': 68.0, 'lon_max': 75.0,
        'lat_min': -72.0, 'lat_max': -68.0,
        'title': 'Amery Ice Shelf',
        'cross_dateline': False,
        'transect_lat': -70.0,
        'transect_lon': 71.0,
        'central_lon': 71.0
    }
}

# Data paths
input_path = spinup_path + '/fesom/'
year = spinup_end

print(f"Loading data for year {year}...")

# Load velocity files
u_file = f"{input_path}u.fesom.{year}.nc"
v_file = f"{input_path}v.fesom.{year}.nc"
temp_file = f"{input_path}temp.fesom.{year}.nc"

print(f"Loading: {u_file}")
ds_u = xr.open_dataset(u_file)
ds_v = xr.open_dataset(v_file)

# Try to load temperature for context
try:
    print(f"Loading: {temp_file}")
    ds_temp = xr.open_dataset(temp_file)
    has_temp = True
except:
    print("Temperature file not available, skipping T panels")
    has_temp = False

# Get time-averaged data
u_data = ds_u['u'].mean(dim='time').values  # (nz, elem)
v_data = ds_v['v'].mean(dim='time').values
depths = ds_u['nz1'].values
layer_thickness = np.diff(np.concatenate([[0], depths]))

if has_temp:
    temp_data = ds_temp['temp'].mean(dim='time').values

print(f"Velocity shape: {u_data.shape}, Depths: {len(depths)}")

# Element centroids (for velocity on elements)
elem_lon = mesh.x2[mesh.elem].mean(axis=1)
elem_lat = mesh.y2[mesh.elem].mean(axis=1)

# Node coordinates (for temperature on nodes)
node_lon = mesh.x2
node_lat = mesh.y2

def select_cavity_elements(cavity_info, lon_arr, lat_arr):
    """Select elements/nodes within cavity bounding box"""
    if cavity_info['cross_dateline']:
        lon_mask = (lon_arr >= cavity_info['lon_min']) | (lon_arr <= cavity_info['lon_max'])
    else:
        lon_mask = (lon_arr >= cavity_info['lon_min']) & (lon_arr <= cavity_info['lon_max'])
    lat_mask = (lat_arr >= cavity_info['lat_min']) & (lat_arr <= cavity_info['lat_max'])
    return lon_mask & lat_mask

def compute_depth_integrated_velocity(u, v, depths, max_depth=None):
    """Compute depth-integrated (vertically averaged) velocity"""
    layer_thick = np.diff(np.concatenate([[0], depths]))
    
    if max_depth is not None:
        valid_layers = depths <= max_depth
        layer_thick = layer_thick[valid_layers]
        u = u[valid_layers, :]
        v = v[valid_layers, :]
    
    # Weighted average by layer thickness
    weights = layer_thick[:, np.newaxis]
    total_depth = np.nansum(weights * ~np.isnan(u), axis=0)
    
    u_avg = np.nansum(u * weights, axis=0) / total_depth
    v_avg = np.nansum(v * weights, axis=0) / total_depth
    
    return u_avg, v_avg

def create_multi_panel_figure(cavity_name, cavity_info, elem_lon, elem_lat, 
                               u_data, v_data, depths, temp_data=None,
                               node_lon=None, node_lat=None):
    """
    Create publication-quality multi-panel figure following Naughten et al. (2018) style:
    (a) Vertically-averaged velocity with arrows
    (b) Velocity magnitude cross-section (N-S or along cavity)
    (c) Bottom temperature (if available)
    (d) Surface velocity detail
    """
    
    print(f"\nProcessing {cavity_name}...")
    
    # Velocity is on elements
    mask = select_cavity_elements(cavity_info, elem_lon, elem_lat)
    n_elements = np.sum(mask)
    print(f"  Found {n_elements} elements")
    
    if n_elements < 10:
        print(f"  WARNING: Too few elements for {cavity_name}, skipping...")
        return None
    
    # Extract cavity velocity data (on elements)
    cav_lon = elem_lon[mask]
    cav_lat = elem_lat[mask]
    cav_u = u_data[:, mask]
    cav_v = v_data[:, mask]
    
    # Temperature is on nodes - need separate mask
    if temp_data is not None and node_lon is not None:
        node_mask = select_cavity_elements(cavity_info, node_lon, node_lat)
        cav_temp = temp_data[:, node_mask]
        cav_temp_lon = node_lon[node_mask]
        cav_temp_lat = node_lat[node_mask]
        print(f"  Found {np.sum(node_mask)} nodes for temperature")
    
    # Compute depth-integrated velocity
    u_avg, v_avg = compute_depth_integrated_velocity(cav_u, cav_v, depths)
    vel_mag = np.sqrt(u_avg**2 + v_avg**2)
    
    # Create figure
    if temp_data is not None:
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2)
    else:
        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(1, 2, wspace=0.2)
    
    # ===== Panel (a): Vertically-averaged velocity with arrows =====
    if temp_data is not None:
        ax1 = fig.add_subplot(gs[0, 0])
    else:
        ax1 = fig.add_subplot(gs[0, 0])
    
    # Create regular grid for interpolation
    lon_grid = np.linspace(cav_lon.min(), cav_lon.max(), 100)
    lat_grid = np.linspace(cav_lat.min(), cav_lat.max(), 100)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    
    # Interpolate velocity magnitude to grid
    valid = ~np.isnan(vel_mag)
    if np.sum(valid) > 10:
        vel_interp = griddata((cav_lon[valid], cav_lat[valid]), vel_mag[valid], 
                              (LON, LAT), method='linear')
        u_interp = griddata((cav_lon[valid], cav_lat[valid]), u_avg[valid], 
                            (LON, LAT), method='linear')
        v_interp = griddata((cav_lon[valid], cav_lat[valid]), v_avg[valid], 
                            (LON, LAT), method='linear')
        
        # Plot velocity magnitude as filled contours
        levels = np.linspace(0, 0.1, 21)
        cf = ax1.contourf(LON, LAT, vel_interp, levels=levels, cmap=cmo.cm.speed, extend='max')
        
        # Add velocity arrows (subsampled)
        skip = 5
        ax1.quiver(LON[::skip, ::skip], LAT[::skip, ::skip],
                   u_interp[::skip, ::skip], v_interp[::skip, ::skip],
                   scale=1.5, width=0.004, headwidth=4, headlength=5,
                   color='black', alpha=0.7)
    else:
        cf = ax1.scatter(cav_lon, cav_lat, c=vel_mag, cmap=cmo.cm.speed, 
                        s=20, vmin=0, vmax=0.1)
    
    ax1.set_xlabel('Longitude (°)', fontsize=11)
    ax1.set_ylabel('Latitude (°)', fontsize=11)
    ax1.set_title('(a) Vertically-Averaged Velocity', fontsize=12, fontweight='bold')
    cbar1 = plt.colorbar(cf, ax=ax1, shrink=0.8)
    cbar1.set_label('Velocity (m/s)', fontsize=10)
    ax1.set_aspect('equal')
    
    # ===== Panel (b): Velocity cross-section =====
    if temp_data is not None:
        ax2 = fig.add_subplot(gs[0, 1])
    else:
        ax2 = fig.add_subplot(gs[0, 1])
    
    # Sort by latitude for N-S section
    sort_idx = np.argsort(cav_lat)
    sorted_lat = cav_lat[sort_idx]
    sorted_vel = np.sqrt(cav_u**2 + cav_v**2)[:, sort_idx]
    
    # Bin into latitude bands
    n_bins = min(50, n_elements // 5)
    if n_bins < 5:
        n_bins = 5
    lat_bins = np.linspace(sorted_lat.min(), sorted_lat.max(), n_bins + 1)
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    
    binned_vel = np.full((len(depths), n_bins), np.nan)
    for i in range(n_bins):
        bin_mask = (sorted_lat >= lat_bins[i]) & (sorted_lat < lat_bins[i+1])
        if np.sum(bin_mask) > 0:
            binned_vel[:, i] = np.nanmean(sorted_vel[:, bin_mask], axis=1)
    
    LAT_SEC, DEPTH_SEC = np.meshgrid(lat_centers, -depths)
    
    levels_sec = np.linspace(0, 0.1, 21)
    cf2 = ax2.contourf(LAT_SEC, DEPTH_SEC, binned_vel, levels=levels_sec, 
                       cmap=cmo.cm.speed, extend='max')
    ax2.contour(LAT_SEC, DEPTH_SEC, binned_vel, levels=[0.02, 0.04, 0.06, 0.08],
                colors='black', linewidths=0.5, alpha=0.5)
    
    ax2.set_xlabel('Latitude (°)', fontsize=11)
    ax2.set_ylabel('Depth (m)', fontsize=11)
    ax2.set_title('(b) Velocity Cross-Section (N-S)', fontsize=12, fontweight='bold')
    ax2.set_ylim(-1500, 0)
    cbar2 = plt.colorbar(cf2, ax=ax2, shrink=0.8)
    cbar2.set_label('Velocity (m/s)', fontsize=10)
    
    if temp_data is not None and node_lon is not None:
        # ===== Panel (c): Bottom temperature =====
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Get bottom temperature (deepest non-NaN value for each node)
        n_nodes = cav_temp.shape[1]
        bottom_temp = np.full(n_nodes, np.nan)
        for i in range(n_nodes):
            col = cav_temp[:, i]
            valid_idx = np.where(~np.isnan(col))[0]
            if len(valid_idx) > 0:
                bottom_temp[i] = col[valid_idx[-1]]
        
        # Create grid for temperature (using node coordinates)
        LON_T, LAT_T = np.meshgrid(
            np.linspace(cav_temp_lon.min(), cav_temp_lon.max(), 100),
            np.linspace(cav_temp_lat.min(), cav_temp_lat.max(), 100))
        
        valid = ~np.isnan(bottom_temp)
        if np.sum(valid) > 10:
            temp_interp = griddata((cav_temp_lon[valid], cav_temp_lat[valid]), bottom_temp[valid],
                                   (LON_T, LAT_T), method='linear')
            cf3 = ax3.contourf(LON_T, LAT_T, temp_interp, levels=20, cmap=cmo.cm.thermal)
        else:
            cf3 = ax3.scatter(cav_temp_lon, cav_temp_lat, c=bottom_temp, cmap=cmo.cm.thermal, s=20)
        
        ax3.set_xlabel('Longitude (°)', fontsize=11)
        ax3.set_ylabel('Latitude (°)', fontsize=11)
        ax3.set_title('(c) Bottom Temperature', fontsize=12, fontweight='bold')
        cbar3 = plt.colorbar(cf3, ax=ax3, shrink=0.8)
        cbar3.set_label('Temperature (°C)', fontsize=10)
        ax3.set_aspect('equal')
        
        # ===== Panel (d): Temperature cross-section =====
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Sort temperature by latitude (using node coordinates)
        temp_sort_idx = np.argsort(cav_temp_lat)
        sorted_temp_lat = cav_temp_lat[temp_sort_idx]
        sorted_temp = cav_temp[:, temp_sort_idx]
        
        # Bin temperature data
        temp_lat_bins = np.linspace(sorted_temp_lat.min(), sorted_temp_lat.max(), n_bins + 1)
        temp_lat_centers = (temp_lat_bins[:-1] + temp_lat_bins[1:]) / 2
        binned_temp = np.full((len(depths), n_bins), np.nan)
        for i in range(n_bins):
            bin_mask = (sorted_temp_lat >= temp_lat_bins[i]) & (sorted_temp_lat < temp_lat_bins[i+1])
            if np.sum(bin_mask) > 0:
                binned_temp[:, i] = np.nanmean(sorted_temp[:, bin_mask], axis=1)
        
        LAT_T_SEC, DEPTH_T_SEC = np.meshgrid(temp_lat_centers, -depths)
        cf4 = ax4.contourf(LAT_T_SEC, DEPTH_T_SEC, binned_temp, levels=20, cmap=cmo.cm.thermal)
        ax4.contour(LAT_T_SEC, DEPTH_T_SEC, binned_temp, levels=10, colors='black', 
                   linewidths=0.3, alpha=0.5)
        
        ax4.set_xlabel('Latitude (°)', fontsize=11)
        ax4.set_ylabel('Depth (m)', fontsize=11)
        ax4.set_title('(d) Temperature Cross-Section', fontsize=12, fontweight='bold')
        ax4.set_ylim(-1500, 0)
        cbar4 = plt.colorbar(cf4, ax=ax4, shrink=0.8)
        cbar4.set_label('Temperature (°C)', fontsize=10)
    
    plt.suptitle(f"{cavity_info['title']} Cavity - Year {year}", fontsize=14, fontweight='bold')
    
    ofile = out_path + f'ice_cavity_{cavity_name}_panels.png'
    plt.savefig(ofile, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {ofile}")
    
    return ofile

def create_streamline_figure(cavity_name, cavity_info, elem_lon, elem_lat, u_data, v_data, depths):
    """Create streamline visualization showing circulation patterns"""
    
    print(f"\nCreating streamline plot for {cavity_name}...")
    
    mask = select_cavity_elements(cavity_info, elem_lon, elem_lat)
    n_elements = np.sum(mask)
    
    if n_elements < 10:
        return None
    
    cav_lon = elem_lon[mask]
    cav_lat = elem_lat[mask]
    cav_u = u_data[:, mask]
    cav_v = v_data[:, mask]
    
    # Depth-integrated velocity
    u_avg, v_avg = compute_depth_integrated_velocity(cav_u, cav_v, depths)
    vel_mag = np.sqrt(u_avg**2 + v_avg**2)
    
    # Create regular grid
    n_grid = 80
    lon_grid = np.linspace(cav_lon.min(), cav_lon.max(), n_grid)
    lat_grid = np.linspace(cav_lat.min(), cav_lat.max(), n_grid)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    
    valid = ~np.isnan(vel_mag)
    if np.sum(valid) < 20:
        print(f"  Not enough valid data for streamlines")
        return None
    
    # Interpolate to grid
    U = griddata((cav_lon[valid], cav_lat[valid]), u_avg[valid], (LON, LAT), method='linear')
    V = griddata((cav_lon[valid], cav_lat[valid]), v_avg[valid], (LON, LAT), method='linear')
    speed = np.sqrt(U**2 + V**2)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Background velocity magnitude
    levels = np.linspace(0, 0.1, 21)
    cf = ax.contourf(LON, LAT, speed, levels=levels, cmap=cmo.cm.speed, extend='max')
    
    # Streamlines
    # Replace NaN with 0 for streamplot
    U_clean = np.nan_to_num(U, nan=0)
    V_clean = np.nan_to_num(V, nan=0)
    speed_clean = np.nan_to_num(speed, nan=0)
    
    strm = ax.streamplot(lon_grid, lat_grid, U_clean, V_clean, 
                         color='white', density=1.5, linewidth=0.8,
                         arrowsize=1.2, arrowstyle='->')
    
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)
    ax.set_title(f"{cavity_info['title']}\nDepth-Integrated Circulation (Year {year})", 
                fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
    cbar.set_label('Velocity Magnitude (m/s)', fontsize=11)
    
    plt.tight_layout()
    
    ofile = out_path + f'ice_cavity_{cavity_name}_streamlines.png'
    plt.savefig(ofile, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {ofile}")
    
    return ofile

# ===== Main Processing =====
print("\n" + "="*60)
print("Processing Antarctic Ice Shelf Cavities")
print("="*60)

for cavity_name, cavity_info in ice_cavities.items():
    try:
        if has_temp:
            create_multi_panel_figure(cavity_name, cavity_info, elem_lon, elem_lat,
                                     u_data, v_data, depths, temp_data,
                                     node_lon, node_lat)
        else:
            create_multi_panel_figure(cavity_name, cavity_info, elem_lon, elem_lat,
                                     u_data, v_data, depths, None, None, None)
        
        create_streamline_figure(cavity_name, cavity_info, elem_lon, elem_lat,
                                u_data, v_data, depths)
    except Exception as e:
        print(f"Error processing {cavity_name}: {e}")
        import traceback
        traceback.print_exc()

# ===== Summary figure with all cavities =====
print("\nCreating summary figure...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, (cavity_name, cavity_info) in enumerate(ice_cavities.items()):
    ax = axes[idx]
    
    mask = select_cavity_elements(cavity_info, elem_lon, elem_lat)
    if np.sum(mask) < 10:
        ax.set_title(f"{cavity_info['title']}\n(Insufficient data)", fontsize=11)
        ax.axis('off')
        continue
    
    cav_lon = elem_lon[mask]
    cav_lat = elem_lat[mask]
    cav_u = u_data[:, mask]
    cav_v = v_data[:, mask]
    
    u_avg, v_avg = compute_depth_integrated_velocity(cav_u, cav_v, depths)
    vel_mag = np.sqrt(u_avg**2 + v_avg**2)
    
    # Grid interpolation
    n_grid = 60
    lon_grid = np.linspace(cav_lon.min(), cav_lon.max(), n_grid)
    lat_grid = np.linspace(cav_lat.min(), cav_lat.max(), n_grid)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    
    valid = ~np.isnan(vel_mag)
    if np.sum(valid) > 10:
        vel_interp = griddata((cav_lon[valid], cav_lat[valid]), vel_mag[valid],
                              (LON, LAT), method='linear')
        u_interp = griddata((cav_lon[valid], cav_lat[valid]), u_avg[valid],
                            (LON, LAT), method='linear')
        v_interp = griddata((cav_lon[valid], cav_lat[valid]), v_avg[valid],
                            (LON, LAT), method='linear')
        
        levels = np.linspace(0, 0.1, 21)
        cf = ax.contourf(LON, LAT, vel_interp, levels=levels, cmap=cmo.cm.speed, extend='max')
        
        skip = 4
        ax.quiver(LON[::skip, ::skip], LAT[::skip, ::skip],
                  u_interp[::skip, ::skip], v_interp[::skip, ::skip],
                  scale=1.5, width=0.005, color='black', alpha=0.7)
        
        plt.colorbar(cf, ax=ax, shrink=0.8, label='m/s')
    
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title(cavity_info['title'], fontsize=12, fontweight='bold')
    ax.set_aspect('equal')

plt.suptitle(f'Antarctic Ice Shelf Cavity Circulation - Year {year}\nDepth-Integrated Velocity', 
            fontsize=14, fontweight='bold')
plt.tight_layout()

ofile = out_path + 'ice_cavities_circulation_summary.png'
plt.savefig(ofile, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"Summary saved: {ofile}")

# Clean up
ds_u.close()
ds_v.close()
if has_temp:
    ds_temp.close()

print(f"\n{'='*60}")
print("Ice Cavity Velocity Visualization Complete")
print(f"Output directory: {out_path}")
print(f"{'='*60}")

# Mark as completed
update_status(SCRIPT_NAME, " Completed")
