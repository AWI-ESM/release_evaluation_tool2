"""
Helper functions for reading and processing LPJ-GUESS output data.
These functions can be reused across different LPJ-GUESS analysis scripts.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from cartopy.io import shapereader

# Cache for land mask to avoid recomputing
_land_mask_cache = {}


def read_lpjg_output(base_path, filename, year):
    """
    Read LPJ-GUESS output files from run directories.
    LPJ-GUESS outputs are typically in: outdata/lpj_guess/run1/*.out, run2/*.out, etc.
    
    Parameters:
    -----------
    base_path : Path or str
        Base path (outdata directory) that contains lpj_guess subdirectory
    filename : str
        Name of output file (e.g., 'lai.out', 'cmass.out')
    year : int
        Year to extract data for
        
    Returns:
    --------
    pd.DataFrame or None
        DataFrame with Lon, Lat, Year, and variable columns
    """
    all_data = []
    base = Path(base_path)
    
    # Search for LPJ-GUESS directories: outdata/lpj_guess/run*
    lpjg_dirs = [
        base / "lpj_guess",
        base / "lpjg",
        base / "lpj-guess",
    ]
    
    lpjg_dir = None
    for d in lpjg_dirs:
        if d.exists():
            lpjg_dir = d
            break
    
    if lpjg_dir is None:
        print(f"Warning: Could not find lpj_guess directory under {base_path}")
        return None
    
    # Find all run directories (run1, run2, run3, etc.)
    run_dirs = sorted(glob.glob(str(lpjg_dir / "run*")))
    if not run_dirs:
        print(f"Warning: No run directories found in {lpjg_dir}")
        return None
    
    print(f"Found {len(run_dirs)} run directories in {lpjg_dir}")
    
    # Read from all run directories
    for run_dir in tqdm(run_dirs, desc=f"Reading {filename}", unit="dir"):
        run_path = Path(run_dir)
        
        # Check for period files (e.g., lai.out_13500101-13511231)
        period_files = sorted(glob.glob(str(run_path / f"{filename}_*")))
        
        # If period files exist, use them; otherwise fall back to the direct file
        files_to_read = period_files if period_files else [run_path / filename]
        
        for filepath in files_to_read:
            filepath = Path(filepath)
            if not filepath.exists() or filepath.is_symlink():
                continue
                
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                continue
            
            header = lines[0].split()
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        yr = int(parts[2])
                        if yr == year:
                            row = {'Lon': float(parts[0]), 'Lat': float(parts[1]), 'Year': yr}
                            for i, col in enumerate(header[3:]):
                                if i + 3 < len(parts):
                                    row[col] = float(parts[i + 3])
                            all_data.append(row)
                    except (ValueError, IndexError):
                        continue
    
    print(f"Extracted {len(all_data)} data points for year {year}")
    return pd.DataFrame(all_data) if all_data else None


def create_land_mask(lon_grid, lat_grid):
    """
    Create a land mask using Natural Earth data.
    
    Parameters:
    -----------
    lon_grid : array-like
        Longitude grid values
    lat_grid : array-like
        Latitude grid values
        
    Returns:
    --------
    np.ndarray
        Boolean mask array
    """
    import shapely.geometry as sgeom
    cache_key = (tuple(lon_grid), tuple(lat_grid))
    if cache_key in _land_mask_cache:
        return _land_mask_cache[cache_key]
    land_shp = shapereader.natural_earth(resolution='110m', category='physical', name='land')
    land = sgeom.MultiPolygon(list(shapereader.Reader(land_shp).geometries()))
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    mask = np.array([[land.contains(sgeom.Point(lon_mesh[i,j], lat_mesh[i,j])) 
                      for j in range(lon_mesh.shape[1])] for i in range(lon_mesh.shape[0])])
    _land_mask_cache[cache_key] = mask
    return mask


def interpolate_to_grid(df, var, year, grid_res=1.0):
    """
    Interpolate variable to regular grid.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Lon, Lat, Year, and variable columns
    var : str
        Variable name to interpolate
    year : int
        Year to extract
    grid_res : float
        Grid resolution in degrees (default: 1.0)
        
    Returns:
    --------
    tuple
        (lon_grid, lat_grid, grid_values, data)
    """
    print(f"  Interpolating {var} to regular grid...", end='', flush=True)
    data = df[df['Year'] == year].copy()
    lon_grid = np.arange(-180, 180, grid_res)
    lat_grid = np.arange(-90, 90, grid_res)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    points = np.column_stack((data['Lon'].values, data['Lat'].values))
    grid_values = griddata(points, data[var].values, (lon_mesh, lat_mesh), method='linear')
    grid_values = np.where(create_land_mask(lon_grid, lat_grid), grid_values, np.nan)
    print(" done")
    return lon_grid, lat_grid, grid_values, data


def get_dominant_pft(df, year, pfts, barren_threshold=0.2):
    """
    Calculate dominant PFT for each grid point.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with PFT LAI columns
    year : int
        Year to extract
    pfts : list
        List of PFT names to consider
    barren_threshold : float
        Threshold below which to classify as barren (default: 0.2)
        
    Returns:
    --------
    tuple
        (data DataFrame with dominant_pft column, list of available PFTs)
    """
    data = df[df['Year'] == year].copy()
    available_pfts = [p for p in pfts if p in data.columns]
    pft_data = data[available_pfts]
    data['dominant_pft'] = pft_data.idxmax(axis=1)
    data['total_lai'] = pft_data.sum(axis=1)
    data.loc[data['total_lai'] < barren_threshold, 'dominant_pft'] = 'Barren'
    all_pfts = available_pfts + ['Barren']
    data['dominant_pft_num'] = data['dominant_pft'].map({pft: i for i, pft in enumerate(all_pfts)})
    return data, all_pfts


def interpolate_dominant_pft(df, year, pfts, grid_res=1.0, barren_threshold=0.2, max_distance=5.0):
    """
    Interpolate dominant PFT to regular grid.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with PFT LAI columns
    year : int
        Year to extract
    pfts : list
        List of PFT names to consider
    grid_res : float
        Grid resolution in degrees (default: 1.0)
    barren_threshold : float
        Threshold below which to classify as barren (default: 0.2)
    max_distance : float
        Maximum distance for nearest neighbor extrapolation (default: 5.0)
        
    Returns:
    --------
    tuple
        (lon_grid, lat_grid, grid_values, data, available_pfts)
    """
    data, available_pfts = get_dominant_pft(df, year, pfts, barren_threshold)
    lon_grid = np.arange(-180, 180, grid_res)
    lat_grid = np.arange(-90, 90, grid_res)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    points = np.column_stack((data['Lon'].values, data['Lat'].values))
    grid_values = griddata(points, data['dominant_pft_num'].values, (lon_mesh, lat_mesh), method='nearest')
    tree = cKDTree(points)
    distances, _ = tree.query(np.column_stack((lon_mesh.ravel(), lat_mesh.ravel())))
    distance_mask = distances.reshape(lon_mesh.shape) <= max_distance
    combined_mask = create_land_mask(lon_grid, lat_grid) & distance_mask
    grid_values = np.where(combined_mask, grid_values, np.nan)
    return lon_grid, lat_grid, grid_values, data, available_pfts
