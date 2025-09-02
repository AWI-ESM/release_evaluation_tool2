#!/usr/bin/env python3
"""
Debug the grid size mismatch between area weights and data
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *
import xarray as xr
import numpy as np

def debug_grid_mismatch():
    print("=== Debugging Grid Mismatch ===")
    
    # Check data file
    test_file = f"{spinup_path}/oifs/atm_remapped_1m_ssr_1m_{spinup_start}-{spinup_start}.nc"
    ds = xr.open_dataset(test_file, decode_times=False, engine="netcdf4")
    print(f"Data grid: {ds.ssr.shape} = {ds.sizes['lat']}×{ds.sizes['lon']} = {ds.sizes['lat'] * ds.sizes['lon']} cells")
    
    # Check area weights file
    area_file = f"{spinup_path}/../restart/oasis3mct/areas.nc"
    area_ds = xr.open_dataset(area_file, decode_times=False, engine="netcdf4")
    
    print(f"\nArea file variables: {list(area_ds.variables.keys())}")
    
    # Check all area variables
    for var in area_ds.variables:
        if 'srf' in var:
            area_var = area_ds[var]
            print(f"{var}: shape={area_var.shape}, dims={area_var.dims}, size={area_var.size}")
    
    # Check if there's a different grid resolution
    area_weights = area_ds[f"{oasis_oifs_grid_name}.srf"]
    print(f"\nArea weights: {area_weights.shape} = {area_weights.size} cells")
    
    # Calculate what grid size would give 40320 cells
    import math
    possible_grids = []
    for lat in range(50, 300):
        for lon in range(50, 500):
            if lat * lon == 40320:
                possible_grids.append((lat, lon))
    
    print(f"Possible grids for 40320 cells: {possible_grids}")
    
    # Check if 40320 = 144 × 280 or similar
    print(f"40320 factorization:")
    n = 40320
    factors = []
    for i in range(2, int(math.sqrt(n)) + 1):
        while n % i == 0:
            factors.append(i)
            n //= i
    if n > 1:
        factors.append(n)
    print(f"Prime factors: {factors}")
    
    # Common climate model grids
    common_grids = [(144, 280), (160, 252), (128, 315), (180, 224)]
    for lat, lon in common_grids:
        if lat * lon == 40320:
            print(f"Match found: {lat}×{lon} = {lat*lon}")
    
    ds.close()
    area_ds.close()

if __name__ == "__main__":
    debug_grid_mismatch()
