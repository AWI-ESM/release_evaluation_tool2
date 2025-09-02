#!/usr/bin/env python3
"""
Fix area weighting to match CDO results
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *
import xarray as xr
import numpy as np

def test_proper_area_weighting():
    print("=== Testing Proper Area Weighting ===")
    
    # Test file
    test_file = f"{spinup_path}/oifs/atm_remapped_1m_ssr_1m_{spinup_start}-{spinup_start}.nc"
    
    # Load data
    ds = xr.open_dataset(test_file, decode_times=False, engine="netcdf4")
    da = ds['ssr']
    
    # Load area weights
    area_file = f"{spinup_path}/../restart/oasis3mct/areas.nc"
    area_ds = xr.open_dataset(area_file, decode_times=False, engine="netcdf4")
    area_weights = area_ds[f"{oasis_oifs_grid_name}.srf"]
    
    print(f"Data shape: {da.shape}")
    print(f"Area weights shape: {area_weights.shape}")
    print(f"Expected grid size: {da.sizes['lat'] * da.sizes['lon']} = {da.sizes['lat']} x {da.sizes['lon']}")
    
    # Reshape area weights to match data grid
    if area_weights.shape == (1, da.sizes['lat'] * da.sizes['lon']):
        # Reshape from (1, N) to (lat, lon)
        area_2d = area_weights.values.reshape(da.sizes['lat'], da.sizes['lon'])
        area_weights_matched = xr.DataArray(
            area_2d, 
            dims=['lat', 'lon'], 
            coords={'lat': da.lat, 'lon': da.lon}
        )
        print("Successfully reshaped area weights")
    else:
        print(f"Area weights shape mismatch: {area_weights.shape}")
        return
    
    # Calculate yearly mean first
    yearly_mean = da.mean(dim='time_counter')
    
    # Apply area weighting
    weighted_sum = (yearly_mean * area_weights_matched).sum()
    total_area = area_weights_matched.sum()
    area_weighted_mean = weighted_sum / total_area
    
    print(f"Area weighted mean: {area_weighted_mean.values:.0f} J/m²")
    print(f"Converted to W/m²: {area_weighted_mean.values/86400:.2f} W/m²")
    
    # Compare with simple mean
    simple_mean = yearly_mean.mean()
    print(f"Simple mean: {simple_mean.values:.0f} J/m²")
    print(f"Simple mean W/m²: {simple_mean.values/86400:.2f} W/m²")
    
    print(f"CDO reference: 40.2 W/m²")
    print(f"Difference from CDO: {area_weighted_mean.values/86400 - 40.2:.2f} W/m²")
    
    ds.close()
    area_ds.close()

if __name__ == "__main__":
    test_proper_area_weighting()
