#!/usr/bin/env python3
"""
Test script to verify area weighting matches CDO results
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *
import xarray as xr
import numpy as np

def test_area_weighting():
    print("=== Testing Area Weighting ===")
    
    # Test file
    test_file = f"{spinup_path}/oifs/atm_remapped_1m_ssr_1m_{spinup_start}-{spinup_start}.nc"
    print(f"Test file: {test_file}")
    
    # Load data
    ds = xr.open_dataset(test_file, decode_times=False, engine="netcdf4")
    da = ds['ssr']
    print(f"Data shape: {da.shape}")
    print(f"Data range: {da.min().values:.0f} to {da.max().values:.0f} J/m²")
    
    # Load area weights
    area_file = f"{spinup_path}/../restart/oasis3mct/areas.nc"
    print(f"Area file: {area_file}")
    
    try:
        area_ds = xr.open_dataset(area_file, decode_times=False, engine="netcdf4")
        print(f"Area file variables: {list(area_ds.variables.keys())}")
        
        area_variable = f"{oasis_oifs_grid_name}.srf"
        if area_variable in area_ds:
            area_weights = area_ds[area_variable]
            print(f"Area weights shape: {area_weights.shape}")
            print(f"Area weights dims: {area_weights.dims}")
            print(f"Area weights range: {area_weights.min().values:.2e} to {area_weights.max().values:.2e}")
        else:
            print(f"Area variable {area_variable} not found")
            print(f"Available variables: {list(area_ds.variables.keys())}")
            
        area_ds.close()
    except Exception as e:
        print(f"Error loading area file: {e}")
        # Try alternative locations
        alt_paths = [
            f"{spinup_path}/../restart/areas.nc",
            f"{spinup_path}/../../areas.nc",
            f"{spinup_path}/areas.nc"
        ]
        for alt_path in alt_paths:
            print(f"Trying: {alt_path}")
            if os.path.exists(alt_path):
                print(f"Found: {alt_path}")
                break
    
    # Test simple mean vs area-weighted mean
    yearly_mean = da.mean(dim='time_counter')
    simple_global_mean = yearly_mean.mean()
    print(f"Simple global mean: {simple_global_mean.values:.0f} J/m²")
    print(f"Converted to W/m²: {simple_global_mean.values/86400:.2f} W/m²")
    
    ds.close()
    
    print("\n=== CDO Reference ===")
    print("CDO fldmean -yearmean result: 3,473,800 J/m² = 40.2 W/m²")
    print("Python simple mean result: {:.0f} J/m² = {:.2f} W/m²".format(
        simple_global_mean.values, simple_global_mean.values/86400))

if __name__ == "__main__":
    test_area_weighting()
