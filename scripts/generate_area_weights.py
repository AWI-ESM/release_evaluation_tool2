#!/usr/bin/env python3
"""
Generate correct area weights for the 192x400 grid using CDO
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *
import subprocess

def generate_area_weights():
    print("=== Generating Area Weights for 192x400 Grid ===")
    
    # Use a sample data file to get the grid
    sample_file = f"{spinup_path}/oifs/atm_remapped_1m_ssr_1m_{spinup_start}-{spinup_start}.nc"
    output_weights = "/tmp/grid_area_weights.nc"
    
    # Use CDO to generate grid area weights
    cmd = f"cdo gridarea {sample_file} {output_weights}"
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("Successfully generated area weights")
            
            # Check the result
            info_cmd = f"cdo info {output_weights}"
            info_result = subprocess.run(info_cmd, shell=True, capture_output=True, text=True)
            print("Area weights info:")
            print(info_result.stdout)
            
            return output_weights
        else:
            print(f"Error: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Exception: {e}")
        return None

if __name__ == "__main__":
    weights_file = generate_area_weights()
    if weights_file:
        print(f"Area weights saved to: {weights_file}")
    else:
        print("Failed to generate area weights")
