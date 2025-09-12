#!/usr/bin/env python3
"""
Utility to generate missing CDO weight files for FESOM2 remapping.

This script checks for the existence of weights_unstr_2_r{resolution}.nc files
and generates them using CDO if they don't exist.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the parent directory to sys.path and load config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

def generate_weight_file(resolution, meshpath, mesh_file, variable='temp'):
    """
    Generate CDO weight file for remapping from unstructured to regular grid.
    
    Parameters:
    -----------
    resolution : str
        Target resolution (e.g., '360x180', '512x256')
    meshpath : str
        Path to mesh directory
    mesh_file : str
        Name of mesh file
    variable : str
        Variable name to use for weight generation (default: 'temp')
    
    Returns:
    --------
    str : Path to generated weight file
    """
    weight_file = f"{meshpath}/weights_unstr_2_r{resolution}.nc"
    
    # Check if weight file already exists
    if os.path.exists(weight_file):
        print(f"Weight file already exists: {weight_file}")
        return weight_file
    
    print(f"Generating weight file: {weight_file}")
    
    # Find a sample FESOM file to use for grid definition
    sample_files = []
    
    # Try different common FESOM variable files
    common_vars = ['temp', 'salt', 'u', 'v', 'ssh']
    for var in common_vars:
        # Look for files in typical FESOM output locations
        for path_candidate in [spinup_path, historic_path]:
            if path_candidate and os.path.exists(path_candidate):
                fesom_dir = os.path.join(path_candidate, 'fesom')
                if os.path.exists(fesom_dir):
                    # Look for files with this variable
                    for file in os.listdir(fesom_dir):
                        if var in file and file.endswith('.nc'):
                            sample_files.append(os.path.join(fesom_dir, file))
                            break
                    if sample_files:
                        break
        if sample_files:
            break
    
    if not sample_files:
        raise FileNotFoundError(f"No FESOM sample files found to generate weights for {resolution}")
    
    sample_file = sample_files[0]
    print(f"Using sample file: {sample_file}")
    
    # Generate CDO command based on your example
    # cdo genycon,r180x91 -selname,${var} -setgrid,$atm_gridfile_path $var weights_unstr_2_r180x91_${var}.nc
    atm_gridfile_path = f"{meshpath}/{mesh_file}"
    
    cmd = [
        'cdo', 
        f'genycon,r{resolution}',
        '-selname,' + variable,
        '-setgrid,' + atm_gridfile_path,
        sample_file,
        weight_file
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully generated: {weight_file}")
        return weight_file
    except subprocess.CalledProcessError as e:
        print(f"Error generating weight file: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise

def check_and_generate_weights(resolution, meshpath, mesh_file):
    """
    Check if weight file exists and generate if missing.
    
    Parameters:
    -----------
    resolution : str
        Target resolution (e.g., '360x180', '512x256')
    meshpath : str
        Path to mesh directory
    mesh_file : str
        Name of mesh file
    
    Returns:
    --------
    str : Path to weight file (existing or newly generated)
    """
    weight_file = f"{meshpath}/weights_unstr_2_r{resolution}.nc"
    
    if os.path.exists(weight_file):
        print(f"Using existing weight file: {weight_file}")
        return weight_file
    else:
        print(f"Weight file missing: {weight_file}")
        return generate_weight_file(resolution, meshpath, mesh_file)

def main():
    """Main function to generate weights for common resolutions."""
    print("=== CDO Weight File Generator ===")
    
    # Common resolutions used in the tool
    resolutions = ['360x180', '512x256', '180x91', '720x361']
    
    for resolution in resolutions:
        try:
            weight_file = check_and_generate_weights(resolution, meshpath, mesh_file)
            print(f"✓ {resolution}: {weight_file}")
        except Exception as e:
            print(f"✗ {resolution}: Failed - {e}")
    
    print("\nWeight file generation complete!")

if __name__ == "__main__":
    main()
