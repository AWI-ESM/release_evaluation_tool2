#!/usr/bin/env python3
"""
Utility functions for the release evaluation tool.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the parent directory to sys.path and load config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

import fcntl
import time

def ensure_weight_file(resolution, meshpath, mesh_file, variable='temp'):
    """
    Ensure CDO weight file exists for remapping from unstructured to regular grid.
    Generate if missing using CDO genycon. Thread-safe with file locking.
    
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
    str : Path to weight file
    
    Raises:
    -------
    FileNotFoundError : If no sample FESOM files found
    subprocess.CalledProcessError : If CDO command fails
    """
    weight_file = f"{meshpath}/weights_unstr_2_r{resolution}.nc"
    lock_file = f"{weight_file}.lock"
    
    # Return existing file if found
    if os.path.exists(weight_file):
        return weight_file
    
    # Use file locking to prevent concurrent generation
    try:
        with open(lock_file, 'w') as lock:
            # Try to acquire exclusive lock
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # Check again after acquiring lock (another process might have created it)
            if os.path.exists(weight_file):
                return weight_file
            
            print(f"Generating missing weight file: {weight_file}")
            
            # Find a sample FESOM file for grid definition
            sample_file = None
            common_vars = ['temp', 'salt', 'u', 'v', 'ssh']
            
            for var in common_vars:
                for path_candidate in [spinup_path, historic_path]:
                    if path_candidate and os.path.exists(path_candidate):
                        fesom_dir = os.path.join(path_candidate, 'fesom')
                        if os.path.exists(fesom_dir):
                            for file in os.listdir(fesom_dir):
                                if var in file and file.endswith('.nc'):
                                    sample_file = os.path.join(fesom_dir, file)
                                    break
                            if sample_file:
                                break
                    if sample_file:
                        break
                if sample_file:
                    break
            
            if not sample_file:
                raise FileNotFoundError(f"No FESOM sample files found to generate weights for {resolution}")
            
            # Generate weight file using CDO
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
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if not os.path.exists(weight_file):
                raise FileNotFoundError(f"Weight file generation failed: {weight_file}")
            
            print(f"Generated: {weight_file}")
            
    except BlockingIOError:
        # Another process is generating the file, wait for it
        print(f"Waiting for weight file generation by another process: {weight_file}")
        max_wait = 300  # 5 minutes max wait
        wait_time = 0
        while not os.path.exists(weight_file) and wait_time < max_wait:
            time.sleep(1)
            wait_time += 1
        
        if not os.path.exists(weight_file):
            raise TimeoutError(f"Timeout waiting for weight file generation: {weight_file}")
    
    finally:
        # Clean up lock file
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except:
                pass
    
    return weight_file
