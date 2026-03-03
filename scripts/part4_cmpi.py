#!/usr/bin/env python
# coding: utf-8
"""
CMIP Performance Index (CMPI) Analysis

Uses cmiptool to evaluate model performance against CMIP6 ensemble.
Preprocessing must be run first using preprocessing_examples/preprocess_AWI-CM3-XIOS_monthly.sh
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bg_routines.config_loader import *
from bg_routines.update_status import update_status

# Add cmpitool repo (cloned into bg_routines/cmpitool/) to path
_cmpitool_repo = os.path.join(os.path.dirname(__file__), '..', 'bg_routines', 'cmpitool')
sys.path.insert(0, os.path.abspath(_cmpitool_repo))

SCRIPT_NAME = os.path.basename(__file__)
update_status(SCRIPT_NAME, " Started")


try:
    # Import cmiptool module
    from cmpitool import cmpitool, cmpisetup
    
    print(f"Model version: {model_version}")
    print(f"Historic path: {historic_path}")
    print(f"Historic years: {historic_start}-{historic_end}")
    print(f"Mesh path: {meshpath}")
    print(f"Grid file: {mesh_file}")
    
    # Setup cmiptool variables and models
    variable, region, climate_model, siconc, tas, clt, pr, rlut, uas, vas, ua, zg, zos, mlotst, thetao, so = cmpisetup()
    
    # Path for preprocessed CMPI input data
    cmpi_input_path = os.path.join(tool_path, 'input', 'cmpi')
    os.makedirs(cmpi_input_path, exist_ok=True)
    
    print(f"\nCMPI input directory: {cmpi_input_path}")
    print(f"Output directory: {out_path}")


    # ======================================
    # Preprocessing Information
    # ======================================
    print("\n" + "="*70)
    print("PREPROCESSING REQUIRED BEFORE RUNNING THIS SCRIPT")
    print("="*70)
    print("\nPreprocessing will run automatically with these parameters:")
    print(f"  Script: preprocess_AWI-CM3-XIOS.sh")
    print(f"  Data path: {historic_path}")
    print(f"  Output: {cmpi_input_path}")
    print(f"  Model: {model_version}")
    print(f"  Years: {historic_start}-{historic_end}")
    print(f"  Grid: {meshpath}/{mesh_file}")
    print("\n" + "="*70 + "\n")

    # ======================================
    # Run Preprocessing
    # ======================================
    
    import subprocess
    
    print("\n" + "="*70)
    print("RUNNING PREPROCESSING")
    print("="*70 + "\n")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preprocess_script = os.path.join(project_root, "preprocessing_examples", "preprocess_AWI-CM3-XIOS.sh")
    gridfile = os.path.join(meshpath, mesh_file)
    if not os.path.exists(preprocess_script):
        print(f"ERROR: Preprocessing script not found: {preprocess_script}")
        update_status(SCRIPT_NAME, " Failed - preprocessing script missing")
        sys.exit(1)
    flux_scale = "21600"
    cmd = ["bash", preprocess_script, historic_path, cmpi_input_path,
           model_version, str(historic_start), str(historic_end),
           gridfile, "true", flux_scale]
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\nERROR: Preprocessing failed with exit code {result.returncode}")
        update_status(SCRIPT_NAME, f" Failed - preprocessing error")
        sys.exit(1)
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")

    # ======================================
    # Configure Models for CMPI Analysis
    # ======================================
    
    # Define which variables to evaluate for this model
    # Exclude siconc if sea ice data not available
    model_variables = [tas, clt, pr, rlut, uas, vas, ua, zg, zos, mlotst, thetao, so]
    
    # Define the model to evaluate
    models = [
        climate_model(name=model_version, variables=model_variables)
    ]
    
    print(f"\nEvaluating model: {model_version}")
    print(f"Variables: {len(model_variables)}")

    # ======================================
    # Define Bias Map Color Limits
    # ======================================
    
    # Fixed limits for bias map colorbars (optional, can use dynamic)
    fixed_limits = {
        'siconc': 60.0,     # Sea Ice Area Fraction (percent)
        'tas': 5.0,         # Near-Surface Air Temperature (K)
        'clt': 30.0,        # Total Cloud Area Fraction (percent)
        'pr': 5.0,          # Precipitation Rate (mm/day)
        'rlut': 20.0,       # TOA Outgoing Longwave Radiation (W/m²)
        'uas': 3.0,         # Eastward Near-Surface Wind Speed (m/s)
        'vas': 3.0,         # Northward Near-Surface Wind Speed (m/s)
        'ua': 5.0,          # Eastward Wind Component (m/s)
        'zg': 100.0,        # Geopotential Height (m)
        'zos': 0.3,         # Sea Surface Height Above Geoid (m)
        'mlotst': 100.0,    # Ocean Mixed Layer Thickness (m)
        'thetao': 3.0,      # Sea Water Potential Temperature (K)
        'so': 1.0           # Sea Water Practical Salinity (psu)
    }
    
    # ======================================
    # Run CMPI Tool
    # ======================================
    
    print("\n" + "="*70)
    print("RUNNING CMPI ANALYSIS")
    print("="*70 + "\n")
    
    # Resolve obs and eval paths relative to the cmpitool repo root (bg_routines/cmpitool/)
    cmpitool_repo_root = os.path.abspath(_cmpitool_repo)
    obs_path = os.path.join(cmpitool_repo_root, 'obs')
    eval_path = os.path.join(cmpitool_repo_root, 'eval', 'ERA5')
    
    # Use the configured output directory and create required subdirs
    cmpi_out_path = os.path.join(out_path, 'cmpi')
    for subdir in ['abs', 'frac', 'plot', 'plot/maps']:
        os.makedirs(os.path.join(cmpi_out_path, subdir), exist_ok=True)
    
    print(f"Obs data path: {obs_path}")
    print(f"Eval data path: {eval_path}")
    print(f"CMPI output path: {cmpi_out_path}")
    
    # Run cmiptool with bias maps enabled
    cmpitool(
        cmpi_input_path,
        models,
        verbose=True,
        biasmaps=True,
        biasmap_limits=fixed_limits,
        obs_path=obs_path,
        eval_path=eval_path,
        out_path=cmpi_out_path,
        use_for_eval=False  # Set to True to add this model to eval database
    )
    
    print("\n" + "="*70)
    print("CMPI ANALYSIS COMPLETED")
    print("="*70)
    
    # Copy heatmap to standard output location as cmpi.png
    import shutil
    heatmap_src = os.path.join(cmpi_out_path, 'plot', model_version + '.png')
    if os.path.exists(heatmap_src):
        shutil.copy2(heatmap_src, os.path.join(out_path, 'cmpi.png'))
        print(f"Copied heatmap to {out_path}cmpi.png")
    
    print(f"\nCMPI outputs saved to: {cmpi_out_path}")
    print("- CMPI heatmap: cmpi.png")
    print("- Bias maps: cmpi/plot/maps/*.png")
    print("- Performance data: cmpi/abs/*.csv, cmpi/frac/*.csv")
    
    update_status(SCRIPT_NAME, " Completed")
    
except ImportError as e:
    print(f"ERROR: Failed to import cmiptool: {e}")
    print("Please ensure cmiptool is installed in your environment.")
    update_status(SCRIPT_NAME, f" Failed: {e}")
    sys.exit(1)
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    update_status(SCRIPT_NAME, f" Failed: {e}")
    raise


