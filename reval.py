"""
AWI-CM3 Release Evaluation Tool (Reval.py)

This script provides a comprehensive evaluation framework for analyzing and visualizing 
data from the AWI-CM-v3.3 climate model. It integrates scientific libraries for data 
processing, statistical analysis, and high-quality visualizations, enabling effective 
assessment of model performance against observations and reanalysis datasets.

Key Features:
- Data Processing & Analysis:
  - Uses PyFESOM2, xarray, SciPy, and scikit-learn for structured climate data handling.
- Visualization:
  - Leverages Matplotlib, Seaborn, Cartopy, and cmocean for high-quality plots.
- FESOM-Specific Routines:
  - Includes functions for handling FESOM2 mesh structures, model data, and 
    meridional overturning circulation (MOC).
- Automated Job Submission:
  - Supports SLURM-based batch processing for large-scale evaluations.
- Multi-Experiment Support:
  - Handles spin-up, preindustrial control, and historical simulations with 
    configurable paths and settings.


2021-12-10: Jan Streffing:                First jupyter notebook version for https://doi.org/10.5194/gmd-15-6399-2022
2024-04-03: Jan Streffing:                Addition of significance metrics for https://doi.org/10.5194/egusphere-2024-2491
2025-02-04: Jan Streffing:                Re-write has parallel scripts
"""

############################
# Module loading         #
############################

#Data access and structures
import pyfesom2 as pf
import xarray as xr
from cdo import *   
#cdo = Cdo(cdo='/home/awiiccp2/miniconda3/envs/pyfesom2/bin/cdo')
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from collections import OrderedDict
import csv
from bg_routines.update_status import update_status

#Plotting
import math as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.ticker import Locator
from matplotlib import ticker
from matplotlib import cm
import seaborn as sns
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from mpl_toolkits.basemap import Basemap
import cmocean as cmo
from cmocean import cm as cmof
import matplotlib.pylab as pylab
import matplotlib.patches as Polygon
import matplotlib.ticker as mticker


#Science
import math
from math import sqrt
from sklearn.metrics import mean_squared_error
from eofs.standard import Eof
from eofs.examples import example_data_path
import shapely
from scipy import signal
from scipy.stats import linregress
from scipy.spatial import cKDTree
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator

#Misc
import os
import warnings
from tqdm import tqdm
import logging
import joblib
import dask
from dask.delayed import delayed
from dask.diagnostics import ProgressBar
import random as rd
import time
import copy as cp
import subprocess

#Fesom related routines
from set_inputarray  import *
from sub_fesom_mesh  import * 
from sub_fesom_data  import * 
from sub_fesom_moc   import *
from colormap_c2c    import *


############################
# Simulation Configuration #
############################

#Name of model release
model_version  = 'AWI-CM-v3.3'

#Spinup
spinup_path    = '/work/ab0246/a270092/runtime/awicm3-v3.3/SPIN/outdata/'
spinup_name    = model_version+'_spinup'
spinup_start   = 1350
spinup_end     = 1849

#Preindustrial Control
pi_ctrl_path   = '/work/ab0246/a270092/runtime/awicm3-v3.3/PI/outdata/'
pi_ctrl_name   = model_version+'_pi-control'
pi_ctrl_start  = 1850
pi_ctrl_end    = 2014

#Historic
historic_path  = '/work/ab0246/a270092/runtime/awicm3-v3.3/HIST/outdata//outdata/'
historic_name  = model_version+'_historic'
historic_start = 1850
historic_end   = 2014


#Misc
reanalysis             = 'ERA5'
remap_resolution       = '512x256'
dpi                    = 300
historic_last25y_start = 1989
historic_last25y_end   = historic_end
status_csv             = "log/status.csv"

#Mesh
mesh_name      = 'CORE2'
grid_name      = 'TCo95'
meshpath       = '/work/ab0246/a270092/input/fesom2/core2/'
mesh_file      = 'mesh.nc'
griddes_file   = 'mesh.nc'
abg            = [0, 0, 0]
reference_path = '/work/ab0246/a270092/postprocessing/climatologies/fdiag/'
reference_name = 'clim'
reference_years= 1958

observation_path = '/work/ab0246/a270092/obs/'

tool_path      = os.getcwd()
out_path       = tool_path+'/output/plot/'+model_version+'/'
mesh = pf.load_mesh(meshpath)
data = xr.open_dataset(meshpath+'/mesh.nc')




############################
# Slurm Configuration      #
############################

SBATCH_SETTINGS = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}.out
#SBATCH --error=logs/{job_name}.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute
#SBATCH -A bb1469
"""



############################
# Script Execution         #
############################

# Ensure required directories exist
os.makedirs("logs", exist_ok=True)

# Locate all Python scripts in the "python_scripts" subfolder
script_files = [f for f in os.listdir("scripts") if f.endswith(".py")]

# Default: Disable all scripts (set to True to enable)
SCRIPTS = {script: False for script in script_files}  # All disabled by default

# Enable scripts manually here:
SCRIPTS.update({
    "part1_mesh_plot.py":           True,
    "part2_rad_balance.py":         False,
    "part3_hovm_temp.py":           False,  
    "part4_cmpi.py":                False,
    "part5_sea_ice_thickness.py":   False,
    "part6_ice_conc_timeseries.py": False,
    "part7_mld.py":                 False,
    "part8_t2m_vs_era5.py":         False,
    "part9_rad_vs_ceres.py":        False,
    "part10_clt_vs_modis.py":       False,
    "part11_zonal_plots.py":        False,
    "part12_qbo.py":                False,
    "part13_fesom_bias_maps.py":    False,
    "part14_fesom_salt.py":         False,
    "part15_enso.py":               False,
    "part16_clim_change.py":        False,
    "part17_moc.py":                False,
})

# Submit jobs
for script, run in SCRIPTS.items():
    if run:
        job_script = f"slurm_{script}.sh"
        script_path = os.path.join("python_scripts", script)

        # Write the SLURM script
        with open(job_script, "w") as f:
            f.write(SBATCH_SETTINGS.format(job_name=script))
            f.write("\nmodule load python\n")  # Load Python module if required
            f.write(f"python {script_path}\n")

        # Submit job
        subprocess.run(["sbatch", job_script])
        print(f"Submitted {script}")
    else:
        print(f"Skipped {script} (disabled)")

