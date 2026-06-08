############################
# Module loading           #
############################

#Misc
import os
import sys
import warnings
from tqdm import tqdm
import logging
import joblib
import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import random as rd
import time
import copy as cp
import subprocess


#Data access and structures
import pyfesom2 as pf
import xarray as xr
from cdo import *
cdo = Cdo(cdo=os.path.join(sys.prefix, 'bin')+'/cdo')
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


#Fesom related routines
from bg_routines.set_inputarray  import *
from bg_routines.sub_fesom_mesh  import *
from bg_routines.sub_fesom_data  import *
from bg_routines.sub_fesom_moc   import *
from bg_routines.colormap_c2c    import *


############################
# Simulation Configuration #
############################

# AWI-ESM3-VEG-HR (TCO199 atm / HR FESOM mesh ~ 3.15M nodes).
# Three runs unified into one workspace tree under /work/bb1469/a270092/
# by preprocessing_examples/setup_AWI-ESM3-VEG-HR_workspaces.sh which
# symlinks the source dirs (Spinup_cont2 + Spinup_cont3 + piControl +
# historical) with the reval-expected file names. See that script for
# the year-by-year provenance.
model_version  = 'AWI-ESM3-VEG-HR'
oasis_oifs_grid_name = 'A199'

# Spinup: Spinup_cont2 (1350-1649, awiesm3-v3.4.1) merged with
# Spinup_cont3 (1650-1679, awiesm3-v3.4.2). 330 yr total.
spinup_path    = '/work/bb1469/a270092/runtime/awiesm3-v3.4.2/AWI-ESM3-VEG-HR-Spinup/outdata/'
spinup_name    = model_version + '_spinup'
spinup_start   = 1350
spinup_end     = 1679

# piControl: 16 yr at 1850-1865. clim_window_years is bumped down to 16
# below so the last-25y window doesn't reach before the run starts.
pi_ctrl_path   = '/work/bb1469/a270092/runtime/awiesm3-v3.4.2/AWI-ESM3-VEG-HR-piControl/outdata/'
pi_ctrl_name   = model_version + '_pi-control'
pi_ctrl_start  = 1850
pi_ctrl_end    = 1865

# historical: 30 yr at 1850-1879. Last 25 yr = 1855-1879.
historic_path  = '/work/bb1469/a270092/runtime/awiesm3-v3.4.2/AWI-ESM3-VEG-HR-historical/outdata/'
historic_name  = model_version + '_historic'
historic_start = 1850
historic_end   = 1879


#Misc
reanalysis             = 'ERA5'
remap_resolution       = '512x256'
dpi                    = 300
# piControl is only 16 yr, so cap the climatology window to its length.
# Real AWI-ESM3 configs at LR use the default 25 yr; this falls back to
# the AWI-ESM2 / ICON pattern where clim_window_years overrides the
# default through globals().get('clim_window_years', 25) in the scripts.
clim_window_years      = min(25, pi_ctrl_end - pi_ctrl_start + 1)
historic_last25y_start = historic_end - (clim_window_years - 1)
historic_last25y_end   = historic_end
status_csv             = "log/status.csv"

# HR FESOM2 mesh (~3.15M nod2). dist_2304 partitioning matches the run.
mesh_name      = 'HR'
grid_name      = 'TCO199'
meshpath       = '/work/ab0246/a270092/input/fesom2/HR/'
mesh_file      = 'mesh.nc'
griddes_file   = 'mesh.nc'
abg            = [0, 0, 0]
# CORE3 climatology dir is the closest reference set we have prebaked at
# 512x256. Swap to an HR-specific one if/when it exists.
reference_path = '/work/ab0246/a270092/postprocessing/climatologies/CORE3/'
reference_name = 'clim'
reference_years= 1958

observation_path = '/work/ab0246/a270092/obs/'

# AWI-ESM3 XIOS emits OIFS accumulated fluxes integrated over 6h
# intervals (J/m^2 for SW/LW, m water-equivalent for snowfall). Same
# convention as the LR config.
accumulation_period = 21600

tool_path      = os.getcwd()
out_path       = tool_path+'/output/'+model_version+'/'
os.makedirs(out_path, exist_ok=True)
mesh = pf.load_mesh(meshpath)
data = xr.open_dataset(meshpath+'/fesom.mesh.diag.nc') if os.path.exists(meshpath+'/fesom.mesh.diag.nc') else None
