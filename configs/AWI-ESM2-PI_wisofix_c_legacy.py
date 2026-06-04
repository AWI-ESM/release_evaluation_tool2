# Temporary config pointing at the legacy single-source PI_wisofix_c
# workspace, used only to validate fixes (e.g. radiation_budget unit)
# while the new multi-source workspaces are being built.
# Once the multi-source setup is done, switch back to the main config.

############################
# Module loading           #
############################
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

import math as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, Locator
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

from bg_routines.set_inputarray  import *
from bg_routines.sub_fesom_mesh  import *
from bg_routines.sub_fesom_data  import *
from bg_routines.sub_fesom_moc   import *
from bg_routines.colormap_c2c    import *


############################
# Simulation Configuration #
############################
model_version  = 'AWI-ESM2-PI_wisofix_c'
oasis_oifs_grid_name = 'atmo'

# Legacy single-source layout (PI_wisofix_c only). 5369-5839 spinup,
# last 165 years (5840-6004) double as pict and historic placeholder.
spinup_path    = '/work/ab0246/a270092/runtime/PI_wisofix_c/outdata/'
spinup_name    = model_version+'_spinup'
spinup_start   = 5369
spinup_end     = 5839

pi_ctrl_path   = spinup_path
pi_ctrl_name   = model_version+'_pi-control'
pi_ctrl_start  = 5840
pi_ctrl_end    = 6004

historic_path  = spinup_path
historic_name  = model_version+'_historic'
historic_start = 5840
historic_end   = 6004

reanalysis             = 'ERA5'
remap_resolution       = '512x256'
dpi                    = 300
historic_last25y_start = historic_end-24
historic_last25y_end   = historic_end
status_csv             = "log/status.csv"

mesh_name      = 'CORE2'
grid_name      = 'T63'
meshpath       = '/work/ab0246/a270092/input/fesom2/core2/'
mesh_file      = 'mesh.nc'
griddes_file   = 'mesh.nc'
abg            = [0, 0, 0]
reference_path = '/work/ab0246/a270092/postprocessing/climatologies/CORE2/'
reference_name = 'clim'
reference_years= 1958

observation_path = '/work/ab0246/a270092/obs/'

accumulation_period   = 1
precip_to_mm_per_day  = 86400.0

tool_path      = os.getcwd()
out_path       = tool_path+'/output/'+model_version+'/'
os.makedirs(out_path, exist_ok=True)
mesh = pf.load_mesh(meshpath)
data = xr.open_dataset(meshpath+'/fesom.mesh.diag.nc')
