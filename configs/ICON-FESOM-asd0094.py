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
cdo = Cdo(cdo='/work/ab0246/a270092/software/cdo_build/cdo-1.9.10/src/cdo')
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

# ICON-FESOM smoke-test config. The same single source run (a 150-year
# spinup `asd0094`) is pointed at all three of spinup / pi_ctrl /
# historic for the test — so historic-vs-obs plots will look nonsensical,
# but every script gets exercised end-to-end on the new model.
model_version  = 'ICON-FESOM-asd0094-test3y'
oasis_oifs_grid_name = 'atmo'

# Single source for atm + ocean. Atm side was pre-remapped from ICON's
# unstructured R02B04 to a 360x180 lat/lon grid by
# preprocessing_examples/preprocess_ICON-FESOM_atm.sh. FESOM side stays
# on its CORE2 mesh and is read directly.
spinup_path    = '/work/ab0246/a270092/runtime/ICON_FESOM_asd0094/outdata/'
spinup_name    = model_version + '_spinup'
spinup_start   = 2105
spinup_end     = 2107

pi_ctrl_path   = spinup_path
pi_ctrl_name   = model_version + '_pi-control'
pi_ctrl_start  = spinup_start
pi_ctrl_end    = spinup_end

historic_path  = spinup_path
historic_name  = model_version + '_historic'
historic_start = spinup_start
historic_end   = spinup_end


#Misc
reanalysis             = 'ERA5'
remap_resolution       = '512x256'
dpi                    = 300
# Real historic configs (AWI-ESM2/3) are 170 yr long and want the last
# 25 yr only so the mean represents the modern (not mid-transient)
# climate. For this 3-year smoke test the window is the whole run.
clim_window_years      = spinup_end - spinup_start + 1
historic_last25y_start = historic_end - (clim_window_years - 1)
historic_last25y_end   = historic_end
status_csv             = "log/status.csv"

# FESOM2 CORE2 mesh shipped alongside the ICON-FESOM run.
mesh_name      = 'CORE2'
grid_name      = 'CORE2'
meshpath       = '/work/ab0246/a270092/input/fesom2/core2_icon_fesom/'
mesh_file      = 'core2_griddes_nodes.nc'
griddes_file   = 'core2_griddes_nodes.nc'
abg            = [0, 0, 0]
reference_path = '/work/ab0246/a270092/postprocessing/climatologies/CORE2/'
reference_name = 'clim'
reference_years= 1958

observation_path = '/work/ab0246/a270092/obs/'

# ICON 2D atm fields are instantaneous (W/m^2, kg/m^2/s, K) and the
# preprocessor monmeans them. No accumulation divide needed.
# part2_rad_balance dispatches the sf->heat-flux constant from
# sf.attrs['units'] (the preproc emits 'kg m-2 s-1'), so it'll pick
# Lf = 334000 automatically.
accumulation_period   = 1
precip_to_mm_per_day  = 86400.0

tool_path      = os.getcwd()
out_path       = tool_path+'/output/'+model_version+'/'
os.makedirs(out_path, exist_ok=True)
mesh = pf.load_mesh(meshpath)
data = xr.open_dataset(meshpath+'/fesom.mesh.diag.nc') if os.path.exists(meshpath+'/fesom.mesh.diag.nc') else None

# Per-config script overrides. ICON atm_2d_ml doesn't emit clear-sky
# radiation (tsrc/ttrc); the atm_3d_ml fields live on model levels, and
# the pressure-level interp for QBO / zonal plots is now produced by
# preprocess_ICON-FESOM_atm_pl.sh (cdo ap2pl), so part11/part12 are
# re-enabled.
#
# Re-enabled (preprocessor now synthesises these):
#   - part18 / part16 precip branch: `cp` is emitted as a zero field
#     (lsp carries all precip; cp+lsp = total precip still holds).
#   - part9 SW vs CERES: `ssrd` is derived from sob_s and the four
#     surface albedos, ssrd = sob_s / max(1 - mean(alb), 0.05).
#   - part11 zonal plots, part12 QBO: pressure-level u/v/t/q now
#     produced by preprocess_ICON-FESOM_atm_pl.sh.
scripts_overrides = {
    "part4_cmpi.py":             False,
    "part22_masks.py":           False,
    # No land-surface model output is exposed in this stream; skip both
    # the LPJ-GUESS (AWI-ESM3) and JSBACH (AWI-ESM2) land scripts.
    "part24_lpjg_lai.py":        False,
    "part25_lpjg_carbon.py":     False,
    "part26_lpjg_pft.py":        False,
    "part24_jsbach_lai.py":      False,
    "part25_jsbach_carbon.py":   False,
    "part26_jsbach_pft.py":      False,
}
