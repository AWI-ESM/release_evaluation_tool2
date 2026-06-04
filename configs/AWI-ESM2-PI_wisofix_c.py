############################
# Module loading         #
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
# hist_1x1 jsbach output is GRIB-szip-compressed and the conda env's cdo
# was built without szlib support; use the dedicated szip-capable build.
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

#Name of model release
model_version  = 'AWI-ESM2-PI_wisofix_c'
oasis_oifs_grid_name = 'atmo'

# AWI-ESM2 multi-source layout. Three separate workspaces built by
# /work/ab0246/a270092/runtime/setup_AWIESM2_workspaces.sh symlink the
# source data with a normalized expname prefix per workspace so reval
# auto-detection picks one experiment per workspace:
#   spinup  : PI_251 + PI_wiso + PI_wisofix + PI_wisofix_c, 2001-5830
#             (per-month FESOM in PI_251/PI_wiso was cdo-cat'd to per-year
#              under /work/ab0246/a270092/downloads/AWIESM2_fesom_cat/.)
#   pi_ctrl : last 170 years of PI_wisofix_c (5831-6000)
#   historic: hist_1x1 (1850-2019, length 170y -> same as pi_ctrl)
spinup_path    = '/work/ab0246/a270092/runtime/AWIESM2_spinup/outdata/'
spinup_name    = model_version+'_spinup'
spinup_start   = 2001
spinup_end     = 5830

pi_ctrl_path   = '/work/ab0246/a270092/runtime/AWIESM2_pict/outdata/'
pi_ctrl_name   = model_version+'_pi-control'
pi_ctrl_start  = 5831
pi_ctrl_end    = 6000

historic_path  = '/work/ab0246/a270092/runtime/AWIESM2_hist/outdata/'
historic_name  = model_version+'_historic'
historic_start = 1850
historic_end   = 2019


#Misc
reanalysis             = 'ERA5'
remap_resolution       = '512x256'
dpi                    = 300
historic_last25y_start = historic_end-24
historic_last25y_end   = historic_end
status_csv             = "log/status.csv"

#Mesh - core2. Use /work/ab0246/a270092/input/fesom2/core2 (has mesh.nc
#with cell_area and the pre-baked CDO grid), not the raw ICEBERGS dir.
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

# echam6 emits fluxes already as W/m**2 and precip already as kg/m**2/s
# (instantaneous, not accumulated), so we don't divide by an
# accumulation interval and the precip conversion is just *86400.
# For AWI-CM3 XIOS the corresponding values were 21600 / ~4000.
accumulation_period   = 1
precip_to_mm_per_day  = 86400.0

tool_path      = os.getcwd()
out_path       = tool_path+'/output/'+model_version+'/'
os.makedirs(out_path, exist_ok=True)
mesh = pf.load_mesh(meshpath)
data = xr.open_dataset(meshpath+'/fesom.mesh.diag.nc')

# Per-config script overrides (read by reval.py via a lightweight ast
# walk, so only bool-literal values are accepted). For AWI-ESM2 we swap
# the LPJ-GUESS land scripts for jsbach replacements and disable
# part22_masks (OASIS grid configs differ; not worth porting for now).
scripts_overrides = {
    # part11/part12 require 3D atmosphere fields on pressure levels
    # (atm_remapped_1m_pl_<u|t>_1m_pl_*.nc). Phase 2b (cdo afterburner
    # for spectral->grid + pressure-level interp + zg hydrostatic
    # integration) hasn't been built yet, so disable until then.
    "part11_zonal_plots.py":     False,
    "part12_qbo.py":             False,
    "part22_masks.py":           False,
    # part4_cmpi.py expects to re-run the AWI-CM3 XIOS preprocessor against
    # `model_version` -- for this run, cmpitool was already executed via
    # cmpitool/run_PI_wisofix_c.py and the result lives in
    # cmpitool/eval/ERA5/PI_wisofix_c.csv (note the name without prefix).
    # Wiring part4 to consume that pre-existing csv is Phase 3 follow-up.
    "part4_cmpi.py":             False,
    "part24_lpjg_lai.py":        False,
    "part25_lpjg_carbon.py":     False,
    "part26_lpjg_pft.py":        False,
    "part24_jsbach_lai.py":      True,
    "part25_jsbach_carbon.py":   True,
    "part26_jsbach_pft.py":      True,
}
