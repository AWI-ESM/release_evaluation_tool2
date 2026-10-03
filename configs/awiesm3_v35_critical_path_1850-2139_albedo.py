# Critical path of the AWI-ESM3 v3.5 tuning campaign, 1850-2139, evaluated on gmhemi1800 2125-2139.
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
model_version  = 'awiesm3_v35_critical_path_1850-2139_albedo'
# THE CRITICAL PATH OF THE TUNING CAMPAIGN AS ONE RECORD, 1850-2139 (2026-10-03).
# The spin-up plots (radiation balance, Hovmoeller, sea-ice extent, AMOC, Gregory) run along the
# five experiments the production line passed through; everything else is evaluated on the last
# 15 years before REcoM was switched on, PICAL_crunveg_gmhemi1800 2125-2139, as HIST and PICT.
#     1850-1919  PICAL                       v3.5 CORE3 defaults, KPP
#     1920-1939  PICAL_momixoff              use_momix off
#     1940-2099  PICAL_ccnice                sea-ice-aware marine CCN; see spinup_annotations
#     2100-2119  PICAL_crunveg               on albedo; LPJ-GUESS from the CRUNCEP state, canopy code
#     2120-2139  PICAL_crunveg_gmhemi1800    hemispheric GM, ocean step 1800 s
# spinup_path is a symlink view built by the campaign repo's scripts/sync/build_critical_path_view.sh.
# Before 2100 it holds only the monthly OpenIFS surface fields and a_ice (levante mirror); the 3D
# ocean diagnostics for those years come from spinup_cache_path, filled on levante with
# scripts/precompute_spinup_cache.py.
# OpenIFS was cold-started at every branch (1920, 1940, 2100, 2120): the first year of each
# segment is an atmosphere spinning up, not a response to the change.
oasis_oifs_grid_name = 'A096'
spinup_path    = '/albedo/work/projects/p_awiesm3_cmip7/jstreffi/reval/views/critical_path_1850-2139/outdata/'
spinup_name    = model_version+'_spinup'
spinup_start   = 1850
spinup_end     = 2124   # the evaluation window starts at 2125 (sea-ice plot: SPIN | HIST & PICT)
# spin-up time series and Hovmoeller continue through the evaluation window, which is shaded
spinup_timeseries_end = 2139
spinup_cache_path     = '/albedo/work/projects/p_awiesm3_cmip7/jstreffi/reval/spinup_cache/critical_path/'
# what changed where along the line; drawn on the spin-up plots (bg_routines/spinup_tools.py)
spinup_annotations = [
    (1920, 'momix off'),
    (1940, 'CCN over sea ice; S4, RSNOWLIN2 dropped'),
    (1970, 'albsn 0.75 to 0.83'),
    (1980, 'albsn 0.80'),
    (2000, 'h0min 1.0'),
    (2010, 'h0min 0.5'),
    (2020, 'aerosol fixed at 1850'),
    (2060, 'cvmix_TKE'),
    (2080, '+ IDEMIX'),
    (2100, 'albedo; CRUNCEP land, canopy'),
    (2120, 'hemispheric GM; dt 1800 s'),
]
_eval_path     = '/albedo/work/projects/p_awiesm3_cmip7/jstreffi/runtime/awiesm3-v3.4/PICAL_crunveg_gmhemi1800/outdata/'
#Preindustrial Control
pi_ctrl_path   = _eval_path
pi_ctrl_name   = model_version+'_pi-control'
pi_ctrl_start  = 2125   # gmhemi1800: last 15 years before REcoM
pi_ctrl_end    = 2139
#Historic
historic_path  = _eval_path
historic_name  = model_version+'_historic'
historic_start = pi_ctrl_start
historic_end   = pi_ctrl_end
#Misc
reanalysis             = 'ERA5'
remap_resolution       = '512x256'
dpi                    = 300
# Use the whole 19-year window as the "last-25y" climatology window.
clim_window_years      = pi_ctrl_end - pi_ctrl_start + 1
# ENSO is read from historic_path, which only holds gmhemi1800: 2121-2139 (2120 is a cold start)
enso_start             = 2121
enso_end               = 2139
historic_last25y_start = historic_end - (clim_window_years - 1)
historic_last25y_end   = historic_end
status_csv             = "log/status.csv"

#Mesh
mesh_name      = 'CORE3'
grid_name      = 'TCO95'
# Symlink mirror of /albedo/work/projects/p_awiesm3_cmip7/input/fesom2/core3/ (220509 nodes,
# md5-identical to levante core3) plus fesom.mesh.diag.nc and core3_griddes_nodes.nc copied
# from levante; kept separate so pyfesom2's pickle cache does not write into the model input.
meshpath       = '/albedo/work/projects/p_awiesm3_cmip7/jstreffi/reval_obs/mesh/core3/'
mesh_file      = 'mesh.nc'
griddes_file   = 'mesh.nc'
abg            = [0, 0, 0]
# PHC3 on the NEW core3 mesh (220509), rsynced from levante climatologies/CORE3_220509.
# NOT /albedo/work/user/jstreffi/climatologies/CORE3 -- that is the 211567-node beta mesh.
reference_path = '/albedo/work/projects/p_awiesm3_cmip7/jstreffi/reval_obs/climatologies/CORE3_220509/'
reference_name = 'clim'
reference_years= 1958
accumulation_period = 3600   # NOT 21600: verified empirically on levante, /3600 -> ASR 240.4 W/m2
precip_to_mm_per_day = 86400000.0   # OpenIFS cp/lsp are metres of water per output step: /3600 * 1000 mm/m * 86400 s/day

# View holding only the complete LPJ-GUESS decades (2100s, 2110s) of THIS run.
# find_lpjg_latest_year() probes the last run dir only, and the live 2120s and later dirs are
# incomplete.
lpjg_path      = '/albedo/work/projects/p_awiesm3_cmip7/jstreffi/reval/views/PICAL_crunveg_gmhemi1800/'

# All files reval reads here are byte-identical (size + md5) to levante /work/ab0246/a270092/obs/.
observation_path = '/albedo/work/user/jstreffi/obs/'

tool_path      = os.getcwd()
out_path       = '/albedo/work/projects/p_awiesm3_cmip7/jstreffi/reval/critical_path_1850-2139/'
os.makedirs(out_path, exist_ok=True)
mesh = pf.load_mesh(meshpath)
data = xr.open_dataset(meshpath+'/fesom.mesh.diag.nc')
