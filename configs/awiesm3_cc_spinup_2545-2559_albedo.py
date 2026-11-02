# v3.5.0-cc CMIP7 spin-up (AWI-ESM3-CC-LR-CMIP7-Spinup, run by lahajd001), 1850-2559 (v3.5.0-cc to 2229, v3.5.1-cc from 2230); climatology window = last 15 years 2545-2559.
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
model_version  = 'awiesm3_cc_spinup_2545-2559_albedo'
# PICAL_crunveg on albedo (AWI): sibling of PICAL_ccnice, same CORE3 (220509) + TCO95 setup
# (see configs/awiesm3_v35_pical_ccnice_2100s_albedo.py for the albedo-specific choices).
#
# WINDOW 2101-2119: outdata starts at 2100 (cold-started atmosphere, excluded); 19 years,
# not decade aligned.  The run continues past 2119 (run_2120*, lpj_guess/2120* and later are
# incomplete): every path below is capped at 2119, and LPJ-GUESS is read through a symlink
# view that only holds the complete decades.

oasis_oifs_grid_name = 'A096'

# Whole series == climatology window here: there is no earlier record on albedo.
spinup_path    = '/albedo/work/projects/p_awiesm3_cmip7/lahajd001/runtime/awiesm3-v3.5.0-cc/AWI-ESM3-CC-LR-CMIP7-Spinup/outdata/'
spinup_name    = model_version+'_spinup'
spinup_start   = 1850
spinup_end     = 2559

#Preindustrial Control
pi_ctrl_path   = spinup_path
pi_ctrl_name   = model_version+'_pi-control'
pi_ctrl_start  = 2545   # last 15 years
pi_ctrl_end    = 2559

#Historic
historic_path  = spinup_path
historic_name  = model_version+'_historic'
historic_start = pi_ctrl_start
historic_end   = pi_ctrl_end


#Misc
reanalysis             = 'ERA5'
remap_resolution       = '512x256'
dpi                    = 300
# Use the whole 19-year window as the "last-25y" climatology window.
clim_window_years      = pi_ctrl_end - pi_ctrl_start + 1
enso_start             = pi_ctrl_start
enso_end               = pi_ctrl_end
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
# The spin-up continued on v3.5.1-cc in a second experiment from 2230. Read by part 27 (carbon pools).
spinup_segments = [
    ('/albedo/work/projects/p_awiesm3_cmip7/lahajd001/runtime/awiesm3-v3.5.0-cc/AWI-ESM3-CC-LR-CMIP7-Spinup/outdata/', 1850, 2229, 'v3.5.0-cc'),
    ('/albedo/work/projects/p_awiesm3_cmip7/lahajd001/runtime/awiesm3-v3.5.1-cc/AWI-ESM3-CC-LR-CMIP7-Spinup_v351_from2230/outdata/', 2230, 2559, 'v3.5.1-cc'),
]
lpjg_path      = '/albedo/work/projects/p_awiesm3_cmip7/lahajd001/runtime/awiesm3-v3.5.0-cc/AWI-ESM3-CC-LR-CMIP7-Spinup/outdata/lpj_guess/'

# All files reval reads here are byte-identical (size + md5) to levante /work/ab0246/a270092/obs/.
observation_path = '/albedo/work/user/jstreffi/obs/'

tool_path      = os.getcwd()
out_path       = '/albedo/work/projects/p_awiesm3_cmip7/jstreffi/reval/SPINUP_v351cc_2545-2559/'
os.makedirs(out_path, exist_ok=True)
mesh = pf.load_mesh(meshpath)
data = xr.open_dataset(meshpath+'/fesom.mesh.diag.nc')
