############################
# Module loading           #
############################

import os, sys, warnings, time, math, csv, subprocess
from tqdm import tqdm
import logging, joblib
import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import random as rd
import copy as cp

import pyfesom2 as pf
import xarray as xr
from cdo import *
cdo = Cdo(cdo='/work/ab0246/a270092/software/cdo_build/cdo-1.9.10/src/cdo')
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from collections import OrderedDict
from bg_routines.update_status import update_status

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, Locator)
from matplotlib import ticker, cm
import seaborn as sns
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import cmocean as cmo
from cmocean import cm as cmof
import matplotlib.pylab as pylab
import matplotlib.ticker as mticker

from math import sqrt
from sklearn.metrics import mean_squared_error
from eofs.standard import Eof
from eofs.examples import example_data_path
import shapely
from scipy import signal
from scipy.stats import linregress
from scipy.spatial import cKDTree
from scipy.interpolate import (CloughTocher2DInterpolator,
    LinearNDInterpolator, NearestNDInterpolator)

from bg_routines.set_inputarray  import *
from bg_routines.sub_fesom_mesh  import *
from bg_routines.sub_fesom_data  import *
from bg_routines.sub_fesom_moc   import *
from bg_routines.colormap_c2c    import *

############################
# Simulation Configuration #
############################

model_version  = 'AWIESM2-pi_ctl'

# ENSO/QBO-only run: historic_* is the one we use; spinup/pi_ctrl set
# to historic so reval doesn't complain.
historic_path  = '/work/ab0246/a270092/runtime/ENSOQBO_pi_ctl/outdata/'
historic_name  = model_version+'_historic'
historic_start = 2121
historic_end   = 2320

spinup_path    = historic_path
spinup_name    = historic_name
spinup_start   = historic_start
spinup_end     = historic_end

pi_ctrl_path   = historic_path
pi_ctrl_name   = historic_name
pi_ctrl_start  = historic_start
pi_ctrl_end    = historic_end

reanalysis             = 'ERA5'
remap_resolution       = '360x180'
dpi                    = 150
historic_last25y_start = historic_end - 24
historic_last25y_end   = historic_end
status_csv             = 'log/status.csv'

# CORE2 mesh — all 1x1 experiments share it.
mesh_name      = 'CORE2'
grid_name      = 'T63'
meshpath       = '/work/ab0246/a270092/input/fesom2/core2/'
mesh_file      = 'mesh.nc'
griddes_file   = 'mesh.nc'
abg            = [0, 0, 0]
reference_path = '/work/ab0246/a270092/postprocessing/climatologies/CORE2/'
reference_name = 'clim'
reference_years= 1958
accumulation_period   = 1
precip_to_mm_per_day  = 86400.0

observation_path = '/work/ab0246/a270092/obs/'

tool_path = os.getcwd()
out_path  = tool_path + '/output_enso_qbo/' + model_version + '/'
os.makedirs(out_path, exist_ok=True)
mesh = pf.load_mesh(meshpath)
data = xr.open_dataset(meshpath + '/fesom.mesh.diag.nc')

# Only ENSO and QBO for this driver; everything else is disabled.
scripts_overrides = {
    'part1_mesh_plot.py': False,
    'part2_rad_balance.py': False,
    'part3_hovm_temp.py': False,
    'part4_cmpi.py': False,
    'part5_sea_ice_thickness.py': False,
    'part6_ice_conc_timeseries.py': False,
    'part7_mld.py': False,
    'part8_t2m_vs_era5.py': False,
    'part9_rad_vs_ceres.py': False,
    'part10_clt_vs_modis.py': False,
    'part11_zonal_plots.py': False,
    'part12_qbo.py': True,
    'part13_fesom_temp_bias.py': False,
    'part14_fesom_salt_bias.py': False,
    'part15_enso.py': True,
    'part16_clim_change.py': False,
    'part17_moc.py': False,
    'part18_precip_vs_gpcp.py': False,
    'part19_ocean_temp_sections.py': False,
    'part20_gregory_plot.py': False,
    'part21_crf_bias_maps.py': False,
    'part22_masks.py': False,
    'part23_ice_cavity_velocities.py': False,
    'part24_lpjg_lai.py': False,
    'part24_jsbach_lai.py': False,
    'part25_lpjg_carbon.py': False,
    'part25_jsbach_carbon.py': False,
    'part26_lpjg_pft.py': False,
    'part26_jsbach_pft.py': False,
}
