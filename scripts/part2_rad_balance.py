#!/usr/bin/env python
# coding: utf-8

# # Paths and config

# In[9]:


#Name of model release
model_version  = 'TCo319-HIST'

#Spinup
spinup_path    = '/scratch/awiiccp5/ctl1950d/outdata/'
spinup_name    = model_version+'_spinup'
spinup_start   = 1850
spinup_end     = 2134

#Preindustrial Control
pi_ctrl_path   = '/scratch/awiiccp5/ctl1950d/outdata/'
pi_ctrl_name   = model_version+'_pi-control'
pi_ctrl_start  = 1850
pi_ctrl_end    = 2134

#Historic
historic_path  = '/scratch/awiiccp5/hi1950d/outdata/'
historic_name  = model_version+'_historic'
historic_start = 1950
historic_end   = 2014


# In[2]:


#Misc
reanalysis             = 'ERA5'
remap_resolution       = '360x180'
dpi                    = 600
historic_last25y_start = historic_end-24
historic_last25y_end   = historic_end

#Mesh
mesh_name      = 'DART'
meshpath       = '/proj/awi/input/fesom2/dart/'
mesh_file      = 'dart_griddes_nodes.nc'
griddes_file   = 'dart_griddes_nodes.nc'
abg            = [0, 0, 0]
reference_path = '/proj/awiiccp5/climatologies/'
reference_name = 'clim'
reference_years= 1990

observation_path = '/proj/awi/'


# # Import libraries

# In[3]:


#Data access and structures
import pyfesom2 as pf
import xarray as xr
from cdo import *   # python version
cdo = Cdo(cdo='/home/awiiccp2/miniconda3/envs/pyfesom2/bin/cdo')
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from collections import OrderedDict
import csv

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

#Fesom related routines
from set_inputarray  import *
from sub_fesom_mesh  import * 
from sub_fesom_data  import * 
from sub_fesom_moc   import *
from colormap_c2c    import *

tool_path      = os.getcwd()
out_path       = tool_path+'/output/plot/'+model_version+'/'



# # Spinup radiative balance

# In[11]:


#Config
climatology_path = ['/p/project/chhb19/streffing1/obs/era5/netcdf/']
accumulation_period = 21600 # output frequency of OpenIFS in seconds
figsize=(7.2, 3.8)
var = ['ssr', 'str', 'tsr', 'ttr', 'sf', 'slhf', 'sshf'] 
exps = list(range(spinup_start, spinup_end+1))
ofile = "radiation_budget.png"
#var must have order:  
#1. Surface net solar radiation
#2. Surface net thermal radiation
#3. Top net solar radiation
#4. Top net thermal radiation

# Load model Data
def load_parallel(variable,path):
    data1 = cdo.yearmean(input='-fldmean '+str(path),returnArray=variable)/accumulation_period
    return data1

data = OrderedDict()

for v in var:

    datat = []
    t = []
    temporary = []
    for exp in tqdm(exps):

        path = spinup_path+'/oifs/atm_remapped_1m_'+v+'_1m_'+f'{exp:04d}-{exp:04d}.nc'
        temporary = dask.delayed(load_parallel)(v,path)
        t.append(temporary)

    with ProgressBar():
        datat = dask.compute(t)
    data[v] = np.squeeze(datat)

#Calculate budget:
surface =   np.squeeze(data['ssr']).flatten() + \
            np.squeeze(data['str']).flatten() + \
            np.squeeze(data['sshf']).flatten() + \
            np.squeeze(data['slhf']).flatten() - \
            np.squeeze(data['sf']).flatten()*333550000 
#multiply by heat of fusion: 333550000 mJ/kg - then we get the flux in W/m2
toa = np.squeeze(data['tsr']).flatten() + \
      np.squeeze(data['ttr']).flatten()

#Plot:
def smooth(x,beta):
    """ kaiser window smoothing """
    window_len=11
    beta=10
    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w = np.kaiser(window_len,beta)
    y = np.convolve(w/w.sum(),s,mode='valid')
    return y[5:len(y)-5]

fig, axes = plt.subplots(figsize=figsize)
years = range(spinup_start, spinup_end+1)

plt.plot(years,surface,linewidth=1,color='darkblue', label='_nolegend_')
plt.plot(years,toa,linewidth=1,color='orange', label='_nolegend_')
plt.plot(years,(toa-surface),linewidth=1,color='grey', label='_nolegend_')

plt.plot(years,smooth(surface,len(surface)),color='darkblue')
plt.plot(years,smooth(toa,len(toa)),color='orange')
plt.plot(years,smooth((toa-surface),len(toa-surface)),color='grey')

axes.set_title('Radiative balance',fontweight="bold")

plt.axhline(y=0, color='black', linestyle='-')
plt.ylabel('W/m²',size='13')
plt.xlabel('Year',size='13')

#plt.axvline(x=1650,color='grey',alpha=0.6)

plt.axhline(y=0,color='grey',alpha=0.6)

axes2 = axes.twinx()
axes2.set_ylim(axes.get_ylim())

axes.xaxis.set_minor_locator(MultipleLocator(10))
axes.yaxis.set_minor_locator(MultipleLocator(0.2))
axes2.yaxis.set_minor_locator(MultipleLocator(0.2))

axes.tick_params(labelsize='12')
axes2.tick_params(labelsize='12')

axes.legend(['Net SFC', 'Net TOA', '\u0394(SFC - TOA)'],fontsize=11)
plt.tight_layout()

if ofile is not None:
    plt.savefig(out_path+ofile, dpi=dpi,bbox_inches='tight')
    os.system(f'convert {ofile} -trim {ofile}_trimmed.png')
    os.system(f'mv {ofile}_trimmed.png {ofile}')


