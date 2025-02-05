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
spinup_end     = 1950

#Preindustrial Control
pi_ctrl_path   = '/scratch/awiiccp5/ctl1950d/outdata/'
pi_ctrl_name   = model_version+'_pi-control'
pi_ctrl_start  = 1950
pi_ctrl_end    = 2014

#Historic
historic_path  = '/scratch/awiiccp5/hi1950d/outdata/'
historic_name  = model_version+'_historic'
historic_start = 1950
historic_end   = 2014


# In[2]:


#Misc
reanalysis             = 'ERA5'
remap_resolution       = '360x180'
dpi                    = 300
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

mesh = pf.load_mesh(meshpath)
data = xr.open_dataset(meshpath+'/fesom.mesh.diag.nc')

#%%capture
runs=[spinup_name, historic_name, pi_ctrl_name]

runid ='fesom'
str_id='a_ice'
fig, ax1 = plt.subplots(1, sharex=True,figsize=(22,5.8))

class MinorSymLogLocator(mticker.Locator):
    """
    Dynamically find minor tick positions based on the positions of major ticks for a symlog scaling.
    
    Attributes
    ----------
    linthresh : float
        The same linthresh value used when setting the symlog scale.
        
    """
    
    def __init__(self, linthresh):
        #super().__init__()
        self.linthresh = linthresh

    def __call__(self):
        majorlocs = self.axis.get_majorticklocs()
        # iterate through minor locs
        minorlocs = []
        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)
        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a {0} type.'.format(type(self)))

        

for exp in runs:  
    if exp == pi_ctrl_name: 
        datapath   = pi_ctrl_path+'/fesom'
        year_start = pi_ctrl_start
        year_end   = pi_ctrl_end
    elif exp == historic_name: 
        datapath   = historic_path+'/fesom'
        year_start = historic_start
        year_end   = historic_end
    elif exp == spinup_name: 
        datapath   = spinup_path+'/fesom'
        year_start = spinup_start
        year_end   = spinup_end
 
    extent_north_march = []
    extent_south_march = []
    extent_north_sep = []
    extent_south_sep = []       
        
    for y in tqdm(range(year_start, year_end)): 
        for x in range(2, 3):
            data = pf.get_data(datapath, str_id, y, mesh, records=[x])
            extent_north_march.append(pf.ice_ext(data, mesh, hemisphere="N"))
            extent_south_march.append(pf.ice_ext(data, mesh, hemisphere="S"))
        for x in range(8, 9):
            data = pf.get_data(datapath, str_id, y, mesh, records=[x])
            extent_north_sep.append(pf.ice_ext(data, mesh, hemisphere="N"))
            extent_south_sep.append(pf.ice_ext(data, mesh, hemisphere="S"))
          
    #if exp == 'SPIN':
    #    years = np.linspace(year_start-700, year_end-700,year_end-year_start+1)
    #else:
    years = np.linspace(year_start, year_end,year_end-year_start+1)

    years = years[:len(years)-1]
    extent_north_march = np.squeeze(np.asarray(extent_north_march))
    extent_south_march = np.squeeze(np.asarray(extent_south_march))
    extent_north_sep = np.squeeze(np.asarray(extent_north_sep))
    extent_south_sep = np.squeeze(np.asarray(extent_south_sep))


    if exp == pi_ctrl_name:
        linestyle=':'
        alpha=0.5
    elif exp == spinup_name:
        linestyle='-'
        alpha=0.6
    elif exp == historic_name:
        linestyle='-'
        alpha=1
        
    ax1.plot(years,extent_north_march/1000000000000,linewidth=2,color='grey',linestyle=linestyle,alpha=alpha,label='Arctic March');
    ax1.tick_params(axis='both', labelsize=17)
        
    ax1.plot(years,extent_north_sep/1000000000000,linewidth=2,color='Orange',linestyle=linestyle,alpha=alpha,label='Arctic September');
    ax1.tick_params(axis='both', labelsize=17)
    
    ax1.plot(years,extent_south_march/1000000000000,linewidth=2,color='Darkblue',linestyle=linestyle,alpha=alpha,label='Antarctic March');
    ax1.tick_params(axis='both', labelsize=17)

    ax1.plot(years,extent_south_sep/1000000000000,linewidth=2,color='black',linestyle=linestyle,alpha=alpha,label='Antarctic September');
    ax1.tick_params(axis='both', labelsize=17)
    
ax1.set_title('Sea ice extent', fontsize=17,fontweight='bold')

#fig.text(-0.04, 0.5, 'Sea ice extent [$1000 km^2$]', fontsize=13, va='center', rotation=90)
#fig.text(0.5, -0.02, 'Year', fontsize=13, ha='center', rotation=0)
ax1.set_ylabel('Sea ice extent [$1000 km^2$]', fontsize=17)
ax1.set_xlabel('Year', fontsize=17)

ax1.yaxis.grid(color='gray', linestyle='dashed')

plt.axvline(x=1950,color='black',alpha=0.7,linewidth=3)
#plt.axvline(x=1650,color='grey',alpha=0.5,linewidth=3)
plt.text(1960,ax1.get_ylim()[1]-2,'HIST & PICT',fontsize=15)
plt.text(1910,ax1.get_ylim()[1]-2,'SPIN',fontsize=15)

ax1.xaxis.set_major_locator(MultipleLocator(50))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.tick_params(axis='both', which='minor', labelsize=12)

# For the minor ticks, use no labels; default NullFormatter.
ax1.xaxis.set_minor_locator(MultipleLocator(10))

legend=['Arctic March','Arctic September','Antarctic March','Antarctic September']
plt.legend(legend,loc='upper left',fontsize=15)
plt.savefig(out_path+"sea_ice_extent_comparison.png",dpi=300,bbox_inches = "tight")#%%capture
runs=[spinup_name, historic_name, pi_ctrl_name]

runid ='fesom'
str_id='a_ice'
fig, ax1 = plt.subplots(1, sharex=True,figsize=(22,5.8))

class MinorSymLogLocator(mticker.Locator):
    """
    Dynamically find minor tick positions based on the positions of major ticks for a symlog scaling.
    
    Attributes
    ----------
    linthresh : float
        The same linthresh value used when setting the symlog scale.
        
    """
    
    def __init__(self, linthresh):
        #super().__init__()
        self.linthresh = linthresh

    def __call__(self):
        majorlocs = self.axis.get_majorticklocs()
        # iterate through minor locs
        minorlocs = []
        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)
        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a {0} type.'.format(type(self)))

        

for exp in runs:  
    if exp == pi_ctrl_name: 
        datapath   = pi_ctrl_path+'/fesom'
        year_start = pi_ctrl_start
        year_end   = pi_ctrl_end
    elif exp == historic_name: 
        datapath   = historic_path+'/fesom'
        year_start = historic_start
        year_end   = historic_end
    elif exp == spinup_name: 
        datapath   = spinup_path+'/fesom'
        year_start = spinup_start
        year_end   = spinup_end
 
    extent_north_march = []
    extent_south_march = []
    extent_north_sep = []
    extent_south_sep = []       
        
    for y in tqdm(range(year_start, year_end)): 
        for x in range(2, 3):
            data = pf.get_data(datapath, str_id, y, mesh, records=[x])
            extent_north_march.append(pf.ice_ext(data, mesh, hemisphere="N"))
            extent_south_march.append(pf.ice_ext(data, mesh, hemisphere="S"))
        for x in range(8, 9):
            data = pf.get_data(datapath, str_id, y, mesh, records=[x])
            extent_north_sep.append(pf.ice_ext(data, mesh, hemisphere="N"))
            extent_south_sep.append(pf.ice_ext(data, mesh, hemisphere="S"))
          
    #if exp == 'SPIN':
    #    years = np.linspace(year_start-700, year_end-700,year_end-year_start+1)
    #else:
    years = np.linspace(year_start, year_end,year_end-year_start+1)

    years = years[:len(years)-1]
    extent_north_march = np.squeeze(np.asarray(extent_north_march))
    extent_south_march = np.squeeze(np.asarray(extent_south_march))
    extent_north_sep = np.squeeze(np.asarray(extent_north_sep))
    extent_south_sep = np.squeeze(np.asarray(extent_south_sep))


    if exp == pi_ctrl_name:
        linestyle=':'
        alpha=0.5
    elif exp == spinup_name:
        linestyle='-'
        alpha=0.6
    elif exp == historic_name:
        linestyle='-'
        alpha=1
        
    ax1.plot(years,extent_north_march/1000000000000,linewidth=2,color='grey',linestyle=linestyle,alpha=alpha,label='Arctic March');
    ax1.tick_params(axis='both', labelsize=17)
        
    ax1.plot(years,extent_north_sep/1000000000000,linewidth=2,color='Orange',linestyle=linestyle,alpha=alpha,label='Arctic September');
    ax1.tick_params(axis='both', labelsize=17)
    
    ax1.plot(years,extent_south_march/1000000000000,linewidth=2,color='Darkblue',linestyle=linestyle,alpha=alpha,label='Antarctic March');
    ax1.tick_params(axis='both', labelsize=17)

    ax1.plot(years,extent_south_sep/1000000000000,linewidth=2,color='black',linestyle=linestyle,alpha=alpha,label='Antarctic September');
    ax1.tick_params(axis='both', labelsize=17)
    
ax1.set_title('Sea ice extent', fontsize=17,fontweight='bold')

#fig.text(-0.04, 0.5, 'Sea ice extent [$1000 km^2$]', fontsize=13, va='center', rotation=90)
#fig.text(0.5, -0.02, 'Year', fontsize=13, ha='center', rotation=0)
ax1.set_ylabel('Sea ice extent [$1000 km^2$]', fontsize=17)
ax1.set_xlabel('Year', fontsize=17)

ax1.yaxis.grid(color='gray', linestyle='dashed')

plt.axvline(x=1950,color='black',alpha=0.7,linewidth=3)
#plt.axvline(x=1650,color='grey',alpha=0.5,linewidth=3)
plt.text(1960,ax1.get_ylim()[1]-2,'HIST & PICT',fontsize=15)
plt.text(1910,ax1.get_ylim()[1]-2,'SPIN',fontsize=15)

ax1.xaxis.set_major_locator(MultipleLocator(50))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.tick_params(axis='both', which='minor', labelsize=12)

# For the minor ticks, use no labels; default NullFormatter.
ax1.xaxis.set_minor_locator(MultipleLocator(10))

legend=['Arctic March','Arctic September','Antarctic March','Antarctic September']
plt.legend(legend,loc='upper left',fontsize=15)
plt.savefig(out_path+"sea_ice_extent_comparison.png",dpi=300,bbox_inches = "tight")
