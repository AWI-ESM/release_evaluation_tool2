#!/usr/bin/env python
# coding: utf-8

# # Paths and config

# In[9]:

figsize=(6,4.5)

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



# # Hovmöller diagram Temperature

# In[12]:


# Load model Data
data = OrderedDict()

mesh = pf.load_mesh(meshpath)
variable='temp'
ofile = 'Hovmoeller_'+variable+'.png'

input_paths = [spinup_path+'/fesom/']
input_names = [spinup_name]
years = range(spinup_start, spinup_end+1)
#years = range(1581, 1583)

maxdepth = 10000

levels = [-1.5, 1.5, 11]
mapticks = np.arange(levels[0],levels[1],0.1)


# Load reference data
path=reference_path+'/'+variable+'.fesom.'+str(reference_years)+'.nc'
data_ref = cdo.yearmean(input='-fldmean -setctomiss,0 -remap,r'+remap_resolution+','+meshpath+'/weights_unstr_2_r'+remap_resolution+'.nc -setgrid,'+meshpath+'/'+mesh_file+' '+str(path),returnArray=variable)
data_ref = np.squeeze(data_ref)

def load_parallel(variable,path,remap_resolution,meshpath,mesh_file):
    data1 = cdo.yearmean(input='-fldmean -setctomiss,0 -remap,r'+remap_resolution+','+meshpath+'/weights_unstr_2_r'+remap_resolution+'.nc -setgrid,'+meshpath+'/'+mesh_file+' '+str(path),returnArray=variable)
    return data1


for exp_path, exp_name  in zip(input_paths, input_names):

    datat = []
    t = []
    temporary = []
    for year in tqdm(years):
        path = exp_path+'/'+variable+'.fesom.'+str(year)+'.nc'
        temporary = dask.delayed(load_parallel)(variable,path,remap_resolution,meshpath,mesh_file)
        t.append(temporary)

    with ProgressBar():
        datat = dask.compute(t)
    data[exp_name] = np.squeeze(datat)

    
# Read depths from 3D file, since mesh.zlevs is empty..
depths = pd.read_csv(meshpath+'/aux3d.out',  nrows=mesh.nlev) 

# Reshape data and expand reference climatology to length for data in preparation for Hovmöller diagram
data_ref_expand = OrderedDict()
for exp_name  in input_names:
    data[exp_name] = np.flip(np.squeeze(data[exp_name]),axis=1)
    
data_ref_expand = np.tile(data_ref,(np.shape(data[exp_name])[0],1)).T

# Flip data for contourf plot
data_diff = OrderedDict()
for exp_name in input_names:
    data_diff[exp_name]=np.flip(data[exp_name].T,axis=0)-data_ref_expand
    
# Prepare coordianates for contourf plot
X,Y = np.meshgrid(years,depths[:len(depths)-1])

# Calculate number of rows and columns for plot
def define_rowscol(input_paths, columns=len(input_paths), reduce=0):
    number_paths = len(input_paths) - reduce
#     columns = columns
    if number_paths < columns:
        ncol = number_paths
    else:
        ncol = columns
    nrows = math.ceil(number_paths / columns)
    return [nrows, ncol]

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

        
# Plot data and save it.
mapticks = np.arange(levels[0],levels[1],0.1)

nrows, ncols = define_rowscol(input_paths)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols,figsize[1]*nrows))


if isinstance(axes, np.ndarray):
    axes = axes.flatten()
else:
    axes = [axes]


i = 0
for exp_name in data_diff:
    im = axes[i].contourf(X,-Y,data_diff[exp_name],levels=mapticks, cmap=cm.PuOr_r, extend='both')
    axes[i].set_title('Global ocean temperature bias',fontweight="bold")
    axes[i].set_ylabel('Depth [m]',size=13)
    axes[i].set_xlabel('Year',size=13)
    axes[i].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axes[i].set_ylim(-maxdepth)
    axes[i].set_yscale('symlog')
    axes[i].xaxis.set_minor_locator(MultipleLocator(10))

    axes[i].yaxis.set_minor_locator(MinorSymLogLocator(-10e4))
    # turn minor ticks off
    #axes[i].yaxis.set_minor_locator(NullLocator())



    i = i+1
fig.tight_layout()

if variable == "temp":
    label='°C'
elif variable == "salt":
    label='PSU'
    
try:
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),location='bottom', label=label, shrink=0.8, aspect=30, pad=0.2)
except:
    cbar = fig.colorbar(im, ax=axes,location='bottom', label=label, shrink=0.8, aspect=25, pad=0.16)

cbar.ax.tick_params(labelsize=12) 
cbar.ax.tick_params(labelsize=12) 

if out_path+ofile is not None:
    plt.savefig(out_path+ofile, dpi=dpi,bbox_inches='tight')
    os.system(f'convert {ofile} -trim {ofile}_trimmed.png')
    os.system(f'mv {ofile}_trimmed.png {ofile}')


