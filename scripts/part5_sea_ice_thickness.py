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
pi_ctrl_start  = 1989
pi_ctrl_end    = 2014

#Historic
historic_path  = '/scratch/awiiccp5/hi1950d/outdata/'
historic_name  = model_version+'_historic'
historic_start = 1989
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


# # Mesh plot
# 

# In[10]:


figsize=(6,4.5)


# # Sea ice thickness

# In[16]:


variable = 'm_ice'
input_paths = [historic_path, pi_ctrl_path]
input_names = [historic_name, pi_ctrl_name]
years = range(historic_last25y_start, historic_last25y_end+1)

res=[180,180]
figsize=(6,6)
levels = [0.1,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
units = r'$^\circ$C'
columns = 2
dpi = 300
ofile = variable
region = "Global Ocean"

# Obtain input names from path if not set explicitly
if input_names is None:
    input_names = []
    for run in input_paths:
        run = os.path.join(run, '')
        input_names.append(run.split('/')[-2])
        
# Load fesom2 mesh
mesh = pf.load_mesh(meshpath, abg=abg, 
                    usepickle=True, usejoblib=False)

# Set number of columns, in case of multiple variables
def define_rowscol(input_paths, columns=len(input_paths), reduce=0):
    number_paths = len(input_paths) - reduce
#     columns = columns
    if number_paths < columns:
        ncol = number_paths
    else:
        ncol = columns
    nrows = math.ceil(number_paths / columns)
    return [nrows, ncol]

# Load model Data
data = OrderedDict()

def load_parallel(variable,path,remap_resolution,meshpath,mesh_file):
    data1 = cdo.copy(input='-setmissval,nan -setctomiss,0 -remap,r'+remap_resolution+','+meshpath+'/weights_unstr_2_r'+remap_resolution+'.nc -setgrid,'+meshpath+'/'+mesh_file+' '+str(path),returnArray=variable)
    return data1


for exp_path, exp_name  in zip(input_paths, input_names):

    datat = []
    t = []
    temporary = []
    for year in tqdm(years):
        path = exp_path+'/fesom/'+variable+'.fesom.'+str(year)+'.nc'
        temporary = dask.delayed(load_parallel)(variable,path,remap_resolution,meshpath,mesh_file)
        t.append(temporary)

    with ProgressBar():
        datat = dask.compute(t)
    data[exp_name] = np.squeeze(datat)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

data_model_mean = OrderedDict()

for exp_name in input_names:
    data_model_mean[exp_name] = data[exp_name]
    if len(np.shape(data_model_mean[exp_name])) > 2:
        data_model_mean[exp_name] = np.nanmean(data_model_mean[exp_name],axis=0)

print(np.shape(data_model_mean[exp_name]))

lon = np.arange(0, 360, 1)
lat = np.arange(-90, 90, 1)
data_model_mean[historic_name], lon = add_cyclic_point(data_model_mean[historic_name], coord=lon)
lon = np.arange(0, 360, 1)
data_model_mean[pi_ctrl_name], lon = add_cyclic_point(data_model_mean[pi_ctrl_name], coord=lon)

nrows, ncol = define_rowscol(input_paths)


figsize=(6,6)

new_cmap = truncate_colormap(cmo.cm.ice, 0.15, 1)

for seas in ['September','March']:
    if seas == 'March':
        nseas=2
    elif seas == 'September':
        nseas=8
    for hemi in ['SH','NH']:
        for exp_name in input_names:
            
            data_nonan = np.nan_to_num(data_model_mean[exp_name][nseas],0)

            fig =plt.figure(figsize=(6,6))

            if hemi == 'SH':
                levels=[0.1,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]   
                ax=plt.axes(projection=ccrs.SouthPolarStereo())
                ax.set_extent([-180,180,-55,-90], ccrs.PlateCarree())

            if hemi == 'NH':
                levels=[0.1,0.5,1,1.5,2,2.5,3,3.5,4]
                ax=plt.axes(projection=ccrs.NorthPolarStereo())
                ax.set_extent([-180,180,50,90], ccrs.PlateCarree())
            
            imf=ax.contourf(lon, lat, data_nonan, cmap=new_cmap, 
                             levels=levels, extend='both',
                             transform = ccrs.PlateCarree(),zorder=1)
            lines=ax.contour(lon, lat, data_nonan, 
                             levels=levels, colors='black', linewidths=0.5,
                             transform = ccrs.PlateCarree(),zorder=2)

            ax.set_title(exp_name+ "\n "+seas+" "+hemi+" sea ice thickness", fontsize=13,fontweight='bold')

            cb = plt.colorbar(imf, orientation='horizontal',ticks=levels, fraction=0.046, pad=0.04)
            cb.set_label(label="m", size='12')
            cb.ax.tick_params(labelsize='11')
            
            ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='lightgrey'),zorder=3)
            #ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', '50m',color='black'),zorder=4)
            #ax.add_feature(cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '110m',color='black'),zorder=4)
            #ax.rivers(resolution='50m', color='black', linewidth=1,zorder=6)

            ax.coastlines(resolution='50m', color='black', linewidth=1,zorder=6)

            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=1, color='gray', alpha=0.2, linestyle='-')
            gl.xlabels_bottom = False
            plt.tight_layout() 

            if exp_name == historic_name:
                plt.savefig(out_path+"historic_"+seas+"_"+hemi+"_sea_ice_thickness.png",dpi=300,bbox_inches='tight')
            elif exp_name == pi_ctrl_name:
                plt.savefig(out_path+"pi-control_"+seas+"_"+hemi+"_sea_ice_thickness.png",dpi=300,bbox_inches='tight')

            
# Load GIOMAS
#cmap = cmo.ice
import cmocean as cmo

#levels = np.linspace(0,100,11).astype(int)
#factor=100
new_cmap = truncate_colormap(cmo.cm.ice, 0.15, 1)
extend='both'

# Load model data
import xarray as xr
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

for exp in ['GIOMAS']:
    path =observation_path+'/GIOMAS/GIOMAS_heff_miss_time_mon.nc'
    if exp == 'GIOMAS':
        var = 'heff'
        year_start = 1990
        year_end = 2008
        
# Load model Data
data = OrderedDict()
paths = []

intermediate = []
intermediate = xr.open_mfdataset(path, combine="by_coords", engine="netcdf4", use_cftime=True)
data[var] = intermediate.compute()
data2=data[var].to_array()
x = np.asarray(data[var].lon_scaler).flatten()
y = np.asarray(data[var].lat_scaler).flatten()

#interpolate
lon = np.linspace(0,360,res[0])
lat = np.linspace(-90,90,res[1])
lon2, lat2 = np.meshgrid(lon, lat)


# interpolate data onto regular grid
sit = []
points = np.vstack((x,y)).T
for t in tqdm(range(0, np.shape(data['heff']['heff'])[0])):
    nn_interpolation = NearestNDInterpolator(points, np.nan_to_num(np.asarray(data['heff']['heff'][t,:,:]).flatten(),0))
    sit.append(nn_interpolation((lon2, lat2)))
sit=np.asarray(sit)



for seas in ['September','March']:
    if seas == 'March':
        nseas=2
    elif seas == 'September':
        nseas=8
    for hemi in ['NH','SH']:
        for key in input_names:
            data_nonan = np.nan_to_num(sit[nseas,:,:],0)
            #data_nonan = sit[nseas,:,:]

            fig =plt.figure(figsize=(6,6))

            if hemi == 'SH':
                levels=[0.1,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]   
                ax=plt.axes(projection=ccrs.SouthPolarStereo())
                ax.set_extent([-180,180,-55,-90], ccrs.PlateCarree())

            if hemi == 'NH':
                levels=[0.1,0.5,1,1.5,2,2.5,3,3.5,4]
                ax=plt.axes(projection=ccrs.NorthPolarStereo())
                ax.set_extent([-180,180,50,90], ccrs.PlateCarree())
            
            imf=ax.contourf(lon2, lat2, data_nonan, cmap=new_cmap, 
                             levels=levels, extend='both',
                             transform = ccrs.PlateCarree(),zorder=1)
            lines=ax.contour(lon2, lat2, data_nonan, 
                             levels=levels, colors='black', linewidths=0.5,
                             transform = ccrs.PlateCarree(),zorder=2)
            
            ax.set_title("GIOMAS "+seas+" "+hemi+" sea ice thickness", fontsize=13,fontweight='bold')

            cb = plt.colorbar(imf, orientation='horizontal',ticks=levels, fraction=0.046, pad=0.04)
            cb.set_label(label="m", size='12')
            cb.ax.tick_params(labelsize='11')
            
            ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='lightgrey'),zorder=3)
            ax.coastlines(resolution='50m', color='black', linewidth=1,zorder=6)

            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=1, color='gray', alpha=0.2, linestyle='-')
            gl.xlabels_bottom = False
            plt.tight_layout() 

            plt.savefig(out_path+"GIOMAS_"+seas+"_"+hemi+"_sea_ice_thickness.png",dpi=300,bbox_inches='tight')


