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

# parameters cell
variables = ['MLD2', 'a_ice']
input_paths = [historic_path+'/fesom/']
input_names = [historic_name]
years = range(historic_last25y_start, historic_last25y_end+1)

figsize=(6,4.5)
levels = [0, 3000, 11]
units = r'$^\circ$C'
columns = 2
ofile = 'mld.png'
region = "Global Ocean"

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

for variable in variables:

    # Load model Data
    data = OrderedDict()

    def load_parallel(variable,path,remap_resolution,meshpath,mesh_file):
        data1 = cdo.yseasmean(input='-setmissval,nan -setctomiss,0 -remap,r'+remap_resolution+','+meshpath+'/weights_unstr_2_r'+remap_resolution+'.nc -setgrid,'+meshpath+'/'+mesh_file+' '+str(path),returnArray=variable)
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


    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap


    if variable == 'a_ice':
        data[exp_name][0][np.isnan(data[exp_name][0])] = 0
        data[exp_name][3][np.isnan(data[exp_name][3])] = 0

    data_model_mean = OrderedDict()
    figsize=(6,6)

    if variable == 'a_ice':
        levels = np.linspace(0,100,11).astype(int)
        levels = [1,10,20,30,40,50,60,70,80,90,100]
        factor=100
        #new_cmap = truncate_colormap(plt.cm.PuOr_r, 0.08, 0.5)
        new_cmap = truncate_colormap(cmo.cm.ice, 0.15, 1)
        extend='min'

    else:
        levels = [0, 0.2, 0.5,  1,  2, 2.5,  3, 3.5, 4]
        factor=-0.001
        new_cmap = truncate_colormap(plt.cm.PuOr, 0.5, 1)
        extend='both'



    for exp_name in input_names:
        data_model_mean[exp_name] = data[exp_name]
        if len(np.shape(data_model_mean[exp_name])) > 2:
            data_model_mean[exp_name] = np.nanmean(data_model_mean[exp_name],axis=0)

    print(np.shape(data_model_mean[exp_name]))

    lon = np.arange(0, 360, 1)
    lat = np.arange(-90, 90, 1)
    data_model_mean[historic_name], lon = add_cyclic_point(data_model_mean[historic_name], coord=lon)

    nrows, ncol = define_rowscol(input_paths)
    fig =plt.figure(figsize=(6,6))


    ax=plt.axes(projection=ccrs.SouthPolarStereo())
    ax.add_feature(cfeature.COASTLINE,zorder=3)
    ax.set_extent([-180,180,-55,-90], ccrs.PlateCarree())

    imf=ax.contourf(lon, lat, factor*data_model_mean[exp_name][3], cmap=new_cmap, 
                     levels=levels, extend=extend,
                     transform = ccrs.PlateCarree(),zorder=1)
    lines=ax.contour(lon, lat, factor*data_model_mean[exp_name][3], 
                     levels=levels, colors='black', linewidths=0.2,
                     transform = ccrs.PlateCarree(),zorder=1)

    ax.set_ylabel('K')
    if variable == 'a_ice':
        ax.set_title("Sea ice concentration",fontweight="bold")
        cb = plt.colorbar(imf, orientation='horizontal',ticks=levels, fraction=0.046, pad=0.04)
        cb.set_label(label="Concentration [%]", size='12')
        cb.ax.tick_params(labelsize='11')
        #cb.ax.set_xticklabels(levels, rotation=90)
    else:
        ax.set_title("Mixed layer depth "+ variable,fontweight="bold")
        cb = plt.colorbar(imf, orientation='horizontal',ticks=levels, fraction=0.046, pad=0.04)
        cb.set_label(label="Depth [km]", size='12')
        cb.ax.tick_params(labelsize='11')
        #cb.ax.set_xticklabels(levels, rotation=90)


    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='lightgrey'))

    ax.add_patch(mpatches.Rectangle(xy=[-40, -72], width=10, height=5,
                                    edgecolor='red',
                                    facecolor='none',
                                    linewidth=1.5,
                                    alpha=1,
                                    transform=ccrs.Geodetic())
                 )

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=1, color='gray', alpha=0.2, linestyle='-')
    gl.xlabels_bottom = False



    #for label in cb.ax.xaxis.get_ticklabels()[::2]:
    #    label.set_visible(False)
    plt.tight_layout() 


    ofile=variable+'_SH'

    if ofile is not None:
        plt.savefig(out_path+ofile, dpi=dpi,bbox_inches='tight')
        os.system(f'convert {ofile} -trim {ofile}_trimmed.png')
        os.system(f'mv {ofile}_trimmed.png {ofile}')


    data_model_mean = OrderedDict()
    figsize=(6,6)

    if variable == 'a_ice':
        levels = np.linspace(0,100,11).astype(int)
        levels = [1,10,20,30,40,50,60,70,80,90,100]
        factor=100
        new_cmap = truncate_colormap(cmo.cm.ice, 0.15, 1)
        extend='min'

    else:
        levels = [0, 0.2, 0.5,  1,  2, 2.5,  3, 3.5, 4]
        factor=-0.001
        new_cmap = truncate_colormap(plt.cm.PuOr, 0.5, 1)
        extend='both'



    for exp_name in input_names:
        data_model_mean[exp_name] = data[exp_name]
        if len(np.shape(data_model_mean[exp_name])) > 2:
            data_model_mean[exp_name] = np.nanmean(data_model_mean[exp_name],axis=0)

    print(np.shape(data_model_mean[exp_name]))

    lon = np.arange(0, 360, 1)
    lat = np.arange(-90, 90, 1)
    data_model_mean[exp_name], lon = add_cyclic_point(data_model_mean[exp_name], coord=lon)


    nrows, ncol = define_rowscol(input_paths)
    fig =plt.figure(figsize=(6,6))


    ax=plt.axes(projection=ccrs.NorthPolarStereo())
    ax.add_feature(cfeature.COASTLINE,zorder=3)
    ax.set_extent([-180,180,50,90], ccrs.PlateCarree())

    imf=ax.contourf(lon, lat, factor*data_model_mean[exp_name][3], cmap=new_cmap, 
                     levels=levels, extend=extend,
                     transform = ccrs.PlateCarree(),zorder=1)
    lines=ax.contour(lon, lat, factor*data_model_mean[exp_name][3], 
                     levels=levels, colors='black', linewidths=0.2,
                     transform = ccrs.PlateCarree(),zorder=1)


    ax.set_ylabel('K')
    if variable == 'a_ice':
        ax.set_title("Sea ice concentration",fontweight="bold")
        cb = plt.colorbar(imf, orientation='horizontal',ticks=levels, fraction=0.046, pad=0.04)
        cb.set_label(label="Concentration [%]", size='12')
        cb.ax.tick_params(labelsize='11')
        #cb.ax.set_xticklabels(levels, rotation=90)
    else:
        ax.set_title("Mixed layer depth "+ variable,fontweight="bold")
        cb = plt.colorbar(imf, orientation='horizontal',ticks=levels, fraction=0.046, pad=0.04)
        cb.set_label(label="Depth [km]", size='12')
        cb.ax.tick_params(labelsize='11')
        #cb.ax.set_xticklabels(levels, rotation=90)


    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='lightgrey'))


    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=1, color='gray', alpha=0.2, linestyle='-')
    gl.xlabels_bottom = False



    #for label in cb.ax.xaxis.get_ticklabels()[::2]:
    #    label.set_visible(False)
    plt.tight_layout() 


    ofile=variable+'_NH'

    if ofile is not None:
        plt.savefig(out_path+ofile, dpi=dpi,bbox_inches='tight')
        os.system(f'convert {ofile} -trim {ofile}_trimmed.png')
        os.system(f'mv {ofile}_trimmed.png {ofile}')
