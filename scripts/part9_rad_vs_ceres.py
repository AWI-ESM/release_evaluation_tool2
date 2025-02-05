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
# parameters cell
input_paths = [historic_path]
input_names = [historic_name]
exps = list(range(historic_last25y_start, historic_last25y_end+1))

climatology_files = ['CERES_EBAF_Ed4.1_Subset_CLIM01-CLIM12.nc']
climatology_path =  observation_path+'/CERES/'
res = [360, 180]
accumulation_period = 21600 # For fluxes that are accumulated -> output frequency of OpenIFS in seconds e.g. 21600s = 3h

figsize=(6, 4.5)

def define_rowscol(input_paths, columns=len(input_paths), reduce=0):
    number_paths = len(input_paths) - reduce
#     columns = columns
    if number_paths < columns:
        ncol = number_paths
    else:
        ncol = columns
    nrows = math.ceil(number_paths / columns)
    return [nrows, ncol]

# Mean Deviation
def md(predictions, targets):
    return (predictions - targets).mean()

# Mean Deviation weighted
def md(predictions, targets, wgts):
    output_errors = np.average((predictions - targets), axis=0, weights=wgts)
    return (output_errors).mean()

for variable in ['str', 'ssr', 'ssrd']:
    if variable == 'str':
        variable_clim = 'sfc_net_lw_all_clim'
        title='Surface net long-wave radiation vs. CERES-EBAF'
    elif variable == 'ssr':
        variable_clim = 'sfc_net_sw_all_clim'
        title='Surface net short-wave radiation vs. CERES-EBAF'
    elif variable == 'ssrd':
        variable_clim = 'sfc_sw_down_all_clim'
        title='Surface downward short-wave radiation vs. CERES-EBAF'

    mapticks = [-50,-30,-20,-10,-6,-2,2,6,10,20,30,50]
    contour_outline_thickness = 0
    


    # Load CERES satobs data (https://doi.org/10.1175/JCLI-D-17-0208.1)

    CERES_path = climatology_path+climatology_files[0]
    CERES_Dataset = Dataset(CERES_path)
    CERES_Data = OrderedDict()
    CERES_CRF = CERES_Dataset.variables[variable_clim][:]


    # Load model Data
    def load_parallel(variable,path):
        data1 = cdo.timmean(input="-remapcon,r"+str(res[0])+"x"+str(res[1])+" "+str(path),returnArray=variable)/accumulation_period
        return data1

    data = OrderedDict()
    for exp_path, exp_name  in zip(input_paths, input_names):
        data[exp_name] = {}
        datat = []
        t = []
        temporary = []
        for exp in tqdm(exps):

            path = exp_path+'/oifs/atm_remapped_1m_'+variable+'_1m_'+f'{exp:04d}-{exp:04d}.nc'
            temporary = dask.delayed(load_parallel)(variable,path)
            t.append(temporary)

        with ProgressBar():
            datat = dask.compute(t)
        data[exp_name][variable] = np.squeeze(datat)
            
    crf_sw_model = OrderedDict()
    crf_sw_model_mean = OrderedDict()


    for exp_name in input_names:
        crf_sw_model[exp_name] = np.squeeze(data[exp_name][variable]) 
        crf_sw_model_mean[exp_name] = np.mean(crf_sw_model[exp_name],axis=0)
        if len(np.shape(crf_sw_model_mean[exp_name])) > 2:
            crf_sw_model_mean[exp_name] = np.mean(crf_sw_model_mean[exp_name],axis=0)
    crf_sw_satobs_mean = np.mean(CERES_CRF,axis=0)

    print(np.shape(crf_sw_model_mean[exp_name]))
    print(np.shape(crf_sw_satobs_mean))

    lon = np.arange(0, 360, 1)
    lat = np.arange(-90, 90, 1)
    crf_sw_model_mean[exp_name], lon = add_cyclic_point(crf_sw_model_mean[exp_name], coord=lon)

    lon = np.arange(0, 360, 1)
    lat = np.arange(-90, 90, 1)
    crf_sw_satobs_mean, lon = add_cyclic_point(crf_sw_satobs_mean, coord=lon)

    print(np.shape(crf_sw_model_mean[exp_name]))
    print(np.shape(crf_sw_satobs_mean))

    coslat = np.cos(np.deg2rad(lat))
    wgts = np.squeeze(np.sqrt(coslat)[..., np.newaxis])
    rmsdval = sqrt(mean_squared_error(crf_sw_model_mean[exp_name],crf_sw_satobs_mean,sample_weight=wgts))
    mdval = md(crf_sw_model_mean[exp_name],crf_sw_satobs_mean,wgts)

    nrows, ncol = define_rowscol(input_paths)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=figsize)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    i = 0


    for key in input_names:

        axes[i]=plt.subplot(nrows,ncol,i+1,projection=ccrs.PlateCarree())
        axes[i].add_feature(cfeature.COASTLINE,zorder=3)


        imf=plt.contourf(lon, lat, crf_sw_model_mean[exp_name]-
                         crf_sw_satobs_mean, cmap='PuOr_r', 
                         levels=mapticks, extend='both',
                         transform=ccrs.PlateCarree(),zorder=1)
        line_colors = ['black' for l in imf.levels]
        imc=plt.contour(lon, lat, crf_sw_model_mean[exp_name]-
                        crf_sw_satobs_mean, colors=line_colors, 
                        levels=mapticks, linewidths=contour_outline_thickness,
                        transform=ccrs.PlateCarree(),zorder=1)

        axes[i].set_ylabel('W/m²')
        axes[i].set_xlabel('Simulation Year')

        axes[i].set_title(title,fontweight="bold")
        plt.tight_layout() 

        cbar_ax_abs = fig.add_axes([0.15, 0.11, 0.7, 0.05])
        cbar_ax_abs.tick_params(labelsize=12)
        cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal',ticks=mapticks)
        cb.set_label(label="W/m²", size='14')
        cb.ax.tick_params(labelsize='12')

        gl = axes[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.2, linestyle='-')

        gl.xlabels_bottom = False

        textrsmd='rmsd='+str(round(rmsdval,3))
        textbias='bias='+str(round(mdval,3))
        props = dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.5)
        axes[i].text(0.02, 0.35, textrsmd, transform=axes[i].transAxes, fontsize=13,
            verticalalignment='top', bbox=props, zorder=4)
        axes[i].text(0.02, 0.25, textbias, transform=axes[i].transAxes, fontsize=13,
            verticalalignment='top', bbox=props, zorder=4)
        i = i+1


    ofile=variable+'_vs_ceres'

    if ofile is not None:
        plt.savefig(out_path+ofile, dpi=dpi,bbox_inches='tight')
        os.system(f'convert {ofile} -trim {ofile}_trimmed.png')
        os.system(f'mv {ofile}_trimmed.png {ofile}')


# parameters cell
input_paths = [historic_path]
input_names = [historic_name]
exps = list(range(historic_last25y_start, historic_last25y_end+1))


climatology_files = ['clt_MODIS_yearmean.nc']
climatology_path =  observation_path+'/MODIS/'

figsize=(6, 4.5)
dpi = 300
ofile = None
res = [180, 91]
variable = ['tcc']
variable_clim = 'clt'
title='Cloud area fraction vs. MODIS'
mapticks = [-50,-30,-20,-10,-6,-2,2,6,10,20,30,50]

contour_outline_thickness = 0
levels = np.linspace(-5, 5, 21)


def define_rowscol(input_paths, columns=len(input_paths), reduce=0):
    number_paths = len(input_paths) - reduce
#     columns = columns
    if number_paths < columns:
        ncol = number_paths
    else:
        ncol = columns
    nrows = math.ceil(number_paths / columns)
    return [nrows, ncol]

# Calculate Root Mean Square Deviation (RMSD)
def rmsd(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Mean Deviation weighted
def md(predictions, targets, wgts):
    output_errors = np.average((predictions - targets), axis=0, weights=wgts)
    return (output_errors).mean()

mdval2 = md(crf_sw_model_mean[exp_name],crf_sw_satobs_mean,wgts)
# Load CERES satobs data (https://doi.org/10.1175/JCLI-D-17-0208.1)

CERES_path = climatology_path+climatology_files[0]
CERES_Dataset = Dataset(CERES_path)
CERES_Data = OrderedDict()
CERES_CRF = CERES_Dataset.variables[variable_clim][:]

# Load model Data
def load_parallel(variable,path):
    data1 = cdo.timmean(input="-remapcon,r"+str(res[0])+"x"+str(res[1])+" "+str(path),returnArray=variable)*100
    return data1

data = OrderedDict()
for exp_path, exp_name  in zip(input_paths, input_names):
    data[exp_name] = {}
    for v in variable:
        datat = []
        t = []
        temporary = []
        for exp in tqdm(exps):

            path = exp_path+'/oifs/atm_remapped_1m_'+v+'_1m_'+f'{exp:04d}-{exp:04d}.nc'
            temporary = dask.delayed(load_parallel)(v,path)
            t.append(temporary)

        with ProgressBar():
            datat = dask.compute(t)
        data[exp_name][v] = np.squeeze(datat)

crf_sw_model = OrderedDict()
crf_sw_model_mean = OrderedDict()


for exp_name in input_names:
    crf_sw_model[exp_name] = np.squeeze(data[exp_name]['tcc']) 
    crf_sw_model_mean[exp_name] = np.mean(crf_sw_model[exp_name],axis=0)
    if len(np.shape(crf_sw_model_mean[exp_name])) > 2:
        crf_sw_model_mean[exp_name] = np.mean(crf_sw_model_mean[exp_name],axis=0)
crf_sw_satobs_mean = np.mean(CERES_CRF,axis=0)

print(np.shape(crf_sw_model_mean[exp_name]))
print(np.shape(crf_sw_satobs_mean))

lon = np.arange(0, 360, 2)
lat = np.arange(-90, 90, 180/91)
crf_sw_model_mean[exp_name], lon = add_cyclic_point(crf_sw_model_mean[exp_name], coord=lon)

lon = np.arange(0, 360, 2)
lat = np.arange(-90, 90, 180/91)
crf_sw_satobs_mean, lon = add_cyclic_point(crf_sw_satobs_mean, coord=lon)

print(np.shape(crf_sw_model_mean[exp_name]))
print(np.shape(crf_sw_satobs_mean))


coslat = np.cos(np.deg2rad(lat))
wgts = np.squeeze(np.sqrt(coslat)[..., np.newaxis])
rmsdval = sqrt(mean_squared_error(crf_sw_model_mean[exp_name],crf_sw_satobs_mean,sample_weight=wgts))
mdval = md(crf_sw_model_mean[exp_name],crf_sw_satobs_mean,wgts)



nrows, ncol = define_rowscol(input_paths)
fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=figsize)
if isinstance(axes, np.ndarray):
    axes = axes.flatten()
else:
    axes = [axes]
i = 0


for key in input_names:

    axes[i]=plt.subplot(nrows,ncol,i+1,projection=ccrs.PlateCarree())
    axes[i].add_feature(cfeature.COASTLINE,zorder=3)
    
    
    imf=plt.contourf(lon, lat, crf_sw_model_mean[exp_name]-
                     crf_sw_satobs_mean, cmap='PuOr_r', 
                     levels=mapticks, extend='both',
                     transform=ccrs.PlateCarree(),zorder=1)
    line_colors = ['black' for l in imf.levels]
    imc=plt.contour(lon, lat, crf_sw_model_mean[exp_name]-
                    crf_sw_satobs_mean, colors=line_colors, 
                    levels=mapticks, linewidths=contour_outline_thickness,
                    transform=ccrs.PlateCarree(),zorder=1)

    axes[i].set_ylabel('W/m²')
    axes[i].set_xlabel('Simulation Year')
    
    axes[i].set_title(title,fontweight="bold")
    plt.tight_layout() 
    gl = axes[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='-')

    gl.xlabels_bottom = False
    
    textrsmd='rmsd='+str(round(rmsdval,3))
    textbias='bias='+str(round(mdval,3))
    props = dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.5)
    axes[i].text(0.02, 0.35, textrsmd, transform=axes[i].transAxes, fontsize=13,
        verticalalignment='top', bbox=props, zorder=4)
    axes[i].text(0.02, 0.25, textbias, transform=axes[i].transAxes, fontsize=13,
        verticalalignment='top', bbox=props, zorder=4)
    
    i = i+1
    
    cbar_ax_abs = fig.add_axes([0.15, 0.11, 0.7, 0.05])
    cbar_ax_abs.tick_params(labelsize=12)
    cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal',ticks=mapticks)
    cb.set_label(label="%", size='14')
    cb.ax.tick_params(labelsize='12')

ofile=variable[0]+'_vs_MODIS'
    
if ofile is not None:
    plt.savefig(out_path+ofile, dpi=dpi,bbox_inches='tight')
    os.system(f'convert {ofile} -trim {ofile}_trimmed.png')
    os.system(f'mv {ofile}_trimmed.png {ofile}')
