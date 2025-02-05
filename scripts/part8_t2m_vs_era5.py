#Name of model release
model_version  = 'TCo1279-DART'

#Spinup
spinup_path    = '/scratch/awicm3/TCo1279-DART-1950/outdata/'
spinup_name    = model_version+'_spinup'
spinup_start   = 1955
spinup_end     = 1969

#Preindustrial Control
pi_ctrl_path   = '/scratch/awicm3/TCo1279-DART-1950/outdata/'
pi_ctrl_name   = model_version+'_pi-control'
pi_ctrl_start  = 1955
pi_ctrl_end    = 1969

#Historic
historic_path  = '/scratch/awicm3/TCo1279-DART-2000/outdata/'
historic_name  = model_version+'_historic'
historic_start = 2002
historic_end   = 2009


#Misc
reanalysis             = 'ERA5'
remap_resolution       = '5124x2560'
dpi                    = 900
historic_last25y_start = historic_start
historic_last25y_end   = historic_end

#Mesh
mesh_name      = 'DART'
grid_name      = 'TCo1279'
meshpath       = '/proj/awi/input/fesom2/dart/'
mesh_file      = 'dart_griddes_nodes.nc'
griddes_file   = 'dart_griddes_nodes.nc'
abg            = [0, 0, 0]
reference_path = '/proj/awiiccp5/climatologies/'
reference_name = 'clim'
reference_years= 1990

observation_path = '/proj/awi/'


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

# parameters cell
input_paths = [historic_path]
input_names = [historic_name]

if reanalysis=='ERA5':
    clim='ERA5'
    clim_var='t2m'
    climatology_files = ['T2M_yearmean_hr.nc']
    title='Near surface (2m) air tempereature vs. ERA5'
    climatology_path = observation_path+'/era5/netcdf/'
elif reanalysis=='NCEP2':
    clim='NCEP2'
    clim_var='air'
    climatology_files = ['air.2m.timemean.nc']
    title='Near surface (2m) air tempereature vs. NCEP2'
    climatology_path =  observation_path+'/NCEP2/'

exps=[]
for year in range(historic_last25y_start, historic_last25y_end + 1):
    # Loop through each month (1 through 12)
    for month in range(1, 13):
        # Append year and month in the format YYYYMM
        exps.append(year * 100 + month)
        
figsize=(6, 4.5)
ofile = None
res = [1440, 720]
var = ['2t']
levels = [-5.0,-3.0,-2.0,-1.0,-.6,-.2,.2,.6,1.0,2.0,3.0,5.0]
contour_outline_thickness = 0

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

# Mean Deviation weighted
def md(predictions, targets, wgts):
    output_errors = np.average((predictions - targets), axis=0, weights=wgts)
    return (output_errors).mean()

# Load reanalysis data

reanalysis_path = climatology_path+climatology_files[0]
data_reanalysis_mean = np.squeeze(cdo.timmean(input="-remapcon,r"+str(res[0])+"x"+str(res[1])+" "+str(reanalysis_path),returnArray=clim_var))

# Load model Data
def load_parallel(variable,path):
    data1 = cdo.timmean(input="-remapcon,r"+str(res[0])+"x"+str(res[1])+" "+str(path),returnArray=variable)
    return data1

data = OrderedDict()
for exp_path, exp_name  in zip(input_paths, input_names):
    data[exp_name] = {}
    for v in var:
        datat = []
        t = []
        temporary = []
        for exp in tqdm(exps):

            path = exp_path+'/oifs/atm_reduced_1m_'+v+'_1m_'+f'{exp:04d}-{exp:04d}.nc'
            temporary = dask.delayed(load_parallel)(v,path)
            t.append(temporary)

        with ProgressBar():
            datat = dask.compute(t)
        data[exp_name][v] = np.squeeze(datat)
        
data_model = OrderedDict()
data_model_mean = OrderedDict()


for exp_name in input_names:
    data_model[exp_name] = np.squeeze(data[exp_name][v]) 
    data_model_mean[exp_name] = data_model[exp_name]
    if len(np.shape(data_model_mean[exp_name])) > 2:
        data_model_mean[exp_name] = np.mean(data_model_mean[exp_name],axis=0)
  

print(np.shape(data_model_mean[exp_name]))
print(np.shape(data_reanalysis_mean))

lon = np.arange(0, 360, 0.25)
lat = np.arange(-90, 90, 0.25)
data_model_mean[exp_name], lon = add_cyclic_point(data_model_mean[exp_name], coord=lon)


lon = np.arange(0, 360, 0.25)
lat = np.arange(-90, 90, 0.25)
data_reanalysis_mean, lon = add_cyclic_point(data_reanalysis_mean, coord=lon)

print(np.shape(data_model_mean[exp_name]))
print(np.shape(data_reanalysis_mean))


coslat = np.cos(np.deg2rad(lat))
wgts = np.squeeze(np.sqrt(coslat)[..., np.newaxis])
rmsdval = sqrt(mean_squared_error(data_model_mean[exp_name],data_reanalysis_mean,sample_weight=wgts))
mdval = md(data_model_mean[exp_name],data_reanalysis_mean,wgts)




nrows, ncol = define_rowscol(input_paths)
fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=figsize, subplot_kw={'projection': ccrs.Robinson(central_longitude=-160)})
if isinstance(axes, np.ndarray):
    axes = axes.flatten()
else:
    axes = [axes]
i = 0


for key in input_names:

    axes[i]=plt.subplot(nrows,ncol,i+1,projection=ccrs.PlateCarree())
    axes[i].add_feature(cfeature.COASTLINE,zorder=3)
    
    
    imf=plt.contourf(lon, lat, data_model_mean[exp_name]-
                    data_reanalysis_mean, cmap=plt.cm.PuOr_r, 
                     levels=levels, extend='both',
                     transform=ccrs.PlateCarree(),zorder=1)
    line_colors = ['black' for l in imf.levels]
    imc=plt.contour(lon, lat, data_model_mean[exp_name]-
                    data_reanalysis_mean, colors=line_colors, 
                    levels=levels, linewidths=contour_outline_thickness,
                    transform=ccrs.PlateCarree(),zorder=1)

    axes[i].set_ylabel('K')
    axes[i].set_xlabel('Simulation Year')
    
    axes[i].set_title(title,fontweight="bold")
    plt.tight_layout() 
    gl = axes[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='-')

    gl.xlabels_bottom = False

    
    cbar_ax_abs = fig.add_axes([0.15, 0.11, 0.7, 0.05])
    cbar_ax_abs.tick_params(labelsize=12)
    cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal',ticks=levels)
    cb.set_label(label="°C", size='14')
    cb.ax.tick_params(labelsize='12')
    
    textrsmd='rmsd='+str(round(rmsdval,3))
    textbias='bias='+str(round(mdval,3))
    props = dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.5)
    axes[i].text(0.02, 0.4, textrsmd, transform=axes[i].transAxes, fontsize=13,
        verticalalignment='top', bbox=props, zorder=4)
    axes[i].text(0.02, 0.3, textbias, transform=axes[i].transAxes, fontsize=13,
        verticalalignment='top', bbox=props, zorder=4)
    
    i = i+1
    
for label in cb.ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)

    
ofile='t2m_vs_'+clim
    
if ofile is not None:
    plt.savefig(out_path+ofile, dpi=dpi,bbox_inches='tight')
    os.system(f'convert {ofile} -trim {ofile}_trimmed.png')
    os.system(f'mv {ofile}_trimmed.png {ofile}')
