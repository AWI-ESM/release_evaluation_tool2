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

#+_____________________________________________________________________________+
#|                                                                             |
#|                         *** LOAD FVSOM MESH ***                             |
#|                                                                             |
#+_____________________________________________________________________________+
# for more options look in set_inputarray.py
tool_path = os.getcwd()
inputarray=set_inputarray()
inputarray['save_fig'],inputarray['save_figpath'] = True, out_path+'/'
inputarray['mesh_id'],inputarray['mesh_dir'] = 'COREv2', meshpath
inputarray['mesh_rotate' ]==False
try:
    mesh
except NameError:
    mesh = fesom_init_mesh(inputarray)
else:
    #if mesh.id!=inputarray['mesh_id']:
    mesh = fesom_init_mesh(inputarray)
    #else:
    #    print(' --> ___FOUND {} FESOM MESH --> will use it!___________________________'.format(mesh.id))   


#%%prun -s cumulative -q -l 100 -D profile.bin
#____________________________________________________________________________________________________
# load vertical velocity data
data1           = fesom_data(inputarray) 
data1.var       = 'w'
data1.descript,data1.path = 'pict_amoc' , pi_ctrl_path+'/fesom/'
data1.year, data1.month= [historic_last25y_start,historic_last25y_end], [1,2,3,4,5,6,7,8,9,10,11,12]
data1.cmap,data1.cnumb = 'red2blue', 10
add_bolusw = False
#____________________________________________________________________________________________________
# load vertical velocity datas for big meshes using xarray
fesom_load_data3d_4bm(mesh,data1,do_output=False)

#____________________________________________________________________________________________________
# add GM bolus velocity
if add_bolusw:
    data_bw     = cp.deepcopy(data1)
    data_bw.var ='bolus_w'
    fesom_load_data_horiz(mesh,data_bw,do_output=False)
    data1.value = data1.value+data_bw.value
    del data_bw

#%%prun -s cumulative -q -l 100 -D profile.bin #write out profile file usable with snakeviz profile.bin
#____________________________________________________________________________________________________
# select XMOC
which_moc = 'amoc'

#____________________________________________________________________________________________________
# calc XMOC
moc1,lat,bottom,elemidx  = calc_xmoc(mesh,data1,which_moc=which_moc,out_elemidx=True)
#moc1,lat,bottom  = calc_xmoc(mesh,data1,which_moc=which_moc,in_elemidx=elemidx)

# moc1,lat,bottom,elemidx  = calc_xmoc(mesh,data1,which_moc=amoc,out_elemidx=True)
# --> writes out elem index to use for AMOC or PMOC can be directly read into next calucation of amoc 
# --> moc2,lat,bottom      =calc_xmoc(mesh,data2,which_moc=amoc,in_elemidx=elemidx)
#____________________________________________________________________________________________________
# plot XMOC
# fig,ax=plot_xmoc(lat,mesh.zlev,moc1,bottom=bottom,which_moc=which_moc,str_descript=data1.descript,str_time=data1.str_time)
fig,ax=plot_xmoc(lat,mesh.zlev,moc1,bottom=bottom,which_moc=which_moc,str_descript=data1.descript,str_time=data1.str_time,crange=[],cnumb=15)

#____________________________________________________________________________________________________
# save XMOC
if inputarray['save_fig']==True:
    print(' --> save figure: png')
    str_times= data1.str_time.replace(' ','').replace(':','') 
    sdname, sfname = inputarray['save_figpath'], 'plot_'+data1.descript+'_'+which_moc+'_'+str_times+'.png'
    plt.savefig(sdname+sfname, format='png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, frameon=True)
    
#%%prun -s cumulative -q -l 100 -D profile.bin
#____________________________________________________________________________________________________
# load vertical velocity data
data1           = fesom_data(inputarray) 
data1.var       = 'w'
data1.descript,data1.path = 'hist_amoc' , historic_path+'/fesom/'
data1.year, data1.month= [historic_last25y_start,historic_last25y_end], [1,2,3,4,5,6,7,8,9,10,11,12]
data1.cmap,data1.cnumb = 'red2blue', 10
add_bolusw = False
#____________________________________________________________________________________________________
# load vertical velocity datas for big meshes using xarray
fesom_load_data3d_4bm(mesh,data1,do_output=False)

#____________________________________________________________________________________________________
# add GM bolus velocity
if add_bolusw:
    data_bw     = cp.deepcopy(data1)
    data_bw.var ='bolus_w'
    fesom_load_data_horiz(mesh,data_bw,do_output=False)
    data1.value = data1.value+data_bw.value
    del data_bw

#%%prun -s cumulative -q -l 100 -D profile.bin #write out profile file usable with snakeviz profile.bin
#____________________________________________________________________________________________________
# select XMOC
which_moc = 'amoc'

#____________________________________________________________________________________________________
# calc XMOC
moc1,lat,bottom,elemidx  = calc_xmoc(mesh,data1,which_moc=which_moc,out_elemidx=True)
#moc1,lat,bottom  = calc_xmoc(mesh,data1,which_moc=which_moc,in_elemidx=elemidx)

# moc1,lat,bottom,elemidx  = calc_xmoc(mesh,data1,which_moc=amoc,out_elemidx=True)
# --> writes out elem index to use for AMOC or PMOC can be directly read into next calucation of amoc 
# --> moc2,lat,bottom      =calc_xmoc(mesh,data2,which_moc=amoc,in_elemidx=elemidx)
#____________________________________________________________________________________________________
# plot XMOC
# fig,ax=plot_xmoc(lat,mesh.zlev,moc1,bottom=bottom,which_moc=which_moc,str_descript=data1.descript,str_time=data1.str_time)
fig,ax=plot_xmoc(lat,mesh.zlev,moc1,bottom=bottom,which_moc=which_moc,str_descript=data1.descript,str_time=data1.str_time,crange=[],cnumb=15)

#____________________________________________________________________________________________________
# save XMOC
if inputarray['save_fig']==True:
    print(' --> save figure: png')
    str_times= data1.str_time.replace(' ','').replace(':','') 
    sdname, sfname = inputarray['save_figpath'], 'plot_'+data1.descript+'_'+which_moc+'_'+str_times+'.png'
    plt.savefig(sdname+sfname, format='png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, frameon=True)
    
which_moc='amoc'
which_lat=[26.0, 40.0,'max']
#____________________________________________________________________________________________________
# load vertical velocity data
data1      = fesom_data(inputarray) 
data1.var = 'w'
data1.descript,data1.path = 'spin_amoc_timeseries' , spinup_path+'/fesom/'
data1.year, data1.month= [spinup_start,spinup_end], [1,2,3,4,5,6,7,8,9,10,11,12]

#____________________________________________________________________________________________________
# be sure mesh has the right focus 
if which_moc=='amoc2' or which_moc=='amoc':
    # for calculation of amoc mesh focus must be on 0 degree longitude
    if mesh.focus!=0:
       mesh.focus=0
       mesh.fesom_grid_rot_r2g(str_mode='focus')
elif which_moc=='pmoc':
     if mesh.focus!=180:
        mesh.focus=180
        mesh.fesom_grid_rot_r2g(str_mode='focus')
#____________________________________________________________________________________________________
# calc MOC time-series
count=0
print(' --> CALC YEAR:')
datayr = cp.deepcopy(data1)
moc_t = np.zeros((data1.year[1]-data1.year[0]+1,len(which_lat)))
time  = np.zeros((data1.year[1]-data1.year[0]+1,))
for year in range(data1.year[0],data1.year[1]+1):
    #_______________________________________________________________________________________________
    print('|'+str(year),end='')
    if np.mod(count+1,15)==0: print('|')
        
    #_______________________________________________________________________________________________
    # load vertical velocity data --> calculates yearly means
    datayr.year     = [year,year]
    # fesom_load_data_horiz(mesh,datayr,do_output=False)
    fesom_load_data3d_4bm(mesh,datayr,do_output=False) 
    
    #_______________________________________________________________________________________________
    # calculate AMOC vor every year
    if count==0:
        moc_prof,lat,bottom,elemidx  = calc_xmoc(mesh,datayr,which_moc=which_moc,out_elemidx=True,do_output=False)
    else:
        moc_prof,lat,bottom          = calc_xmoc(mesh,datayr,which_moc=which_moc,in_elemidx=elemidx,do_output=False)
    #_______________________________________________________________________________________________
    # look for maximum value below 500m at certain latitude or between latitudinal range 'max' 
    # (looks between 30°N and 45°N)
    moc_d=moc_prof[np.where(mesh.zlev<=-500)[0],:]
    count_lat=0
    for lati in which_lat:
        if lati=='max':
            moc_l= moc_d[:,np.where((lat>=30) & (lat<=45))[0]]
        else:
            moc_l= moc_d[:,np.where(lat>=lati)[0][0]]
        moc_t[count,count_lat]=moc_l.max()
        count_lat=count_lat+1
    time[count]=year    
    count=count+1
    
#____________________________________________________________________________________________________
# plot MOC time-series
fig,ax=plot_xmoc_tseries(time,moc_t,which_lat,which_moc)    
if inputarray['save_fig']==True:
    print(' --> save figure: png')
    str_times= data1.str_time.replace(' ','').replace(':','') 
    sdname, sfname = inputarray['save_figpath'], 'plot_'+data1.descript+'_'+which_moc+'_timeseries.png'
    fig.savefig(sdname+sfname, format='png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, frameon=True)
    
#___________________

which_moc='amoc'
which_lat=[26.0, 40.0,'max']
#____________________________________________________________________________________________________
# load vertical velocity data
data1      = fesom_data(inputarray) 
data1.var = 'w'
data1.descript,data1.path = 'hist_amoc_timeseries' , historic_path+'/fesom/'
data1.year, data1.month= [historic_start,historic_end], [1,2,3,4,5,6,7,8,9,10,11,12]

#____________________________________________________________________________________________________
# be sure mesh has the right focus 
if which_moc=='amoc2' or which_moc=='amoc':
    # for calculation of amoc mesh focus must be on 0 degree longitude
    if mesh.focus!=0:
       mesh.focus=0
       mesh.fesom_grid_rot_r2g(str_mode='focus')
elif which_moc=='pmoc':
     if mesh.focus!=180:
        mesh.focus=180
        mesh.fesom_grid_rot_r2g(str_mode='focus')
#____________________________________________________________________________________________________
# calc MOC time-series
count=0
print(' --> CALC YEAR:')
datayr = cp.deepcopy(data1)
moc_t = np.zeros((data1.year[1]-data1.year[0]+1,len(which_lat)))
time  = np.zeros((data1.year[1]-data1.year[0]+1,))
for year in range(data1.year[0],data1.year[1]+1):
    #_______________________________________________________________________________________________
    print('|'+str(year),end='')
    if np.mod(count+1,15)==0: print('|')
        
    #_______________________________________________________________________________________________
    # load vertical velocity data --> calculates yearly means
    datayr.year     = [year,year]
    # fesom_load_data_horiz(mesh,datayr,do_output=False)
    fesom_load_data3d_4bm(mesh,datayr,do_output=False) 
    
    #_______________________________________________________________________________________________
    # calculate AMOC vor every year
    if count==0:
        moc_prof,lat,bottom,elemidx  = calc_xmoc(mesh,datayr,which_moc=which_moc,out_elemidx=True,do_output=False)
    else:
        moc_prof,lat,bottom          = calc_xmoc(mesh,datayr,which_moc=which_moc,in_elemidx=elemidx,do_output=False)
    #_______________________________________________________________________________________________
    # look for maximum value below 500m at certain latitude or between latitudinal range 'max' 
    # (looks between 30°N and 45°N)
    moc_d=moc_prof[np.where(mesh.zlev<=-500)[0],:]
    count_lat=0
    for lati in which_lat:
        if lati=='max':
            moc_l= moc_d[:,np.where((lat>=30) & (lat<=45))[0]]
        else:
            moc_l= moc_d[:,np.where(lat>=lati)[0][0]]
        moc_t[count,count_lat]=moc_l.max()
        count_lat=count_lat+1
    time[count]=year    
    count=count+1
    
#____________________________________________________________________________________________________
# plot MOC time-series
fig,ax=plot_xmoc_tseries(time,moc_t,which_lat,which_moc)    
if inputarray['save_fig']==True:
    print(' --> save figure: png')
    str_times= data1.str_time.replace(' ','').replace(':','') 
    sdname, sfname = inputarray['save_figpath'], 'plot_'+data1.descript+'_'+which_moc+'_timeseries.png'
    fig.savefig(sdname+sfname, format='png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, frameon=True)
    
#___________________
