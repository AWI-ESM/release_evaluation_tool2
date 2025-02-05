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


# # Mesh plot
# 

# In[10]:


levels = np.linspace(5, 30, 11)
figsize=(6,4.5)

mesh = pf.load_mesh(meshpath)
data = xr.open_dataset(meshpath+'/fesom.mesh.diag.nc')
nod_area = data['nod_area'][0,:].values
nod_area = (np.sqrt(nod_area/np.pi)/1e3)*2

data=nod_area,
cmap=cmo.cm.thermal_r,
influence=80000,
box=[-180, 180, -89, 90],
res=[360, 180],
interp="nn",
mapproj="pc",
ptype="cf",
units=None,
titles=None,
distances_path=None,
inds_path=None,
qhull_path=None,
basepath=None,
interpolated_data=None,
lonreg=None,
latreg=None,
no_pi_mask=False,

box=[-180, 180, -89, 90]
res=[360, 180]

if not isinstance(data, list):
    data = [data]
if titles:
    if not isinstance(titles, list):
        titles = [titles]
    if len(titles) != len(data):
        raise ValueError(
            "The number of titles do not match the number of data fields, please adjust titles (or put to None)"
        )


radius_of_influence = influence

left, right, down, up = box
lonNumber, latNumber = res

lonreg = np.linspace(left, right, lonNumber)
latreg = np.linspace(down, up, latNumber)

lonreg2, latreg2 = np.meshgrid(lonreg, latreg)


interpolated = pf.interpolate_for_plot(
    data[0],
    mesh,
    lonreg2,
    latreg2,
    interp=interp[0],
    distances_path=distances_path[0],
    inds_path=inds_path[0],
    radius_of_influence=radius_of_influence[0],
    basepath=basepath[0],
    qhull_path=qhull_path[0],
    )

data_model_mean = OrderedDict()

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)
if isinstance(axes, np.ndarray):
    axes = axes.flatten()
else:
    axes = [axes]
i = 0


axes[i]=plt.subplot(1,1,i+1,projection=ccrs.PlateCarree())
axes[i].add_feature(cfeature.COASTLINE,zorder=3)

imf=plt.contourf(lonreg, latreg, np.squeeze(interpolated), cmap=cmo.cm.thermal_r, 
                 levels=levels, extend='both',
                 transform=ccrs.PlateCarree(),zorder=1)

axes[i].set_ylabel('K')
axes[i].set_title(mesh_name+" mesh resolution",fontweight="bold")
axes[i].add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='lightgrey'))

plt.tight_layout() 

gl = axes[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
              linewidth=1, color='gray', alpha=0.2, linestyle='-')
gl.xlabels_bottom = False

cbar_ax_abs = fig.add_axes([0.15, 0.11, 0.7, 0.05])
cbar_ax_abs.tick_params(labelsize=12)
cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal',ticks=levels)
cb.set_label(label="Horizontal Resolution [km]", size='12')
cb.ax.tick_params(labelsize='11')
axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=30)
    
ofile=out_path+'mesh_resolution'
    
if ofile is not None:
    plt.savefig(ofile, dpi=dpi,bbox_inches='tight')
    os.system(f'convert {ofile} -trim {ofile}_trimmed.png')
    os.system(f'mv {ofile}_trimmed.png {ofile}')


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

#plt.plot(years,smooth(surface,len(surface)),color='darkblue')
#plt.plot(years,smooth(toa,len(toa)),color='orange')
#plt.plot(years,smooth((toa-surface),len(toa-surface)),color='grey')

axes.set_title('Radiative balance',fontweight="bold")

plt.axhline(y=0, color='black', linestyle='-')
plt.ylabel('W/m²',size='13')
plt.xlabel('Year',size='13')

#plt.axvline(x=1650,color='grey',alpha=0.6)

plt.axhline(x=0,color='grey',alpha=0.6)

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


# # Perfomance indicies

# In[13]:


tool_path = os.getcwd()
cmd = './preprocessing_examples/noncmore_preprocess_AWI-CM3-XIOS_6h.sh '+historic_path+' '+tool_path+'/input/ '+model_version+' '+str(historic_start+39)+' '+str(historic_end)+' '+meshpath+'/'+griddes_file
print(cmd)
#os.system(cmd)

#Verbose?
verbose='true'

#Choose ERA5 or NCEP2. This switch also selects the eval/???? subfolders, so do not mix and match as this 
#would lead to incorrect results.

#Define paths
obs_path='obs/'
model_path=tool_path+'/input/'
save=out_path
out_path='output/'
eval_path='eval/'+reanalysis+'/'
time = '198912-201411'


#Define the name and evaluated variables for your model run

#Default CMIP6 evaluation set
cmip6 = {
    'ACCESS-CM2':   [ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so'],
    'AWI-CM1-MR':   [           'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so'],
    'BCC-SM2-MR':   [ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so'],
    'CAMS':         [ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so'],
    'CanESM5':      [ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so'],
    'CAS-ESM2-0':   [ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos',                         ],
    'CESM2':        [ 'siconc', 'tas', 'clt', 'pr', 'rlut',               'ua', 'zg', 'zos', 'tos', 'mlotst'                ],
    'CIESM':        [           'tas', 'clt', 'pr', 'rlut',               'ua', 'zg', 'zos', 'tos',           'thetao', 'so'],
    'CMCC-CM2-SR5': [ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so'],
    'CNRM-CM6-1-HR':[ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst'                ],
    'E3SM-1-1':     [ 'siconc', 'tas', 'clt', 'pr', 'rlut',               'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so'],
    'EC-Earth3':    [ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so'],
    'FGOALS-f3-L':  [ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so'],
    'FIO-ESM-2-0':  [ 'siconc', 'tas', 'clt', 'pr', 'rlut',               'ua', 'zg', 'zos', 'tos',           'thetao', 'so'],
    'GISS-E2-1-G':  [ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so'],
    'HadGEM3MM':    [ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so'],
    'ICON-ESM-LR':  [ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so'],
    'IITM-ESM':     [           'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg',        'tos',                         ],
    'INM5':         [ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg',                         'thetao', 'so'],
    'IPSL-CM6A-LR': [ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so'],
    'KIOST-ESM':    [ 'siconc', 'tas', 'clt',       'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst',               ],
    'MCMUA1':       [           'tas',        'pr', 'rlut', 'uas', 'vas', 'ua', 'zg',        'tos',           'thetao', 'so'],
    'MIROC6':       [ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos',                         ],
    'MPI-ESM1-2-HR':[ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so'],
    'MRI-ESM2-0':   [ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas',             'zos', 'tos', 'mlotst', 'thetao', 'so'],
    'NESM3':        [ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so'],   
    'NOAA-GFDL':    [ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos',           'thetao', 'so'],
    'NorESM2-MM':   [ 'siconc', 'tas', 'clt', 'pr', 'rlut',               'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so'],
    'SNU':          [ 'siconc', 'tas', 'clt', 'pr', 'rlut',               'ua', 'zg', 'zos', 'tos',           'thetao', 'so'],
    'TaiESM1':      [ 'siconc', 'tas', 'clt', 'pr', 'rlut',               'ua', 'zg', 'zos', 'tos',           'thetao', 'so'],
}

awi_cm3_0_ref= {
    'AWI-CM3-REF':[ 'siconc', 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so']
}

awi_cm3_new_release= {
    model_version:[ 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so']
}
awi_cm3_new_release2= {
    model_version:[ 'tas', 'clt', 'pr', 'rlut', 'uas', 'vas', 'ua', 'zg', 'zos', 'tos', 'mlotst', 'thetao', 'so']
}

models = awi_cm3_new_release2
eval_models = cmip6

#Select for each variable which vertical levels shall be taken into account
var_depths ={    
        #'siconc':['surface'],
        'tas':['surface'],
        'clt':['surface'],
        'pr':['surface'],
        'rlut':['surface'],
        'uas':['surface'],
        'vas':['surface'],
        'ua':['300hPa'],
        'zg':['500hPa'],
        'zos':['surface'],
        'tos':['surface'],
        'mlotst':['surface'],
        'thetao':['10m','100m','1000m'],
        'so':['10m','100m','1000m'],
}


#Define which observational dataset biases are computed against for each variable
obs = { 
    #'siconc':'OSISAF',
    'tas':reanalysis,
    'clt':'MODIS',
    'pr':'GPCP',
    'rlut':'CERES',
    'uas':reanalysis,
    'vas':reanalysis,
    'ua':reanalysis,
    'zg':reanalysis,
    'zos':'NESDIS',
    'tos':'HadISST2',
    'mlotst':'C-GLORSv7',
    'thetao':'EN4',
    'so':'EN4',
}

#Select seasons
seasons = ['MAM', 'JJA', 'SON', 'DJF']

#Define regions
regions={
    #'glob' : {
    #'lat_min':-90,
    #'lat_max':90,
    #'lon_min':0,
    #'lon_max':360,
    #'plot_color':'none',},
         
    'arctic' : {
    'lat_min':60,
    'lat_max':90,
    'lon_min':0,
    'lon_max':360,
    'plot_color':'red',},
         
    'northmid' : {
    'lat_min':30,
    'lat_max':60,
    'lon_min':0,
    'lon_max':360,
    'plot_color':'lightgrey',},
         
    'tropics' : {
    'lat_min':-30,
    'lat_max':30,
    'lon_min':0,
    'lon_max':360,
    'plot_color':'green',},
         
    #'innertropics' : {
    #'lat_min':-15,
    #'lat_max':15,
    #'lon_min':0,
    #'lon_max':360,
    #'plot_color':'green',},
        
    'nino34' : {
    'lat_min':-5,
    'lat_max':5,
    'lon_min':190,
    'lon_max':240,
    'plot_color':'yellow',},
         
    'southmid' : {
    'lat_min':-60,
    'lat_max':-30,
    'lon_min':0,
    'lon_max':360,
    'plot_color':'pink',},
         
    'antarctic' : {
    'lat_min':-90,
    'lat_max':-60,
    'lon_min':0,
    'lon_max':360,
    'plot_color':'blue',},
          
}

# This stores all regions for which the evaluation data has been generated
all_regions=[ 'glob', 'arctic', 'northmid', 'tropics', 'innertropics', 'nino34', 'southmid', 'antarctic']

# Visulatize regions

if verbose == 'true':
    projection = ccrs.PlateCarree()

    # Plot the leading EOF expressed as correlation in the Pacific domain.
    plt.figure(figsize=(12,9))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

    ax.add_feature(cfeature.LAND, color='lightgrey')
    ax.add_feature(cfeature.COASTLINE)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    plt.title('Regions', fontsize=13,fontweight="bold")

    ax.set_extent([0, -1, 90, -90])
    for region in regions:
        if region == 'glob':
            continue 
        else:
            lon_min=regions[region]['lon_min']
            lon_max=regions[region]['lon_max']
            lat_min=regions[region]['lat_min']
            lat_max=regions[region]['lat_max']
            ax.add_patch(mpatches.Rectangle(xy=[lon_min-1, lat_min], width=lon_max-lon_min, height=lat_max-lat_min,
                                            facecolor=regions[region]['plot_color'],
                                            alpha=0.5,
                                            edgecolor=regions[region]['plot_color'],
                                            lw='2',
                                            transform=ccrs.PlateCarree())
                         )
            plt.text(lon_min-177,lat_max-7,region,weight='bold')

    ax.tick_params(labelsize=13)

print('Loading obs data')

ds_obs = OrderedDict()

for var,depths in zip(obs,var_depths):
    for depth in np.arange(0,len(var_depths[depths])):
        for seas in seasons:
            if verbose == 'true':
                print('loading '+obs_path+var+'_'+obs[var]+'_'+var_depths[depths][depth]+'_'+seas+'.nc')

            intermediate = xr.open_dataset(obs_path+var+'_'+obs[var]+'_'+var_depths[depths][depth]+'_'+seas+'.nc')
            ds_obs[var,var_depths[depths][depth],seas] = intermediate.compute()
            try:
                ds_obs[var,var_depths[var][depth],seas]=ds_obs[var,var_depths[var][depth],seas].drop('time_bnds')
            except:
                pass
            try:
                ds_obs[var,var_depths[var][depth],seas]=ds_obs[var,var_depths[var][depth],seas].drop('time_bnds_2')
            except:
                pass
            try:
                ds_obs[var,var_depths[var][depth],seas]=ds_obs[var,var_depths[var][depth],seas].drop('depth')
            except:
                pass

print('Loading model data')

ds_model = OrderedDict()

for model in tqdm(models):
    for var in models[model]:
        for depth in np.arange(0,len(var_depths[var])):
            for seas in seasons:
                if verbose == 'true':
                    print('loading '+model_path+var+'_'+model+'_'+time+'_'+var_depths[var][depth]+'_'+seas+'.nc')
                intermediate = xr.open_mfdataset(model_path+var+'_'+model+'_'+time+'_'+var_depths[var][depth]+'_'+seas+'.nc')
                intermediate = intermediate.squeeze(drop=True)
                ds_model[var,var_depths[var][depth],seas,model] = intermediate.compute()
                try:
                    ds_model[var,var_depths[var][depth],seas,model]=ds_model[var,var_depths[var][depth],seas,model].drop('time_bnds')
                except:
                    pass
                try:
                    ds_model[var,var_depths[var][depth],seas,model]=ds_model[var,var_depths[var][depth],seas,model].drop('depth')
                except:
                    pass


print('Calculating absolute error and field mean of abs error')

# Returns equvalent to cdo fldmean ()
def fldmean(ds):
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"
    ds_weighted = ds.weighted(weights)
    return ds.mean(("lon", "lat"))


abs_error = OrderedDict()
mean_error = OrderedDict()

for model in tqdm(models):
    for var in models[model]:
        for depth in np.arange(0,len(var_depths[var])):
            for region in regions:
                filter1 = ds_model[var,var_depths[var][depth],seas,model].lat>regions[region]['lat_min']
                filter2 = ds_model[var,var_depths[var][depth],seas,model].lat<regions[region]['lat_max']
                filter3 = ds_model[var,var_depths[var][depth],seas,model].lon>regions[region]['lon_min']
                filter4 = ds_model[var,var_depths[var][depth],seas,model].lon<regions[region]['lon_max']

                for seas in seasons:
                    abs_error[var,var_depths[var][depth],seas,model,region]=np.sqrt((ds_model[var,var_depths[var][depth],seas,model].where(filter1 & filter2 & filter3 & filter4)-
                                                       ds_obs[var,var_depths[var][depth],seas]).where(filter1 & filter2 & filter3 & filter4)*
                                                      (ds_model[var,var_depths[var][depth],seas,model].where(filter1 & filter2 & filter3 & filter4)-
                                                       ds_obs[var,var_depths[var][depth],seas].where(filter1 & filter2 & filter3 & filter4)))
                    mean_error[var,var_depths[var][depth],seas,model,region] = fldmean(abs_error[var,var_depths[var][depth],seas,model,region])

                    
print('Writing field mean of errors into csv files')

for model in tqdm(models):
    with open(out_path+'abs/'+model+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Variable','Region','Level','Season','AbsMeanError'])
        for var in models[model]:
            for region in regions:
                for depth in np.arange(0,len(var_depths[var])):
                    for seas in seasons:
                        writer.writerow([var,region,var_depths[var][depth],seas,np.squeeze(mean_error[var,var_depths[var][depth],seas,model,region].to_array(var).values[0])])

                        
                        
print('Reading precalculated cmip6 field mean of errors from csv files')

max_depth=0
for var in var_depths:
    if len(var_depths[var]) > max_depth:
        max_depth = len(var_depths[var])

collect = np.empty([len(eval_models),len(obs),len(regions),max_depth,len(seasons)])*np.nan
i=0
for eval_model in tqdm(eval_models):
    df = pd.read_csv(eval_path+eval_model+'.csv', delimiter=' ')
    values = df['AbsMeanError']
    regions_csv = df['Region']
    var_csv = df['Variable']
    j=0
    r=0
    for var in obs:
        k=0
        a=(df['Variable']==var).to_list()
        if verbose == 'true':
            if any(a): # Check if variable appears in list. If not, skip it.
                print('reading: ',eval_model,var)
            else:
                print('filling: ',eval_model,var)
        for region in regions:
            l=0
            for depth in np.arange(0,len(var_depths[var])):
                m=0
                for seas in seasons:
                    if any(a): # Check if variable appears in csv. If not, skip it.
                        if regions_csv[r] not in regions: # Check if region from csv part of the analysis. Else advance
                            while True:
                                r+=1
                                if regions_csv[r] in regions:
                                    break
                        collect[i,j,k,l,m]=values[r]
                        r+=1
                    m+=1
                l+=1
            k+=1
        j+=1
    i+=1
# Ignoring non useful warning:
# /tmp/ipykernel_19478/363568120.py:37: RuntimeWarning: Mean of empty slice
#  ensmean=np.nanmean(collect,axis=0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    ensmean=np.nanmean(collect,axis=0)

    
    
print('Placing sums of error into easier to inspect dictionary')

eval_error_mean = OrderedDict()

j=0
for var in tqdm(obs):
    k=0
    for region in regions:
        l=0
        for depth in np.arange(0,len(var_depths[var])):
            m=0
            for seas in seasons:
                eval_error_mean[var,region,var_depths[var][depth],seas]=ensmean[j,k,l,m]
                m+=1
            l+=1
        k+=1
    j+=1
    

print('Calculating ratio of current model error to evaluation model error')

error_fraction = OrderedDict()

sum=0
for model in tqdm(models):
    for var in models[model]:
        for region in regions:
            for depth in np.arange(0,len(var_depths[var])):
                for seas in seasons:
                    error_fraction[var,var_depths[var][depth],seas,model,region] = mean_error[var,var_depths[var][depth],seas,model,region] / eval_error_mean[var,region,var_depths[var][depth],seas]


print('Writing ratio of field mean of errors into csv files and sum up error fractions for cmpi score')

cmpi = OrderedDict()

for model in tqdm(models):
    sum=0
    iter=0
    with open(out_path+'frac/'+model+'_fraction.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Variable','Region','Level','Season','FracMeanError'])
        for var in models[model]:
            for depth in np.arange(0,len(var_depths[var])):
                for region in regions:
                    for seas in seasons:
                        writer.writerow([var,region,var_depths[var][depth],seas,np.squeeze(error_fraction[var,var_depths[var][depth],seas,model,region].to_array(var).values[0])])
                        if ma.isnan(np.squeeze(error_fraction[var,var_depths[var][depth],seas,model,region].to_array(var).values[0])):
                            pass
                        else:
                            sum+=np.squeeze(error_fraction[var,var_depths[var][depth],seas,model,region].to_array(var).values[0])
                            iter+=1
        cmpi[model]=np.squeeze(sum)/iter
        writer.writerow(['CMPI','global','yearly',cmpi[model]])
        
        
print('Reading precalculated evaluation field means of errors from csv files and plotting heatmap(s)')

max_depth=0
for var in var_depths:
    if len(var_depths[var]) > max_depth:
        max_depth = len(var_depths[var])

plt.rcParams.update({'figure.max_open_warning': 0})
collect_frac_non = OrderedDict()
for model in tqdm(models):
    df = pd.read_csv(out_path+'frac/'+model+'_fraction.csv', delimiter=' ')
    values = df['FracMeanError'] #you can also use df['column_name']
    r=0
    for var in obs:
        a=(df['Variable']==var).to_list()
        if verbose == 'true':
            if any(a): # Check if variable appears in list. If not, skip it.
                print('reading: ',model,var)
            else:
                print('filling: ',model,var)
        for depth in np.arange(0,len(var_depths[var])):
            for region in regions:
                for seas in seasons:
                    if any(a):
                        collect_frac_non[var+' '+region,var_depths[var][depth]+' '+seas]=values[r]
                        r+=1
                    else:
                        collect_frac_non[var+' '+region,var_depths[var][depth]+' '+seas]=np.nan


    seasons_plot = [' MAM', ' JJA', ' SON', ' DJF'] #adding spaces in front
    a=seasons_plot*len(regions)
    b=np.repeat(list(regions.keys()),len(seasons_plot))
    coord=[n+str(m) for m,n in zip(a,b)]
    
    index_obs=[]
    for var in obs:
        for depth in np.arange(0,len(var_depths[var])):
            if var_depths[var][depth] == 'surface':
                levelname=''
            else:
                levelname=var_depths[var][depth]+' '
            if var == 'zos' or var == 'tos':
                levelname='st. dev. '
            index_obs.append(levelname+var)
    if verbose == 'true':
        print(model,'number of values: ',len(list(collect_frac_non.values())),'; shape:',len(index_obs),'x',len(regions)*len(seasons))
    collect_frac_reshaped = np.array(list(collect_frac_non.values()) ).reshape(len(index_obs),len(regions)*len(seasons)) # transform to 2D
    collect_frac_dataframe = pd.DataFrame(data=collect_frac_reshaped, index=index_obs, columns=coord)

    fig, ax = plt.subplots(figsize=((len(regions)*len(seasons))/1.5,len(index_obs)/1.5))
    fig.patch.set_facecolor('white')
    plt.rcParams['axes.facecolor'] = 'white'
    ax = sns.heatmap(collect_frac_dataframe, vmin=0.5, vmax=1.5,center=1,annot=True,fmt='.2f',cmap="PiYG_r",cbar=False,linewidths=1)
    plt.xticks(rotation=90,fontsize=14)
    plt.yticks(rotation=0, ha='right',fontsize=14)
    plt.title(model+' CMPI: '+str(round(cmpi[model],3)), fontsize=18)
    
    plt.savefig(save+'/CMPI.png',dpi=dpi,bbox_inches='tight')
    i+=1

out_path=save


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


# # Sea ice concentration timeseries

# In[ ]:


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
plt.savefig(out_path+"sea_ice_extent_comparison.png",dpi=300,bbox_inches = "tight")


# # Mixed layer depth and sea ice concentration maps

# In[ ]:


# parameters cell
variables = ['MLD2', 'a_ice']
input_paths = [historic_path+'/fesom/']
input_names = [historic_name]
years = range(historic_last25y_start, historic_last25y_end+1)

figsize=(6,4.5)
levels = [0, 3000, 11]
units = r'$^\circ$C'
columns = 2
ofile = variable
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


# # Weddell sea salinity profile

# In[ ]:


# parameters cell
variable = 'salt'
input_paths = [historic_path+'/fesom/']
input_names = [historic_name]
years = range(historic_last25y_start, historic_last25y_end+1)

figsize=(7,4.3)
levels = [-0.15, 0.15, 11]
maxdepth = 10000
units = r'$^\circ$C'
columns = 2
ofile='Weddell_'+variable+'_profile'


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

        


# Load model Data
data = OrderedDict()

def load_parallel(variable,path,remap_resolution,meshpath,mesh_file):
    print('-fldmean -sellonlatbox,-55,0,-78,-60 -setmissval,nan -setctomiss,0 -remap,r'+remap_resolution+','+meshpath+'/weights_unstr_2_r'+remap_resolution+'.nc -setgrid,'+meshpath+'/'+mesh_file+' '+str(path))

    data1 = cdo.yseasmean(input='-fldmean -sellonlatbox,-40,-30,-72,-67 -remap,r'+remap_resolution+','+meshpath+'/weights_unstr_2_r'+remap_resolution+'.nc -setgrid,'+meshpath+'/'+mesh_file+' -setmissval,nan -setctomiss,0  '+str(path),returnArray=variable)
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
# Load reference data
path=reference_path+'/'+variable+'.fesom.'+str(reference_years)+'.nc'
data_ref = cdo.yseasmean(input='-fldmean -sellonlatbox,-40,-30,-72,-67 -remap,r'+remap_resolution+','+meshpath+'/weights_unstr_2_r'+remap_resolution+'.nc -setgrid,'+meshpath+'/'+mesh_file+' -setmissval,nan -setctomiss,0  '+str(path),returnArray=variable)
data_ref = np.squeeze(data_ref)

data_ref_expand = np.tile(data_ref,(np.shape(data[exp_name][:,1,:])[0],1)).T

# Flip data for contourf plot
data_diff = OrderedDict()
data_abs = OrderedDict()
data_abs_ref = OrderedDict()
data_abs_hov = OrderedDict()

for exp_name in input_names:
    data_diff[exp_name]=np.flip(data[exp_name][:,1,:].T,axis=0)-data_ref_expand
    data_abs_hov[exp_name]=np.flip(data[exp_name][:,1,:].T,axis=0)

    data_abs[exp_name]=np.mean(data[exp_name][:,1,:].T,axis=1)
    data_abs_ref[exp_name]=np.mean(data_ref_expand,axis=1)

depths_plot=np.squeeze(depths)[:47]

fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=([4.5,4.5]))

plt.plot(data_abs[exp_name],depths_plot,color='darkblue',linewidth=2,alpha=0.8)
plt.plot(data_abs_ref[exp_name],depths_plot,color='black',linewidth=2)
plt.legend(['PICT','PHC3'],fontsize=14)
axes.set_yscale('symlog')
axes.tick_params(axis='both', which='major', labelsize=14)
axes.set_title("Weddell salinity profile",fontweight="bold", fontsize=14)

axes.yaxis.set_minor_locator(MinorSymLogLocator(-10e4))
axes.set_ylabel('Depth [m]',size=14)
plt.xlabel('Salinity [PSU]',size=14)

ofile=variable+'_Weddell_abs_profile'

if ofile is not None:
    plt.savefig(out_path+ofile, dpi=dpi,bbox_inches='tight')
    os.system(f'convert {ofile} -trim {ofile}_trimmed.png')
    os.system(f'mv {ofile}_trimmed.png {ofile}')


# # T2M vs. ERA5 or NCEP2 bias map

# In[ ]:


# parameters cell
input_paths = [historic_path]
input_names = [historic_name]

if reanalysis=='ERA5':
    clim='ERA5'
    clim_var='t2m'
    climatology_files = ['T2M_yearmean.nc']
    title='Near surface (2m) air tempereature vs. ERA5'
    climatology_path = observation_path+'/era5/netcdf/'
elif reanalysis=='NCEP2':
    clim='NCEP2'
    clim_var='air'
    climatology_files = ['air.2m.timemean.nc']
    title='Near surface (2m) air tempereature vs. NCEP2'
    climatology_path =  observation_path+'/NCEP2/'


exps = list(range(historic_last25y_start, historic_last25y_end+1))

figsize=(6, 4.5)
dpi = 300
ofile = None
res = [192, 94]
var = ['2t']
levels = [-8.0,-5.0,-3.0,-2.0,-1.0,-.6,-.2,.2,.6,1.0,2.0,3.0,5.0,8.0]
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

            path = exp_path+'/oifs/atm_remapped_1m_'+v+'_1m_'+f'{exp:04d}-{exp:04d}.nc'
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

lon = np.arange(0, 360, 1.875)
lat = np.arange(-90, 90, 1.875)
data_model_mean[exp_name], lon = add_cyclic_point(data_model_mean[exp_name], coord=lon)


lon = np.arange(0, 360, 1.875)
lat = np.arange(-90, 90, 180/94)
data_reanalysis_mean, lon = add_cyclic_point(data_reanalysis_mean, coord=lon)

print(np.shape(data_model_mean[exp_name]))
print(np.shape(data_reanalysis_mean))


coslat = np.cos(np.deg2rad(lat))
wgts = np.squeeze(np.sqrt(coslat)[..., np.newaxis])
rmsdval = sqrt(mean_squared_error(data_model_mean[exp_name],data_reanalysis_mean,sample_weight=wgts))
mdval = md(data_model_mean[exp_name],data_reanalysis_mean,wgts)




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


# In[ ]:


# parameters cell
input_paths = [historic_path]
input_names = [historic_name]
exps = list(range(historic_last25y_start, historic_last25y_end+1))

climatology_files = ['precip.mon.mean_timmean.nc']
climatology_path =  observation_path+'/gpcp/'
figsize=(6, 4.5)
res = [180, 90]
var = ['cp', 'lsp']
variable_clim = 'pr'
levels = np.linspace(-5, 5, 11)
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

# Calculate Root Mean Square Deviation (RMSD)
def rmsd(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Mean Deviation weighted
def md(predictions, targets, wgts):
    output_errors = np.average((predictions - targets), axis=0, weights=wgts)
    return (output_errors).mean()

# Load GPCP reanalysis data

GPCP_path = climatology_path+climatology_files[0]
GPCP_data = cdo.yearmean(input="-remapcon,r"+str(res[0])+"x"+str(res[1])+" "+str(GPCP_path),returnArray=variable_clim)*86400


# Load model Data
def load_parallel(variable,path):
    data1 = cdo.timmean(input="-remapcon,r"+str(res[0])+"x"+str(res[1])+" "+str(path),returnArray=variable)*3600
    return data1

data = OrderedDict()
for exp_path, exp_name  in zip(input_paths, input_names):
    data[exp_name] = {}
    for v in var:
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
        
data_model_mean = np.mean(data[historic_name]['cp'] + \
                          data[historic_name]['lsp'],axis=0)
        
levels = [-4,-3,-2,-1,-.5,-0.2,0.2,.5,1,2,3,4]
data_reanalysis_mean = np.mean(GPCP_data,axis=0)

print(np.shape(data_model_mean))
print(np.shape(data_reanalysis_mean))

lon = np.arange(0, 360, 2)
lat = np.arange(-90, 90, 2)
data_model_mean, lon = add_cyclic_point(data_model_mean, coord=lon)

lon = np.arange(0, 360, 2)
lat = np.arange(-90, 90, 2)
data_reanalysis_mean, lon = add_cyclic_point(data_reanalysis_mean, coord=lon)

print(np.shape(data_model_mean))
print(np.shape(data_reanalysis_mean))

coslat = np.cos(np.deg2rad(lat))
wgts = np.squeeze(np.sqrt(coslat)[..., np.newaxis])
rmsdval = sqrt(mean_squared_error(data_model_mean,data_reanalysis_mean,sample_weight=wgts))
mdval = md(data_model_mean,data_reanalysis_mean,wgts)


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
    
    
    imf=plt.contourf(lon, lat, data_model_mean-
                    data_reanalysis_mean, cmap=plt.cm.PuOr, 
                     levels=levels, extend='both',
                     transform=ccrs.PlateCarree(),zorder=1)
    line_colors = ['black' for l in imf.levels]
    imc=plt.contour(lon, lat, data_model_mean-
                    data_reanalysis_mean, colors=line_colors, 
                    levels=levels, linewidths=contour_outline_thickness,
                    transform=ccrs.PlateCarree(),zorder=1)

    axes[i].set_xlabel('Simulation Year')
    
    axes[i].set_title("Precipitation vs. GPCP",fontweight="bold")
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
    cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal',ticks=levels)
    cb.set_label(label="mm/day", size='14')
    cb.ax.tick_params(labelsize='12')

    
for label in cb.ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)

    
ofile='precip_vs_GPCP'
    
if ofile is not None:
    plt.savefig(out_path+ofile, dpi=dpi,bbox_inches='tight')
    os.system(f'convert {ofile} -trim {ofile}_trimmed.png')
    os.system(f'mv {ofile}_trimmed.png {ofile}')


# In[ ]:


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


# In[ ]:


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


# # Zonal plots

# In[ ]:


# parameters cell
input_paths = [historic_path]
input_names = [historic_name]
exps = list(range(historic_last25y_start, historic_last25y_end+1))
variables=['u','t']
res=[320, 160]

clim=reanalysis
figsize=(6, 4)
contour_outline_thickness = .2

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
# Calculate Root Mean Square Deviation (RMSD)
def rmsd(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
# Mean Deviation
def md(predictions, targets):
    return (predictions - targets).mean()

class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh, nints=10):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically. nints gives the number of
        intervals that will be bounded by the minor ticks.
        """
        self.linthresh = linthresh
        self.nintervals = nints

    def __call__(self):
        # Return the locations of the ticks
        majorlocs = self.axis.get_majorticklocs()

        if len(majorlocs) == 1:
            return self.raise_if_exceeds(np.array([]))

        # add temporary major tick locs at either end of the current range
        # to fill in minor tick gaps
        dmlower = majorlocs[1] - majorlocs[0]    # major tick difference at lower end
        dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper end

        # add temporary major tick location at the lower end
        if majorlocs[0] != 0. and ((majorlocs[0] != self.linthresh and dmlower > self.linthresh) or (dmlower == self.linthresh and majorlocs[0] < 0)):
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]*10.)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]-self.linthresh)

        # add temporary major tick location at the upper end
        if majorlocs[-1] != 0. and ((np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh) or (dmupper == self.linthresh and majorlocs[-1] > 0)):
            majorlocs = np.append(majorlocs, majorlocs[-1]*10.)
        else:
            majorlocs = np.append(majorlocs, majorlocs[-1]+self.linthresh)

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in xrange(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = self.nintervals
            else:
                ndivs = self.nintervals - 1.

            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                          '%s type.' % type(self))
        
        
for variable in variables:
    if variable=='t':
        clim_var='T'
        climatology_file = 'ERA5_T.nc'
        title='Zonal mean temperate bias vs. ERA5'
        climatology_path =  observation_path+'/era5/netcdf/'
        levels = np.linspace(-8, 8, 17)
        levels2=np.linspace(190,300,12)
        labels='°C'
    if variable=='u':
        clim_var='U'
        climatology_file = 'ERA5_U.nc'
        title='Zonal mean zonal wind bias vs. ERA5'
        climatology_path =  observation_path+'/era5/netcdf/'
        levels = np.linspace(-8, 8, 17)
        levels2=np.linspace(-10,30,6)
        labels='m/s'

    # Load NCEP2 reanalysis data
    ERA5_CRF = cdo.timmean(input=str(climatology_path)+'/'+str(climatology_file),returnArray=clim_var)

    # Load model Data
    def load_parallel(variable,path):
        data1 = cdo.timmean(input="-zonmean -remapcon,r"+str(res[0])+"x"+str(res[1])+" "+str(path),returnArray=variable)
        return data1

    data = OrderedDict()
    for exp_path, exp_name  in zip(input_paths, input_names):
        data[exp_name] = {}
        datat = []
        t = []
        temporary = []
        for exp in tqdm(exps):

            path = exp_path+'/oifs/atm_remapped_1m_pl_'+variable+'_1m_pl_'+f'{exp:04d}-{exp:04d}.nc'
            temporary = dask.delayed(load_parallel)(variable,path)
            t.append(temporary)

        with ProgressBar():
            datat = dask.compute(t)
        data[exp_name][variable] = np.squeeze(datat)
        
    data_model = OrderedDict()
    data_model_mean = OrderedDict()


    for exp_name in input_names:
        data_model[exp_name] = np.mean(data[exp_name][variable],axis=0) 
        data_model_mean[exp_name] = data_model[exp_name]
    data_reanalysis_mean = np.squeeze(np.fliplr(ERA5_CRF))

    print(np.shape(data_model_mean[exp_name]))
    print(np.shape(data_reanalysis_mean))

    rmsdval = rmsd(data_model_mean[exp_name],data_reanalysis_mean)
    mdval = md(data_model_mean[exp_name],data_reanalysis_mean)


    nrows, ncol = define_rowscol(input_paths)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=figsize)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    i = 0

    x = [100,92.5,85,70,60,50,40,30,25,20,15,10,7,5,3,2,1,0.5,0.1]
    lat = np.arange(-90, 90, 1.125)

    for key in input_names:

        axes[i]=plt.subplot(nrows,ncol,i+1)
        imf=plt.contourf(lat, x, data_model_mean[exp_name]-
                        data_reanalysis_mean, cmap=plt.cm.PuOr_r, 
                         levels=levels, extend='both',
                         zorder=1)
        line_colors = ['black' for l in imf.levels]
        imc=plt.contour(lat, x, data_model_mean[exp_name]-
                        data_reanalysis_mean, colors=line_colors, 
                        levels=levels, linewidths=contour_outline_thickness,
                        zorder=2)
        ima=plt.contour(lat, x, data_model_mean[exp_name], colors=line_colors, 
                        levels=levels2, linewidths=.5,
                        zorder=3)
        plt.clabel(ima, inline=1, fontsize=8, fmt='%2.0f')
        axes[i].set_ylabel('Pressure [hPa]',fontsize=12)
        axes[i].set_xlabel('Latitude [°]',fontsize=12)

        axes[i].set_title(title,fontweight="bold")
        axes[i].set(xlim=[min(lat), max(lat)], ylim=[min(x), max(x)])
        #axes[i].invert_xaxis()
        axes[i].invert_yaxis()
        axes[i].set_yscale('symlog')


        plt.tight_layout() 

        i = i+1

        cbar_ax_abs = fig.add_axes([0.15, -0.05, 0.8, 0.05])
        cbar_ax_abs.tick_params(labelsize=12)
        cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal',ticks=levels)
        cb.set_label(label=labels, size='14')
        cb.ax.tick_params(labelsize='12')
        #plt.text(5, 168, r'rmsd='+str(round(rmsdval,3)))
        #plt.text(-7.5, 168, r'bias='+str(round(mdval,3)))

    for label in cb.ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)


    ofile=variable[0]+'zonal_mean_vs_ERA5'

    if ofile is not None:
        plt.savefig(out_path+ofile, dpi=dpi,bbox_inches='tight')
        os.system(f'convert {ofile} -trim {ofile}_trimmed.png')
        os.system(f'mv {ofile}_trimmed.png {ofile}')


# # QBO

# In[ ]:


# parameters cell
input_paths = [historic_path]
input_names = [historic_name]
exps = list(range(historic_last25y_start, historic_last25y_end+1))
variables=['u','t']
res=[320, 160]

clim='ERA5'

clim_var='U'
title='AWI-CM3 equatorial zonal wind evolution'
climatology_path =  observation_path+'/era5/netcdf/'
variable = 'u'
levels = np.linspace(-20, 20, 11)
levels2=np.linspace(-10,30,12)
labels='m/s'

figsize=(6, 3)

contour_outline_thickness = .2

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

# Calculate Root Mean Square Deviation (RMSD)
def rmsd(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Mean Deviation
def md(predictions, targets):
    return (predictions - targets).mean()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import Locator
class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh, nints=10):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically. nints gives the number of
        intervals that will be bounded by the minor ticks.
        """
        self.linthresh = linthresh
        self.nintervals = nints

    def __call__(self):
        # Return the locations of the ticks
        majorlocs = self.axis.get_majorticklocs()

        if len(majorlocs) == 1:
            return self.raise_if_exceeds(np.array([]))

        # add temporary major tick locs at either end of the current range
        # to fill in minor tick gaps
        dmlower = majorlocs[1] - majorlocs[0]    # major tick difference at lower end
        dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper end

        # add temporary major tick location at the lower end
        if majorlocs[0] != 0. and ((majorlocs[0] != self.linthresh and dmlower > self.linthresh) or (dmlower == self.linthresh and majorlocs[0] < 0)):
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]*10.)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]-self.linthresh)

        # add temporary major tick location at the upper end
        if majorlocs[-1] != 0. and ((np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh) or (dmupper == self.linthresh and majorlocs[-1] > 0)):
            majorlocs = np.append(majorlocs, majorlocs[-1]*10.)
        else:
            majorlocs = np.append(majorlocs, majorlocs[-1]+self.linthresh)

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in xrange(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = self.nintervals
            else:
                ndivs = self.nintervals - 1.

            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                          '%s type.' % type(self))

        
# Load model Data
def load_parallel(variable,path):
    data1 = cdo.zonmean(input="-mermean -sellonlatbox,0,360,-10,10 "+str(path),returnArray=variable)
    return data1

data = OrderedDict()
for exp_path, exp_name  in zip(input_paths, input_names):
    data[exp_name] = {}
    for variable in variable:
        datat = []
        t = []
        temporary = []
        for exp in tqdm(exps):

            path = exp_path+'/oifs/atm_remapped_1m_pl_'+variable+'_1m_pl_'+f'{exp:04d}-{exp:04d}.nc'
            temporary = dask.delayed(load_parallel)(variable,path)
            t.append(temporary)

        with ProgressBar():
            datat = dask.compute(t)
        data[exp_name][variable] = np.squeeze(datat)
        
data_model = OrderedDict()
data_model_mean = OrderedDict()


for exp_name in input_names:
    data_model[exp_name] = np.squeeze(data[exp_name][variable]) 
    data_model_mean[exp_name] = data[exp_name][variable].reshape(np.shape(data[exp_name][variable])[0]*np.shape(data[exp_name][variable])[1],np.shape(data[exp_name][variable])[2])

print(np.shape(data_model_mean[exp_name]))


nrows, ncol = define_rowscol(input_paths)
fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=figsize)
if isinstance(axes, np.ndarray):
    axes = axes.flatten()
else:
    axes = [axes]
i = 0

x = [100,92.5,85,70,60,50,40,30,25,20,15,10,7,5,3,2,1,0.5,0.1]
x=np.asarray(x)
time = np.arange(historic_last25y_start, historic_last25y_end+1, 0.0834)


for key in input_names:

    axes[i]=plt.subplot(nrows,ncol,i+1)
    imf=plt.contourf(time,x, data_model_mean[exp_name].T, cmap=plt.cm.PuOr_r, 
                     levels=levels, extend='both',
                     zorder=1)
    line_colors = ['black' for l in imf.levels]
    imc=plt.contour(time,x, data_model_mean[exp_name].T, colors=line_colors, 
                    levels=levels, linewidths=contour_outline_thickness,
                    zorder=2)

    axes[i].set_ylabel('Pressure [hPa]',fontsize=12)
    axes[i].set_xlabel('Year',fontsize=12)
    
    axes[i].set_title(title,fontweight="bold")
    axes[i].set_yscale('symlog')
    axes[i].invert_yaxis()
    axes[i].set(ylim=[20, 0.5])

    plt.tight_layout() 

    i = i+1
    
    cbar_ax_abs = fig.add_axes([0.15, -0.05, 0.8, 0.08])
    cbar_ax_abs.tick_params(labelsize=12)
    cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal',ticks=levels)
    cb.set_label(label=labels, size='14')
    cb.ax.tick_params(labelsize='12')
    #plt.text(5, 168, r'rmsd='+str(round(rmsdval,3)))
    #plt.text(-7.5, 168, r'bias='+str(round(mdval,3)))
    
for label in cb.ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)

    
ofile='AWI-CM3-qbo.png'
    
if ofile is not None:
    plt.savefig(out_path+ofile, dpi=dpi,bbox_inches='tight')
    os.system(f'convert {ofile} -trim {ofile}_trimmed.png')
    os.system(f'mv {ofile}_trimmed.png {ofile}')
    
# Load ERA5 reanalysis data

climatology_file = 'qbo_sel_zonmermean.nc'
ERA5_CRF = cdo.copy(input=str(climatology_path)+'/'+str(climatology_file),returnArray=clim_var)


data_model = OrderedDict()
data_model_mean = OrderedDict()

title2='ERA5 equatorial zonal wind evolution'


for exp_name in input_names:
    data_model[exp_name] = np.squeeze(ERA5_CRF) 
    data_model_mean[exp_name] = np.fliplr(data_model[exp_name])

print(np.shape(data_model_mean[exp_name]))


nrows, ncol = define_rowscol(input_paths)
fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=figsize)
if isinstance(axes, np.ndarray):
    axes = axes.flatten()
else:
    axes = [axes]
i = 0

x = [100,92.5,85,70,60,50,40,30,25,20,15,10,7,5,3,2,1,0.5,0.1]
x=np.asarray(x)
time = np.arange(1989, 2014, 0.0803)


for key in input_names:

    axes[i]=plt.subplot(nrows,ncol,i+1)
    imf=plt.contourf(time,x, data_model_mean[exp_name].T, cmap=plt.cm.PuOr_r, 
                     levels=levels, extend='both',
                     zorder=1)
    line_colors = ['black' for l in imf.levels]
    imc=plt.contour(time,x, data_model_mean[exp_name].T, colors=line_colors, 
                    levels=levels, linewidths=contour_outline_thickness,
                    zorder=2)

    axes[i].set_ylabel('Pressure [hPa]',fontsize=12)
    axes[i].set_xlabel('Year',fontsize=12)
    
    axes[i].set_title(title2,fontweight="bold")
    axes[i].set_yscale('symlog')
    axes[i].invert_yaxis()
    axes[i].set(ylim=[20, 0.5])

    plt.tight_layout() 

    i = i+1
    
    cbar_ax_abs = fig.add_axes([0.15, -0.05, 0.8, 0.08])
    cbar_ax_abs.tick_params(labelsize=12)
    cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal',ticks=levels)
    cb.set_label(label=labels, size='14')
    cb.ax.tick_params(labelsize='12')
    #plt.text(5, 168, r'rmsd='+str(round(rmsdval,3)))
    #plt.text(-7.5, 168, r'bias='+str(round(mdval,3)))
    
for label in cb.ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)

    
ofile='ERA5-qbo.png'
    
if ofile is not None:
    plt.savefig(out_path+ofile, dpi=dpi,bbox_inches='tight')
    os.system(f'convert {ofile} -trim {ofile}_trimmed.png')
    os.system(f'mv {ofile}_trimmed.png {ofile}')


# # FESOM2 bias maps

# In[ ]:


# parameters cell
input_paths = [historic_path+'/fesom']
input_names = [historic_name]
years = range(historic_last25y_start, historic_last25y_end+1)

variable = 'temp'

rowscol=[1,1]
bbox = [-180, 180, -80, 90]
res = [360, 180]
mapproj='pc'
figsize=(6, 4.5)

levels = [-5, 5, 21]
units = r'$^\circ$C'
how="mean"



def data_to_plot(plotds, depth):
    plot_data = []
    plot_names = []
    for key, value in plotds[depth].items():
        if value['nodiff'] is False:
            plot_data.append(value['data'])
            plot_names.append(key)
                
    return plot_data, plot_names

# Mean Deviation weighted
def md(predictions, targets, wgts):
    output_errors = np.average((predictions - targets), axis=0, weights=wgts)
    return (output_errors).mean()

def get_cmap(cmap=None):
    """Get the color map.
    Parameters
    ----------
    cmap: str, mpl.colors.Colormap
        The colormap can be provided as the name (should be in matplotlib or cmocean colormaps),
        or as matplotlib colormap object.
    Returns
    -------
    colormap:mpl.colors.Colormap
        Matplotlib colormap object.
    """
    if cmap:
        if isinstance(cmap, (mpl.colors.Colormap)):
            colormap = cmap
        elif cmap in cmof.cmapnames:
            colormap = cmo.cmap_d[cmap]
        elif cmap in plt.colormaps():
            colormap = plt.get_cmap(cmap)
        else:
            raise ValueError(
                "Get unrecognised name for the colormap `{}`. Colormaps should be from standard matplotlib set of from cmocean package.".format(
                    cmap
                )
            )
    else:
        colormap = plt.get_cmap("Spectral_r")

    return colormap

def interpolate_for_plot(
    data,
    mesh,
    lonreg2,
    latreg2,
    interp="nn",
    distances_path=None,
    inds_path=None,
    radius_of_influence=None,
    basepath=None,
    qhull_path=None,
):
    """Interpolate for the plot.
    Parameters
    ----------
    mesh: mesh object
        FESOM2 mesh object
    data: np.array or list of np.arrays
        FESOM 2 data on nodes (for u,v,u_ice and v_ice one have to first interpolate from elements to nodes).
        Can be ether one np.ndarray or list of np.ndarrays.
    lonreg2: 2D numpy array
        Longitudes of the regular grid.
    latreg2: 2D numpy array
        Latitudes of the regular grid.
    interp: str
        Interpolation method. Options are 'nn' (nearest neighbor), 'idist' (inverce distance), "linear" and "cubic".
    distances_path : string
        Path to the file with distances. If not provided and dumpfile=True, it will be created.
    inds_path : string
        Path to the file with inds. If not provided and dumpfile=True, it will be created.
    qhull_path : str
         Path to the file with qhull (needed for linear and cubic interpolations). If not provided and dumpfile=True, it will be created.
    basepath: str
        path where to store additional interpolation files. If None (default),
        the path of the mesh will be used.
    """
    interpolated = []
    for datainstance in data:

        if interp == "nn":
            ofesom = fesom2regular(
                datainstance,
                mesh,
                lonreg2,
                latreg2,
                distances_path=distances_path,
                inds_path=inds_path,
                radius_of_influence=radius_of_influence,
                basepath=basepath,
            )
            interpolated.append(ofesom)
        elif interp == "idist":
            ofesom = fesom2regular(
                datainstance,
                mesh,
                lonreg2,
                latreg2,
                distances_path=distances_path,
                inds_path=inds_path,
                radius_of_influence=radius_of_influence,
                how="idist",
                k=5,
                basepath=basepath,
            )
            interpolated.append(ofesom)
        elif interp == "linear":
            ofesom = fesom2regular(
                datainstance,
                mesh,
                lonreg2,
                latreg2,
                how="linear",
                qhull_path=qhull_path,
                basepath=basepath,
            )
            interpolated.append(ofesom)
        elif interp == "cubic":
            ofesom = fesom2regular(
                datainstance, mesh, lonreg2, latreg2, basepath=basepath, how="cubic"
            )
            interpolated.append(ofesom)
    return interpolated

def create_indexes_and_distances(mesh, lons, lats, k=1, n_jobs=2):
    """
    Creates KDTree object and query it for indexes of points in FESOM mesh that are close to the
    points of the target grid. Also return distances of the original points to target points.

    Parameters
    ----------
    mesh : fesom_mesh object
        pyfesom mesh representation
    lons/lats : array
        2d arrays with target grid values.
    k : int
        k-th nearest neighbors to return.
    n_jobs : int, optional
        Number of jobs to schedule for parallel processing. If -1 is given
        all processors are used. Default: 1.

    Returns
    -------
    distances : array of floats
        The distances to the nearest neighbors.
    inds : ndarray of ints
        The locations of the neighbors in data.

    """
    xs, ys, zs = lon_lat_to_cartesian(mesh.x2, mesh.y2)
    xt, yt, zt = lon_lat_to_cartesian(lons.flatten(), lats.flatten())

    tree = cKDTree(list(zip(xs, ys, zs)))
    distances, inds = tree.query(list(zip(xt, yt, zt)), k=k, n_jobs=n_jobs)

    return distances, inds

def fesom2regular(
    data,
    mesh,
    lons,
    lats,
    distances_path=None,
    inds_path=None,
    qhull_path=None,
    how="nn",
    k=5,
    radius_of_influence=100000,
    n_jobs=2,
    dumpfile=True,
    basepath=None,
):
    """
    Interpolates data from FESOM mesh to target (usually regular) mesh.
    Parameters
    ----------
    data : array
        1d array that represents FESOM data at one
    mesh : fesom_mesh object
        pyfesom mesh representation
    lons/lats : array
        2d arrays with target grid values.
    distances_path : string
        Path to the file with distances. If not provided and dumpfile=True, it will be created.
    inds_path : string
        Path to the file with inds. If not provided and dumpfile=True, it will be created.
    qhull_path : str
         Path to the file with qhull (needed for linear and cubic interpolations). If not provided and dumpfile=True, it will be created.
    how : str
       Interpolation method. Options are 'nn' (nearest neighbor), 'idist' (inverce distance), "linear" and "cubic".
    k : int
        k-th nearest neighbors to use. Only used when how==idist
    radius_of_influence : int
        Cut off distance in meters, only used in nn and idist.
    n_jobs : int, optional
        Number of jobs to schedule for parallel processing. If -1 is given
        all processors are used. Default: 1. Only used for nn and idist.
    dumpfile: bool
        wether to dump resulted distances and inds to the file.
    basepath: str
        path where to store additional interpolation files. If None (default),
        the path of the mesh will be used.
    Returns
    -------
    data_interpolated : 2d array
        array with data interpolated to the target grid.
    """

    left, right, down, up = np.min(lons), np.max(lons), np.min(lats), np.max(lats)
    lonNumber, latNumber = lons.shape[1], lats.shape[0]

    if how == "nn":
        kk = 1
    else:
        kk = k

    distances_paths = []
    inds_paths = []
    qhull_paths = []

    MESH_BASE = os.path.basename(mesh.path)
    MESH_DIR = mesh.path
    CACHE_DIR = os.environ.get("PYFESOM_CACHE", os.path.join(os.getcwd(), "MESH_cache"))
    CACHE_DIR = os.path.join(CACHE_DIR, MESH_BASE)

    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    distances_file = "distances_{}_{}_{}_{}_{}_{}_{}_{}".format(
        mesh.n2d, left, right, down, up, lonNumber, latNumber, kk
    )
    inds_file = "inds_{}_{}_{}_{}_{}_{}_{}_{}".format(
        mesh.n2d, left, right, down, up, lonNumber, latNumber, kk
    )
    qhull_file = "qhull_{}".format(mesh.n2d)

    distances_paths.append(os.path.join(mesh.path, distances_file))
    distances_paths.append(os.path.join(CACHE_DIR, distances_file))

    inds_paths.append(os.path.join(mesh.path, inds_file))
    inds_paths.append(os.path.join(CACHE_DIR, inds_file))

    qhull_paths.append(os.path.join(mesh.path, qhull_file))
    qhull_paths.append(os.path.join(CACHE_DIR, qhull_file))

    # if distances_path is provided, use it first
    if distances_path is not None:
        distances_paths.insert(0, distances_path)

    if inds_path is not None:
        inds_paths.insert(0, inds_path)

    if qhull_path is not None:
        qhull_paths.insert(0, qhull_path)

    loaded_distances = False
    loaded_inds = False
    loaded_qhull = False
    if how == "nn":
        for distances_path in distances_paths:
            if os.path.isfile(distances_path):
                logging.info(
                    "Note: using precalculated file from {}".format(distances_path)
                )
                try:
                    distances = joblib.load(distances_path)
                    loaded_distances = True
                    break
                except PermissionError:
                    # who knows, something didn't work. Try the next path:
                    continue
        for inds_path in inds_paths:
            if os.path.isfile(inds_path):
                logging.info("Note: using precalculated file from {}".format(inds_path))
                try:
                    inds = joblib.load(inds_path)
                    loaded_inds = True
                    break
                except PermissionError:
                    # Same as above...something is wrong
                    continue
        if not (loaded_distances and loaded_inds):
            distances, inds = create_indexes_and_distances(
                mesh, lons, lats, k=kk, n_jobs=n_jobs
            )
            if dumpfile:
                for distances_path in distances_paths:
                    try:
                        joblib.dump(distances, distances_path)
                        break
                    except PermissionError:
                        # Couldn't dump the file, try next path
                        continue
                for inds_path in inds_paths:
                    try:
                        joblib.dump(inds, inds_path)
                        break
                    except PermissionError:
                        # Couldn't dump inds file, try next
                        continue

        data_interpolated = data[inds]
        data_interpolated[distances >= radius_of_influence] = np.nan
        data_interpolated = data_interpolated.reshape(lons.shape)
        data_interpolated = np.ma.masked_invalid(data_interpolated)
        return data_interpolated

    elif how == "idist":
        for distances_path in distances_paths:
            if os.path.isfile(distances_path):
                logging.info(
                    "Note: using precalculated file from {}".format(distances_path)
                )
                try:
                    distances = joblib.load(distances_path)
                    loaded_distances = True
                    break
                except PermissionError:
                    # who knows, something didn't work. Try the next path:
                    continue
        for inds_path in inds_paths:
            if os.path.isfile(inds_path):
                logging.info("Note: using precalculated file from {}".format(inds_path))
                try:
                    inds = joblib.load(inds_path)
                    loaded_inds = True
                    break
                except PermissionError:
                    # Same as above...something is wrong
                    continue
        if not (loaded_distances and loaded_inds):
            distances, inds = create_indexes_and_distances(
                mesh, lons, lats, k=kk, n_jobs=n_jobs
            )
            if dumpfile:
                for distances_path in distances_paths:
                    try:
                        joblib.dump(distances, distances_path)
                        break
                    except PermissionError:
                        # Couldn't dump the file, try next path
                        continue
                for inds_path in inds_paths:
                    try:
                        joblib.dump(inds, inds_path)
                        break
                    except PermissionError:
                        # Couldn't dump inds file, try next
                        continue

        distances_ma = np.ma.masked_greater(distances, radius_of_influence)

        w = 1.0 / distances_ma ** 2
        data_interpolated = np.ma.sum(w * data[inds], axis=1) / np.ma.sum(w, axis=1)
        data_interpolated.shape = lons.shape
        data_interpolated = np.ma.masked_invalid(data_interpolated)
        return data_interpolated

    elif how == "linear":
        for qhull_path in qhull_paths:
            if os.path.isfile(qhull_path):
                logging.info(
                    "Note: using precalculated file from {}".format(qhull_path)
                )
                try:
                    qh = joblib.load(qhull_path)
                    loaded_qhull = True
                    break
                except PermissionError:
                    # who knows, something didn't work. Try the next path:
                    continue
        if not loaded_qhull:
            points = np.vstack((mesh.x2, mesh.y2)).T
            qh = qhull.Delaunay(points)
            if dumpfile:
                for qhull_path in qhull_paths:
                    try:
                        joblib.dump(qh, qhull_path)
                        break
                    except PermissionError:
                        continue
        data_interpolated = LinearNDInterpolator(qh, data)((lons, lats))
        data_interpolated = np.ma.masked_invalid(data_interpolated)
        return data_interpolated

    elif how == "cubic":
        for qhull_path in qhull_paths:
            if os.path.isfile(qhull_path):
                logging.info(
                    "Note: using precalculated file from {}".format(qhull_path)
                )
                logging.info(
                    "Note: using precalculated file from {}".format(qhull_path)
                )
                try:
                    qh = joblib.load(qhull_path)
                    loaded_qhull = True
                    break
                except PermissionError:
                    # who knows, something didn't work. Try the next path:
                    continue
        if not loaded_qhull:
            points = np.vstack((mesh.x2, mesh.y2)).T
            qh = qhull.Delaunay(points)
            if dumpfile:
                for qhull_path in qhull_paths:
                    try:
                        joblib.dump(qh, qhull_path)
                        break
                    except PermissionError:
                        continue
        data_interpolated = CloughTocher2DInterpolator(qh, data)((lons, lats))
        data_interpolated = np.ma.masked_invalid(data_interpolated)
        return data_interpolated
    else:
        raise ValueError("Interpolation method is not supported")

def mask_ne(lonreg2, latreg2):
    """ Mask earth from lon/lat data using Natural Earth.
    Parameters
    ----------
    lonreg2: float, np.array
        2D array of longitudes
    latreg2: float, np.array
        2D array of latitudes
    Returns
    -------
    m2: bool, np.array
        2D mask with True where the ocean is.
    """
    nearth = cfeature.NaturalEarthFeature("physical", "ocean", "50m")
    main_geom = [contour for contour in nearth.geometries()][0]

    mask = shapely.vectorized.contains(main_geom, lonreg2, latreg2)
    m2 = np.where(((lonreg2 == -180.0) & (latreg2 > 71.5)), True, mask)
    m2 = np.where(
        ((lonreg2 == -180.0) & (latreg2 < 70.95) & (latreg2 > 68.96)), True, m2
    )
    m2 = np.where(((lonreg2 == 180.0) & (latreg2 > 71.5)), True, mask)
    m2 = np.where(
        ((lonreg2 == 180.0) & (latreg2 < 70.95) & (latreg2 > 68.96)), True, m2
    )
    # m2 = np.where(
    #        ((lonreg2 == 180.0) & (latreg2 > -75.0) & (latreg2 < 0)), True, m2
    #    )
    m2 = np.where(((lonreg2 == -180.0) & (latreg2 < 65.33)), True, m2)
    m2 = np.where(((lonreg2 == 180.0) & (latreg2 < 65.33)), True, m2)

    return ~m2

def lon_lat_to_cartesian(lon, lat, R=6371000):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R. Taken from http://earthpy.org/interpolation_between_grids_with_ckdtree.html
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x, y, z


def create_proj_figure(mapproj, rowscol, figsize):
    """ Create figure and axis with cartopy projection.
    Parameters
    ----------
    mapproj: str
        name of the projection:
            merc: Mercator
            pc: PlateCarree (default)
            np: NorthPolarStereo
            sp: SouthPolarStereo
            rob: Robinson
    rowcol: (int, int)
        number of rows and columns of the figure.
    figsize: (float, float)
        width, height in inches.
    Returns
    -------
    fig, ax
    """
    if mapproj == "merc":
        fig, ax = plt.subplots(
            rowscol[0],
            rowscol[1],
            subplot_kw=dict(projection=ccrs.Mercator()),
            constrained_layout=True,
            figsize=figsize,
        )
    elif mapproj == "pc":
        fig, ax = plt.subplots(
            rowscol[0],
            rowscol[1],
            subplot_kw=dict(projection=ccrs.PlateCarree()),
            constrained_layout=True,
            figsize=figsize,
        )
    elif mapproj == "np":
        fig, ax = plt.subplots(
            rowscol[0],
            rowscol[1],
            subplot_kw=dict(projection=ccrs.NorthPolarStereo()),
            constrained_layout=True,
            figsize=figsize,
        )
    elif mapproj == "sp":
        fig, ax = plt.subplots(
            rowscol[0],
            rowscol[1],
            subplot_kw=dict(projection=ccrs.SouthPolarStereo()),
            constrained_layout=True,
            figsize=figsize,
        )
    elif mapproj == "rob":
        fig, ax = plt.subplots(
            rowscol[0],
            rowscol[1],
            subplot_kw=dict(projection=ccrs.Robinson()),
            constrained_layout=True,
            figsize=figsize,
        )
    else:
        raise ValueError(f"Projection {mapproj} is not supported.")
    return fig, ax

def get_plot_levels(levels, data, lev_to_data=False):
    """Returns levels for the plot.
    Parameters
    ----------
    levels: list, numpy array
        Can be list or numpy array with three or more elements.
        If only three elements provided, they will b einterpereted as min, max, number of levels.
        If more elements provided, they will be used directly.
    data: numpy array of xarray
        Data, that should be plotted with this levels.
    lev_to_data: bool
        Switch to correct the levels to the actual data range.
        This is needed for safe plotting on triangular grid with cartopy.
    Returns
    -------
    data_levels: numpy array
        resulted levels.
    """
    if levels is not None:
        if len(levels) == 3:
            mmin, mmax, nnum = levels
            if lev_to_data:
                mmin, mmax = levels_to_data(mmin, mmax, data)
            nnum = int(nnum)
            data_levels = np.linspace(mmin, mmax, nnum)
        elif len(levels) < 3:
            raise ValueError(
                "Levels can be the list or numpy array with three or more elements."
            )
        else:
            data_levels = np.array(levels)
    else:
        mmin = np.nanmin(data)
        mmax = np.nanmax(data)
        nnum = 40
        data_levels = np.linspace(mmin, mmax, nnum)
    return data_levels


def plot(
    mesh,
    data,
    cmap=None,
    influence=80000,
    box=[-180, 180, -89, 90],
    res=[360, 180],
    interp="nn",
    mapproj="pc",
    levels=None,
    ptype="cf",
    units=None,
    figsize=(6, 4.5),
    rowscol=(1, 1),
    titles=None,
    distances_path=None,
    inds_path=None,
    qhull_path=None,
    basepath=None,
    interpolated_data=None,
    lonreg=None,
    latreg=None,
    no_pi_mask=False,
    rmsdval=None,
    mdval=None,
):
    """
    Plots interpolated 2d field on the map.
    Parameters
    ----------
    mesh: mesh object
        FESOM2 mesh object
    data: np.array or list of np.arrays
        FESOM 2 data on nodes (for u,v,u_ice and v_ice one have to first interpolate from elements to nodes).
        Can be ether one np.ndarray or list of np.ndarrays.
    cmap: str
        Name of the colormap from cmocean package or from the standard matplotlib set.
        By default `Spectral_r` will be used.
    influence: float
        Radius of influence for interpolation, in meters.
    box: list
        Map boundaries in -180 180 -90 90 format that will be used for interpolation (default [-180 180 -89 90]).
    res: list
        Number of points along each axis that will be used for interpolation (for lon and lat),
        default [360, 180].
    interp: str
        Interpolation method. Options are 'nn' (nearest neighbor), 'idist' (inverce distance), "linear" and "cubic".
    mapproj: str
        Map projection. Options are Mercator (merc), Plate Carree (pc),
        North Polar Stereo (np), South Polar Stereo (sp),  Robinson (rob)
    levels: list
        Levels for contour plot in format (min, max, numberOfLevels). List with more than
        3 values will be interpreted as just a list of individual level values.
        If not provided min/max values from data will be used with 40 levels.
    ptype: str
        Plot type. Options are contourf (\'cf\') and pcolormesh (\'pcm\')
    units: str
        Units for color bar.
    figsize: tuple
        figure size in inches
    rowscol: tuple
        number of rows and columns.
    titles: str or list
        Title of the plot (if string) or subplots (if list of strings)
    distances_path : string
        Path to the file with distances. If not provided and dumpfile=True, it will be created.
    inds_path : string
        Path to the file with inds. If not provided and dumpfile=True, it will be created.
    qhull_path : str
         Path to the file with qhull (needed for linear and cubic interpolations).
         If not provided and dumpfile=True, it will be created.
    interpolated_data: np.array
         data interpolated to regular grid (you also have to provide lonreg and latreg).
         If provided, data will be plotted directly, without interpolation.
    lonreg: np.array
         1D array of longitudes. Used in combination with `interpolated_data`,
         when you need to plot interpolated data directly.
    latreg: np.array
         1D array of latitudes. Used in combination with `interpolated_data`,
         when you need to plot interpolated data directly.
    basepath: str
        path where to store additional interpolation files. If None (default),
        the path of the mesh will be used.
    no_pi_mask: bool
        Mask PI by default or not.
    """
    if not isinstance(data, list):
        data = [data]
    if titles:
        if not isinstance(titles, list):
            titles = [titles]
        if len(titles) != len(data):
            raise ValueError(
                "The number of titles do not match the number of data fields, please adjust titles (or put to None)"
            )

    if (rowscol[0] * rowscol[1]) < len(data):
        raise ValueError(
            "Number of rows*columns is smaller than number of data fields, please adjust rowscol."
        )

    colormap = get_cmap(cmap=cmap)

    radius_of_influence = influence

    left, right, down, up = box
    lonNumber, latNumber = res

    if lonreg is None:
        lonreg = np.linspace(left, right, lonNumber)
        latreg = np.linspace(down, up, latNumber)

    lonreg2, latreg2 = np.meshgrid(lonreg, latreg)

    if interpolated_data is None:
        interpolated = interpolate_for_plot(
            data,
            mesh,
            lonreg2,
            latreg2,
            interp=interp,
            distances_path=distances_path,
            inds_path=inds_path,
            radius_of_influence=radius_of_influence,
            basepath=basepath,
            qhull_path=qhull_path,
        )
    else:
        interpolated = [interpolated_data]

    m2 = mask_ne(lonreg2, latreg2)

    for i in range(len(interpolated)):
        if not no_pi_mask:
            interpolated[i] = np.ma.masked_where(m2, interpolated[i])
        interpolated[i] = np.ma.masked_equal(interpolated[i], 0)

    fig, ax = create_proj_figure(mapproj, rowscol, figsize)

    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
    else:
        ax = [ax]

    for ind, data_int in enumerate(interpolated):
        ax[ind].set_extent([left, right, down, up], crs=ccrs.PlateCarree())

        data_levels = get_plot_levels(levels, data_int, lev_to_data=False)

        if ptype == "cf":
            data_int_cyc, lon_cyc = add_cyclic_point(data_int, coord=lonreg)
            image = ax[ind].contourf(
                lon_cyc,
                latreg,
                data_int_cyc,
                levels=data_levels,
                transform=ccrs.PlateCarree(),
                cmap=colormap,
                extend="both",
            )
        elif ptype == "pcm":
            mmin = data_levels[0]
            mmax = data_levels[-1]
            data_int_cyc, lon_cyc = add_cyclic_point(data_int, coord=lonreg)
            image = ax[ind].pcolormesh(
                lon_cyc,
                latreg,
                data_int_cyc,
                vmin=mmin,
                vmax=mmax,
                transform=ccrs.PlateCarree(),
                cmap=colormap,
            )
        else:
            raise ValueError("Inknown plot type {}".format(ptype))

        # ax.coastlines(resolution = '50m',lw=0.5)
        ax[ind].add_feature(
            cfeature.GSHHSFeature(levels=[1], scale="low", facecolor="lightgray")
        )
        if titles:
            titles = titles.copy()
            ax[ind].set_title(titles.pop(0), fontweight='bold')

    for delind in range(ind + 1, len(ax)):
        fig.delaxes(ax[delind])


    gl = ax[ind].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='-')

    gl.xlabels_bottom = False
        
    textrsmd='rmsd='+str(round(rmsdval,3))
    textbias='bias='+str(round(mdval,3))
    props = dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.5)
    ax[ind].text(0.02, 0.4, textrsmd, transform=ax[ind].transAxes, fontsize=13,
        verticalalignment='top', bbox=props, zorder=4)
    ax[ind].text(0.02, 0.3, textbias, transform=ax[ind].transAxes, fontsize=13,
        verticalalignment='top', bbox=props, zorder=4)

    cbar_ax_abs = fig.add_axes([0.15, 0.10, 0.7, 0.05])
    cbar_ax_abs.tick_params(labelsize=12)
    cb = fig.colorbar(image, cax=cbar_ax_abs, orientation='horizontal',ticks=levels)
    cb.set_label(label=units, size='14')
    cb.ax.tick_params(labelsize='12')
    for label in cb.ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    return ax,latreg

for depth in [0,100,1000,4000]:

    if depth==0:
        title2 = "Sea surface temperature bias vs. PHC3"
        ofile=str(depth)+"m-temp-phc3"
    elif depth==100:
        title2 = "100 meter temperature bias vs. PHC3"
        ofile=str(depth)+"m-temp-phc3"
    elif depth==1000:
        title2 = "1000 meter temperature bias vs. PHC3"
        ofile=str(depth)+"m-temp-phc3"
    elif depth==4000:
        title2 = "4000 meter temperature bias vs. PHC3"
        ofile=str(depth)+"m-temp-phc3"


    if input_names is None:
        input_names = []
        for run in input_paths:
            run = os.path.join(run, '')
            input_names.append(run.split('/')[-2])

    mesh = pf.load_mesh(meshpath, abg=abg, 
                        usepickle=True, usejoblib=False)

    from pprint import pprint
    pprint(vars(mesh))

    plotds = OrderedDict()
    data_reference = pf.get_data(reference_path, variable, reference_years, mesh, depth = depth, how=how, compute=True, silent=True)
    plotds[depth] = {}
    for exp_path, exp_name  in zip(input_paths, input_names):
        data_test      = pf.get_data(exp_path, variable, years, mesh, depth = depth, how=how, compute=True, silent=True)
        data_difference= data_test - data_reference
        title = exp_name+" - "+reference_name
        plotds[depth][title] = {}
        plotds[depth][title]['data'] = data_difference
        if (data_difference.max() == data_difference.min() == 0):
            plotds[depth][title]['nodiff'] = True
        else:
            plotds[depth][title]['nodiff'] = False

    mesh_data = Dataset(meshpath+'/'+mesh_file)
    wgts=mesh_data['cell_area']
    rmsdval = sqrt(mean_squared_error(data_test,data_reference,sample_weight=wgts))
    mdval = md(data_test,data_reference,wgts)

    sfmt = ticker.ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((-3, 4))

    levels = [-5.0,-3.0,-2.0,-1.0,-.6,-.2,.2,.6,1.0,2.0,3.0,5.0]

    figsize=(6, 4.5)
    dpi=300

    plot_data, plot_names = data_to_plot(plotds, depth)
    if not plot_data:
        print('There is no difference between fields')
        identical = True
    else:
        identical = False

    if len(plot_data) == 1:
        plot_data = plot_data[0]
        plot_names = plot_names[0]


    if not identical:
        plot(mesh, 
            plot_data,
            rowscol=rowscol,
            mapproj=mapproj,
            cmap='PuOr_r', 
            levels=levels,
            figsize = figsize, 
            box=bbox, 
            res = res,
            units = units,
            titles = title2,
            rmsdval = rmsdval,
            mdval = mdval);


    if ofile is not None:
        plt.savefig(out_path+ofile, dpi=dpi, bbox_inches='tight')
        os.system(f'convert {ofile} -trim {ofile}_trimmed.png')
        os.system(f'mv {ofile}_trimmed.png {ofile}')


# # FESOM2 salinity biases

# In[ ]:


# parameters cell
input_paths = [historic_path+'/fesom']
input_names = [historic_name]
years = range(historic_last25y_start, historic_last25y_end+1)

variable = 'salt'

rowscol=[1,1]
bbox = [-180, 180, -80, 90]
res = [360, 180]
mapproj='pc'
figsize=(6, 4.5)

levels = [-5, 5, 21]
units = r'PSU'
how="mean"



def data_to_plot(plotds, depth):
    plot_data = []
    plot_names = []
    for key, value in plotds[depth].items():
        if value['nodiff'] is False:
            plot_data.append(value['data'])
            plot_names.append(key)
                
    return plot_data, plot_names

# Mean Deviation weighted
def md(predictions, targets, wgts):
    output_errors = np.average((predictions - targets), axis=0, weights=wgts)
    return (output_errors).mean()

def get_cmap(cmap=None):
    """Get the color map.
    Parameters
    ----------
    cmap: str, mpl.colors.Colormap
        The colormap can be provided as the name (should be in matplotlib or cmocean colormaps),
        or as matplotlib colormap object.
    Returns
    -------
    colormap:mpl.colors.Colormap
        Matplotlib colormap object.
    """
    if cmap:
        if isinstance(cmap, (mpl.colors.Colormap)):
            colormap = cmap
        elif cmap in cmof.cmapnames:
            colormap = cmo.cmap_d[cmap]
        elif cmap in plt.colormaps():
            colormap = plt.get_cmap(cmap)
        else:
            raise ValueError(
                "Get unrecognised name for the colormap `{}`. Colormaps should be from standard matplotlib set of from cmocean package.".format(
                    cmap
                )
            )
    else:
        colormap = plt.get_cmap("Spectral_r")

    return colormap

def interpolate_for_plot(
    data,
    mesh,
    lonreg2,
    latreg2,
    interp="nn",
    distances_path=None,
    inds_path=None,
    radius_of_influence=None,
    basepath=None,
    qhull_path=None,
):
    """Interpolate for the plot.
    Parameters
    ----------
    mesh: mesh object
        FESOM2 mesh object
    data: np.array or list of np.arrays
        FESOM 2 data on nodes (for u,v,u_ice and v_ice one have to first interpolate from elements to nodes).
        Can be ether one np.ndarray or list of np.ndarrays.
    lonreg2: 2D numpy array
        Longitudes of the regular grid.
    latreg2: 2D numpy array
        Latitudes of the regular grid.
    interp: str
        Interpolation method. Options are 'nn' (nearest neighbor), 'idist' (inverce distance), "linear" and "cubic".
    distances_path : string
        Path to the file with distances. If not provided and dumpfile=True, it will be created.
    inds_path : string
        Path to the file with inds. If not provided and dumpfile=True, it will be created.
    qhull_path : str
         Path to the file with qhull (needed for linear and cubic interpolations). If not provided and dumpfile=True, it will be created.
    basepath: str
        path where to store additional interpolation files. If None (default),
        the path of the mesh will be used.
    """
    interpolated = []
    for datainstance in data:

        if interp == "nn":
            ofesom = fesom2regular(
                datainstance,
                mesh,
                lonreg2,
                latreg2,
                distances_path=distances_path,
                inds_path=inds_path,
                radius_of_influence=radius_of_influence,
                basepath=basepath,
            )
            interpolated.append(ofesom)
        elif interp == "idist":
            ofesom = fesom2regular(
                datainstance,
                mesh,
                lonreg2,
                latreg2,
                distances_path=distances_path,
                inds_path=inds_path,
                radius_of_influence=radius_of_influence,
                how="idist",
                k=5,
                basepath=basepath,
            )
            interpolated.append(ofesom)
        elif interp == "linear":
            ofesom = fesom2regular(
                datainstance,
                mesh,
                lonreg2,
                latreg2,
                how="linear",
                qhull_path=qhull_path,
                basepath=basepath,
            )
            interpolated.append(ofesom)
        elif interp == "cubic":
            ofesom = fesom2regular(
                datainstance, mesh, lonreg2, latreg2, basepath=basepath, how="cubic"
            )
            interpolated.append(ofesom)
    return interpolated

def create_indexes_and_distances(mesh, lons, lats, k=1, n_jobs=2):
    """
    Creates KDTree object and query it for indexes of points in FESOM mesh that are close to the
    points of the target grid. Also return distances of the original points to target points.

    Parameters
    ----------
    mesh : fesom_mesh object
        pyfesom mesh representation
    lons/lats : array
        2d arrays with target grid values.
    k : int
        k-th nearest neighbors to return.
    n_jobs : int, optional
        Number of jobs to schedule for parallel processing. If -1 is given
        all processors are used. Default: 1.

    Returns
    -------
    distances : array of floats
        The distances to the nearest neighbors.
    inds : ndarray of ints
        The locations of the neighbors in data.

    """
    xs, ys, zs = lon_lat_to_cartesian(mesh.x2, mesh.y2)
    xt, yt, zt = lon_lat_to_cartesian(lons.flatten(), lats.flatten())

    tree = cKDTree(list(zip(xs, ys, zs)))
    distances, inds = tree.query(list(zip(xt, yt, zt)), k=k, n_jobs=n_jobs)

    return distances, inds

def fesom2regular(
    data,
    mesh,
    lons,
    lats,
    distances_path=None,
    inds_path=None,
    qhull_path=None,
    how="nn",
    k=5,
    radius_of_influence=100000,
    n_jobs=2,
    dumpfile=True,
    basepath=None,
):
    """
    Interpolates data from FESOM mesh to target (usually regular) mesh.
    Parameters
    ----------
    data : array
        1d array that represents FESOM data at one
    mesh : fesom_mesh object
        pyfesom mesh representation
    lons/lats : array
        2d arrays with target grid values.
    distances_path : string
        Path to the file with distances. If not provided and dumpfile=True, it will be created.
    inds_path : string
        Path to the file with inds. If not provided and dumpfile=True, it will be created.
    qhull_path : str
         Path to the file with qhull (needed for linear and cubic interpolations). If not provided and dumpfile=True, it will be created.
    how : str
       Interpolation method. Options are 'nn' (nearest neighbor), 'idist' (inverce distance), "linear" and "cubic".
    k : int
        k-th nearest neighbors to use. Only used when how==idist
    radius_of_influence : int
        Cut off distance in meters, only used in nn and idist.
    n_jobs : int, optional
        Number of jobs to schedule for parallel processing. If -1 is given
        all processors are used. Default: 1. Only used for nn and idist.
    dumpfile: bool
        wether to dump resulted distances and inds to the file.
    basepath: str
        path where to store additional interpolation files. If None (default),
        the path of the mesh will be used.
    Returns
    -------
    data_interpolated : 2d array
        array with data interpolated to the target grid.
    """

    left, right, down, up = np.min(lons), np.max(lons), np.min(lats), np.max(lats)
    lonNumber, latNumber = lons.shape[1], lats.shape[0]

    if how == "nn":
        kk = 1
    else:
        kk = k

    distances_paths = []
    inds_paths = []
    qhull_paths = []

    MESH_BASE = os.path.basename(mesh.path)
    MESH_DIR = mesh.path
    CACHE_DIR = os.environ.get("PYFESOM_CACHE", os.path.join(os.getcwd(), "MESH_cache"))
    CACHE_DIR = os.path.join(CACHE_DIR, MESH_BASE)

    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    distances_file = "distances_{}_{}_{}_{}_{}_{}_{}_{}".format(
        mesh.n2d, left, right, down, up, lonNumber, latNumber, kk
    )
    inds_file = "inds_{}_{}_{}_{}_{}_{}_{}_{}".format(
        mesh.n2d, left, right, down, up, lonNumber, latNumber, kk
    )
    qhull_file = "qhull_{}".format(mesh.n2d)

    distances_paths.append(os.path.join(mesh.path, distances_file))
    distances_paths.append(os.path.join(CACHE_DIR, distances_file))

    inds_paths.append(os.path.join(mesh.path, inds_file))
    inds_paths.append(os.path.join(CACHE_DIR, inds_file))

    qhull_paths.append(os.path.join(mesh.path, qhull_file))
    qhull_paths.append(os.path.join(CACHE_DIR, qhull_file))

    # if distances_path is provided, use it first
    if distances_path is not None:
        distances_paths.insert(0, distances_path)

    if inds_path is not None:
        inds_paths.insert(0, inds_path)

    if qhull_path is not None:
        qhull_paths.insert(0, qhull_path)

    loaded_distances = False
    loaded_inds = False
    loaded_qhull = False
    if how == "nn":
        for distances_path in distances_paths:
            if os.path.isfile(distances_path):
                logging.info(
                    "Note: using precalculated file from {}".format(distances_path)
                )
                try:
                    distances = joblib.load(distances_path)
                    loaded_distances = True
                    break
                except PermissionError:
                    # who knows, something didn't work. Try the next path:
                    continue
        for inds_path in inds_paths:
            if os.path.isfile(inds_path):
                logging.info("Note: using precalculated file from {}".format(inds_path))
                try:
                    inds = joblib.load(inds_path)
                    loaded_inds = True
                    break
                except PermissionError:
                    # Same as above...something is wrong
                    continue
        if not (loaded_distances and loaded_inds):
            distances, inds = create_indexes_and_distances(
                mesh, lons, lats, k=kk, n_jobs=n_jobs
            )
            if dumpfile:
                for distances_path in distances_paths:
                    try:
                        joblib.dump(distances, distances_path)
                        break
                    except PermissionError:
                        # Couldn't dump the file, try next path
                        continue
                for inds_path in inds_paths:
                    try:
                        joblib.dump(inds, inds_path)
                        break
                    except PermissionError:
                        # Couldn't dump inds file, try next
                        continue

        data_interpolated = data[inds]
        data_interpolated[distances >= radius_of_influence] = np.nan
        data_interpolated = data_interpolated.reshape(lons.shape)
        data_interpolated = np.ma.masked_invalid(data_interpolated)
        return data_interpolated

    elif how == "idist":
        for distances_path in distances_paths:
            if os.path.isfile(distances_path):
                logging.info(
                    "Note: using precalculated file from {}".format(distances_path)
                )
                try:
                    distances = joblib.load(distances_path)
                    loaded_distances = True
                    break
                except PermissionError:
                    # who knows, something didn't work. Try the next path:
                    continue
        for inds_path in inds_paths:
            if os.path.isfile(inds_path):
                logging.info("Note: using precalculated file from {}".format(inds_path))
                try:
                    inds = joblib.load(inds_path)
                    loaded_inds = True
                    break
                except PermissionError:
                    # Same as above...something is wrong
                    continue
        if not (loaded_distances and loaded_inds):
            distances, inds = create_indexes_and_distances(
                mesh, lons, lats, k=kk, n_jobs=n_jobs
            )
            if dumpfile:
                for distances_path in distances_paths:
                    try:
                        joblib.dump(distances, distances_path)
                        break
                    except PermissionError:
                        # Couldn't dump the file, try next path
                        continue
                for inds_path in inds_paths:
                    try:
                        joblib.dump(inds, inds_path)
                        break
                    except PermissionError:
                        # Couldn't dump inds file, try next
                        continue

        distances_ma = np.ma.masked_greater(distances, radius_of_influence)

        w = 1.0 / distances_ma ** 2
        data_interpolated = np.ma.sum(w * data[inds], axis=1) / np.ma.sum(w, axis=1)
        data_interpolated.shape = lons.shape
        data_interpolated = np.ma.masked_invalid(data_interpolated)
        return data_interpolated

    elif how == "linear":
        for qhull_path in qhull_paths:
            if os.path.isfile(qhull_path):
                logging.info(
                    "Note: using precalculated file from {}".format(qhull_path)
                )
                try:
                    qh = joblib.load(qhull_path)
                    loaded_qhull = True
                    break
                except PermissionError:
                    # who knows, something didn't work. Try the next path:
                    continue
        if not loaded_qhull:
            points = np.vstack((mesh.x2, mesh.y2)).T
            qh = qhull.Delaunay(points)
            if dumpfile:
                for qhull_path in qhull_paths:
                    try:
                        joblib.dump(qh, qhull_path)
                        break
                    except PermissionError:
                        continue
        data_interpolated = LinearNDInterpolator(qh, data)((lons, lats))
        data_interpolated = np.ma.masked_invalid(data_interpolated)
        return data_interpolated

    elif how == "cubic":
        for qhull_path in qhull_paths:
            if os.path.isfile(qhull_path):
                logging.info(
                    "Note: using precalculated file from {}".format(qhull_path)
                )
                logging.info(
                    "Note: using precalculated file from {}".format(qhull_path)
                )
                try:
                    qh = joblib.load(qhull_path)
                    loaded_qhull = True
                    break
                except PermissionError:
                    # who knows, something didn't work. Try the next path:
                    continue
        if not loaded_qhull:
            points = np.vstack((mesh.x2, mesh.y2)).T
            qh = qhull.Delaunay(points)
            if dumpfile:
                for qhull_path in qhull_paths:
                    try:
                        joblib.dump(qh, qhull_path)
                        break
                    except PermissionError:
                        continue
        data_interpolated = CloughTocher2DInterpolator(qh, data)((lons, lats))
        data_interpolated = np.ma.masked_invalid(data_interpolated)
        return data_interpolated
    else:
        raise ValueError("Interpolation method is not supported")

def mask_ne(lonreg2, latreg2):
    """ Mask earth from lon/lat data using Natural Earth.
    Parameters
    ----------
    lonreg2: float, np.array
        2D array of longitudes
    latreg2: float, np.array
        2D array of latitudes
    Returns
    -------
    m2: bool, np.array
        2D mask with True where the ocean is.
    """
    nearth = cfeature.NaturalEarthFeature("physical", "ocean", "50m")
    main_geom = [contour for contour in nearth.geometries()][0]

    mask = shapely.vectorized.contains(main_geom, lonreg2, latreg2)
    m2 = np.where(((lonreg2 == -180.0) & (latreg2 > 71.5)), True, mask)
    m2 = np.where(
        ((lonreg2 == -180.0) & (latreg2 < 70.95) & (latreg2 > 68.96)), True, m2
    )
    m2 = np.where(((lonreg2 == 180.0) & (latreg2 > 71.5)), True, mask)
    m2 = np.where(
        ((lonreg2 == 180.0) & (latreg2 < 70.95) & (latreg2 > 68.96)), True, m2
    )
    # m2 = np.where(
    #        ((lonreg2 == 180.0) & (latreg2 > -75.0) & (latreg2 < 0)), True, m2
    #    )
    m2 = np.where(((lonreg2 == -180.0) & (latreg2 < 65.33)), True, m2)
    m2 = np.where(((lonreg2 == 180.0) & (latreg2 < 65.33)), True, m2)

    return ~m2

def lon_lat_to_cartesian(lon, lat, R=6371000):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R. Taken from http://earthpy.org/interpolation_between_grids_with_ckdtree.html
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x, y, z


def create_proj_figure(mapproj, rowscol, figsize):
    """ Create figure and axis with cartopy projection.
    Parameters
    ----------
    mapproj: str
        name of the projection:
            merc: Mercator
            pc: PlateCarree (default)
            np: NorthPolarStereo
            sp: SouthPolarStereo
            rob: Robinson
    rowcol: (int, int)
        number of rows and columns of the figure.
    figsize: (float, float)
        width, height in inches.
    Returns
    -------
    fig, ax
    """
    if mapproj == "merc":
        fig, ax = plt.subplots(
            rowscol[0],
            rowscol[1],
            subplot_kw=dict(projection=ccrs.Mercator()),
            constrained_layout=True,
            figsize=figsize,
        )
    elif mapproj == "pc":
        fig, ax = plt.subplots(
            rowscol[0],
            rowscol[1],
            subplot_kw=dict(projection=ccrs.PlateCarree()),
            constrained_layout=True,
            figsize=figsize,
        )
    elif mapproj == "np":
        fig, ax = plt.subplots(
            rowscol[0],
            rowscol[1],
            subplot_kw=dict(projection=ccrs.NorthPolarStereo()),
            constrained_layout=True,
            figsize=figsize,
        )
    elif mapproj == "sp":
        fig, ax = plt.subplots(
            rowscol[0],
            rowscol[1],
            subplot_kw=dict(projection=ccrs.SouthPolarStereo()),
            constrained_layout=True,
            figsize=figsize,
        )
    elif mapproj == "rob":
        fig, ax = plt.subplots(
            rowscol[0],
            rowscol[1],
            subplot_kw=dict(projection=ccrs.Robinson()),
            constrained_layout=True,
            figsize=figsize,
        )
    else:
        raise ValueError(f"Projection {mapproj} is not supported.")
    return fig, ax

def get_plot_levels(levels, data, lev_to_data=False):
    """Returns levels for the plot.
    Parameters
    ----------
    levels: list, numpy array
        Can be list or numpy array with three or more elements.
        If only three elements provided, they will b einterpereted as min, max, number of levels.
        If more elements provided, they will be used directly.
    data: numpy array of xarray
        Data, that should be plotted with this levels.
    lev_to_data: bool
        Switch to correct the levels to the actual data range.
        This is needed for safe plotting on triangular grid with cartopy.
    Returns
    -------
    data_levels: numpy array
        resulted levels.
    """
    if levels is not None:
        if len(levels) == 3:
            mmin, mmax, nnum = levels
            if lev_to_data:
                mmin, mmax = levels_to_data(mmin, mmax, data)
            nnum = int(nnum)
            data_levels = np.linspace(mmin, mmax, nnum)
        elif len(levels) < 3:
            raise ValueError(
                "Levels can be the list or numpy array with three or more elements."
            )
        else:
            data_levels = np.array(levels)
    else:
        mmin = np.nanmin(data)
        mmax = np.nanmax(data)
        nnum = 40
        data_levels = np.linspace(mmin, mmax, nnum)
    return data_levels


def plot(
    mesh,
    data,
    cmap=None,
    influence=80000,
    box=[-180, 180, -89, 90],
    res=[360, 180],
    interp="nn",
    mapproj="pc",
    levels=None,
    ptype="cf",
    units=None,
    figsize=(6, 4.5),
    rowscol=(1, 1),
    titles=None,
    distances_path=None,
    inds_path=None,
    qhull_path=None,
    basepath=None,
    interpolated_data=None,
    lonreg=None,
    latreg=None,
    no_pi_mask=False,
    rmsdval=None,
    mdval=None,
):
    """
    Plots interpolated 2d field on the map.
    Parameters
    ----------
    mesh: mesh object
        FESOM2 mesh object
    data: np.array or list of np.arrays
        FESOM 2 data on nodes (for u,v,u_ice and v_ice one have to first interpolate from elements to nodes).
        Can be ether one np.ndarray or list of np.ndarrays.
    cmap: str
        Name of the colormap from cmocean package or from the standard matplotlib set.
        By default `Spectral_r` will be used.
    influence: float
        Radius of influence for interpolation, in meters.
    box: list
        Map boundaries in -180 180 -90 90 format that will be used for interpolation (default [-180 180 -89 90]).
    res: list
        Number of points along each axis that will be used for interpolation (for lon and lat),
        default [360, 180].
    interp: str
        Interpolation method. Options are 'nn' (nearest neighbor), 'idist' (inverce distance), "linear" and "cubic".
    mapproj: str
        Map projection. Options are Mercator (merc), Plate Carree (pc),
        North Polar Stereo (np), South Polar Stereo (sp),  Robinson (rob)
    levels: list
        Levels for contour plot in format (min, max, numberOfLevels). List with more than
        3 values will be interpreted as just a list of individual level values.
        If not provided min/max values from data will be used with 40 levels.
    ptype: str
        Plot type. Options are contourf (\'cf\') and pcolormesh (\'pcm\')
    units: str
        Units for color bar.
    figsize: tuple
        figure size in inches
    rowscol: tuple
        number of rows and columns.
    titles: str or list
        Title of the plot (if string) or subplots (if list of strings)
    distances_path : string
        Path to the file with distances. If not provided and dumpfile=True, it will be created.
    inds_path : string
        Path to the file with inds. If not provided and dumpfile=True, it will be created.
    qhull_path : str
         Path to the file with qhull (needed for linear and cubic interpolations).
         If not provided and dumpfile=True, it will be created.
    interpolated_data: np.array
         data interpolated to regular grid (you also have to provide lonreg and latreg).
         If provided, data will be plotted directly, without interpolation.
    lonreg: np.array
         1D array of longitudes. Used in combination with `interpolated_data`,
         when you need to plot interpolated data directly.
    latreg: np.array
         1D array of latitudes. Used in combination with `interpolated_data`,
         when you need to plot interpolated data directly.
    basepath: str
        path where to store additional interpolation files. If None (default),
        the path of the mesh will be used.
    no_pi_mask: bool
        Mask PI by default or not.
    """
    if not isinstance(data, list):
        data = [data]
    if titles:
        if not isinstance(titles, list):
            titles = [titles]
        if len(titles) != len(data):
            raise ValueError(
                "The number of titles do not match the number of data fields, please adjust titles (or put to None)"
            )

    if (rowscol[0] * rowscol[1]) < len(data):
        raise ValueError(
            "Number of rows*columns is smaller than number of data fields, please adjust rowscol."
        )

    colormap = get_cmap(cmap=cmap)

    radius_of_influence = influence

    left, right, down, up = box
    lonNumber, latNumber = res

    if lonreg is None:
        lonreg = np.linspace(left, right, lonNumber)
        latreg = np.linspace(down, up, latNumber)

    lonreg2, latreg2 = np.meshgrid(lonreg, latreg)

    if interpolated_data is None:
        interpolated = interpolate_for_plot(
            data,
            mesh,
            lonreg2,
            latreg2,
            interp=interp,
            distances_path=distances_path,
            inds_path=inds_path,
            radius_of_influence=radius_of_influence,
            basepath=basepath,
            qhull_path=qhull_path,
        )
    else:
        interpolated = [interpolated_data]

    m2 = mask_ne(lonreg2, latreg2)

    for i in range(len(interpolated)):
        if not no_pi_mask:
            interpolated[i] = np.ma.masked_where(m2, interpolated[i])
        interpolated[i] = np.ma.masked_equal(interpolated[i], 0)

    fig, ax = create_proj_figure(mapproj, rowscol, figsize)

    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
    else:
        ax = [ax]

    for ind, data_int in enumerate(interpolated):
        ax[ind].set_extent([left, right, down, up], crs=ccrs.PlateCarree())

        data_levels = get_plot_levels(levels, data_int, lev_to_data=False)

        if ptype == "cf":
            data_int_cyc, lon_cyc = add_cyclic_point(data_int, coord=lonreg)
            image = ax[ind].contourf(
                lon_cyc,
                latreg,
                data_int_cyc,
                levels=data_levels,
                transform=ccrs.PlateCarree(),
                cmap=colormap,
                extend="both",
            )
        elif ptype == "pcm":
            mmin = data_levels[0]
            mmax = data_levels[-1]
            data_int_cyc, lon_cyc = add_cyclic_point(data_int, coord=lonreg)
            image = ax[ind].pcolormesh(
                lon_cyc,
                latreg,
                data_int_cyc,
                vmin=mmin,
                vmax=mmax,
                transform=ccrs.PlateCarree(),
                cmap=colormap,
            )
        else:
            raise ValueError("Inknown plot type {}".format(ptype))

        # ax.coastlines(resolution = '50m',lw=0.5)
        ax[ind].add_feature(
            cfeature.GSHHSFeature(levels=[1], scale="low", facecolor="lightgray")
        )
        if titles:
            titles = titles.copy()
            ax[ind].set_title(titles.pop(0), fontweight='bold')

    for delind in range(ind + 1, len(ax)):
        fig.delaxes(ax[delind])


    gl = ax[ind].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='-')

    gl.xlabels_bottom = False
        
    textrsmd='rmsd='+str(round(rmsdval,3))
    textbias='bias='+str(round(mdval,3))
    props = dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.5)
    ax[ind].text(0.02, 0.4, textrsmd, transform=ax[ind].transAxes, fontsize=13,
        verticalalignment='top', bbox=props, zorder=4)
    ax[ind].text(0.02, 0.3, textbias, transform=ax[ind].transAxes, fontsize=13,
        verticalalignment='top', bbox=props, zorder=4)

    cbar_ax_abs = fig.add_axes([0.15, 0.10, 0.7, 0.05])
    cbar_ax_abs.tick_params(labelsize=12)
    cb = fig.colorbar(image, cax=cbar_ax_abs, orientation='horizontal',ticks=levels)
    cb.set_label(label=units, size='14')
    cb.ax.tick_params(labelsize='12')
    for label in cb.ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    return ax,latreg

for depth in [0,100,1000,4000]:

    if depth==0:
        title2 = "Surface salinity bias vs. PHC3"
        ofile=str(depth)+"m-salt-phc3"
    elif depth==100:
        title2 = "100 meter salinity bias vs. PHC3"
        ofile=str(depth)+"m-salt-phc3"
    elif depth==1000:
        title2 = "1000 meter salinity bias vs. PHC3"
        ofile=str(depth)+"m-salt-phc3"
    elif depth==4000:
        title2 = "4000 meter salinity bias vs. PHC3"
        ofile=str(depth)+"m-salt-phc3"


    if input_names is None:
        input_names = []
        for run in input_paths:
            run = os.path.join(run, '')
            input_names.append(run.split('/')[-2])

    mesh = pf.load_mesh(meshpath, abg=abg, 
                        usepickle=True, usejoblib=False)

    from pprint import pprint
    pprint(vars(mesh))

    plotds = OrderedDict()
    data_reference = pf.get_data(reference_path, variable, reference_years, mesh, depth = depth, how=how, compute=True, silent=True)
    plotds[depth] = {}
    for exp_path, exp_name  in zip(input_paths, input_names):
        data_test      = pf.get_data(exp_path, variable, years, mesh, depth = depth, how=how, compute=True, silent=True)
        data_difference= data_test - data_reference
        title = exp_name+" - "+reference_name
        plotds[depth][title] = {}
        plotds[depth][title]['data'] = data_difference
        if (data_difference.max() == data_difference.min() == 0):
            plotds[depth][title]['nodiff'] = True
        else:
            plotds[depth][title]['nodiff'] = False

    mesh_data = Dataset(meshpath+'/'+mesh_file)
    wgts=mesh_data['cell_area']
    rmsdval = sqrt(mean_squared_error(data_test,data_reference,sample_weight=wgts))
    mdval = md(data_test,data_reference,wgts)

    sfmt = ticker.ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((-3, 4))

    levels = [-2.0,-1.0,-.6,-.2,-.1,.1,.2,.6,1.0,2.0]

    figsize=(6, 4.5)
    dpi=300

    plot_data, plot_names = data_to_plot(plotds, depth)
    if not plot_data:
        print('There is no difference between fields')
        identical = True
    else:
        identical = False

    if len(plot_data) == 1:
        plot_data = plot_data[0]
        plot_names = plot_names[0]


    if not identical:
        plot(mesh, 
            plot_data,
            rowscol=rowscol,
            mapproj=mapproj,
            cmap='PuOr_r', 
            levels=levels,
            figsize = figsize, 
            box=bbox, 
            res = res,
            units = units,
            titles = title2,
            rmsdval = rmsdval,
            mdval = mdval);


    if ofile is not None:
        plt.savefig(out_path+ofile, dpi=dpi, bbox_inches='tight')
        os.system(f'convert {ofile} -trim {ofile}_trimmed.png')
        os.system(f'mv {ofile}_trimmed.png {ofile}')


# # ENSO

# In[ ]:


# parameters cell
variable = 'sst'
input_paths = [spinup_path+'/fesom/']
years = range(spinup_start, spinup_end+1)
figsize=(10, 5)


# load mesh and data
mesh = pf.load_mesh(meshpath, abg=abg, 
                    usepickle=True, usejoblib=False)

data_raw = pf.get_data(input_paths[0], 'sst', years, mesh, how=None, compute=False, silent=True)

model_lon = mesh.x2
model_lon = np.where(model_lon < 0, model_lon+360, model_lon)
model_lat = mesh.y2

# TODO better to detrend and use monthly?
data = []
steps_per_year=int(np.shape(data_raw)[0]/len(years))
for y in tqdm(range(len(years))):
    data.append(np.mean(data_raw[y*steps_per_year:y*steps_per_year+steps_per_year-1,:],axis=0))
data = np.asarray(data)
data = signal.detrend(data)

# Detrend linearly to remove forcing or spinup induced trends
# TODO: probably better to detrend with something like a 50 year running mean
data_raw = signal.detrend(data_raw)

# Reshape to add monthly time axis
data_raw_reshape = data_raw.reshape(data_raw.shape[0]//12,data_raw.shape[1], 12)

# Calculate seasonal cycle
data_season_cycle = np.mean(data_raw_reshape,axis=0)

# Repeat seasonal cycle
data_season_cycle_repeat = np.repeat(data_season_cycle[np.newaxis,...],np.shape(data_raw_reshape)[0],axis=0)

# Reshape into original format
data_season_cycle_repeat_reshape = data_season_cycle_repeat.reshape(np.shape(data_raw))

# Remove seasonal cycle from data
data = data_raw - data_season_cycle_repeat_reshape

#select ENSO region
lon = np.linspace(110, 290, 181)
lat = np.linspace(-46, 46, 92)
lon2, lat2 = np.meshgrid(lon, lat)

# interpolate data onto regular grid
sst = []
points = np.vstack((model_lon, model_lat)).T
for t in tqdm(range(0, np.shape(data)[0])):
    nn_interpolation = NearestNDInterpolator(points, data[t,:])
    sst.append(nn_interpolation((lon2, lat2)))
sst=np.asarray(sst)

# Create an EOF solver to do the EOF analysis. Square-root of cosine of
# latitude weights are applied before the computation of EOFs.
coslat = np.cos(np.deg2rad(lat))
wgts = np.sqrt(coslat)[..., np.newaxis]
solver = Eof(sst, weights=wgts)

# Retrieve the leading EOF, expressed as the correlation between the leading
# PC time series and the input SST anomalies at each grid point, and the
# leading PC time series itself.
eof1_corr = solver.eofsAsCorrelation(neofs=1)
eof1 = solver.eofs(neofs=1, eofscaling=0)

eof_abs = solver.eofs(neofs=1)
pc1 = -np.squeeze(solver.pcs(npcs=1, pcscaling=1))

# Sign of correlation is arbitrary, but plot should be positive
if np.mean(eof1_corr) < 0:
    eof1_corr = eof1_corr
if np.mean(eof1) < 0:
    eof1 = -eof1
    
title='EOF1 as correlation between PC1 time series and the input data'
    
    
fig =plt.figure(figsize=(9,5.56))
colormap=plt.cm.PuOr_r

# Plot the leading EOF in the Pacific domain.
clevs = np.linspace(-1, 1, 21)
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=190))
fill = ax.contourf(lon2, lat2, eof1_corr.squeeze(), clevs,
                   transform=ccrs.PlateCarree(), cmap=colormap,zorder=-1)
line_colors = ['black' for l in fill.levels]
con = ax.contour(lon2, lat2, eof1_corr.squeeze(), clevs, colors=line_colors, linewidths=0.3,
                   transform=ccrs.PlateCarree(),zorder=-1)
ax.add_feature(cfeature.LAND, color='lightgrey')
ax.add_feature(cfeature.COASTLINE)


box = 'Nino34'
lon_min=190 #(-170+360)
lon_max=240 #(-120+360)
lat_min=-5
lat_max= 5

plt.title('Box: '+box, fontsize=13,fontweight="bold")
plt.text(lon_min-202,lat_min-2,str(lon_min)+'/'+str(lat_min)+'°')
plt.text(lon_min-202,lat_max-2,str(lon_min)+'/'+str(lat_max)+'°')
plt.text(lon_max-189,lat_max-2,str(lon_max)+'/'+str(lat_max)+'°')
plt.text(lon_max-189,lat_min-2,str(lon_max)+'/'+str(lat_min)+'°')


ax.add_patch(mpatches.Rectangle(xy=[lon_min, lat_min], width=lon_max-lon_min, height=lat_max-lat_min,
                                    facecolor='none',
                                    #alpha=0.5,
                                    edgecolor='Black',
                                    lw='2',
                                    transform=ccrs.PlateCarree(),
                                    zorder=6)
            )

textstr='Nino34 box'
props = dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7)

ax.text(0.506, 0.65, textstr, transform=ax.transAxes, fontsize=13,
        verticalalignment='top', bbox=props, zorder=4)

cbar_ax_abs = fig.add_axes([0.15, 0.1, 0.7, 0.05])
cbar_ax_abs.tick_params(labelsize=12)

ax.set_title(title,fontweight="bold")

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
              linewidth=1, color='gray', alpha=0.2, linestyle='-')
gl.xlabels_bottom = False

cb = fig.colorbar(fill, cax=cbar_ax_abs, orientation='horizontal')
#cb.set_label(label=unit, size='14')
cb.ax.tick_params(labelsize='12')
cb.set_label('Correlation coefficient', fontsize=12)

ofile='HIST'

print(ofile)
if ofile is not None:
    ofile_long = f"{ofile}_enso_eof_corr.png"
    plt.savefig(f"{out_path+ofile_long}", dpi=300)
    os.system(f'convert {ofile_long} -trim {ofile_long}_trimmed.png')
    os.system(f'mv {ofile_long}_trimmed.png {ofile_long}')


    
# interpolate data onto regular grid
# select Nino index region
box = 'Nino34'


if box == 'Nino12':
    lon_min=270 #(-90+360) 
    lon_max=280 #(-80+360) 
    lat_min=-10
    lat_max= 0
elif box == 'Nino3':
    lon_min=210 #(-150+360)
    lon_max=270 #(-90+360)
    lat_min=-5
    lat_max= 5
elif box == 'Nino34':
    lon_min=190 #(-170+360)
    lon_max=240 #(-120+360)
    lat_min=-5
    lat_max= 5
elif box == 'Nino4':
    lon_min=160 
    lon_max=210 #(-150+360)
    lat_min=-5
    lat_max= 5

    
    
lon = np.linspace(lon_min, lon_max, lon_max-lon_min)
lat = np.linspace(lat_min, lat_max, lat_max-lat_min)
lon2, lat2 = np.meshgrid(lon, lat)

sst = []
points = np.vstack((model_lon, model_lat)).T
for t in tqdm(range(0, np.shape(data_raw)[0])):
    nn_interpolation = NearestNDInterpolator(points, data[t,:])
    sst.append(nn_interpolation((lon2, lat2)))
sst=np.asarray(sst)
sst_area_mean = np.mean(np.mean(sst,axis=2),axis=1)
sst_nino = sst_area_mean.reshape(len(sst_area_mean)//12, 12)
sst_nino_ano = sst_nino - np.mean(sst_nino)

obs_path = observation_path+'/hadisst2/box'



from cdo import *   # python version
cdo = Cdo()
obs_raw = cdo.copy(input=str(obs_path),returnArray='sst')
del cdo
from scipy import signal
obs_raw = obs_raw[0:1812]

# Detrend linearly to remove forcing or spinup induced trends
# TODO: probably better to detrend with something like a 50 year running mean
data_raw = signal.detrend(data_raw)
#obs_raw = signal.detrend(obs_raw)

# Reshape to add monthly time axis
data_raw_reshape = data_raw.reshape(data_raw.shape[0]//12,data_raw.shape[1], 12)
obs_raw_reshape = obs_raw.reshape(obs_raw.shape[0]//12,obs_raw.shape[1],obs_raw.shape[2], 12)

# Calculate seasonal cycle
data_season_cycle = np.mean(data_raw_reshape,axis=0)
obs_season_cycle = np.mean(obs_raw_reshape,axis=0)

# Repeat seasonal cycle
data_season_cycle_repeat = np.repeat(data_season_cycle[np.newaxis,...],np.shape(data_raw_reshape)[0],axis=0)
obs_season_cycle_repeat = np.repeat(obs_season_cycle[np.newaxis,...],np.shape(obs_raw_reshape)[0],axis=0)

# Reshape into original format
data_season_cycle_repeat_reshape = data_season_cycle_repeat.reshape(np.shape(data_raw))
obs_season_cycle_repeat_reshape = obs_season_cycle_repeat.reshape(np.shape(obs_raw))

# Remove seasonal cycle from data
data = data_raw - data_season_cycle_repeat_reshape
obs = obs_raw - obs_season_cycle_repeat_reshape



def smooth3(x,beta):
    """ kaiser window smoothing """
    window_len=3
    beta=2
    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w = np.kaiser(window_len,beta)
    y = np.convolve(w/w.sum(),s,mode='valid')
    return y[1:len(y)-1]

obs_nino = obs.reshape(len(obs)//12, 12)
obs_nino_ano = obs_nino - np.mean(obs_nino)

# Seasonal smoothing
sst_nino_ano_smooth=smooth3(sst_nino_ano.flatten(),len(sst_nino_ano.flatten()))
obs_nino_ano_smooth=smooth3(obs_nino_ano.flatten(),len(obs_nino_ano.flatten()))

# Plot the leading PC time series.

plt.figure(figsize=figsize)
plt.plot(sst_nino_ano_smooth, color='black', linewidth=1) 
plt.axhline(0, color='k')
plt.title(historic_name+' '+box+' Index Time Series',fontweight="bold")
plt.xlabel('Month',fontsize=13)
plt.ylabel('°C',fontsize=13)
ax.tick_params(labelsize=13)

plt.ylim(-2.5, 2.5)
plt.axhline(y=1, color='grey', linestyle='--')
plt.axhline(y=-1, color='grey', linestyle='--')

months = np.arange(len(sst_nino_ano.flatten()))

plt.fill_between(months, sst_nino_ano_smooth, 0, where = (sst_nino_ano_smooth > 0), color='Orange',alpha=0.25)
plt.fill_between(months, sst_nino_ano_smooth, -0, where = (sst_nino_ano_smooth < -0), color='darkblue',alpha=0.25)

plt.fill_between(months, sst_nino_ano_smooth, 1, where = (sst_nino_ano_smooth > 1), color='Orange')
plt.fill_between(months, sst_nino_ano_smooth, -1, where = (sst_nino_ano_smooth < -1), color='darkblue')


if ofile is not None:
    ofile_long = f"{ofile}_"+box+"_enso_box_index.png"
    plt.savefig(f"{out_path+ofile_long}", dpi=dpi,bbox_inches='tight')
    os.system(f'convert {ofile_long} -trim {ofile_long}_trimmed.png')
    os.system(f'mv {ofile_long}_trimmed.png {ofile_long}')
    
    
# Plot the leading PC time series.

plt.figure(figsize=figsize)
plt.plot(obs_nino_ano_smooth, color='black', linewidth=1) 
plt.axhline(0, color='k')
plt.title('HadISST '+box+' Index Time Series',fontweight="bold")
plt.xlabel('Month',fontsize=13)
plt.ylabel('°C',fontsize=13)
ax.tick_params(labelsize=13)

plt.ylim(-2.5, 2.5)
plt.axhline(y=1, color='grey', linestyle='--')
plt.axhline(y=-1, color='grey', linestyle='--')

months = np.arange(len(obs_nino_ano.flatten()))

plt.fill_between(months, obs_nino_ano_smooth, 0, where = (obs_nino_ano_smooth > 0), color='Orange',alpha=0.25)
plt.fill_between(months, obs_nino_ano_smooth, -0, where = (obs_nino_ano_smooth < -0), color='darkblue',alpha=0.25)
plt.fill_between(months, obs_nino_ano_smooth, 1, where = (obs_nino_ano_smooth > 1), color='Orange')
plt.fill_between(months, obs_nino_ano_smooth, -1, where = (obs_nino_ano_smooth < -1), color='darkblue')


if ofile is not None:
    ofile_long = f"HadISST_"+box+"_enso_box_index.png"
    plt.savefig(f"{out_path+ofile_long}", dpi=dpi,bbox_inches='tight')
    os.system(f'convert {ofile_long} -trim {ofile_long}_trimmed.png')
    os.system(f'mv {ofile_long}_trimmed.png {ofile_long}')
    
    
    
# Obtain data
Ntotal = len(sst_nino_ano_smooth)
data = sst_nino_ano_smooth

# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('PuOr_r')

# Get the histogramp
nbins = 13
minbin = -3
maxbin = 3
bins = np.linspace(minbin,maxbin,nbins)

Y,X = np.histogram(data, bins=bins)
Y = (Y*100)/np.sum(Y)
x_span = X.max()-X.min()
corr=(x_span/nbins)/2
C = [cm(((x-X.min()+corr)/x_span)) for x in X]

fig, ax = plt.subplots(figsize=figsize)

plt.bar(X[:-1],Y,color=C,width=X[1]-X[0],edgecolor='black',align='edge')
plt.xlim((minbin, maxbin))
plt.title(historic_name+' '+box+' temperature anomaly distribution',fontweight="bold")
plt.ylabel("Occurance [%]",fontsize=13)
plt.xlabel("Temperature anomaly [°C]",fontsize=13)
ax.tick_params(labelsize=13)


if ofile is not None:
    ofile_long = f"{ofile}_"+box+"_enso_temperature_distribution.png"
    plt.savefig(f"{out_path+ofile_long}", dpi=dpi,bbox_inches='tight')
    os.system(f'convert {ofile_long} -trim {ofile_long}_trimmed.png')
    os.system(f'mv {ofile_long}_trimmed.png {ofile_long}')
    
    
    
# Obtain data
Ntotal = len(obs_nino_ano_smooth)
data = obs_nino_ano_smooth

# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('PuOr_r')

# Get the histogramp
nbins = 13
minbin = -3
maxbin = 3
bins = np.linspace(minbin,maxbin,nbins)

Y,X = np.histogram(data, bins=bins)
Y = (Y*100)/np.sum(Y)
x_span = X.max()-X.min()
corr=(x_span/nbins)/2
C = [cm(((x-X.min()+corr)/x_span)) for x in X]

fig, ax = plt.subplots(figsize=figsize)

plt.bar(X[:-1],Y,color=C,width=X[1]-X[0],edgecolor='black',align='edge')
plt.xlim((minbin, maxbin))
plt.title('HadISST '+box+' temperature anomaly distribution',fontweight="bold")
plt.ylabel("Occurance [%]",fontsize=13)
plt.xlabel("Temperature anomaly [°C]",fontsize=13)

if ofile is not None:
    ofile_long = f"HadISST_"+box+"_enso_temperature_distribution.png"
    plt.savefig(f"{out_path+ofile_long}", dpi=dpi,bbox_inches='tight')
    os.system(f'convert {ofile_long} -trim {ofile_long}_trimmed.png')
    os.system(f'mv {ofile_long}_trimmed.png {ofile_long}')
    
    

def smooth(x,beta):
    """ kaiser window smoothing """
    window_len=201
    beta=200
    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w = np.kaiser(window_len,beta)
    y = np.convolve(w/w.sum(),s,mode='valid')
    return y[100:len(y)-100]




f, Pxx_den = signal.periodogram(sst_nino_ano.flatten(),nfft=8000)
f_obs, Pxx_den_obs = signal.periodogram(obs_nino_ano.flatten(),nfft=8000)
#f_obsn, Pxx_den_obsn = signal.periodogram(obs_nino_ano.flatten(),nfft=250)

fig, ax = plt.subplots(figsize=figsize)

#ax.plot(f_obsn,Pxx_den_obsn/np.mean(Pxx_den_obsn),linewidth=1,color='orange',label='HadISST')

ax.semilogx(f_obs,smooth(Pxx_den_obs/np.mean(Pxx_den_obs),len(Pxx_den)),linewidth=2,color='darkblue',label='HadISST')
ax.semilogx(f,smooth(Pxx_den/np.mean(Pxx_den),len(Pxx_den)),linewidth=2,color='orange',label='AWI-CM3 HIST')


ax.set_xlim([0.0015, 0.1])
#ax.set_ylim([0.01, 25])
plt.xlabel('Frequency [Cycles/Month]',fontsize=13)
plt.ylabel('Normalized PSD',fontsize=13)
plt.legend(loc='upper left',fontsize=13)
#ax.set_xscale('log')
ax.tick_params(labelsize=13)

def twelve_over(x):
    """Vectorized 12/x, treating x==0 manually"""
    x = np.array(x*12).astype(float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = np.inf
    x[~near_zero] = 1 / x[~near_zero]
    return x

# the function "12/x" is its own inverse
inverse = twelve_over

#secax.set_xticks([50,12,8,6,5,4,3,2,1])
secax = ax.secondary_xaxis('top', functions=(twelve_over, inverse))
secax.set_xlabel('Period [Years]',fontsize=13)
secax.set_xlabel('Period [Years]',fontsize=13)
secax.xaxis.set_major_formatter(FormatStrFormatter("%1.f"))
secax.xaxis.set_minor_formatter(FormatStrFormatter("%1.f"))
secax.tick_params(axis='x', which='major', labelsize=11)
secax.tick_params(axis='x', which='minor', labelsize=11)

if ofile is not None:
    ofile_long = f"{ofile}_enso_"+box+"_box_norm_psd.png"
    plt.savefig(f"{out_path+ofile_long}", dpi=dpi,bbox_inches='tight')
    os.system(f'convert {ofile_long} -trim {ofile_long}_trimmed.png')
    os.system(f'mv {ofile_long}_trimmed.png {ofile_long}')


# # Historic and pi-control timeseries

# In[ ]:


# parameters cell
input_paths = [historic_path]
input_names = [historic_name]

ctrl_input_paths = [pi_ctrl_path]
ctrl_input_names = [pi_ctrl_name]

exps = list(range(historic_start, historic_end+1))

climatology_path = [observation_path+'/HadCRUT5/']
climatology_names = ['HadCRUT5']

figsize=(9,5.56)
var = '2t'

params = {'legend.fontsize': 'large',
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

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


# Load Data
def load_parallel(variable,path):
    data1 = cdo.yearmean(input='-fldmean '+str(path),returnArray=variable)
    return data1

data = OrderedDict()
v=var
paths = []
data[v] = []
for exp_path, exp_name  in zip(input_paths, input_names):

    datat = []
    t = []
    temporary = []
    for exp in tqdm(exps):

        path = exp_path+'/oifs/atm_remapped_1m_'+v+'_1m_'+f'{exp:04d}-{exp:04d}.nc'
        temporary = dask.delayed(load_parallel)(var,path)
        t.append(temporary)

    with ProgressBar():
        datat = dask.compute(t)
    data[v] = np.squeeze(datat)
    
    
ctrl_data = OrderedDict()
v=var
paths = []
ctrl_data[v] = []
for exp_path, exp_name  in zip(ctrl_input_paths, ctrl_input_names):

    datat = []
    t = []
    temporary = []
    for exp in tqdm(exps):

        path = exp_path+'/oifs/atm_remapped_1m_'+v+'_1m_'+f'{exp:04d}-{exp:04d}.nc'
        temporary = dask.delayed(load_parallel)(var,path)
        t.append(temporary)

    with ProgressBar():
        datat = dask.compute(t)
    ctrl_data[v] = np.squeeze(datat)
    
    
# extract data
hist = np.squeeze(np.vstack(np.squeeze(data[var]).flatten()-273.15))
pict = np.squeeze(np.vstack(np.squeeze(ctrl_data[var]).flatten()-273.15))
pict = pict[0:len(hist)]

hist = hist - np.mean(pict)
pict = pict - np.mean(pict)

years = np.arange(1850,len(hist)+1850,1)


# Load reference data
path=climatology_path[0]+'/HadCRUT.5.0.1.0.analysis.summary_series.global.annual.nc'
data_ref = cdo.copy(input=str(path),returnArray='tas_mean')
data_ref = np.squeeze(data_ref)
data_ref = data_ref[0:len(hist)]


# calculate linear regression
res = linregress(years,pict)

# correct historic simulation by lin regression slope
correction = np.arange(0,len(hist)*res.slope,res.slope)
correction = correction[0:len(hist)]
obs_correction = np.mean(data_ref[0:60])

hist_c = hist - correction + np.mean(correction)
pict = pict + np.mean(correction)
pict_c = pict - correction
data_ref = data_ref - obs_correction



fig, ax = plt.subplots(figsize=figsize)
# plot running mean
plt.plot(years,smooth(hist,len(hist)),color='orange')
plt.plot(years,smooth(pict,len(pict)),color='darkblue')

# plot raw data
plt.plot(years,hist,linewidth=0.5,color='orange')
plt.plot(years,pict,linewidth=0.5,color='darkblue')



# plot linear regression
x_vals = np.array((years[0],years[len(years)-1]))
y_vals = res.intercept + res.slope * x_vals
plt.plot(x_vals, y_vals+np.mean(correction), '--',color='darkblue')


ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='minor', length=4)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='minor', length=4)
ax.tick_params(direction='out', length=6, width=2, grid_alpha=0.5,labelright=True)
ax.yaxis.set_ticks_position('both')
ax = plt.gca()

plt.ylabel('Change in near surface (2m) air temperature $\Delta$T [C°]')
plt.xlabel('Year')

ax.legend([input_names[0], ctrl_input_names[0]])
plt.savefig(out_path+'T2M_hist-pict.png', dpi=dpi)



fig, ax = plt.subplots(figsize=figsize)

# plot running mean
plt.plot(years,smooth(data_ref,len(data_ref)),color='black')
plt.plot(years,smooth(pict_c,len(pict_c)),color='darkblue')
plt.plot(years,smooth(hist_c,len(hist_c)),color='orange')


# plot raw data
plt.plot(years,data_ref,linewidth=0.5,color='black')
plt.plot(years,pict_c,linewidth=0.5,color='darkblue')
plt.plot(years,hist_c,linewidth=0.5,color='orange')


ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='minor', length=4)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='minor', length=4)
ax.tick_params(direction='out', length=6, width=2, grid_alpha=0.5,labelright=True)
ax.yaxis.set_ticks_position('both')
ax = plt.gca()

plt.ylabel('Change in near surface (2m) air temperature $\Delta$T [C°]')
plt.xlabel('Year')

ax.legend([ climatology_names[0] ,ctrl_input_names[0],input_names[0]])
plt.savefig(out_path+'T2M_hist-pict_corrected.png', dpi=dpi)


# In[ ]:


# parameters cell
input_paths = [historic_path]
input_names = [historic_name]

ctrl_input_paths = [pi_ctrl_path]
ctrl_input_names = [pi_ctrl_name]

exps = list(range(historic_last25y_start, historic_last25y_end+1))


figsize=(6, 4.5)
dpi = 300
ofile = None
res = [720, 360]

data = OrderedDict()
data_ctrl = OrderedDict()

data_ctrl[historic_name] = {}
data_ctrl[pi_ctrl_name] = {}
data[historic_name] = {}
data[pi_ctrl_name] = {}

for var in ['precip','temp']:
    if var == 'precip':
        variables = ['lsp','cp']

        levels = [-15,-10,-7,-4,-2,-1,1,2,4,7,10,15]
        levels2 = [-30,-25,-20,-15,-10,-3,3,10,15,20,25,30]

        title = 'Precipitation anomaly'
        accumulation_factor=86400
        colormap=plt.cm.PuOr
        unit='%'
        
    elif var == 'temp':
        variables = ['2t']
        levels = [-5.0,-3.0,-2.0,-1.0,-.6,-.2,.2,.6,1.0,2.0,3.0,5.0]
        title = 'Near surface (2m) air temperature anomaly'
        accumulation_factor=1
        colormap=plt.cm.PuOr_r
        unit='°C'

    '''
    variable = ['SKT']
    levels = [-15,-8.0,-5.0,-3.0,-2.0,-1.0,-.6,-.2,.2,.6,1.0,2.0,3.0,5.0,8.0,15]
    title = 'Sea and land surface temperature anomaly'
    accumulation_factor=1
    colormap=plt.cm.PuOr_r
    unit='°C'
    '''
    '''
    variable = ['U10M']
    levels = [-1.0,-0.7,-0.5,-0.2,-0.1,0.1,.2,.5,0.7,1.0]
    title = '10m eastward wind anonmaly'
    accumulation_factor=1
    colormap=plt.cm.PuOr_r
    unit='m/s'
    '''
    '''
    variable = ['V10M']
    levels = [-3.0,-2.0,-1.0,-.6,-.2,.2,.6,1.0,2.0,3.0]
    title = '10m northward wind anonmaly'
    accumulation_factor=1
    colormap=plt.cm.PuOr_r
    unit='m/s'
    '''
    '''
    variable = ['EKE']
    levels = [-30,-20,-10,-6,-2,2,6,10,20,30]
    title = 'Eddy kinetic energy anamoaly'
    accumulation_factor=1
    colormap=plt.cm.PuOr_r
    unit='cm²/s²'
    '''

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

    # Calculate Root Mean Square Deviation (RMSD)
    def rmsd(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    # Mean Deviation
    def md(predictions, targets):
        return (predictions - targets).mean()


    # Load Data
    def load_parallel(variable,path):
        data1 = cdo.yearmean(input="-remapcon,r"+str(res[0])+"x"+str(res[1])+" "+str(path),returnArray=variable)
        return data1

    paths = []
    for exp_path, exp_name  in zip(input_paths, input_names):
        for v in variables:
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

    paths = []
    for exp_path, exp_name  in zip(ctrl_input_paths, ctrl_input_names):

        for v in variables:
            datat = []
            t = []
            temporary = []
            for exp in tqdm(exps):

                path = exp_path+'/oifs/atm_remapped_1m_'+v+'_1m_'+f'{exp:04d}-{exp:04d}.nc'
                temporary = dask.delayed(load_parallel)(v,path)
                t.append(temporary)

            with ProgressBar():
                datat = dask.compute(t)
            data_ctrl[exp_name][v] = np.squeeze(datat)
            
def resample(xyobs,n,m):
    xstar = []
    ystar = []
    for ni in range(n):
        r = rd.randrange(0, xyobs.shape[0])
        xstar.append(xyobs[r])
    for mi in range(m):
        r = rd.randrange(0, xyobs.shape[0])
        ystar.append(xyobs[r])
    xbarstar = np.mean(np.asarray(xstar),axis=0)
    ybarstar = np.mean(np.asarray(ystar),axis=0)
    t = xbarstar - ybarstar
    return t

def bootstrap(xyobs, data1, data2):
    tstarobs = np.asarray(data2 - data1)
    tstar = []
    ta = []
    pvalue = []
    n = xyobs.shape[0]//2
    m = xyobs.shape[0]//2
    B = 20000

    for bi in tqdm(range(B)):
        t = dask.delayed(resample)(xyobs,n,m)
        ta.append(t)
    with ProgressBar():
        tstar = dask.compute(ta)
    tstar = np.squeeze(np.asarray(tstar), axis = 0)
    pvalue = np.empty((tstarobs.shape[0],tstarobs.shape[1]))
    for lat in tqdm(range(0,tstarobs.shape[0])):
        for lon in range(0,tstarobs.shape[1]):
            p1 = tstar[:,lat,lon][tstar[:,lat,lon] >= tstarobs[lat,lon]].shape[0]/B
            p2 = tstar[:,lat,lon][tstar[:,lat,lon] >= -tstarobs[lat,lon]].shape[0]/B
            pvalue[lat,lon] = min(p1,p2)
    return pvalue


for var in ['precip','temp']:
    if var == 'precip':
        variables = ['lsp','cp']

        levels = [-15,-10,-7,-4,-2,-1,1,2,4,7,10,15]
        levels2 = [-30,-25,-20,-15,-10,-3,3,10,15,20,25,30]

        title = 'Precipitation anomaly'
        accumulation_factor=86400
        colormap=plt.cm.PuOr
        unit='mm/day'
        data_plot = (data[historic_name]['lsp']+data[historic_name]['cp'])*accumulation_factor
        data_ctrl_plot = (data_ctrl[pi_ctrl_name]['lsp']+data_ctrl[pi_ctrl_name]['cp'])*accumulation_factor

    elif var == 'temp':
        variables = ['2t']
        levels = [-5.0,-3.0,-2.0,-1.0,-.6,-.2,.2,.6,1.0,2.0,3.0,5.0]
        title = 'Near surface (2m) air temperature anomaly'
        accumulation_factor=1
        colormap=plt.cm.PuOr_r
        unit='°C'
        data_plot = data[historic_name]['2t']
        data_ctrl_plot = data_ctrl[pi_ctrl_name]['2t']

        
    # Bootstrap significance test
    xyobs = np.asarray(np.concatenate([np.squeeze(data_plot),np.squeeze(data_ctrl_plot)]))
    mean_hist=np.mean(np.squeeze(data_plot),axis=0)
    mean_pict=np.mean(np.squeeze(data_ctrl_plot),axis=0)

    pvalue = bootstrap(xyobs, mean_hist, mean_pict)
    data_sig = np.greater(pvalue, 0.025)

    lon = np.arange(0, 360, 0.5)
    lat = np.arange(-90, 90, 0.5)
    data_sig, lon = add_cyclic_point(data_sig, coord=lon)
    
    
    
    
    
    data_model = OrderedDict()
    data_model_mean = OrderedDict()
    
    for exp_name in [historic_name]:
        data_model[exp_name] = np.squeeze(data_plot) 
        data_model_mean[exp_name] = data_model[exp_name]
        if len(np.shape(data_model_mean[exp_name])) > 2:
            data_model_mean[exp_name] = np.mean(data_model_mean[exp_name],axis=0)

    for exp_name in [pi_ctrl_name]:
        data_model[exp_name] = np.squeeze(data_ctrl_plot) 
        data_model_mean[exp_name] = data_model[exp_name]
        if len(np.shape(data_model_mean[exp_name])) > 2:
            data_model_mean[exp_name] = np.mean(data_model_mean[exp_name],axis=0)

            
            
    print(np.shape(data_model_mean[exp_name]))

    for exp_name in [historic_name, pi_ctrl_name]:
        lon = np.arange(0, 360, 0.5)
        lat = np.arange(-90, 90, 0.5)
        data_model_mean[exp_name], lon = add_cyclic_point(data_model_mean[exp_name], coord=lon)

    print(np.shape(data_model_mean[exp_name]))


    #rmsdval = rmsd(data_model_mean[exp_name],data_reanalysis_mean)
    #mdval = md(data_model_mean[exp_name],data_reanalysis_mean)


    nrows, ncol = 1, 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=figsize)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    i = 0

    axes[i]=plt.subplot(nrows,ncol,i+1,projection=ccrs.PlateCarree())
    axes[i].add_feature(cfeature.COASTLINE,zorder=3)

    if var == 'precip':
        imf=plt.contourf(lon, lat, (data_model_mean[historic_name]-
                         data_model_mean[pi_ctrl_name])/data_model_mean[pi_ctrl_name]*100, cmap=colormap, 
                         levels=levels2, extend='both',
                         transform=ccrs.PlateCarree(),zorder=1)
    else: 
        imf=plt.contourf(lon, lat, data_model_mean[historic_name]-
                         data_model_mean[pi_ctrl_name], cmap=colormap, 
                         levels=levels, extend='both',
                         transform=ccrs.PlateCarree(),zorder=1)

    plt.rcParams['hatch.linewidth']=0.15
    cs = plt.contourf(lon, lat, data_sig, 3 , hatches=['\\\\\\', ''],  alpha=0)


    axes[i].set_xlabel('Simulation Year')
    axes[i].set_title(title,fontweight="bold")
    plt.tight_layout() 
    gl = axes[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='-')
    gl.xlabels_bottom = False

    if var == 'temp':
        plt.axhline(y=65, color='black', linestyle='-',alpha=.4)

        # Calculate AAI
        arctic_mean_pict=np.mean(data_model_mean[pi_ctrl_name][310:,:])
        arctic_mean_hist=np.mean(data_model_mean[historic_name][310:,:])
        glob_mean_pict=np.mean(data_model_mean[pi_ctrl_name][180,:])
        glob_mean_hist=np.mean(data_model_mean[historic_name][180,:])
        Arctic_Amplification_Index = (arctic_mean_hist-arctic_mean_pict)/(glob_mean_hist-glob_mean_pict)
        print("AAI:",Arctic_Amplification_Index)
        #plt.text(185, 90, 'AAI:')
        #plt.text(185, 70, str(round(Arctic_Amplification_Index,2)))
        props = dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7)
        textstr='AAI:'+str(round(Arctic_Amplification_Index,2))
        axes[i].text(0.86, 0.98, textstr, transform=axes[i].transAxes, fontsize=13,
            verticalalignment='top', bbox=props, zorder=4)


    cbar_ax_abs = fig.add_axes([0.15, 0.11, 0.7, 0.05])
    cbar_ax_abs.tick_params(labelsize=12)
    cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal',ticks=levels)
    cb.set_label(label=unit, size='14')
    cb.ax.tick_params(labelsize='12')
    #plt.text(5, 168, r'rmsd='+str(round(rmsdval,3)))
    #plt.text(-7.5, 168, r'bias='+str(round(mdval,3)))



    #for label in cb.ax.xaxis.get_ticklabels()[::2]:
    #    label.set_visible(False)

    ofile=var+'_hist-pict.png'

    print(ofile)
    if ofile is not None:
        plt.savefig(out_path+ofile, dpi=dpi,bbox_inches='tight')
        os.system(f'convert {ofile} -trim {ofile}_trimmed.png')
        os.system(f'mv {ofile}_trimmed.png {ofile}')


# In[ ]:


for var in ['precip']:
    if var == 'precip':
        variables = ['lsp','cp']

        levels = [-15,-10,-7,-4,-2,-1,1,2,4,7,10,15]
        levels2 = [-30,-25,-20,-15,-10,-3,3,10,15,20,25,30]

        title = 'Precipitation anomaly'
        accumulation_factor=86400
        colormap=plt.cm.PuOr
        unit='mm/day'
        data_plot = (data[historic_name]['lsp']+data[historic_name]['cp'])*accumulation_factor
        data_ctrl_plot = (data_ctrl[pi_ctrl_name]['lsp']+data_ctrl[pi_ctrl_name]['cp'])*accumulation_factor

    elif var == 'temp':
        variables = ['2t']
        levels = [-5.0,-3.0,-2.0,-1.0,-.6,-.2,.2,.6,1.0,2.0,3.0,5.0]
        title = 'Near surface (2m) air temperature anomaly'
        accumulation_factor=1
        colormap=plt.cm.PuOr_r
        unit='°C'
        data_plot = data[historic_name]['2t']
        data_ctrl_plot = data_ctrl[pi_ctrl_name]['2t']

        
    # Bootstrap significance test
    xyobs = np.asarray(np.concatenate([np.squeeze(data_plot),np.squeeze(data_ctrl_plot)]))
    mean_hist=np.mean(np.squeeze(data_plot),axis=0)
    mean_pict=np.mean(np.squeeze(data_ctrl_plot),axis=0)

    pvalue = bootstrap(xyobs, mean_hist, mean_pict)
    data_sig = np.greater(pvalue, 0.025)

    lon = np.arange(0, 360, 0.5)
    lat = np.arange(-90, 90, 0.5)
    data_sig, lon = add_cyclic_point(data_sig, coord=lon)
    
    
    
    
    
    data_model = OrderedDict()
    data_model_mean = OrderedDict()
    
    for exp_name in [historic_name]:
        data_model[exp_name] = np.squeeze(data_plot) 
        data_model_mean[exp_name] = data_model[exp_name]
        if len(np.shape(data_model_mean[exp_name])) > 2:
            data_model_mean[exp_name] = np.mean(data_model_mean[exp_name],axis=0)

    for exp_name in [pi_ctrl_name]:
        data_model[exp_name] = np.squeeze(data_ctrl_plot) 
        data_model_mean[exp_name] = data_model[exp_name]
        if len(np.shape(data_model_mean[exp_name])) > 2:
            data_model_mean[exp_name] = np.mean(data_model_mean[exp_name],axis=0)

            
            
    print(np.shape(data_model_mean[exp_name]))

    for exp_name in [historic_name, pi_ctrl_name]:
        lon = np.arange(0, 360, 0.5)
        lat = np.arange(-90, 90, 0.5)
        data_model_mean[exp_name], lon = add_cyclic_point(data_model_mean[exp_name], coord=lon)

    print(np.shape(data_model_mean[exp_name]))


    #rmsdval = rmsd(data_model_mean[exp_name],data_reanalysis_mean)
    #mdval = md(data_model_mean[exp_name],data_reanalysis_mean)


    nrows, ncol = 1, 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=figsize)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    i = 0

    axes[i]=plt.subplot(nrows,ncol,i+1,projection=ccrs.PlateCarree())
    axes[i].add_feature(cfeature.COASTLINE,zorder=3)

    if var == 'precip':
        imf=plt.contourf(lon, lat, (data_model_mean[historic_name]-
                         data_model_mean[pi_ctrl_name])/data_model_mean[pi_ctrl_name]*100, cmap=colormap, 
                         levels=levels2, extend='both',
                         transform=ccrs.PlateCarree(),zorder=1)
    else: 
        imf=plt.contourf(lon, lat, data_model_mean[historic_name]-
                         data_model_mean[pi_ctrl_name], cmap=colormap, 
                         levels=levels, extend='both',
                         transform=ccrs.PlateCarree(),zorder=1)

    plt.rcParams['hatch.linewidth']=0.15
    cs = plt.contourf(lon, lat, data_sig, 3 , hatches=['\\\\\\', ''],  alpha=0)


    axes[i].set_xlabel('Simulation Year')
    axes[i].set_title(title,fontweight="bold")
    plt.tight_layout() 
    gl = axes[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='-')
    gl.xlabels_bottom = False

    if var == 'temp':
        plt.axhline(y=65, color='black', linestyle='-',alpha=.4)

        # Calculate AAI
        arctic_mean_pict=np.mean(data_model_mean[pi_ctrl_name][310:,:])
        arctic_mean_hist=np.mean(data_model_mean[historic_name][310:,:])
        glob_mean_pict=np.mean(data_model_mean[pi_ctrl_name][180,:])
        glob_mean_hist=np.mean(data_model_mean[historic_name][180,:])
        Arctic_Amplification_Index = (arctic_mean_hist-arctic_mean_pict)/(glob_mean_hist-glob_mean_pict)
        print("AAI:",Arctic_Amplification_Index)
        #plt.text(185, 90, 'AAI:')
        #plt.text(185, 70, str(round(Arctic_Amplification_Index,2)))
        props = dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7)
        textstr='AAI:'+str(round(Arctic_Amplification_Index,2))
        axes[i].text(0.86, 0.98, textstr, transform=axes[i].transAxes, fontsize=13,
            verticalalignment='top', bbox=props, zorder=4)


    cbar_ax_abs = fig.add_axes([0.15, 0.11, 0.7, 0.05])
    cbar_ax_abs.tick_params(labelsize=12)
    if var == 'precip':
        cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal',ticks=levels2)
    else:
        cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal',ticks=levels)
    cb.set_label(label=unit, size='14')
    cb.ax.tick_params(labelsize='12')
    #plt.text(5, 168, r'rmsd='+str(round(rmsdval,3)))
    #plt.text(-7.5, 168, r'bias='+str(round(mdval,3)))



    #for label in cb.ax.xaxis.get_ticklabels()[::2]:
    #    label.set_visible(False)

    ofile=var+'_hist-pict.png'

    print(ofile)
    if ofile is not None:
        plt.savefig(out_path+ofile, dpi=dpi,bbox_inches='tight')
        os.system(f'convert {ofile} -trim {ofile}_trimmed.png')
        os.system(f'mv {ofile}_trimmed.png {ofile}')


# In[ ]:


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


# # Calculate Meridional Overturning Circulation (MOC) Profile
# 
# Use for the calculation of the Meridional Overturning Circulation (MOC) the equation for the calculation of the "Pseudostreamfunction". Condition for the calculation of the regional MOC (i.e AMOC, PMOC, IMOC) is that the domain over which the caluclation is carried out, is approximately sorounded by a coast (Bering Strait can be accouted as coast its just 30m deep). Since Atlantic, Pacific and Indian Ocean have no southern coastal boundary the AMOC and PMOC can just be calculated until -30°S and the meridional cumulativ integration has to be carried out from North to South instead South to North which leads to an additional minus sign in the calcualtion (see: sub_fesom_moc.py, line:137)
# $${\int_E^W w(x',y,z) dx' = {{\partial\Psi} \over {\partial y}}}$$
# $$ \textrm{GMOC:} ~~~  {\Psi(y,z) = \int_S^N {\int_E^W w(x',y',z) \cdot dx'} dy'} ~~$$
# $$ \textrm{AMOC:} ~~~  {\Psi(y,z) = -\int_N^{-30S^\circ} {\int_E^W w(x',y',z) \cdot dx'} dy'} $$
# $$ \textrm{PMOC:} ~~~  {\Psi(y,z) = -\int_{Bering Strait} ^{-30S^\circ} {\int_E^W w(x',y',z) \cdot dx'} dy'} $$

# In[ ]:


#%%prun -s cumulative -q -l 100 -D profile.bin
#____________________________________________________________________________________________________
# load vertical velocity data
data1 		 	= fesom_data(inputarray) 
data1.var 		= 'w'
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
data1 		 	= fesom_data(inputarray) 
data1.var 		= 'w'
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
data1 	   = fesom_data(inputarray) 
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
    datayr.year		= [year,year]
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
data1 	   = fesom_data(inputarray) 
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
    datayr.year		= [year,year]
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

