# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

SCRIPT_NAME = os.path.basename(__file__)  # Get the current script name

print(SCRIPT_NAME)

# Mark as started
update_status(SCRIPT_NAME, " Started")


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

batch_size = 20

for exp_path, exp_name in zip(input_paths, input_names):

    data_batches = []
    for i in range(0, len(exps), batch_size):
        datat = []
        t = []

        batch = exps[i:i+batch_size]  # Process in chunks of 20
        for exp in tqdm(batch):
            path = f"{exp_path}/oifs/atm_remapped_1m_{v}_1m_{exp:04d}-{exp:04d}.nc"
            temporary = dask.delayed(load_parallel)(var, path)
            t.append(temporary)

        with ProgressBar():
            datat = dask.compute(*t, scheduler='threads')

        data_batches.extend(datat)  # Collect results

    data[v] = np.squeeze(data_batches)
    
    
ctrl_data = OrderedDict()
v=var
paths = []
ctrl_data[v] = []
for exp_path, exp_name  in zip(ctrl_input_paths, ctrl_input_names):

    data_batches = []
    for i in range(0, len(exps), batch_size):
        datat = []
        t = []

        batch = exps[i:i+batch_size]  # Process in chunks of 20
        for exp in tqdm(batch):
            path = f"{exp_path}/oifs/atm_remapped_1m_{v}_1m_{exp:04d}-{exp:04d}.nc"
            temporary = dask.delayed(load_parallel)(var, path)
            t.append(temporary)

        with ProgressBar():
            datat = dask.compute(*t, scheduler='threads')

        data_batches.extend(datat)  # Collect results

    ctrl_data[v] = np.squeeze(data_batches)
    
    
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
plt.savefig(out_path+'T2M_hist-pict.png', dpi=dpi,bbox_inches='tight')



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
plt.savefig(out_path+'T2M_hist-pict_corrected.png', dpi=dpi,bbox_inches='tight')

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
                datat = dask.compute(*t, scheduler='threads')
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
                datat = dask.compute(*t, scheduler='threads')
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
    B = 500

    for bi in tqdm(range(B)):
        t = dask.delayed(resample)(xyobs,n,m)
        ta.append(t)
    with ProgressBar():
        tstar = dask.compute(ta, scheduler='threads')
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
    fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    i = 0

    axes[i].add_feature(cfeature.COASTLINE, zorder=3)

    if var == 'precip':
        imf=axes[i].contourf(lon, lat, (data_model_mean[historic_name]-
                         data_model_mean[pi_ctrl_name])/data_model_mean[pi_ctrl_name]*100, cmap=colormap, 
                         levels=levels2, extend='both',
                         transform=ccrs.PlateCarree(),zorder=1)
    else: 
        imf=axes[i].contourf(lon, lat, data_model_mean[historic_name]-
                         data_model_mean[pi_ctrl_name], cmap=colormap, 
                         levels=levels, extend='both',
                         transform=ccrs.PlateCarree(),zorder=1)

    plt.rcParams['hatch.linewidth'] = 0.15
    cs = axes[i].contourf(lon, lat, data_sig, 3, hatches=['\\\\\\', ''], alpha=0)

    axes[i].set_title(title, fontweight="bold")

    # Hide the borders and tick labels
    axes[i].set_xticks([])
    axes[i].set_yticks([])

    gl = axes[i].gridlines(draw_labels=True,
                           linewidth=1, color='gray', alpha=0.2, linestyle='-')
    gl.xlabels_bottom = False
    gl.bottom_labels = False

    if var == 'temp':
        plt.axhline(y=65, color='black', linestyle='-', alpha=.4)

        # Calculate AAI
        arctic_mean_pict = np.mean(data_model_mean[pi_ctrl_name][310:, :])
        arctic_mean_hist = np.mean(data_model_mean[historic_name][310:, :])
        glob_mean_pict = np.mean(data_model_mean[pi_ctrl_name][180, :])
        glob_mean_hist = np.mean(data_model_mean[historic_name][180, :])
        Arctic_Amplification_Index = (arctic_mean_hist - arctic_mean_pict) / (glob_mean_hist - glob_mean_pict)
        print("AAI:", Arctic_Amplification_Index)
        props = dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7)
        textstr = 'AAI:' + str(round(Arctic_Amplification_Index, 2))
        axes[i].text(0.86, 0.98, textstr, transform=axes[i].transAxes, fontsize=10,
                     verticalalignment='top', bbox=props, zorder=4)

    cbar_ax_abs = fig.add_axes([0.15, 0.15, 0.7, 0.05])
    cbar_ax_abs.tick_params(labelsize=12)
    if var == 'precip':
        cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal', ticks=levels2)
    else:
        cb = fig.colorbar(imf, cax=cbar_ax_abs, orientation='horizontal', ticks=levels)
    cb.set_label(label=unit, size='14')
    cb.ax.tick_params(labelsize='12')

    ofile = var + '_hist-pict.png'
    print(ofile)
    if ofile is not None:
        plt.savefig(out_path + ofile, dpi=dpi, transparent=True)

# Mark as completed
update_status(SCRIPT_NAME, " Completed")
