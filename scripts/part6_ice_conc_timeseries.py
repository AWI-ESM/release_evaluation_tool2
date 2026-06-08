# Add the parent directory to sys.path and load config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bg_routines.config_loader import *

SCRIPT_NAME = os.path.basename(__file__)  # Get the current script name

print(SCRIPT_NAME)

# Mark as started
update_status(SCRIPT_NAME, " Started")


def _read_seasonal_extents(datapath, str_id, y, mesh):
    """Return (extent_N_march, extent_S_march, extent_N_sep, extent_S_sep)
    for one year, handling both 12-record monthly and ~365-record daily
    fesom output. The original records=[2]/records=[8] indexing only
    works for monthly files; for daily output it selects Jan 3 / Jan 9
    and collapses summer and winter into the same value (the bug seen
    from year ~5910 onward in PI_wisofix_c)."""
    fpath = f"{datapath}/{str_id}.fesom.{y}.nc"
    with xr.open_dataset(fpath, decode_times=True, use_cftime=True) as ds:
        time_dim = 'time' if 'time' in ds.dims else list(ds.dims)[0]
        ds_mar = ds.sel({time_dim: ds[time_dim].dt.month == 3}).mean(dim=time_dim)
        ds_sep = ds.sel({time_dim: ds[time_dim].dt.month == 9}).mean(dim=time_dim)
        arr_mar = ds_mar[str_id].values[np.newaxis, :]
        arr_sep = ds_sep[str_id].values[np.newaxis, :]
    return (
        pf.ice_ext(arr_mar, mesh, hemisphere="N"),
        pf.ice_ext(arr_mar, mesh, hemisphere="S"),
        pf.ice_ext(arr_sep, mesh, hemisphere="N"),
        pf.ice_ext(arr_sep, mesh, hemisphere="S"),
    )


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
        ext_nm, ext_sm, ext_ns, ext_ss = _read_seasonal_extents(datapath, str_id, y, mesh)
        extent_north_march.append(ext_nm)
        extent_south_march.append(ext_sm)
        extent_north_sep.append(ext_ns)
        extent_south_sep.append(ext_ss)
          
    # Map historic years onto the model timeline so the plot reads as
    # one chronological sequence: spinup ... pi_ctrl, with historic
    # overlaying pi_ctrl as a separate forced branch from the same start.
    # Without this the historic run (e.g. 1850-2019) and spinup
    # (2001-5830) end up on disjoint x-axes covering 1850-6000.
    # And when the spinup ends well before pi_ctrl_start (e.g. AWI-ESM3-VEG-HR
    # with spinup ending 1679 but historic / pi_ctrl starting 1850), shift the
    # spinup forward so it lands immediately before pi_ctrl_start with no gap.
    if exp == historic_name:
        offset = pi_ctrl_start - year_start
        years = np.linspace(year_start + offset, year_end + offset,
                            year_end - year_start + 1)
    elif exp == spinup_name:
        spinup_shift = max(0, pi_ctrl_start - 1 - spinup_end)
        years = np.linspace(year_start + spinup_shift, year_end + spinup_shift,
                            year_end - year_start + 1)
    else:
        years = np.linspace(year_start, year_end, year_end - year_start + 1)

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
ax1.set_ylabel('Sea ice extent [$10^6$ km$^2$]', fontsize=17)
ax1.set_xlabel('Year', fontsize=17)

ax1.yaxis.grid(color='gray', linestyle='dashed')

# Place the divider where the spinup was visually shifted to end, not at
# the raw spinup_end. For configs where spinup runs right up to pi_ctrl
# (e.g. LR-Spinup with spinup_end=1849, pi_ctrl_start=1824) the shift is
# 0 and _split = spinup_end as before; for HR-style configs where there
# is a gap, the divider sits at pi_ctrl_start - 1.
_split = spinup_end + max(0, pi_ctrl_start - 1 - spinup_end)
plt.axvline(x=_split,color='black',alpha=0.7,linewidth=3)
# Position the SPIN / HIST&PICT labels in *axis-relative* coordinates
# (0-1 along each axis) so they always stay inside the axis bounds
# regardless of the data x-range. Previously they were placed at
# _split+-10/40 in data coords, which for a 3-year smoke-test run sat
# ~40 years outside the data; bbox_inches='tight' then inflated the
# figure past matplotlib's 2**16-pixel Agg limit (Image size 116679x...).
ax1.text(0.97, 0.97, 'HIST & PICT', transform=ax1.transAxes,
         ha='right', va='top', fontsize=15)
ax1.text(0.03, 0.97, 'SPIN', transform=ax1.transAxes,
         ha='left', va='top', fontsize=15)

# Scale x-tick density to the timeline length: 50 yr majors / 10 yr minors
# work for the multi-millennium spinup but produce no ticks on a 3-yr run.
_xspan = abs(ax1.get_xlim()[1] - ax1.get_xlim()[0])
_majstep = max(1, int(_xspan / 10))
_minstep = max(1, int(_majstep / 5))
ax1.xaxis.set_major_locator(MultipleLocator(_majstep))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.tick_params(axis='both', which='minor', labelsize=12)

# For the minor ticks, use no labels; default NullFormatter.
ax1.xaxis.set_minor_locator(MultipleLocator(_minstep))

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
        year_end   = pi_ctrl_end-1
    elif exp == historic_name: 
        datapath   = historic_path+'/fesom'
        year_start = historic_start
        year_end   = historic_end-1
    elif exp == spinup_name: 
        datapath   = spinup_path+'/fesom'
        year_start = spinup_start
        year_end   = spinup_end-1
 
    extent_north_march = []
    extent_south_march = []
    extent_north_sep = []
    extent_south_sep = []       
        
    for y in tqdm(range(year_start, year_end)):
        ext_nm, ext_sm, ext_ns, ext_ss = _read_seasonal_extents(datapath, str_id, y, mesh)
        extent_north_march.append(ext_nm)
        extent_south_march.append(ext_sm)
        extent_north_sep.append(ext_ns)
        extent_south_sep.append(ext_ss)
          
    # Map historic years onto the model timeline so the plot reads as
    # one chronological sequence: spinup ... pi_ctrl, with historic
    # overlaying pi_ctrl as a separate forced branch from the same start.
    # Without this the historic run (e.g. 1850-2019) and spinup
    # (2001-5830) end up on disjoint x-axes covering 1850-6000.
    # And when the spinup ends well before pi_ctrl_start (e.g. AWI-ESM3-VEG-HR
    # with spinup ending 1679 but historic / pi_ctrl starting 1850), shift the
    # spinup forward so it lands immediately before pi_ctrl_start with no gap.
    if exp == historic_name:
        offset = pi_ctrl_start - year_start
        years = np.linspace(year_start + offset, year_end + offset,
                            year_end - year_start + 1)
    elif exp == spinup_name:
        spinup_shift = max(0, pi_ctrl_start - 1 - spinup_end)
        years = np.linspace(year_start + spinup_shift, year_end + spinup_shift,
                            year_end - year_start + 1)
    else:
        years = np.linspace(year_start, year_end, year_end - year_start + 1)

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
ax1.set_ylabel('Sea ice extent [$10^6$ km$^2$]', fontsize=17)
ax1.set_xlabel('Year', fontsize=17)

ax1.yaxis.grid(color='gray', linestyle='dashed')

# Place the divider where the spinup was visually shifted to end, not at
# the raw spinup_end. For configs where spinup runs right up to pi_ctrl
# (e.g. LR-Spinup with spinup_end=1849, pi_ctrl_start=1824) the shift is
# 0 and _split = spinup_end as before; for HR-style configs where there
# is a gap, the divider sits at pi_ctrl_start - 1.
_split = spinup_end + max(0, pi_ctrl_start - 1 - spinup_end)
plt.axvline(x=_split,color='black',alpha=0.7,linewidth=3)
# Position the SPIN / HIST&PICT labels in *axis-relative* coordinates
# (0-1 along each axis) so they always stay inside the axis bounds
# regardless of the data x-range. Previously they were placed at
# _split+-10/40 in data coords, which for a 3-year smoke-test run sat
# ~40 years outside the data; bbox_inches='tight' then inflated the
# figure past matplotlib's 2**16-pixel Agg limit (Image size 116679x...).
ax1.text(0.97, 0.97, 'HIST & PICT', transform=ax1.transAxes,
         ha='right', va='top', fontsize=15)
ax1.text(0.03, 0.97, 'SPIN', transform=ax1.transAxes,
         ha='left', va='top', fontsize=15)

# Scale x-tick density to the timeline length: 50 yr majors / 10 yr minors
# work for the multi-millennium spinup but produce no ticks on a 3-yr run.
_xspan = abs(ax1.get_xlim()[1] - ax1.get_xlim()[0])
_majstep = max(1, int(_xspan / 10))
_minstep = max(1, int(_majstep / 5))
ax1.xaxis.set_major_locator(MultipleLocator(_majstep))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.tick_params(axis='both', which='minor', labelsize=12)

# For the minor ticks, use no labels; default NullFormatter.
ax1.xaxis.set_minor_locator(MultipleLocator(_minstep))

legend=['Arctic March','Arctic September','Antarctic March','Antarctic September']
plt.legend(legend,loc='upper left',fontsize=15)
plt.savefig(out_path+"sea_ice_extent_comparison.png",dpi=300,bbox_inches = "tight")


# Mark as completed
update_status(SCRIPT_NAME, " Completed")
