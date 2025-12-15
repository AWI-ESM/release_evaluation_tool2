# PI Control Drift Analysis (Gregory-style plot)
# ------------------------------------------------
# This script creates a Gregory-style plot for PI control tuning in the AWI-CM3 
# Release Evaluation Tool (part20). For PI control, we plot the relationship 
# between global mean surface temperature drift and TOA radiative imbalance to 
# assess model equilibrium and energy balance closure.
#
# The plot shows:
# - X-axis: Temperature drift from initial state (K)
# - Y-axis: TOA radiative imbalance (W/m²) - positive = heat uptake
# - Green reference boxes indicate acceptable drift ranges
# - Linear regression provides drift rate and equilibrium assessment
#
# For PI control tuning, we expect:
# - Temperature drift: 0 ± 0.5 K over simulation period
# - Radiative imbalance: 0 ± 0.5 W/m² (energy balance closure)
# ------------------------------------------------

import os
import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Add parent directory to path and import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

SCRIPT_NAME = os.path.basename(__file__)
print(SCRIPT_NAME)
update_status(SCRIPT_NAME, " Started")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Spinup experiment path (analyzing spinup as PI control equivalent)
SPINUP_PATH = os.environ.get(
    "AWICM3_SPINUP_PATH", 
    spinup_path + "/oifs/"
)

# Configuration options
USE_SURFACE_BUDGET = True  # Set to True to use surface energy budget instead of TOA

# Years to analyze (use spinup period)
SPINUP_YEARS = list(range(spinup_start, spinup_end + 1))
print(f"Analyzing spinup years: {spinup_start}-{spinup_end} ({len(SPINUP_YEARS)} years)")
print(f"Energy budget method: {'Surface' if USE_SURFACE_BUDGET else 'TOA'}")

# Output file
PLOT_FILE = "pi_control_drift_analysis.png"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def detect_file_pattern(path, var, years):
    """Detect available file pattern (1m or 6h) for given variable."""
    # Try 1m files first
    pattern_1m = f"atm_remapped_1m_{var}_1m_{{year:04d}}-{{year:04d}}.nc"
    test_file = os.path.join(path, pattern_1m.format(year=years[0]))
    if os.path.exists(test_file):
        return pattern_1m, "1m"
    
    # Try 6h files
    pattern_6h = f"atm_remapped_6h_{var}_6h_{{year:04d}}-{{year:04d}}.nc"
    test_file = os.path.join(path, pattern_6h.format(year=years[0]))
    if os.path.exists(test_file):
        return pattern_6h, "6h"
    
    return None, None

def load_yearly_data_simple(path, var, years, pattern, freq):
    """Load and process data to yearly means - simple xarray approach"""
    print(f"Processing variable: {var}")
    
    files = []
    for year in years:
        filepath = os.path.join(path, pattern.format(year=year))
        if os.path.exists(filepath):
            files.append(filepath)
        else:
            print(f"WARNING: Missing file {filepath}")
    
    if not files:
        print(f"ERROR: No files found for variable '{var}'")
        return None
    
    try:
        print(f"Loading {len(files)} files for {var}...")
        
        # Load files with explicit time decoding to handle mixed calendar types
        ds = xr.open_mfdataset(files, combine="by_coords", parallel=False, 
                             chunks={'time_counter': 12}, 
                             decode_times=True, use_cftime=True,
                             combine_attrs='drop_conflicts')
        
        # Get the variable data
        var_data = ds[var]
        
        # Normalize by accumulation period ONLY for flux variables (not temperature)
        if var != '2t':  # Don't normalize temperature
            var_data = var_data / accumulation_period
        
        # Calculate global area mean
        global_mean = global_area_mean(var_data)
        
        # Convert to yearly means using groupby
        yearly_data = global_mean.groupby('time_counter.year').mean()
        
        # Force computation to avoid lazy evaluation
        yearly_data = yearly_data.compute()
        
        print(f"Loaded {var}: {len(files)} files -> {len(yearly_data)} yearly values")
        return yearly_data
        
    except Exception as e:
        print(f"ERROR loading {var}: {e}")
        return None

def global_area_mean(da):
    """Calculate proper area-weighted global mean."""
    # Find latitude coordinate
    lat_coord = None
    for coord in ['lat', 'latitude', 'y']:
        if coord in da.coords:
            lat_coord = coord
            break
    
    if lat_coord is None:
        raise ValueError(f"No latitude coordinate found. Available coords: {list(da.coords.keys())}")
    
    # Calculate cosine latitude weights (proper area weighting for regular lat-lon grid)
    # This accounts for the fact that grid cells get smaller towards the poles
    weights = np.cos(np.deg2rad(da[lat_coord]))
    
    # For regular grids, this is equivalent to proper area weighting since:
    # - All longitude bands have the same Δlon
    # - Area ∝ cos(lat) * Δlat * Δlon, and Δlat is constant
    # - So relative weights are just cos(lat)
    
    # Compute weighted mean over spatial dimensions
    spatial_dims = [lat_coord]
    if 'lon' in da.dims:
        spatial_dims.append('lon')
    elif 'longitude' in da.dims:
        spatial_dims.append('longitude')
    
    da_global = da.weighted(weights).mean(dim=spatial_dims)
    
    return da_global

# -----------------------------------------------------------------------------
# Load and process data
# -----------------------------------------------------------------------------
print("Loading spinup data...")

# Detect file patterns for each variable
patterns = {}
frequencies = {}

if USE_SURFACE_BUDGET:
    print("Loading surface energy budget data...")
    required_vars = ['2t', 'ssr', 'str', 'sshf', 'slhf', 'sf']
    var_descriptions = {
        '2t': '2m temperature',
        'ssr': 'surface solar radiation',
        'str': 'surface thermal radiation', 
        'sshf': 'surface sensible heat flux',
        'slhf': 'surface latent heat flux',
        'sf': 'snowfall'
    }
else:
    print("Loading TOA energy budget data...")
    required_vars = ['2t', 'tsr', 'ttr']
    var_descriptions = {
        '2t': '2m temperature',
        'tsr': 'TOA solar radiation',
        'ttr': 'TOA thermal radiation'
    }

# Detect patterns for all required variables
for var in required_vars:
    pattern, freq = detect_file_pattern(SPINUP_PATH, var, SPINUP_YEARS)
    if pattern is None:
        print(f"ERROR: No files found for {var}")
        sys.exit(1)
    patterns[var] = pattern
    frequencies[var] = freq
    print(f"Found {var} files: {freq} frequency")

# Load temperature data
temp_data = load_yearly_data_simple(SPINUP_PATH, '2t', SPINUP_YEARS, patterns['2t'], frequencies['2t'])
if temp_data is None:
    print("ERROR: Failed to load temperature data")
    sys.exit(1)

# Load energy budget data and calculate imbalance
if USE_SURFACE_BUDGET:
    # Load surface energy budget components
    ssr_data = load_yearly_data_simple(SPINUP_PATH, 'ssr', SPINUP_YEARS, patterns['ssr'], frequencies['ssr'])
    str_data = load_yearly_data_simple(SPINUP_PATH, 'str', SPINUP_YEARS, patterns['str'], frequencies['str'])
    sshf_data = load_yearly_data_simple(SPINUP_PATH, 'sshf', SPINUP_YEARS, patterns['sshf'], frequencies['sshf'])
    slhf_data = load_yearly_data_simple(SPINUP_PATH, 'slhf', SPINUP_YEARS, patterns['slhf'], frequencies['slhf'])
    sf_data = load_yearly_data_simple(SPINUP_PATH, 'sf', SPINUP_YEARS, patterns['sf'], frequencies['sf'])
    
    if any(data is None for data in [ssr_data, str_data, sshf_data, slhf_data, sf_data]):
        print("ERROR: Failed to load surface energy budget data")
        sys.exit(1)
    
    # Data is already global means from parallel processing
    ssr_global = ssr_data
    str_global = str_data
    sshf_global = sshf_data
    slhf_global = slhf_data
    sf_global = sf_data
    
    # Calculate surface energy imbalance (following part2_rad_balance.py)
    # Surface budget = SSR + STR + SSHF + SLHF - SF_heat_flux
    # All fluxes are now in W/m² after normalization by accumulation_period
    # Convert SF from kg/m²/s to W/m² using heat of fusion (same as part2)
    sf_heat_flux = sf_global * 333550000  # Heat of fusion for ice (J/kg) - same as part2
    energy_imbalance = ssr_global + str_global + sshf_global + slhf_global - sf_heat_flux
    
    print("Surface energy budget components loaded and calculated")
    print(f"DEBUG: SSR range: {ssr_global.min().values:.2f} to {ssr_global.max().values:.2f} W/m²")
    print(f"DEBUG: STR range: {str_global.min().values:.2f} to {str_global.max().values:.2f} W/m²") 
    print(f"DEBUG: SSHF range: {sshf_global.min().values:.2f} to {sshf_global.max().values:.2f} W/m²")
    print(f"DEBUG: SLHF range: {slhf_global.min().values:.2f} to {slhf_global.max().values:.2f} W/m²")
    print(f"DEBUG: SF heat flux range: {sf_heat_flux.min().values:.2f} to {sf_heat_flux.max().values:.2f} W/m²")
    
else:
    # Load TOA energy budget components  
    tsr_data = load_yearly_data_simple(SPINUP_PATH, 'tsr', SPINUP_YEARS, patterns['tsr'], frequencies['tsr'])
    ttr_data = load_yearly_data_simple(SPINUP_PATH, 'ttr', SPINUP_YEARS, patterns['ttr'], frequencies['ttr'])
    
    if tsr_data is None or ttr_data is None:
        print("ERROR: Failed to load TOA energy budget data")
        sys.exit(1)
    
    # Data is already global means from parallel processing
    tsr_global = tsr_data
    ttr_global = ttr_data
    
    # Calculate TOA radiative imbalance
    # Convention: positive = heat uptake (energy going into the system)
    # TOA imbalance = -(outgoing SW + outgoing LW) = -(tsr + ttr)
    energy_imbalance = -(tsr_global + ttr_global)
    
    print("TOA energy budget components loaded and calculated")

print("Accumulation period:", accumulation_period, "seconds")

# Data is already global means from parallel processing
print("Data loaded as global means...")
temp_global = temp_data

print("Processing data for analysis...")

# Calculate absolute temperatures and actual energy imbalance
print("Converting temperature to Celsius...")
temp_celsius = temp_global - 273.15  # Convert Kelvin to Celsius
imbalance_actual = energy_imbalance     # Use actual imbalance, not drift

print("Converting to numpy arrays...")
# Convert to numpy arrays for regression
temp_vals = temp_celsius.values
imbalance_vals = imbalance_actual.values

# Remove any NaN values
mask = np.isfinite(temp_vals) & np.isfinite(imbalance_vals)
temp_vals = temp_vals[mask]
imbalance_vals = imbalance_vals[mask]

print(f"Analysis points before outlier removal: {len(temp_vals)} years")

# Regression-based outlier detection
print("Step 1: Computing initial regression from all points...")
initial_reg = linregress(temp_vals, imbalance_vals)
initial_slope = initial_reg.slope
initial_intercept = initial_reg.intercept

# Calculate predicted values and residuals
predicted_vals = initial_slope * temp_vals + initial_intercept
residuals = imbalance_vals - predicted_vals

# Remove outliers based on residuals (more than 2 standard deviations from regression line)
residual_std = np.std(residuals)
residual_threshold = 2 * residual_std

print(f"Initial regression: slope={initial_slope:.3f} W/m²/K, intercept={initial_intercept:.3f} W/m²")
print(f"Residual standard deviation: {residual_std:.3f} W/m²")
print(f"Outlier threshold: ±{residual_threshold:.3f} W/m² from regression line")

# Create mask for points within threshold distance from regression line
outlier_mask = np.abs(residuals) <= residual_threshold

# Apply outlier mask
temp_vals_clean = temp_vals[outlier_mask]
imbalance_vals_clean = imbalance_vals[outlier_mask]

n_outliers = len(temp_vals) - len(temp_vals_clean)
print(f"Step 2: Removed {n_outliers} outliers (>2σ from regression line)")
print(f"Analysis points after outlier removal: {len(temp_vals_clean)} years")
print(f"Temperature range: {np.min(temp_vals):.3f} to {np.max(temp_vals):.3f} °C")
print(f"Energy imbalance range: {np.min(imbalance_vals):.3f} to {np.max(imbalance_vals):.3f} W/m²")

# -----------------------------------------------------------------------------
# Statistical analysis
# -----------------------------------------------------------------------------
if len(temp_vals_clean) > 1:
    print("Step 3: Computing final regression without outliers...")
    reg = linregress(temp_vals_clean, imbalance_vals_clean)
    drift_rate = reg.slope  # W/m²/K
    intercept = reg.intercept  # W/m²
    r_squared = reg.rvalue**2
    p_value = reg.pvalue
    
    print(f"Final regression: slope={drift_rate:.3f} W/m²/K, intercept={intercept:.3f} W/m²")
    print(f"R²={r_squared:.3f}, p-value={p_value:.3e}")
    
    # Initial state (first 10 years mean)
    initial_temp = np.mean(temp_vals_clean[:10])
    initial_imbalance = np.mean(imbalance_vals_clean[:10])
    
    # Final state (last 10 years mean)
    final_temp = np.mean(temp_vals_clean[-10:])
    final_imbalance = np.mean(imbalance_vals_clean[-10:])
    
    print("\nDrift Analysis:")
    print(f"  Drift rate: {drift_rate:.3f} W/m²/K")
    print(f"  Intercept: {intercept:.3f} W/m²")
    print(f"  R²: {r_squared:.3f}")
    
    print(f"\nInitial State (first 10 years: {SPINUP_YEARS[0]}-{SPINUP_YEARS[9]}):")
    print(f"  Temperature: {initial_temp:.3f} °C")
    print(f"  Energy imbalance: {initial_imbalance:.3f} W/m²")
    
    print(f"\nFinal State (last 10 years: {SPINUP_YEARS[-10]}-{SPINUP_YEARS[-1]}):")
    print(f"  Temperature: {final_temp:.3f} °C")
    print(f"  Energy imbalance: {final_imbalance:.3f} W/m²")
else:
    print("ERROR: Insufficient data for analysis")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Create plot
# -----------------------------------------------------------------------------
print("Creating drift analysis plot...")

fig, ax = plt.subplots(figsize=(8, 6))

# Reference boxes for acceptable PI control state
# Temperature: 13.2-14.2°C (13.7 ± 0.5°C)
ax.axvspan(13.2, 14.2, color='green', alpha=0.2, zorder=0, label='Target temperature')
# Energy imbalance: ±0.5 W/m²  
ax.axhspan(-0.5, 0.5, color='green', alpha=0.2, zorder=0, label='Target energy balance')

# Color points by simulation year for temporal context
years_normalized = (np.arange(len(temp_vals)) / len(temp_vals))
scatter = ax.scatter(temp_vals, imbalance_vals, c=years_normalized, 
                    cmap='viridis', s=30, alpha=0.7, zorder=3)

# Highlight outliers in red
if n_outliers > 0:
    outlier_indices = ~outlier_mask
    ax.scatter(temp_vals[outlier_indices], imbalance_vals[outlier_indices], 
              c='red', s=50, alpha=0.8, zorder=4, marker='x', 
              label=f'Outliers (>2σ, n={n_outliers})')

# Regression line through the cleaned point cloud - extend to show equilibrium
temp_range_extended = np.linspace(np.min(temp_vals_clean) - 1, np.max(temp_vals_clean) + 1, 200)
imbalance_fit = drift_rate * temp_range_extended + intercept

# Plot regression line in front (higher zorder)
ax.plot(temp_range_extended, imbalance_fit, 'b-', linewidth=3, alpha=0.9, zorder=5,
        label=f'Regression: {drift_rate:.3f} W/m²/K (R²={r_squared:.3f}, n={len(temp_vals_clean)})')

# Mark the equilibrium temperature (where regression line crosses y=0)
equilibrium_temp = -intercept / drift_rate
ax.axvline(equilibrium_temp, color='orange', linestyle='--', alpha=0.8, zorder=4,
           label=f'Equilibrium: {equilibrium_temp:.1f}°C')
ax.plot(equilibrium_temp, 0, 'ro', markersize=10, zorder=6, label='Final State Projection')

# Add horizontal line at y=0 for reference
ax.axhline(0, color='gray', linestyle='-', alpha=0.5, zorder=1)

# Annotations for first and last years
ax.text(temp_vals[0], imbalance_vals[0], f'{SPINUP_YEARS[0]}', 
        fontsize=9, ha='right', va='bottom', color='black', weight='bold')
ax.text(temp_vals[-1], imbalance_vals[-1], f'{SPINUP_YEARS[-1]}', 
        fontsize=9, ha='left', va='top', color='black', weight='bold')

# Set axis limits dynamically based on data range with reasonable bounds
temp_range = np.max(temp_vals) - np.min(temp_vals)
imbalance_range = np.max(imbalance_vals) - np.min(imbalance_vals)

# Use reasonable limits or data-driven limits
temp_center = np.mean(temp_vals)
imbalance_center = np.mean(imbalance_vals)

temp_width = max(6, temp_range * 1.2)  # At least 6°C width
# Cap imbalance width to prevent plot failures
imbalance_width = max(6, min(imbalance_range * 1.2, 100))  # At least 6, max 100 W/m²

# If imbalance values are extremely large, use fixed reasonable limits
if abs(imbalance_center) > 1000:
    print(f"WARNING: Large imbalance values detected ({imbalance_center:.1f} W/m²)")
    print("Using fixed axis limits to prevent plot failure")
    ax.set_xlim(temp_center - temp_width/2, temp_center + temp_width/2)
    ax.set_ylim(-50, 50)  # Fixed reasonable range for energy imbalance
else:
    ax.set_xlim(temp_center - temp_width/2, temp_center + temp_width/2)
    ax.set_ylim(imbalance_center - imbalance_width/2, imbalance_center + imbalance_width/2)

# Labels and formatting
ax.set_xlabel('Global Mean Temperature (°C)', fontsize=12)
budget_type = 'Surface' if USE_SURFACE_BUDGET else 'TOA'
ax.set_ylabel(f'{budget_type} Energy Imbalance (W m⁻²)', fontsize=12)
ax.set_title(f'Spinup Temperature vs {budget_type} Energy Balance - {model_version}\n({SPINUP_YEARS[0]}-{SPINUP_YEARS[-1]})', 
             fontsize=14, fontweight='bold')

# Colorbar for time evolution
cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label('Simulation Progress', fontsize=10)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(['Start', 'Mid', 'End'])

# Statistics text box
stats_text = (f'Final temp: {final_temp:.1f} °C\n'
              f'Final imbalance: {final_imbalance:.2f} W/m²')

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Legend
ax.legend(loc='lower right', fontsize=10)

plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
output_path = out_path + PLOT_FILE
plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
plt.close()

print(f"PI control drift analysis saved to: {output_path}")

# -----------------------------------------------------------------------------
# Assessment summary
# -----------------------------------------------------------------------------
print(f"\n=== Spinup Assessment Summary ===")
print(f"Model: {model_version}")
print(f"Period: {SPINUP_YEARS[0]}-{SPINUP_YEARS[-1]} ({len(SPINUP_YEARS)} years)")
print(f"")
print(f"Final State (last 10 years mean):")
print(f"  Temperature: {final_temp:.3f} °C")
print(f"  Energy imbalance: {final_imbalance:.3f} W/m²")
print(f"")

# Assessment criteria - check if final state is in target ranges
temp_in_range = 13.2 <= final_temp <= 14.2
imbalance_in_range = abs(final_imbalance) <= 0.5

print(f"Assessment:")
print(f"  Temperature range: {'✓ PASS' if temp_in_range else '✗ FAIL'} ({final_temp:.1f}°C {'in' if temp_in_range else 'outside'} 13.2-14.2°C)")
print(f"  Energy balance: {'✓ PASS' if imbalance_in_range else '✗ FAIL'} (|{final_imbalance:.1f}| {'≤' if imbalance_in_range else '>'} 0.5 W/m²)")
print(f"  Overall: {'✓ ACCEPTABLE' if (temp_in_range and imbalance_in_range) else '✗ NEEDS TUNING'}")

update_status(SCRIPT_NAME, " Completed")
