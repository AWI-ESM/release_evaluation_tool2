# AWI-CM3 Release Evaluation Tool

A comprehensive climate model evaluation framework for analyzing AWI-CM3 (AWI Climate Model version 3) simulations. This tool provides automated diagnostics, comparisons with observations, and publication-quality visualizations for model spinup, preindustrial control, and historical experiments.

## Overview

The Release Evaluation Tool (reval) is designed to streamline the assessment of AWI-CM3 model performance through:
- **Parallel execution** of 20+ diagnostic scripts via SLURM
- **Multi-component analysis** (atmosphere, ocean, sea ice, land vegetation)
- **Observation comparisons** against ERA5, CERES, MODIS, GPCP, HadISST, and more
- **Publication-ready plots** with consistent formatting and metadata

Originally developed for [Streffing et al. (2022, GMD)](https://doi.org/10.5194/gmd-15-6399-2022) and extended for [Streffing et al. (2024, EGUSPHERE)](https://doi.org/10.5194/egusphere-2024-2491).

---

## Quick Start

### Prerequisites
- Conda environment with required packages (see `environment.yaml`)
- Access to AWI-CM3 model output data
- SLURM cluster for parallel job submission

### Setup
```bash
# Clone repository
cd /work/ab0246/a270092/software/release_evaluation_tool2

# Activate environment
source ~/loadconda.sh
conda activate reval

# Configure simulation paths
edit config.py  # Set model_version, paths, years

# Enable/disable scripts
edit reval.py   # Set SCRIPTS = {script: True/False}

# Submit all enabled scripts
python reval.py
```

### Monitor Progress
```bash
# Check job status
squeue -u $USER

# Monitor logs
tail -f logs/<script_name>.log

# View completion status
cat logs/status.csv
```

---

## Directory Structure

```
release_evaluation_tool2/
├── README.md              # This file
├── reval.py               # Main submission script
├── config.py              # Central configuration (imported by configs/)
├── environment.yaml       # Conda dependencies
│
├── configs/               # Experiment-specific configurations
│   ├── AWI-CM3-v3.3.py
│   ├── CORE3_tuning.py
│   ├── HR_tuning.py
│   ├── lpjg_spinup_200y_16c.py
│   └── ...
│
├── scripts/               # Analysis scripts (part##_name.py)
│   ├── part1_mesh_plot.py
│   ├── part2_rad_balance.py
│   ├── part8_t2m_vs_era5.py
│   ├── part23_ice_cavity_velocities.py
│   └── ...
│
├── bg_routines/           # Helper functions
│   ├── update_status.py
│   ├── sub_fesom_mesh.py
│   ├── sub_fesom_data.py
│   └── ...
│
├── logs/                  # SLURM output logs
│   ├── status.csv         # Completion tracking
│   └── part##_*.log       # Individual script logs
│
├── output/                # Generated plots (organized by model_version)
│   └── <model_version>/
│
└── tmp/                   # Temporary SLURM scripts
```

---

## Configuration System

### Main Config (`config.py`)
Acts as a pointer to experiment-specific configs. Simply imports from `configs/` directory:

```python
from configs.AWI-CM3-v3.3 import *  # or CORE3_tuning, HR_tuning, etc.
```

### Experiment Configs (`configs/*.py`)
Define simulation-specific parameters:

```python
# Model identification
model_version = 'AWI-ESM3.4_rc3_HR'
oasis_oifs_grid_name = 'A320'

# Spinup experiment
spinup_path  = '/path/to/spinup/outdata/'
spinup_start = 1354
spinup_end   = 1359

# Preindustrial control
pi_ctrl_path  = '/path/to/pi-control/outdata/'
pi_ctrl_start = 1400
pi_ctrl_end   = 1500

# Historical experiment
historic_path  = '/path/to/historical/outdata/'
historic_start = 1850
historic_end   = 2014

# Mesh configuration
mesh_name = 'DARS2'      # FESOM2 mesh
grid_name = 'TCo319'     # OpenIFS grid
meshpath  = '/path/to/mesh/'

# Analysis settings
reanalysis = 'ERA5'
dpi = 300
```

**Key Variables:**
- `model_version`: Identifier for output directory
- `*_path`: Base paths to simulation outdata
- `*_start`, `*_end`: Year ranges for each experiment
- `historic_last25y_start/end`: Period for climatology (auto-calculated)
- `mesh_name`, `grid_name`: Model component identifiers
- `out_path`: Auto-generated output directory

---

## Analysis Scripts

### Naming Convention
Scripts follow `part##_description.py` format and are executed in natural sort order.

### Script Structure
Each script follows this template:

```python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

SCRIPT_NAME = os.path.basename(__file__)
update_status(SCRIPT_NAME, " Started")

try:
    # Analysis code here
    # - Load data from *_path/component/*.nc
    # - Process using CDO, xarray, pyfesom2
    # - Generate plots to out_path
    
    update_status(SCRIPT_NAME, " Completed")
except Exception as e:
    update_status(SCRIPT_NAME, f" Failed: {e}")
    raise
```

### Categories

**Mesh & Radiation (1-2)**
- `part1_mesh_plot.py`: FESOM2 mesh resolution visualization
- `part2_rad_balance.py`: Top-of-atmosphere radiation budget

**Ocean (3-4, 7, 13-14, 17, 19)**
- `part3_hovm_temp.py`: Hovmöller diagram of ocean temperature
- `part7_mld.py`: Mixed layer depth climatology
- `part13_fesom_bias_maps.py`: Ocean temperature/salinity bias vs observations
- `part14_fesom_salt.py`: Ocean salinity analysis
- `part17_moc.py`: Meridional overturning circulation
- `part19_ocean_temp_sections.py`: Ocean temperature cross-sections

**Sea Ice (5-6, 23)**
- `part5_sea_ice_thickness.py`: Sea ice thickness evolution
- `part6_ice_conc_timeseries.py`: Sea ice extent timeseries
- `part23_ice_cavity_velocities.py`: Antarctic ice shelf cavity circulation (2D sections)
- `part23_ice_cavity_velocities_v2.py`: Ice cavity multi-panel analysis (streamlines, cross-sections)

**Atmosphere vs Obs (8-12, 18)**
- `part8_t2m_vs_era5.py`: 2m temperature vs ERA5
- `part9_rad_vs_ceres.py`: Radiation fluxes vs CERES
- `part10_clt_vs_modis.py`: Cloud cover vs MODIS
- `part11_zonal_plots.py`: Zonal mean diagnostics
- `part12_qbo.py`: Quasi-biennial oscillation
- `part18_precip_vs_gpcp.py`: Precipitation vs GPCP

**Climate Analysis (15-16, 20-21)**
- `part15_enso.py`: ENSO analysis (Niño 3.4 index, EOF, power spectra)
- `part16_clim_change.py`: Climate change signals (PI control vs historical)
- `part20_gregory_plot.py`: Gregory plot (ECS estimation)
- `part21_crf_bias_maps.py`: Cloud radiative forcing bias maps

**Masks & Configuration (22)**
- `part22_masks.py`: Visualization of model grids and land-sea masks

**Land Vegetation - LPJ-GUESS (24-26)**
- `part24_lpjg_lai.py`: Leaf area index by PFT
- `part25_lpjg_carbon.py`: Vegetation and soil carbon stocks
- `part26_lpjg_pft.py`: Plant functional type dominance and diversity

---

## Workflow

### 1. Configure Experiment
Edit `config.py` to point to desired experiment config:
```python
from configs.AWI-CM3-v3.3 import *
```

### 2. Enable Scripts
Edit `reval.py` to enable/disable specific analyses:
```python
SCRIPTS.update({
    "part1_mesh_plot.py":    True,
    "part8_t2m_vs_era5.py":  True,
    "part15_enso.py":        False,  # Skip this one
    ...
})
```

### 3. Submit Jobs
```bash
python reval.py
```

This creates individual SLURM scripts (`slurm_part##_*.sh`) and submits them to the compute partition.

### 4. Monitor Execution
- **Job queue**: `squeue -u $USER`
- **Logs**: `tail -f logs/part##_*.log`
- **Status**: `cat logs/status.csv`
- **Output**: `ls output/<model_version>/`

### 5. Review Results
All plots are saved to `output/<model_version>/` with descriptive filenames:
- `t2m_vs_ERA5.png`
- `ice_cavity_Ross_panels.png`
- `HIST_Nino34_enso_box_index.png`
- etc.

---

## SLURM Configuration

Default settings in `reval.py`:
```bash
#SBATCH --time=00:20:00       # 20 minutes per script
#SBATCH --mem=256G            # 256 GB memory
#SBATCH --ntasks=128          # 128 cores
#SBATCH --partition=compute
#SBATCH -A ab0246             # Account
```

Adjust these for specific scripts if needed (e.g., longer time for ocean sections).

---

## Data Requirements

### Model Output Structure
Expected data organization:
```
<experiment_path>/
├── fesom/
│   ├── temp.fesom.YYYY.nc
│   ├── salt.fesom.YYYY.nc
│   ├── u.fesom.YYYY.nc
│   └── ...
├── oifs/
│   ├── atm_remapped_1d_2t_1d_YYYY-YYYY.nc
│   ├── atm_remapped_1m_lcc_1m_YYYY-YYYY.nc
│   └── ...
└── lpj_guess/ (if using LPJ-GUESS)
    ├── run1/
    │   ├── lai.out
    │   ├── cmass.out
    │   └── ...
    └── run2/
        └── ...
```

### Observation Data
Stored in `observation_path` (default: `/work/ab0246/a270092/obs/`):
- ERA5 reanalysis
- CERES radiation products
- MODIS cloud cover
- GPCP precipitation
- HadISST sea surface temperature
- PHC3 ocean climatology

---

## Troubleshooting

### Common Issues

**Script shows "Started" but not "Completed"**
- Check log file: `cat logs/part##_*.log`
- Common causes: missing data files, insufficient memory, timeout

**"No such file or directory" errors**
- Verify paths in config match actual data location
- Check year ranges - data must exist for all years in `*_start` to `*_end`
- Ensure FESOM/OpenIFS output files follow expected naming conventions

**LPJ-GUESS scripts fail**
- Confirm LPJ-GUESS data exists for requested years
- Check that data follows `outdata/lpj_guess/run*/lai.out` structure
- Note: LPJ-GUESS years may differ from ESM years (e.g., 6734-9999 vs 1354-1359)

**Memory errors**
- Increase `--mem` in SLURM settings for data-heavy scripts
- Consider processing fewer years or coarser temporal resolution

**Plot quality issues**
- Adjust `dpi` in config (default: 300)
- Check colormap limits in individual scripts
- Verify projection settings for polar plots

---

## Key Dependencies

- **Python 3.12+**
- **PyFESOM2**: FESOM2 mesh and data handling
- **xarray**: NetCDF data structures
- **CDO** (Python bindings): Climate Data Operators
- **matplotlib, cartopy**: Plotting and map projections
- **cmocean**: Oceanographic colormaps
- **dask**: Parallel computation
- **scipy, scikit-learn**: Scientific computing
- **tqdm**: Progress bars

See `environment.yaml` for complete list.

---

## Output Examples

Generated plots include:
- **Global maps**: Temperature, precipitation, radiation biases
- **Timeseries**: Sea ice extent, ENSO indices
- **Cross-sections**: Ocean temperature/salinity transects
- **Hovmöller diagrams**: Spatiotemporal evolution
- **Multi-panel figures**: Comprehensive cavity circulation analysis
- **Statistical plots**: Gregory plots, power spectra, EOFs

All plots include:
- Model version and year range in title
- Colorbars with units
- Coastlines and grid references
- High resolution (300 DPI) for publication

---

## Citation

If you use this tool, please cite:

> Streffing, J., Sidorenko, D., Semmler, T., Zampieri, L., Scholz, P., Andrés-Martínez, M., Koldunov, N., Rackow, T., Kjellsson, J., Goessling, H., Athanase, M., Wang, Q., Hegewald, J., Sein, D. V., Mu, L., Hinrichs, C., Kluft, L., Danilov, S., Jungclaus, J., and Jung, T.: AWI-CM3 model output prepared for GMD paper "AWI-CM3 coupled climate model: Description and evaluation", https://doi.org/10.5194/gmd-15-6399-2022, 2022.

---

## Support

For questions or issues:
- Check logs in `logs/` directory
- Review data paths in `config.py` and experiment configs
- Consult script-specific comments in `scripts/`
- Contact: Jan Streffing (jan.streffing@awi.de)

---

## Version History

- **2025-02-04**: Refactored for parallel script execution (v2)
- **2024-04-03**: Added significance metrics for EGUSphere paper
- **2021-12-10**: Initial Jupyter notebook version for GMD paper