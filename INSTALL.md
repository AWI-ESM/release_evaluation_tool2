# Installation Guide for AWI-CM3 Release Evaluation Tool

## Prerequisites
- Access to DKRZ Levante HPC system (or similar SLURM cluster)
- Conda/Mamba package manager
- Access to AWI-CM3 model output data

## Installation Steps

### 1. Clone Repository
```bash
cd /work/ab0246/a270092/software/
git clone <repository_url> release_evaluation_tool2
cd release_evaluation_tool2
```

### 2. Create Conda Environment
The `environment.yaml` file includes all dependencies for reval **and** the integrated cmiptool.

```bash
# Create new environment
conda env create -f environment.yaml

# OR update existing environment
conda env update -f environment.yaml --prune
```

### 3. Activate Environment
```bash
source $HOME/loadconda.sh
conda activate reval
```

### 4. Install CMPI Tool Module
The cmiptool is integrated as a local module:

```bash
pip install -e ./cmpitool
```

### 5. Verify Installation
```bash
# Test all imports
python -c "import pooch; import regionmask; import geopandas; from cmpitool import cmpitool, cmpisetup; print('✓ All dependencies installed')"
```

## Integrated Components

### Main Dependencies (environment.yaml)
- **Core Scientific**: numpy, xarray, pandas, netcdf4
- **Visualization**: matplotlib, seaborn, cartopy, cmocean
- **Ocean/FESOM**: pyfesom2, python-cdo
- **Machine Learning**: scikit-learn, scipy
- **CMPI Tool**: pooch, regionmask, geopandas

### CMPI Tool Integration
The CMIP Performance Index tool (`cmpitool/`) is fully integrated:
- **Module**: `cmpitool/cmpitool/`
- **Preprocessing scripts**: `preprocessing_examples/preprocess_AWI-CM3-XIOS*.sh`
- **Analysis script**: `scripts/part4_cmpi.py`

**Observational Data**: Downloaded automatically by `pooch` package on first run of CMPI analysis.

## Directory Structure After Installation
```
release_evaluation_tool2/
├── environment.yaml       # All conda dependencies (including cmiptool)
├── cmpitool/             # CMPI tool module (installed with pip install -e)
│   ├── cmpitool/         # Python package
│   ├── setup.py
│   └── requirements.txt
├── preprocessing_examples/ # Data preprocessing scripts
├── scripts/              # Analysis scripts (part##_*.py)
└── bg_routines/          # Helper functions
```

## Troubleshooting

### Missing Dependencies
If you encounter import errors:
```bash
conda activate reval
conda install -c conda-forge pooch regionmask geopandas -y
```

### CMPI Tool Issues
Ensure cmiptool is installed in editable mode:
```bash
pip install -e ./cmpitool
```

### CDO Not Found
Ensure python-cdo is installed:
```bash
conda install -c conda-forge python-cdo
```

## Next Steps
1. Create/edit experiment config in `configs/`
2. Run `python reval.py -c configs/<your_config>.py`
3. Monitor jobs with `squeue -u $USER`
4. Check outputs in `output/<model_version>/`
