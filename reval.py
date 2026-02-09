"""
AWI-CM3 Release Evaluation Tool (Reval.py)

This script provides a comprehensive evaluation framework for analyzing and visualizing 
data from the AWI-CM-v3.3 climate model. It integrates scientific libraries for data 
processing, statistical analysis, and high-quality visualizations, enabling effective 
assessment of model performance against observations and reanalysis datasets.

Key Features:
- Data Processing & Analysis:
  - Uses PyFESOM2, xarray, SciPy, and scikit-learn for structured climate data handling.
- Visualization:
  - Leverages Matplotlib, Seaborn, Cartopy, and cmocean for high-quality plots.
- FESOM-Specific Routines:
  - Includes functions for handling FESOM2 mesh structures, model data, and 
    meridional overturning circulation (MOC).
- Automated Job Submission:
  - Supports SLURM-based batch processing for large-scale evaluations.
- Multi-Experiment Support:
  - Handles spin-up, preindustrial control, and historical simulations with 
    configurable paths and settings.


2021-12-10: Jan Streffing:                First jupyter notebook version for https://doi.org/10.5194/gmd-15-6399-2022
2024-04-03: Jan Streffing:                Addition of significance metrics for https://doi.org/10.5194/egusphere-2024-2491
2025-02-04: Jan Streffing:                Re-write has parallel scripts
"""

import os
import sys
import subprocess
import argparse
from natsort import natsorted
import shutil

############################
# Slurm Configuration      #
############################

SBATCH_SETTINGS = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}.log
#SBATCH --error=logs/{job_name}.log
#SBATCH --time=00:20:00
#SBATCH --mem=256G
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute
#SBATCH -A ab0246
"""



############################
# Script Execution         #
############################

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='AWI-CM3 Release Evaluation Tool - Submit analysis jobs',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Examples:
  python reval.py --config configs/AWI-CM3-v3.3.py
  python reval.py --config configs/lpjg_spinup_200y_16c.py
  python reval.py -c configs/HR_tuning.py
''')
parser.add_argument(
    '-c', '--config',
    required=True,
    help='Path to configuration file in configs/ folder (e.g., configs/AWI-CM3-v3.3.py)')
args = parser.parse_args()

# Validate config file exists
if not os.path.exists(args.config):
    print(f"ERROR: Config file not found: {args.config}")
    print("\nAvailable configs in configs/:")
    for f in sorted(os.listdir('configs')):
        if f.endswith('.py'):
            print(f"  - configs/{f}")
    sys.exit(1)

config_path = os.path.abspath(args.config)
print(f"Using configuration: {config_path}")
print(f"{'='*60}\n")

# Ensure required directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("tmp", exist_ok=True)

# Locate all part##_*.py analysis scripts in the "scripts" subfolder
script_files = natsorted(
    [f for f in os.listdir("scripts") if f.endswith(".py") and f.startswith("part")]
)

# Default: Disable all scripts (set to True to enable)
SCRIPTS = {script: False for script in script_files}  # All disabled by default

# Enable scripts manually here:
SCRIPTS.update({
    "part1_mesh_plot.py":           True,
    "part2_rad_balance.py":         True,
    "part3_hovm_temp.py":           True,
    "part5_sea_ice_thickness.py":   True,
    "part6_ice_conc_timeseries.py": True,
    "part7_mld.py":                 True,
    "part8_t2m_vs_era5.py":         True,
    "part9_rad_vs_ceres.py":        True,
    "part10_clt_vs_modis.py":       True,
    "part11_zonal_plots.py":        True,
    "part12_qbo.py":                True,
    "part13_fesom_bias_maps.py":    True,
    "part14_fesom_salt.py":         True,
    "part15_enso.py":               True,
    "part16_clim_change.py":        True,
    "part17_moc.py":                True,
    "part18_precip_vs_gpcp.py":     True,
    "part19_ocean_temp_sections.py":True,
    "part20_gregory_plot.py":       True,
    "part21_crf_bias_maps.py":      True,
    "part22_masks.py":              True,
    "part23_ice_cavity_velocities.py": True,
    "part24_lpjg_lai.py":           True,
    "part25_lpjg_carbon.py":        True,
    "part26_lpjg_pft.py":           True,
})

# Submit jobs
for script, run in SCRIPTS.items():
    if run:
        job_script = f"slurm_{script}.sh"
        script_path = os.path.join("scripts", script)

        # Write the SLURM script
        with open(job_script, "w") as f:
            f.write(SBATCH_SETTINGS.format(job_name=script))
            f.write("\nsource /home/a/a270092/loadconda.sh\n")  # Load Python module if required
            f.write("\nconda activate reval\n")  # Load Python module if required
            f.write(f"\nexport REVAL_CONFIG={config_path}\n")  # Pass config file path
            f.write(f"python {script_path}\n")

        # Submit job
        print(f"Submitting {script} as:")
        subprocess.run(["sbatch", job_script])
        destination = f"tmp/{job_script}"
        shutil.move(job_script, destination)
    else:
        print(f"Skipped {script} (disabled)")

