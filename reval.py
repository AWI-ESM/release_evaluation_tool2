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
from bg_routines.config_loader import *

############################
# Slurm Configuration      #
############################

SBATCH_SETTINGS = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}.log
#SBATCH --error=logs/{job_name}.log
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute
#SBATCH -A ab0995
"""

SBATCH_SETTINGS = SBATCH_SETTINGS.replace("logs", f"logs/{model_version}")



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
  python reval.py --status
''')
parser.add_argument(
    '-c', '--config',
    help='Path to configuration file in configs/ folder (e.g., configs/AWI-CM3-v3.3.py)')
parser.add_argument(
    '-s', '--status',
    action='store_true',
    help='Show status of all scripts and exit')
args = parser.parse_args()

# Handle --status
if args.status:
    from bg_routines.update_status import get_all_status
    status = get_all_status()
    if not status:
        print("No status information found yet.")
    else:
        print(f"{'Script':<40} {'Status'}")
        print("-" * 70)
        for script, stat in status.items():
            print(f"{script:<40} {stat}")
    sys.exit(0)

# Require --config for job submission
if not args.config:
    parser.error("--config is required when submitting jobs")

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
    "part1_mesh_plot.py":           False,
    "part2_rad_balance.py":         False,
    "part3_hovm_temp.py":           False,
    "part5_sea_ice_thickness.py":   False,
    "part6_ice_conc_timeseries.py": False,
    "part7_mld.py":                 False,
    "part8_t2m_vs_era5.py":         False,
    "part9_rad_vs_ceres.py":        False,
    "part10_clt_vs_modis.py":       False,
    "part11_zonal_plots.py":        False,
    "part12_qbo.py":                False,
    "part13_fesom_bias_maps.py":    False,
    "part14_fesom_salt.py":         False,
    "part15_enso.py":               False,
    "part16_clim_change.py":        False,
    "part17_moc.py":                False,
    "part18_precip_vs_gpcp.py":     False,
    "part19_ocean_temp_sections.py":False,
    "part20_gregory_plot.py":       False,
    "part21_crf_bias_maps.py":      False,
    "part22_masks.py":              False,
    "part23_ice_cavity_velocities.py": False,
    "part24_lpjg_lai.py":           True,
    "part25_lpjg_carbon.py":        False,
    "part26_lpjg_pft.py":           False,
})

# Submit jobs
for script, run in SCRIPTS.items():
    if run:
        job_script = f"slurm_{script}.sh"
        script_path = os.path.join("scripts", script)

        # Write the SLURM script
        with open(job_script, "w") as f:
            f.write(SBATCH_SETTINGS.format(job_name=script))
            f.write("\nmodule load python3\n")  # Load Python module if required
            #f.write("\nconda activate reval\n")  # Load Python module if required
            f.write(f"\nexport REVAL_CONFIG={config_path}\n")  # Pass config file path
            f.write(f"python {script_path}\n")

        # Submit job
        print(f"Submitting {script} as:")
        subprocess.run(["sbatch", job_script])
        destination = f"tmp/{job_script}"
        shutil.move(job_script, destination)
    else:
        print(f"Skipped {script} (disabled)")

