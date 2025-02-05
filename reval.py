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
import subprocess
from natsort import natsorted


############################
# Slurm Configuration      #
############################

SBATCH_SETTINGS = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}.log
#SBATCH --error=logs/{job_name}.log
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute
#SBATCH -A bb1469
"""



############################
# Script Execution         #
############################

# Ensure required directories exist
os.makedirs("logs", exist_ok=True)

# Locate all Python scripts in the "scripts" subfolder
script_files = natsorted(
    [f for f in os.listdir("scripts") if f.endswith(".py") and f != "__init__.py"]
)

# Default: Disable all scripts (set to True to enable)
SCRIPTS = {script: False for script in script_files}  # All disabled by default

# Enable scripts manually here:
SCRIPTS.update({
    "part1_mesh_plot.py":           False,
    "part2_rad_balance.py":         True,
    "part3_hovm_temp.py":           False,  
    "part4_cmpi.py":                False,
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
            f.write(f"python {script_path}\n")

        # Submit job
        subprocess.run(["sbatch", job_script])
        print(f"Submitted {script}")
    else:
        print(f"Skipped {script} (disabled)")

