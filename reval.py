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
    "part19_ocean_temp_sections.py": True,
    "part20_gregory_plot.py":       True,
    "part21_crf_bias_maps.py":      True,
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
        print(f"Submitting {script} as:")
        subprocess.run(["sbatch", job_script])
        print(f"___________________________")
        destination = f"tmp/{job_script}"
        shutil.move(job_script, destination)
    else:
        print(f"Skipped {script} (disabled)")
        print(f"___________________________")

