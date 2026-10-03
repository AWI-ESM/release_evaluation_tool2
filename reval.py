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

# Site detection: albedo (AWI) vs levante (DKRZ).  Override with REVAL_SITE.
SITE = os.environ.get("REVAL_SITE") or ("albedo" if os.path.isdir("/albedo") else "levante")

if SITE == "albedo":
    # albedo prod nodes: 128 cores, 256 GB.  smp shares nodes, so ask for a slice
    # rather than a whole node; the scripts are mostly serial + cdo.
    SBATCH_SETTINGS = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}.log
#SBATCH --error=logs/{job_name}.log
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=smp
#SBATCH -A clidyn.clidyn
#SBATCH --qos=12h
"""
    SBATCH_REPORT_SITE = """\
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --partition=smp
#SBATCH -A clidyn.clidyn
#SBATCH --qos=12h
"""
    # RLIMIT_NPROC is 2048 per user: cap BLAS/OpenMP threads or numpy dies when
    # several jobs share a node.
    ENV_SETUP = (
        "export PATH=/albedo/soft/sw/spack-sw/imagemagick/7.0.8-7-xi3o53s/bin:$PATH  # convert (figure trimming); RPATH binary, no LD_LIBRARY_PATH needed\n"
        "source $HOME/loadconda.sh\n"
        "conda activate " + os.environ.get(
            "REVAL_ENV", "/albedo/work/projects/p_awiesm3_cmip7/jstreffi/software/conda_envs/reval") + "\n"
        "export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} "
        "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}\n"
    )
else:
    SBATCH_SETTINGS = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}.log
#SBATCH --error=logs/{job_name}.log
#SBATCH --time=02:00:00
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute
#SBATCH -A ab0246
"""
    SBATCH_REPORT_SITE = """\
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH -A ab0246
"""
    ENV_SETUP = "source $HOME/loadconda.sh\nconda activate reval\n"



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
    "part1_mesh_plot.py":           True,
    "part2_rad_balance.py":         True,
    "part3_hovm_temp.py":           True,
    "part4_cmpi.py":                True,
    "part5_sea_ice_thickness.py":   True,
    "part6_ice_conc_timeseries.py": True,
    "part7_mld.py":                 True,
    "part8_t2m_vs_era5.py":         True,
    "part9_rad_vs_ceres.py":        True,
    "part10_clt_vs_modis.py":       True,
    "part11_zonal_plots.py":        True,
    "part12_qbo.py":                True,
    "part13_fesom_temp_bias.py":    True,
    "part14_fesom_salt_bias.py":    True,
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
    # Off by default: it reads a year of 3D vertical velocity per spin-up year (seconds and a
    # gigabyte each), which a multi-millennium spin-up cannot afford in one job. Enable it per
    # config with scripts_overrides = {"part27_amoc_timeseries.py": True}; results are cached
    # per year (spinup_cache_path), so a long record can be filled in several runs.
    "part27_amoc_timeseries.py":    False,
})

# Read `scripts_overrides` from the config without executing it (heavy
# imports like pyfesom2 mean we don't want to import the config here).
# This is a literal-only ast walk; only bool values are accepted.
def _read_scripts_overrides(path):
    import ast
    try:
        with open(path) as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'scripts_overrides':
                    try:
                        val = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        return {}
                    return val if isinstance(val, dict) else {}
    return {}

_overrides = _read_scripts_overrides(config_path)
if _overrides:
    print(f"Applying {len(_overrides)} scripts_overrides from config")
    for name, enabled in _overrides.items():
        if name not in SCRIPTS and enabled:
            # Allow enabling a script not in the default set (e.g. jsbach
            # replacements). It must still exist on disk.
            if not os.path.exists(os.path.join('scripts', name)):
                print(f"  WARN override '{name}' enabled but scripts/{name} missing - ignoring")
                continue
        SCRIPTS[name] = bool(enabled)
        print(f"  {name}: {'enabled' if enabled else 'disabled'}")

# Submit jobs and collect job IDs for the report dependency
submitted_job_ids = []

for script, run in SCRIPTS.items():
    if run:
        job_script = f"slurm_{script}.sh"
        script_path = os.path.join("scripts", script)

        # Write the SLURM script
        with open(job_script, "w") as f:
            f.write(SBATCH_SETTINGS.format(job_name=script))
            f.write("\n" + ENV_SETUP)  # site-specific conda activation
            f.write(f"\nexport REVAL_CONFIG={config_path}\n")  # Pass config file path
            f.write(f"python -u {script_path}\n")

        # Submit job and capture job ID
        print(f"Submitting {script} as:")
        result = subprocess.run(["sbatch", job_script], capture_output=True, text=True)
        print(result.stdout.strip())
        # Parse job ID from "Submitted batch job 12345678"
        if result.returncode == 0 and "Submitted" in result.stdout:
            try:
                job_id = result.stdout.strip().split()[-1]
                submitted_job_ids.append(job_id)
            except (IndexError, ValueError):
                pass
        destination = f"tmp/{job_script}"
        shutil.move(job_script, destination)
    else:
        print(f"Skipped {script} (disabled)")

# Submit HTML report generation after all analysis jobs finish
if submitted_job_ids:
    print(f"\n{'='*60}")
    print(f"Submitting HTML report generator (after {len(submitted_job_ids)} jobs)...")
    
    SBATCH_REPORT = """\
#!/bin/bash
#SBATCH --job-name=generate_report
#SBATCH --output=logs/generate_report.log
#SBATCH --error=logs/generate_report.log
#SBATCH --time=00:10:00
""" + SBATCH_REPORT_SITE
    report_script = "slurm_generate_report.sh"
    dep_str = ":".join(submitted_job_ids)
    with open(report_script, "w") as f:
        f.write(SBATCH_REPORT)
        f.write("\n" + ENV_SETUP)
        f.write(f"\nexport REVAL_CONFIG={config_path}\n")
        f.write("python -u scripts/generate_report.py\n")

    result = subprocess.run(
        ["sbatch", f"--dependency=afterany:{dep_str}", report_script],
        capture_output=True, text=True)
    print(result.stdout.strip())
    shutil.move(report_script, f"tmp/{report_script}")
    print("Report will be generated after all analysis jobs complete.")

