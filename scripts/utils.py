#!/usr/bin/env python3
"""
Utility functions for the release evaluation tool.
"""

import os
import sys
import glob
import subprocess
import fcntl
from pathlib import Path

# Add the parent directory to sys.path and load config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bg_routines.config_loader import *
from bg_routines.update_status import update_status

import psutil
import math

# -----------------------------------------------------------------------------
# Remapped OpenIFS file naming
# -----------------------------------------------------------------------------
# The frequency tag is not written consistently across model versions:
#   older runs:  atm_remapped_1m_ssr_1m_1850-1850.nc   (tag repeated)
#   newer runs:  atm_remapped_1m_ssr_1850-1850.nc      (tag written once)
# Scripts must not hardcode either form - use oifs_file()/oifs_files(), which
# probe the experiment directory once and remember what it uses.
#
# Pressure level output is addressed by passing the full tag, e.g. freq='1m_pl'.
#
# A directory holding both spellings for one variable aborts by default, since
# either choice silently drops the years written under the other. To override,
# set REVAL_ALLOW_MIXED_NAMING=1 or allow_mixed_oifs_naming=True in the config:
# the first spelling below that matches anything is then used for every year,
# with a warning. Only ever one spelling is used - the two are never combined.

_OIFS_NAME_VARIANTS = (
    "atm_remapped_{freq}_{var}_{freq}_{{year:04d}}-{{year:04d}}.nc",
    "atm_remapped_{freq}_{var}_{{year:04d}}-{{year:04d}}.nc",
)

_oifs_pattern_cache = {}


def oifs_dir(exp_path):
    """Return the oifs output directory of an experiment.

    Accepts either the outdata directory or the oifs directory itself, so
    call sites can pass whatever they already have.
    """
    exp_path = str(exp_path).rstrip("/")
    if os.path.basename(exp_path) == "oifs":
        return exp_path
    return os.path.join(exp_path, "oifs")


class MixedOifsNamingError(RuntimeError):
    """One directory holds two naming conventions for the same variable."""


class OifsCoverageError(RuntimeError):
    """Variables meant to be combined cover different years."""


def _running_under_sbatch():
    """True when running as a SLURM batch job rather than from a terminal.

    reval.py submits every script with sbatch, where stdout goes to the job
    log and nobody watches a traceback - the status file is what the report
    reads. An interactive salloc session also sets SLURM_JOB_ID, so the tty
    check is what separates the two.
    """
    return bool(os.environ.get("SLURM_JOB_ID")) and not sys.stdout.isatty()


def _abort(message, status, error_class):
    """Fail loudly interactively, or with a status file under sbatch."""
    if _running_under_sbatch():
        script = os.path.basename(sys.argv[0]) or "unknown_script"
        update_status(script, status)
        print(f"ERROR: {message}", file=sys.stderr)
        sys.exit(1)
    raise error_class(message)


def require_matching_coverage(exp_path, variables, years, freqs=("1m",)):
    """
    Require that `variables` all cover the same years, and return those years.

    Climatologies of different variables are routinely differenced (CRF is
    all-sky minus clear-sky). If the variables cover different periods the
    difference is not short, it is wrong, and nothing in the result shows it.
    Uniformly short coverage is fine and only warned about by the callers.

    Raises:
    -------
    OifsCoverageError : coverage differs between variables
                        (exits with a failed status under sbatch instead)
    """
    coverage = {var: oifs_available_years(exp_path, var, years, freqs)
                for var in variables}

    if len({tuple(v) for v in coverage.values()}) <= 1:
        return next(iter(coverage.values())) if coverage else []

    lines = [
        "Variables to be combined cover different years in",
        f"  {oifs_dir(exp_path)}",
        "",
    ]
    for var, found in coverage.items():
        lines.append(f"  {var:6s} {len(found):4d} year(s): "
                     f"{_format_year_spans(found) or 'none'}")
    lines += [
        "",
        "Refusing to continue. These fields are differenced against each other,",
        "so climatologies built from different periods would give a bias that",
        "looks plausible but is not physical.",
        "",
        "Restrict the configured year range to the years all variables share,",
        "or complete the preprocessing for the ones that lag behind.",
    ]
    _abort("\n".join(lines), " Failed - variable coverage mismatch", OifsCoverageError)


def _mixed_naming_allowed():
    """True when the caller has opted out of the mixed-naming guard.

    Set REVAL_ALLOW_MIXED_NAMING=1 for a single run, or
    allow_mixed_oifs_naming=True in the config to make it stick. The
    environment variable wins if both are set.
    """
    env = os.environ.get("REVAL_ALLOW_MIXED_NAMING", "").strip().lower()
    if env:
        return env not in ("0", "false", "no", "off")
    return bool(globals().get("allow_mixed_oifs_naming", False))


def _handle_mixed_naming(path, var, freq, matches):
    """Report a directory that uses two conventions for the same variable.

    Aborts unless the guard has been disabled, in which case it warns and
    returns, letting the caller fall through to the first listed spelling.
    That single spelling is then used for every year - the conventions are
    never combined either way.
    """
    lines = [
        f"Mixed file naming for OpenIFS variable '{var}' ({freq}) in",
        f"  {path}",
        "",
        "More than one naming convention is present for this variable:",
    ]
    for template, hits in matches:
        span = f"{min(hits)}-{max(hits)}" if len(hits) > 1 else str(hits[0])
        lines.append(f"  {template.format(year=min(hits))}")
        lines.append(f"      {len(hits)} year(s) matched: {span}")

    if _mixed_naming_allowed():
        chosen, hits = matches[0]
        lines += [
            "",
            "Guard disabled, continuing with the first convention only:",
            f"  {chosen.format(year=min(hits))}",
            "Years written under the other convention are NOT included.",
        ]
        print("WARNING: " + "\n".join(lines), file=sys.stderr)
        return

    lines += [
        "",
        "Refusing to continue. Picking one convention would silently drop the",
        "years written under the other, and combining them would splice output",
        "from two preprocessing runs into a single time series.",
        "",
        "Move the stale files aside so only one convention remains, then re-run.",
        "To use the first convention anyway, set REVAL_ALLOW_MIXED_NAMING=1.",
    ]
    _abort("\n".join(lines),
           f" Failed - mixed file naming for '{var}'",
           MixedOifsNamingError)


def _probe_oifs_pattern(path, var, years, freqs):
    """Find the naming variant actually present in `path` for `var`.

    Frequencies are tried in the given order, so a caller may prefer e.g.
    monthly over 6-hourly output. Within one frequency, however, two spellings
    of the same variable are a hard error rather than something to choose
    between - unless the guard is disabled, in which case the first spelling
    that matched is used for every year.
    """
    for freq in freqs:
        matches = []
        for variant in _OIFS_NAME_VARIANTS:
            template = variant.format(freq=freq, var=var)
            hits = [year for year in years
                    if os.path.exists(os.path.join(path, template.format(year=year)))]
            if hits:
                matches.append((template, hits))

        if len(matches) > 1:
            _handle_mixed_naming(path, var, freq, matches)
        if matches:
            return matches[0][0], freq

    # Nothing matched the known variants: derive the template from whatever
    # file is on disk, so future naming changes do not need a code change.
    for freq in freqs:
        found = {}
        for year in years:
            for hit in glob.glob(os.path.join(
                    path, f"atm_remapped_{freq}_{var}_*{year:04d}-{year:04d}.nc")):
                template = os.path.basename(hit).replace(
                    f"{year:04d}-{year:04d}", "{year:04d}-{year:04d}")
                found.setdefault(template, []).append(year)

        # insertion order, so the reported template is the one used below
        if len(found) > 1:
            _handle_mixed_naming(path, var, freq, list(found.items()))
        if found:
            return next(iter(found)), freq

    return None, None


def detect_oifs_pattern(exp_path, var, years, freqs=("1m",)):
    """
    Detect the filename pattern used for a remapped OpenIFS variable.

    Parameters:
    -----------
    exp_path : str
        Experiment outdata directory (or its oifs subdirectory)
    var : str
        OpenIFS variable name, e.g. 'ssr'
    years : iterable of int
        Years to probe; all are checked, so a directory that switched
        convention part-way through is detected rather than half-read
    freqs : str or sequence of str
        Frequency tags to try in order, e.g. ('1m', '6h') or '1m_pl'

    Returns:
    --------
    (template, freq) : template.format(year=1850) gives the file name,
                       or (None, None) if no file was found for any year

    Raises:
    -------
    MixedOifsNamingError : two spellings of one variable in one directory
                           (exits with a failed status under sbatch instead)
    """
    if isinstance(freqs, str):
        freqs = (freqs,)
    freqs = tuple(freqs)

    path = oifs_dir(exp_path)
    key = (path, var, freqs)
    if key not in _oifs_pattern_cache:
        _oifs_pattern_cache[key] = _probe_oifs_pattern(path, var, list(years), freqs)
    return _oifs_pattern_cache[key]


def oifs_file(exp_path, var, year, freqs=("1m",), years=None):
    """
    Full path of the remapped OpenIFS file holding `var` for `year`.

    `years` may be given so the pattern is probed against the full year range
    rather than a single year (files for individual years may be missing).
    Falls back to the current naming convention if nothing is found, so the
    caller reports a sensible path in its 'file missing' message.
    """
    path = oifs_dir(exp_path)
    template, _ = detect_oifs_pattern(path, var, years if years is not None else [year], freqs)

    if template is None:
        freq = freqs if isinstance(freqs, str) else tuple(freqs)[0]
        template = _OIFS_NAME_VARIANTS[-1].format(freq=freq, var=var)

    return os.path.join(path, template.format(year=year))


def _format_year_spans(years):
    """Render a sorted year list as compact ranges: [1850,1851,1853] -> '1850-1851, 1853'."""
    spans = []
    for year in sorted(years):
        if spans and year == spans[-1][1] + 1:
            spans[-1][1] = year
        else:
            spans.append([year, year])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)


def oifs_available_years(exp_path, var, years, freqs=("1m",)):
    """
    Which of `years` are actually on disk for `var`.

    Use this to check that variables about to be combined cover the same
    period; differencing climatologies built from different years is silently
    wrong rather than merely incomplete.
    """
    years = list(years)
    path = oifs_dir(exp_path)
    template, _ = detect_oifs_pattern(path, var, years, freqs)

    if template is None:
        return []

    return [year for year in years
            if os.path.exists(os.path.join(path, template.format(year=year)))]


def oifs_files(exp_path, var, years, freqs=("1m",), warn_missing=True):
    """
    Existing remapped OpenIFS files for `var` over `years`.

    Missing years are reported as a single summary line rather than one line
    each, which would scroll away in a long run. Set warn_missing=False when
    the caller checks coverage itself.

    Returns:
    --------
    (files, freq) : list of existing paths in year order and the detected
                    frequency tag, or ([], None) if the variable is absent
    """
    years = list(years)
    path = oifs_dir(exp_path)
    template, freq = detect_oifs_pattern(path, var, years, freqs)

    if template is None:
        return [], None

    found = [year for year in years
             if os.path.exists(os.path.join(path, template.format(year=year)))]

    if warn_missing and len(found) < len(years):
        missing = sorted(set(years) - set(found))
        print(f"WARNING: {var} ({freq}): {len(found)} of {len(years)} years found in {path}"
              f"\n         missing: {_format_year_spans(missing)}"
              f"\n         averages will cover {_format_year_spans(found) or 'nothing'}",
              file=sys.stderr)

    return [os.path.join(path, template.format(year=year)) for year in found], freq


def get_optimal_batch_size(file_path, safety_factor=4.0, max_procs=16, min_batch=1):
    """
    Calculate optimal batch size based on available memory and file size.
    
    Parameters:
    -----------
    file_path : str
        Path to a sample input file to check size
    safety_factor : float
        Memory safety factor (default 4.0: assume operation needs 4x file size in RAM)
    max_procs : int
        Maximum number of concurrent processes to allow
    min_batch : int
        Minimum batch size (default 1)
        
    Returns:
    --------
    int : Recommended batch size
    """
    try:
        # Get file size in GB
        if not os.path.exists(file_path):
            return min_batch
            
        file_size_gb = os.path.getsize(file_path) / (1024**3)
        
        # Get available memory in GB
        mem = psutil.virtual_memory()
        available_mem_gb = mem.available / (1024**3)
        
        # Estimate max concurrent processes
        # Formula: (Available RAM) / (File Size * Safety Factor)
        if file_size_gb > 0:
            est_procs = int(available_mem_gb / (file_size_gb * safety_factor))
        else:
            est_procs = max_procs
            
        # Clamp between min_batch and max_procs
        batch_size = max(min_batch, min(est_procs, max_procs))
        
        print(f"Batch sizing: File={file_size_gb:.1f}GB, RAM={available_mem_gb:.1f}GB")
        print(f"              Optimal batch size = {batch_size} (limit={max_procs})")
        
        return batch_size
        
    except Exception as e:
        print(f"Warning: Could not determine optimal batch size ({e}). Using default=4.")
        return 4

def ensure_weight_file(resolution, meshpath, mesh_file, variable='temp'):
    """
    Ensure CDO weight file exists for remapping from unstructured to regular grid.
    Generate if missing using CDO genycon. Thread-safe with file locking.
    
    Parameters:
    -----------
    resolution : str
        Target resolution (e.g., '360x180', '512x256')
    meshpath : str
        Path to mesh directory
    mesh_file : str
        Name of mesh file
    variable : str
        Variable name to use for weight generation (default: 'temp')
    
    Returns:
    --------
    str : Path to weight file
    
    Raises:
    -------
    FileNotFoundError : If no sample FESOM files found
    subprocess.CalledProcessError : If CDO command fails
    """
    weight_file = f"{meshpath}/weights_unstr_2_r{resolution}.nc"
    lock_file = f"{weight_file}.lock"
    
    # Return existing file if found
    if os.path.exists(weight_file):
        return weight_file
    
    # Use file locking to prevent concurrent generation
    try:
        with open(lock_file, 'w') as lock:
            # Try to acquire exclusive lock
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # Check again after acquiring lock (another process might have created it)
            if os.path.exists(weight_file):
                return weight_file
            
            print(f"Generating missing weight file: {weight_file}")
            
            # Find a sample FESOM file for grid definition
            sample_file = None
            common_vars = ['temp', 'salt', 'u', 'v', 'ssh']
            
            for var in common_vars:
                for path_candidate in [spinup_path, historic_path]:
                    if path_candidate and os.path.exists(path_candidate):
                        fesom_dir = os.path.join(path_candidate, 'fesom')
                        if os.path.exists(fesom_dir):
                            for file in os.listdir(fesom_dir):
                                if var in file and file.endswith('.nc'):
                                    sample_file = os.path.join(fesom_dir, file)
                                    break
                            if sample_file:
                                break
                    if sample_file:
                        break
                if sample_file:
                    break
            
            if not sample_file:
                raise FileNotFoundError(f"No FESOM sample files found to generate weights for {resolution}")
            
            # Generate weight file using CDO
            atm_gridfile_path = f"{meshpath}/{mesh_file}"
            cmd = [
                'cdo', 
                f'genycon,r{resolution}',
                '-selname,' + variable,
                '-setgrid,' + atm_gridfile_path,
                sample_file,
                weight_file
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if not os.path.exists(weight_file):
                raise FileNotFoundError(f"Weight file generation failed: {weight_file}")
            
            print(f"Generated: {weight_file}")
            
    except BlockingIOError:
        # Another process is generating the file, wait for it
        print(f"Waiting for weight file generation by another process: {weight_file}")
        max_wait = 300  # 5 minutes max wait
        wait_time = 0
        while not os.path.exists(weight_file) and wait_time < max_wait:
            time.sleep(1)
            wait_time += 1
        
        if not os.path.exists(weight_file):
            raise TimeoutError(f"Timeout waiting for weight file generation: {weight_file}")
    
    finally:
        # Clean up lock file
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except:
                pass
    
    return weight_file
