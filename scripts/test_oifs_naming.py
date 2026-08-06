#!/usr/bin/env python3
"""
Check the OpenIFS filename guards against synthetic run directories.

The guards only ever look at file names - they stat paths and never open a
file - so empty .nc files are enough to exercise every code path. This builds
a set of throwaway directories, each representing one situation the guards are
meant to handle, and reports whether each behaves as intended.

    python scripts/test_oifs_naming.py configs/MR_default.py

A config argument is required only because scripts/utils.py loads one on
import; none of its values are used here. Add --keep to leave the fixture
directories in place for inspection.
"""

import io
import os
import sys
import shutil
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (detect_oifs_pattern, oifs_files, require_matching_coverage,
                   oifs_available_years, MixedOifsNamingError, OifsCoverageError,
                   _oifs_pattern_cache)

YEARS = [1850, 1851, 1852, 1853]


# -----------------------------------------------------------------------------
# Fixture building
# -----------------------------------------------------------------------------

def name(var, year, freq="1m", doubled=False):
    """One remapped OpenIFS file name, in either spelling."""
    if doubled:
        return f"atm_remapped_{freq}_{var}_{freq}_{year:04d}-{year:04d}.nc"
    return f"atm_remapped_{freq}_{var}_{year:04d}-{year:04d}.nc"


def build(root, label, files):
    """Create <root>/<label>/oifs/ holding the given empty files."""
    path = os.path.join(root, label)
    oifs = os.path.join(path, "oifs")
    os.makedirs(oifs, exist_ok=True)
    for filename in files:
        open(os.path.join(oifs, filename), "w").close()
    return path


# -----------------------------------------------------------------------------
# Checks - each returns True when the guard behaved as intended
# -----------------------------------------------------------------------------

def expect_pattern(path, var, doubled, freqs=("1m",)):
    """The resolver picks the spelling that is actually on disk."""
    template, _ = detect_oifs_pattern(path, var, YEARS, freqs=freqs)
    return template == name(var, 0, freqs[0], doubled).replace(
        "0000-0000", "{year:04d}-{year:04d}")


def expect_abort(error_class, call):
    """The guard refuses to continue."""
    try:
        call()
    except error_class:
        return True
    return False


def expect_single_spelling(path, var, doubled):
    """With the override on, exactly one spelling is used - never a blend."""
    files, _ = oifs_files(path, var, YEARS)
    marker = f"_{var}_1m_" if doubled else f"_{var}_1"
    return files and all(marker in os.path.basename(f) for f in files)


def section(title):
    print(f"\n{title}")
    print("-" * len(title))


def run_case(label, description, check):
    _oifs_pattern_cache.clear()          # each case probes a fresh directory
    try:
        ok = check()
    except Exception as exc:             # an unexpected error is a failure
        print(f"  {label:4s} FAIL  {description}\n         unexpected {type(exc).__name__}: {exc}")
        return False
    print(f"  {label:4s} {'PASS' if ok else 'FAIL'}  {description}")
    return ok


# -----------------------------------------------------------------------------

def main():
    keep = "--keep" in sys.argv
    root = tempfile.mkdtemp(prefix="oifs_naming_test_")

    print("Testing how scripts/utils.py resolves remapped OpenIFS file names.")
    print("Model versions spell the frequency tag two ways:")
    print("  atm_remapped_1m_ssr_1m_1850-1850.nc   (repeated)")
    print("  atm_remapped_1m_ssr_1850-1850.nc      (once)")
    print("Each case below is a fake run directory of empty .nc files - the code")
    print("only reads file names - checking that the right one is picked and that")
    print("directories which cannot be read safely stop the run instead.")
    print(f"\nFixtures in {root}")

    crf = ["tsr", "tsrc", "ttr", "ttrc"]
    results = []

    # --- the two normal cases: one spelling, nothing to complain about -------
    single = build(root, "single", [name("ssr", y) for y in YEARS])
    doubled = build(root, "doubled", [name("ssr", y, doubled=True) for y in YEARS])

    section("Ordinary directories: the right spelling is picked")
    results.append(run_case(
        "G1", "new-style single tag resolves, no guard fires",
        lambda: expect_pattern(single, "ssr", doubled=False)))
    results.append(run_case(
        "G2", "old-style doubled tag resolves, no guard fires",
        lambda: expect_pattern(doubled, "ssr", doubled=True)))

    # --- both spellings present: must abort ---------------------------------
    same = build(root, "same_years",
                 [name("ssr", y) for y in YEARS] +
                 [name("ssr", y, doubled=True) for y in YEARS])
    split = build(root, "split_years",
                  [name("ssr", y, doubled=True) for y in YEARS[:2]] +
                  [name("ssr", y) for y in YEARS[2:]])

    section("Both spellings in one directory: the run must stop")
    results.append(run_case(
        "G3", "both spellings, same years -> abort",
        lambda: expect_abort(MixedOifsNamingError,
                             lambda: detect_oifs_pattern(same, "ssr", YEARS))))
    results.append(run_case(
        "G4", "both spellings, different years -> abort (would silently drop half)",
        lambda: expect_abort(MixedOifsNamingError,
                             lambda: detect_oifs_pattern(split, "ssr", YEARS))))

    # --- the override --------------------------------------------------------
    section("REVAL_ALLOW_MIXED_NAMING: opting out, still without blending")
    os.environ["REVAL_ALLOW_MIXED_NAMING"] = "1"
    results.append(run_case(
        "G5", "override on -> continues using ONE spelling only",
        lambda: expect_single_spelling(split, "ssr", doubled=True)))

    os.environ["REVAL_ALLOW_MIXED_NAMING"] = "0"
    results.append(run_case(
        "G6", "override explicitly off -> still aborts",
        lambda: expect_abort(MixedOifsNamingError,
                             lambda: detect_oifs_pattern(split, "ssr", YEARS))))
    del os.environ["REVAL_ALLOW_MIXED_NAMING"]

    # --- two frequencies are a preference, not a conflict --------------------
    freqmix = build(root, "freq_mix",
                    [name("ssr", y, "1m", doubled=True) for y in YEARS] +
                    [name("ssr", y, "6h", doubled=True) for y in YEARS])
    section("Different frequencies are a preference, not a conflict")
    results.append(run_case(
        "G7", "1m and 6h for one variable -> 1m preferred, no abort",
        lambda: detect_oifs_pattern(freqmix, "ssr", YEARS, freqs=("1m", "6h"))[1] == "1m"))

    # --- coverage of variables that get differenced --------------------------
    bad = build(root, "coverage_bad",
                [name(v, y) for v in ("tsr", "ttrc") for y in YEARS] +
                [name("tsrc", y) for y in YEARS[:2]] +
                [name("ttr", y) for y in YEARS[1:]])
    short = build(root, "coverage_short",
                  [name(v, y) for v in crf for y in YEARS[:2]])

    section("part21: variables that get differenced must cover the same years")
    results.append(run_case(
        "G8", "CRF variables covering different years -> abort (unphysical bias)",
        lambda: expect_abort(OifsCoverageError,
                             lambda: require_matching_coverage(bad, crf, YEARS))))
    results.append(run_case(
        "G9", "CRF variables uniformly short -> allowed, only noted",
        lambda: require_matching_coverage(short, crf, YEARS) == YEARS[:2]))

    # --- absent variable is not an error -------------------------------------
    section("A missing variable is not an error")
    results.append(run_case(
        "G11", "absent variable -> (None, None), availability checks still work",
        lambda: detect_oifs_pattern(single, "nosuchvar", YEARS) == (None, None)))

    # --- batch behaviour ----------------------------------------------------
    section("Failures reach you differently under sbatch than in a terminal")
    results.append(run_case(
        "G10", "under sbatch -> failed status file and exit 1, no traceback",
        lambda: check_sbatch_path(split)))

    shutil.rmtree(root) if not keep else print(f"\nFixtures kept in {root}")

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


def check_sbatch_path(path):
    """Probe as a batch job would: SLURM_JOB_ID set and stdout not a terminal.

    The guard should call sys.exit rather than raise, after recording a failed
    status under the running script's name. StringIO stands in for the job log:
    its isatty() is False, which is what tells the guard it is running batch.
    """
    status_name = "oifs_batch_probe"
    status_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "logs", "status", status_name)

    saved = sys.stdout, sys.stderr, sys.argv
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    sys.argv = [f"{status_name}.py"]          # what update_status will name it
    os.environ["SLURM_JOB_ID"] = "1"
    code, raised = None, None
    try:
        detect_oifs_pattern(path, "ssr", YEARS)
    except SystemExit as exc:
        code = exc.code
    except BaseException as exc:              # a traceback is the wrong outcome
        raised = exc
    finally:
        message = sys.stderr.getvalue()
        sys.stdout, sys.stderr, sys.argv = saved
        del os.environ["SLURM_JOB_ID"]

    wrote_status = os.path.exists(status_file)
    if wrote_status:
        os.remove(status_file)                # not a real script, don't leave it
    if raised is not None:
        print(f"         raised {type(raised).__name__} instead of exiting")
    elif code != 1:
        print(f"         exit code was {code!r}, expected 1")
    elif not wrote_status:
        print(f"         no status file at {status_file}")

    return code == 1 and wrote_status and "Mixed file naming" in message


if __name__ == "__main__":
    main()
