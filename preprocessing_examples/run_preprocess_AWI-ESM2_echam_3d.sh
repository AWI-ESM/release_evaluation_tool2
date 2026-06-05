#!/bin/bash
#SBATCH --account=ab0246
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=preproc_ESM2_echam3d
#SBATCH --output=preproc_ESM2_echam3d_%j.out
#SBATCH --error=preproc_ESM2_echam3d_%j.err

# SLURM wrapper for preprocess_AWI-ESM2_echam_3d.sh (afterburner-based
# 3D pressure-level preprocessing for part11/part12).
#
# Override defaults by passing args:
#   sbatch run_preprocess_AWI-ESM2_echam_3d.sh <expname> <indir> <outdir> <starty> <endy> [grid]

set -euo pipefail

# sbatch copies this wrapper to its spool dir, so BASH_SOURCE points
# there. SLURM_SUBMIT_DIR is where sbatch was invoked from.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PREPROCESS_SCRIPT="${SCRIPT_DIR}/preprocess_AWI-ESM2_echam_3d.sh"

EXP=${1:-PI_wisofix_c}
INDIR=${2:-/work/ab0246/a270092/runtime/PI_wisofix_c/outdata/echam}
OUTDIR=${3:-/work/ab0246/a270092/runtime/PI_wisofix_c/outdata/echam_remapped}
STARTY=${4:-5840}
ENDY=${5:-6004}
GRID=${6:-r360x180}

echo "=========================================="
echo "AWI-ESM2 echam 3D pressure-level preprocessing"
echo "  exp:    $EXP"
echo "  indir:  $INDIR"
echo "  outdir: $OUTDIR"
echo "  years:  $STARTY-$ENDY"
echo "  grid:   $GRID"
echo "  job:    ${SLURM_JOB_ID:-(local)}"
echo "=========================================="

bash "$PREPROCESS_SCRIPT" "$EXP" "$INDIR" "$OUTDIR" "$STARTY" "$ENDY" "$GRID"

echo "=========================================="
echo "Done."
echo "=========================================="
