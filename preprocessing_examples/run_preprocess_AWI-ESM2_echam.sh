#!/bin/bash
#SBATCH --account=ab0246
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=preproc_ESM2_echam
#SBATCH --output=preproc_ESM2_echam_%j.out
#SBATCH --error=preproc_ESM2_echam_%j.err

# SLURM wrapper for preprocess_AWI-ESM2_echam.sh.
#
# Defaults to PI_wisofix_c historic block (5840-6004) and links the
# output dir as outdata/oifs so reval atmosphere scripts find it.
#
# Override by passing args (same order as the underlying script):
#   sbatch run_preprocess_AWI-ESM2_echam.sh <expname> <indir> <outdir> <starty> <endy> [grid]

set -euo pipefail

# Under sbatch, BASH_SOURCE points at slurm's spool copy of this wrapper,
# not the original. SLURM_SUBMIT_DIR is where sbatch was invoked from.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PREPROCESS_SCRIPT="${SCRIPT_DIR}/preprocess_AWI-ESM2_echam.sh"

# Defaults for PI_wisofix_c historic block. The indir points through a
# writable workspace mirror (see configs/AWI-ESM2-PI_wisofix_c.py) so we
# don't need write access on the source experiment dir.
EXP=${1:-PI_wisofix_c}
INDIR=${2:-/work/ab0246/a270092/runtime/PI_wisofix_c/outdata/echam}
OUTDIR=${3:-/work/ab0246/a270092/runtime/PI_wisofix_c/outdata/echam_remapped}
STARTY=${4:-5840}
ENDY=${5:-6004}
GRID=${6:-r360x180}

echo "=========================================="
echo "AWI-ESM2 echam preprocessing"
echo "  exp:    $EXP"
echo "  indir:  $INDIR"
echo "  outdir: $OUTDIR"
echo "  years:  $STARTY-$ENDY"
echo "  grid:   $GRID"
echo "  job:    ${SLURM_JOB_ID:-(local)}"
echo "=========================================="

bash "$PREPROCESS_SCRIPT" "$EXP" "$INDIR" "$OUTDIR" "$STARTY" "$ENDY" "$GRID"

# Symlink the output dir as outdata/oifs so reval finds it under the
# hardcoded /oifs/ path the part scripts use. Idempotent.
parent="$(dirname "$OUTDIR")"
target="$(basename "$OUTDIR")"
link="$parent/oifs"
if [ ! -e "$link" ]; then
    ln -s "$target" "$link"
    echo "Linked $link -> $target"
elif [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ]; then
    echo "Symlink $link already points at $target"
else
    echo "WARN: $link exists and is not the expected symlink; leaving it alone" >&2
fi

echo "=========================================="
echo "Done."
echo "=========================================="
