#!/bin/bash
# Builds the full-factorial night dataset (every sequence x every condition) into a
# separate self-contained root at /scratch/Ananya_Kulkarni/Night_balanced_uniform,
# using the current continuous-severity NightEngine. Safe to re-run: both stages
# skip work that's already done.
#
# Stage 1 (build_uniform_night_dataset.py): copies GT subsets (frame-capped at 2000,
#          evenly subsampled for the long GoPro sequences) into the new root, and
#          draws each (sequence, cond slot) its own continuous severity params
#          (persisted to severity_plan.json so re-runs are deterministic, not
#          re-randomized).
# Stage 2 (generate_uniform_night.py): renders 6 condition slots per sequence,
#          parallelized across all available GPUs. Skips any sequence+condition
#          folder already complete.


#SBATCH -A mobility_arfs
#SBATCH -c 48
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --partition=ihub
#SBATCH --output=build_uniform_night_dataset.txt
#SBATCH --nodelist=gnode095
#SBATCH --job-name=build_uniform_night_dataset


# Hardcoded, not derived from $BASH_SOURCE: sbatch copies this script into a spool
# directory (/var/spool/slurmd/jobNNNN/) and runs it from there, so BASH_SOURCE-based
# path resolution points at the spool copy, not this repo.
SCRIPT_DIR="/scratch/Ananya_Kulkarni/JarvisIR/degradation_synthesis"

source ~/.bashrc
conda activate visia

echo "=== Stage 1: copying GT subsets + drawing severity plan ==="
python "$SCRIPT_DIR/build_uniform_night_dataset.py"

echo ""
echo "=== Stage 2: full-factorial night generation (multi-GPU) ==="
python "$SCRIPT_DIR/generate_uniform_night.py"

echo ""
echo "=== Done. Dataset at /scratch/Ananya_Kulkarni/Night_balanced_uniform ==="
