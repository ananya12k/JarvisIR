#!/bin/bash
# Builds the full-factorial fog dataset (every scene x every fog level) into a
# separate root at /scratch/Ananya_Kulkarni/Defog_balanced_fixed, using the fixed
# PhysicalFogEngine. Safe to re-run: both stages skip work that's already done.
#
# Stage 1 (build_uniform_dataset.py): copies GT + Semantic subsets (frame-capped at
#          3500, evenly subsampled for the 3 long GoPro scenes) into the new root,
#          and assigns each scene its own randomly sampled beta per level (persisted
#          to beta_plan.json so re-runs are deterministic, not re-randomized).
# Stage 2 (generate_uniform_haze.py): renders Haze frames for 4 levels — light,
#          light_medium, medium, heavy (no extreme) — for every scene, parallelized
#          across all available GPUs. Skips any scene+level folder already complete.



#SBATCH -A mobility_arfs
#SBATCH -c 48
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --partition=ihub
#SBATCH --output=rebuild_fog_dataset.txt
#SBATCH --nodelist=gnode095
#SBATCH --job-name=rebuild_fog_dataset


# Hardcoded, not derived from $BASH_SOURCE: sbatch copies this script into a spool
# directory (/var/spool/slurmd/jobNNNN/) and runs it from there, so BASH_SOURCE-based
# path resolution points at the spool copy, not this repo — which is exactly what
# broke the last run ("can't open file '/var/spool/slurmd/job.../build_uniform_dataset.py'").
SCRIPT_DIR="/scratch/Ananya_Kulkarni/JarvisIR/degradation_synthesis"

source ~/.bashrc
conda activate visia

echo "=== Stage 1: copying GT + Semantic subsets ==="
python "$SCRIPT_DIR/build_uniform_dataset.py"

echo ""
echo "=== Stage 2: full-factorial Haze generation (multi-GPU) ==="
python "$SCRIPT_DIR/generate_uniform_haze.py"

echo ""
echo "=== Done. Dataset at /scratch/Ananya_Kulkarni/Defog_balanced_fixed ==="
