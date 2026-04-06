#!/bin/bash

#SBATCH -A inai
#SBATCH -c 24
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --partition=ihub
#SBATCH --output=create_fog_data.txt
#SBATCH --nodelist=gnode103
#SBATCH --job-name=create_fog_data

echo "Running on $(hostname)"
echo "Job started at $(date)"

# Activate your conda or venv if needed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate visia

cd /scratch/Ananya_Kulkarni/JarvisIR

# Run the two scripts in parallel using CUDA_VISIBLE_DEVICES
python3 create_fog_data.py


echo "Job completed at $(date)"
echo "All tasks finished successfully."

