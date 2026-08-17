#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 48
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --partition=ihub
#SBATCH --output=create_night_data.txt
#SBATCH --nodelist=gnode103
#SBATCH --job-name=create_night_data

echo "Running on $(hostname)"
echo "Job started at $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate visia

cd /scratch/Ananya_Kulkarni/JarvisIR

# 4 GPUs, all splits (train/val/test), 6 TOD variants per frame
python3 create_night_data.py --num_gpus 4 --splits train val test

echo "Job completed at $(date)"
echo "All tasks finished successfully."
