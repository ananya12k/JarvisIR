#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 28
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --partition=ihub
#SBATCH --output=ablatio.txt
#SBATCH --nodelist=gnode094
#SBATCH --job-name=ablatio

echo "Running on $(hostname)"
echo "Job started at $(date)"

# Activate your conda or venv if needed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate visia

cd /scratch/Ananya_Kulkarni/JarvisIR

python3 ablation_fog_modules_visia.py \
  --input-frames /scratch/Ananya_Kulkarni/Defog_Dataset/train/GT/00 \
  --checkpoint /scratch/Ananya_Kulkarni/VISIA/experiments/p3d_defog_v1/best_model.pth \
  --visia-root /scratch/Ananya_Kulkarni/VISIA \
  --vda-root /scratch/Ananya_Kulkarni/Video-Depth-Anything \
  --output-root /scratch/Ananya_Kulkarni/JarvisIR/ablation_outputs \
  --target-frames 900 \
  --encoder vits \
  --gpu-id 0 \
  --infer-gpu-ids 0,1 \
  --seq-len 7 \
  --tile-size 256 \
  --overlap 32

echo "Job completed at $(date)"
echo "All tasks finished successfully."
