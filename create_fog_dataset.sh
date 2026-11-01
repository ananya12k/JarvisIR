#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 24
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --partition=ihub
#SBATCH --output=create_fog_dataset.txt
#SBATCH --job-name=create_fog_dataset

# Paired foggy/clean video dataset, set up as in the paper's Dataset section:
#   - fog mix: uniform p=0.9 / patchy p=0.1
#   - train 4 fog configs per scene, val/test 2 per scene
#   - MOR ladder repeated and deterministically shuffled so each level gets an
#     equal share, beta = 3 / MOR
#   - native resolution end to end
# The one extension is the ladder: 300 200 150 100 75 50 m (paper) + 30 20 m.
#
# Output, per split (train / val / test):
#   GT/<scene>/        clean frames (copied byte-for-byte from SOURCE)
#   Haze/<scene>_cfg<i>_beta<b>/   foggy frames (lossless PNG) + meta.json
#   Semantic/<scene>/  SegFormer-B5 Cityscapes masks, uint8 trainIds, GT resolution
#   + manifest.csv at the root
#
# Both stages use every allocated GPU through a shared, longest-first job queue
# with several workers per GPU, so no GPU idles while work remains.
#
# Everything below can be overridden from the environment, e.g.
#   MOR_LADDER=300,200,150,100,75,50 sbatch create_fog_dataset.sh   # paper ladder only
#   MAX_FRAMES=3500 sbatch create_fog_dataset.sh                    # cap long scenes like the shipped EDFog
#   SEM_FORMAT=png sbatch create_fog_dataset.sh                     # ~10x smaller masks (loader must read png)
#   RESIZE=480x270 sbatch create_fog_dataset.sh                     # NOT the paper: size of the shipped EDFog

set -Eeuo pipefail
echo "Running on $(hostname)"
echo "Job started at $(date)"

# Call the env's interpreter directly: after `conda activate visia`, plain `python`
# on these nodes resolves to an interpreter without cv2.
PY="${PY:-$HOME/miniconda3/envs/visia/bin/python}"

JARVIS=/scratch/Ananya_Kulkarni/JarvisIR/degradation_synthesis
SOURCE="${SOURCE:-/scratch/Ananya_Kulkarni/Defog_Dataset}"     # <split>/GT/<scene>/<frames>
OUT="${OUT:-/scratch/Ananya_Kulkarni/Defog_fog_8mor}"
MOR_LADDER="${MOR_LADDER:-300,200,150,100,75,50,30,20}"
HET_MIX="${HET_MIX:-patchy:0.1,uniform:0.9}"
TRAIN_CONFIGS="${TRAIN_CONFIGS:-4}"
VAL_CONFIGS="${VAL_CONFIGS:-2}"
TEST_CONFIGS="${TEST_CONFIGS:-2}"
SEED="${SEED:-edfog-8mor}"
RESIZE="${RESIZE:-}"                                            # empty = native resolution (paper)
MAX_FRAMES="${MAX_FRAMES:-0}"                                   # 0 = all frames of every scene
FOG_JOBS_PER_GPU="${FOG_JOBS_PER_GPU:-2}"                       # overlap CPU fog maths with VDA
SEM_JOBS_PER_GPU="${SEM_JOBS_PER_GPU:-1}"
SEM_FORMAT="${SEM_FORMAT:-npy}"                                 # npy = what VISIA's loader reads
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
THREADS=$(( ${SLURM_CPUS_PER_TASK:-$(nproc)} / (NUM_GPUS * FOG_JOBS_PER_GPU) ))
(( THREADS < 1 )) && THREADS=1

mkdir -p "$OUT"

echo "=== [1/2] fog: $NUM_GPUS GPU(s) x $FOG_JOBS_PER_GPU worker(s) -> $OUT ==="
FOG_ARGS=(--engine_file "$JARVIS/fog_synthesis.py" --out_root "$OUT"
          --train_root "$SOURCE/train/GT" --val_root "$SOURCE/val/GT" --test_root "$SOURCE/test/GT"
          --n_configs_train "$TRAIN_CONFIGS" --n_configs_eval "$VAL_CONFIGS" --n_configs_test "$TEST_CONFIGS"
          --mor_ladder "$MOR_LADDER" --heterogeneity_mix "$HET_MIX" --global_seed "$SEED"
          --vda_root /scratch/Ananya_Kulkarni/Video-Depth-Anything --encoder vits
          --max_frames "$MAX_FRAMES" --fps 30 --vda_batch 4 --vda_overlap 2
          --num_gpus "$NUM_GPUS" --jobs_per_gpu "$FOG_JOBS_PER_GPU" --threads_per_worker "$THREADS"
          --copy_gt --skip_existing)
[[ -n "$RESIZE" ]] && FOG_ARGS+=(--resize "$RESIZE")
"$PY" "$JARVIS/build_fog_dataset.py" "${FOG_ARGS[@]}" 2>&1 | sed -u '/it\/s\]/d; /s\/it\]/d'

echo "=== [2/2] semantic masks: $NUM_GPUS GPU(s) x $SEM_JOBS_PER_GPU worker(s), $SEM_FORMAT uint8 ==="
"$PY" "$JARVIS/build_semantic_masks.py" --root "$OUT" --format "$SEM_FORMAT" \
    --num_gpus "$NUM_GPUS" --jobs_per_gpu "$SEM_JOBS_PER_GPU" --threads_per_worker "$THREADS" \
    --max_frames "$MAX_FRAMES" --skip_existing 2>&1 | sed -u '/it\/s\]/d; /s\/it\]/d'

for f in failed_jobs.txt failed_semantic_jobs.txt; do
    [[ -s "$OUT/$f" ]] && { echo "!!! failures listed in $OUT/$f"; cat "$OUT/$f"; }
done
echo "Job completed at $(date)"
echo "Dataset at $OUT  ({train,val,test}/{GT,Haze,Semantic} + manifest.csv)"
