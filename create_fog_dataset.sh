#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 24
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --partition=ihub
#SBATCH --output=create_fog_dataset.txt
#SBATCH --job-name=create_fog_dataset

# Paired foggy/clean video dataset (fog model and parameters as in the paper's
# Dataset section, native resolution end to end), built leak-free:
#
#   [0] leak-free split: SPLIT_JSON assigns scenes to splits; continuity between
#       scenes is detected from the frames and a BUFFER_S buffer is cut at every
#       continuous train|eval boundary; the job ABORTS on any leak (shared scene,
#       unbuffered boundary, cross-split near-duplicate frame)
#   [1] fog: stratified per split, so val and test contain every fog variant
#       train does (all MOR levels, uniform+patchy, all airlights, all fog
#       layers); MOR ladder 300 200 150 100 75 50 m (paper) + 30 20 m
#         train 4 configs/scene | val 3/scene (all 8 MORs) | test 8/scene (every
#         MOR on every test scene)
#   [2] semantic masks
#   [3] leak re-check of the finished dataset
#
# Output, per split (train / val / test):
#   GT/<scene>/        clean frames (copied byte-for-byte from SOURCE, minus buffers)
#   Haze/<scene>_cfg<i>_beta<b>/   foggy frames (lossless PNG) + meta.json
#   Semantic/<scene>/  SegFormer-B5 Cityscapes masks, uint8 trainIds, GT resolution
#   + manifest.csv, split_report.json (split + buffers), leak_check.json at the root
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
SPLIT_JSON="${SPLIT_JSON:-$JARVIS/splits/edfog_leakfree.json}"
BUFFER_S="${BUFFER_S:-15}"
OUT="${OUT:-/scratch/Ananya_Kulkarni/Defog_fog_8mor}"
MOR_LADDER="${MOR_LADDER:-300,200,150,100,75,50,30,20}"
HET_MIX="${HET_MIX:-patchy:0.1,uniform:0.9}"
TRAIN_CONFIGS="${TRAIN_CONFIGS:-4}"
VAL_CONFIGS="${VAL_CONFIGS:-3}"
TEST_CONFIGS="${TEST_CONFIGS:-8}"
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
SPLIT_SRC="$OUT/.split_source"                                  # symlinks to the kept source frames

echo "=== [0/3] leak-free split ($SPLIT_JSON, ${BUFFER_S}s buffers) ==="
"$PY" "$JARVIS/leakfree_split.py" --source "$SOURCE" --split_json "$SPLIT_JSON" \
    --out "$SPLIT_SRC" --buffer_s "$BUFFER_S" --fps 30 \
    || { echo "!!! LEAK CHECK FAILED - nothing was rendered"; exit 1; }
cp "$SPLIT_SRC/split_report.json" "$OUT/split_report.json"

echo "=== [1/3] fog: $NUM_GPUS GPU(s) x $FOG_JOBS_PER_GPU worker(s) -> $OUT ==="
FOG_ARGS=(--engine_file "$JARVIS/fog_synthesis.py" --out_root "$OUT"
          --train_root "$SPLIT_SRC/train/GT" --val_root "$SPLIT_SRC/val/GT" --test_root "$SPLIT_SRC/test/GT"
          --n_configs_train "$TRAIN_CONFIGS" --n_configs_eval "$VAL_CONFIGS" --n_configs_test "$TEST_CONFIGS"
          --mor_ladder "$MOR_LADDER" --heterogeneity_mix "$HET_MIX" --global_seed "$SEED"
          --vda_root /scratch/Ananya_Kulkarni/Video-Depth-Anything --encoder vits
          --max_frames "$MAX_FRAMES" --fps 30 --vda_batch 4 --vda_overlap 2
          --num_gpus "$NUM_GPUS" --jobs_per_gpu "$FOG_JOBS_PER_GPU" --threads_per_worker "$THREADS"
          --copy_gt --skip_existing)
[[ -n "$RESIZE" ]] && FOG_ARGS+=(--resize "$RESIZE")
"$PY" "$JARVIS/build_fog_dataset.py" "${FOG_ARGS[@]}" 2>&1 | sed -u '/it\/s\]/d; /s\/it\]/d'

echo "=== [2/3] semantic masks: $NUM_GPUS GPU(s) x $SEM_JOBS_PER_GPU worker(s), $SEM_FORMAT uint8 ==="
"$PY" "$JARVIS/build_semantic_masks.py" --root "$OUT" --format "$SEM_FORMAT" \
    --num_gpus "$NUM_GPUS" --jobs_per_gpu "$SEM_JOBS_PER_GPU" --threads_per_worker "$THREADS" \
    --max_frames "$MAX_FRAMES" --skip_existing 2>&1 | sed -u '/it\/s\]/d; /s\/it\]/d'

echo "=== [3/3] leak re-check of the finished dataset ==="
"$PY" "$JARVIS/leakfree_split.py" --check_only --root "$OUT" \
    || { echo "!!! LEAK CHECK OF THE FINISHED DATASET FAILED - see $OUT/leak_check.json"; exit 1; }

for f in failed_jobs.txt failed_semantic_jobs.txt; do
    [[ -s "$OUT/$f" ]] && { echo "!!! failures listed in $OUT/$f"; cat "$OUT/$f"; }
done
echo "Job completed at $(date)"
echo "Dataset at $OUT  ({train,val,test}/{GT,Haze,Semantic} + manifest.csv)"
