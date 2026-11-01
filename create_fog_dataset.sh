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
# The one extension is the ladder: 300 200 150 100 75 50 m (paper) + 30 20 m.
#
# Everything below can be overridden from the environment, e.g.
#   MOR_LADDER=300,200,150,100,75,50 sbatch create_fog_dataset.sh   # paper ladder only
#   TEST_CONFIGS=8 sbatch create_fog_dataset.sh                     # every MOR on every test scene
#   RESIZE=480x270 sbatch create_fog_dataset.sh                     # size of the shipped EDFog

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
RESIZE="${RESIZE:-}"                                            # empty = native resolution
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"

ARGS=(--engine_file "$JARVIS/fog_synthesis.py" --out_root "$OUT"
      --train_root "$SOURCE/train/GT" --val_root "$SOURCE/val/GT" --test_root "$SOURCE/test/GT"
      --n_configs_train "$TRAIN_CONFIGS" --n_configs_eval "$VAL_CONFIGS" --n_configs_test "$TEST_CONFIGS"
      --mor_ladder "$MOR_LADDER" --heterogeneity_mix "$HET_MIX" --global_seed "$SEED"
      --vda_root /scratch/Ananya_Kulkarni/Video-Depth-Anything --encoder vits
      --fps 30 --vda_batch 4 --vda_overlap 2 --skip_existing)
[[ -n "$RESIZE" ]] && ARGS+=(--resize "$RESIZE")

mkdir -p "$OUT"
echo "=== building on $NUM_GPUS GPU(s) -> $OUT ==="
for ((g = 0; g < NUM_GPUS; g++)); do
    "$PY" "$JARVIS/build_fog_dataset.py" "${ARGS[@]}" \
        --gpu_id "$g" --shard_index "$g" --num_shards "$NUM_GPUS" \
        > "$OUT/log_gpu${g}.txt" 2>&1 &
done
wait
grep -h "\[plan\]" "$OUT/log_gpu0.txt" || true
grep -h "done\|WARNING\|Error" "$OUT"/log_gpu*.txt || true

echo "Job completed at $(date)"
echo "Dataset at $OUT (manifest.csv lists every clip's fog parameters)"
