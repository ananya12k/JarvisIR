#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 24
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --partition=ihub
#SBATCH --output=create_fog_dataset.txt
#SBATCH --job-name=create_fog_dataset

# Paired foggy/clean video dataset with an 8-level visibility ladder:
#   MOR 300 200 150 100 75 50 30 20 m   (beta = 3 / MOR)
#
#   train : 4 fog configs per scene, MORs balanced over the split
#   val   : 3 configs per scene (3 val scenes x 3 = 9 clips covers all 8 MORs)
#   test  : 8 configs per scene = every MOR on every test scene
#
# Train/val use TRAIN_HET_MIX (uniform + the volumetric patchy_strong fog, for
# transfer to real fog). Test uses TEST_HET_MIX (the paper's original mix) so
# test numbers stay comparable with the shipped EDFog.
#
# Everything below can be overridden from the environment, e.g.
#   RESIZE=480x270 NUM_GPUS=2 sbatch create_fog_dataset.sh

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
TRAIN_HET_MIX="${TRAIN_HET_MIX:-uniform:0.4,patchy_strong:0.6}"
TEST_HET_MIX="${TEST_HET_MIX:-patchy:0.1,uniform:0.9}"
SEED="${SEED:-edfog-8mor}"
RESIZE="${RESIZE:-}"                                            # empty = native resolution
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"

COMMON=(--engine_file "$JARVIS/fog_synthesis.py" --out_root "$OUT"
        --mor_ladder "$MOR_LADDER" --global_seed "$SEED"
        --vda_root /scratch/Ananya_Kulkarni/Video-Depth-Anything --encoder vits
        --fps 30 --vda_batch 4 --vda_overlap 2 --skip_existing)
[[ -n "$RESIZE" ]] && COMMON+=(--resize "$RESIZE")

run_sharded() {   # args: tag, builder args...
    local tag=$1; shift
    echo "=== $tag on $NUM_GPUS GPU(s) ==="
    for ((g = 0; g < NUM_GPUS; g++)); do
        "$PY" "$JARVIS/build_fog_dataset.py" "${COMMON[@]}" "$@" \
            --gpu_id "$g" --shard_index "$g" --num_shards "$NUM_GPUS" \
            > "$OUT/log_${tag}_gpu${g}.txt" 2>&1 &
    done
    wait
    grep -h "done\|WARNING\|Error" "$OUT"/log_${tag}_gpu*.txt || true
}

mkdir -p "$OUT"

run_sharded trainval \
    --train_root "$SOURCE/train/GT" --val_root "$SOURCE/val/GT" \
    --n_configs_train 4 --n_configs_eval "${VAL_CONFIGS:-3}" \
    --heterogeneity_mix "$TRAIN_HET_MIX" --manifest_name manifest_trainval.csv

run_sharded test \
    --test_root "$SOURCE/test/GT" --n_configs_test 8 \
    --heterogeneity_mix "$TEST_HET_MIX" --manifest_name manifest_test.csv

# One manifest.csv for downstream tools (video_dataset.py, regen scripts).
"$PY" - "$OUT" <<'EOF'
import csv, sys, os
out = sys.argv[1]
rows = []
for name in ("manifest_trainval.csv", "manifest_test.csv"):
    with open(os.path.join(out, name)) as f:
        r = csv.DictReader(f); fields = r.fieldnames; rows += list(r)
with open(os.path.join(out, "manifest.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"manifest.csv: {len(rows)} clips")
EOF

echo "Job completed at $(date)"
echo "Dataset at $OUT"
