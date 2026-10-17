#!/bin/bash

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/group/pmc073/huwang/TimeLens}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/pmc073/huwang/timelens}
DATA_ROOT=${DATA_ROOT:-$SCRATCH_ROOT/data/TimeLens-Bench}
MODEL_PATH_FILE=${MODEL_PATH_FILE:-$SCRATCH_ROOT/models/timelens8b_snapshot_path.txt}
RUN_TAG=${RUN_TAG:-timelens8b_dm_smoke_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$SCRATCH_ROOT/outputs/$RUN_TAG}
LOG_DIR=${LOG_DIR:-$SCRATCH_ROOT/logs}

MIN_TOKENS=${MIN_TOKENS:-64}
TOTAL_TOKENS=${TOTAL_TOKENS:-14336}
FPS=${FPS:-2}
DTYPE=${DTYPE:-auto}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
SDPA_BACKEND=${SDPA_BACKEND:-efficient}
USE_CACHE=${USE_CACHE:-auto}
FORCE_REPEAT_KV_FOR_SDPA=${FORCE_REPEAT_KV_FOR_SDPA:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
PIN_MEMORY=${PIN_MEMORY:-false}
DEVICE=${DEVICE:-balanced}
MAX_GPU_MEMORY_GIB=${MAX_GPU_MEMORY_GIB:-14}
CPU_OFFLOAD_GIB=${CPU_OFFLOAD_GIB:-120}
OFFLOAD_STATE_DICT=${OFFLOAD_STATE_DICT:-true}
SMOKE_DATASET=${SMOKE_DATASET:-activitynet-timelens}
SMOKE_MAX_EXAMPLES=${SMOKE_MAX_EXAMPLES:-3}

mkdir -p "$RUN_DIR" "$LOG_DIR"

if [ ! -s "$MODEL_PATH_FILE" ]; then
    echo "Missing TimeLens-8B model path file: $MODEL_PATH_FILE" >&2
    echo "Run scripts/kaya/prepare_kaya_timelens.sh with MODEL_ID=TencentARC/TimeLens-8B first." >&2
    exit 1
fi

job_id=$(
    sbatch --parsable \
        --job-name tl8b_dm_smoke \
        --export "ALL,SCRATCH_ROOT=$SCRATCH_ROOT,DATA_ROOT=$DATA_ROOT,RUN_DIR=$RUN_DIR,DATASET=$SMOKE_DATASET,MIN_TOKENS=$MIN_TOKENS,TOTAL_TOKENS=$TOTAL_TOKENS,FPS=$FPS,DTYPE=$DTYPE,ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION,SDPA_BACKEND=$SDPA_BACKEND,USE_CACHE=$USE_CACHE,FORCE_REPEAT_KV_FOR_SDPA=$FORCE_REPEAT_KV_FOR_SDPA,NUM_WORKERS=$NUM_WORKERS,PIN_MEMORY=$PIN_MEMORY,MODEL_PATH_FILE=$MODEL_PATH_FILE,MAX_EXAMPLES=$SMOKE_MAX_EXAMPLES,COMPUTE_METRICS=0,DEVICE=$DEVICE,MAX_GPU_MEMORY_GIB=$MAX_GPU_MEMORY_GIB,CPU_OFFLOAD_GIB=$CPU_OFFLOAD_GIB,OFFLOAD_STATE_DICT=$OFFLOAD_STATE_DICT,OFFLOAD_FOLDER=$SCRATCH_ROOT/offload/$RUN_TAG" \
        "$REPO_ROOT/scripts/kaya/run_timelens_device_map_smoke.slurm"
)

cat > "$RUN_DIR/device_map_smoke_submission.txt" <<EOF
run_dir=$RUN_DIR
model_path=$(cat "$MODEL_PATH_FILE")
dataset=$SMOKE_DATASET
official_input=min_tokens=$MIN_TOKENS,total_tokens=$TOTAL_TOKENS,fps=$FPS
runtime=device=$DEVICE,max_gpu_memory_gib=$MAX_GPU_MEMORY_GIB,cpu_offload_gib=$CPU_OFFLOAD_GIB,offload_state_dict=$OFFLOAD_STATE_DICT,dtype=$DTYPE,attn_implementation=$ATTN_IMPLEMENTATION,sdpa_backend=$SDPA_BACKEND,num_workers=$NUM_WORKERS,pin_memory=$PIN_MEMORY
smoke_job=$job_id
EOF

echo "smoke_job=$job_id"
echo "run_dir=$RUN_DIR"
echo "summary=$RUN_DIR/device_map_smoke_submission.txt"
