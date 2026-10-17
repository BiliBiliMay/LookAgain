#!/bin/bash

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/group/pmc073/huwang/TimeLens}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/pmc073/huwang/timelens}
DATA_ROOT=${DATA_ROOT:-$SCRATCH_ROOT/data/TimeLens-Bench}
MODEL_PATH_FILE=${MODEL_PATH_FILE:-$SCRATCH_ROOT/models/timelens8b_snapshot_path.txt}
PYTHON_BIN=${PYTHON_BIN:-$SCRATCH_ROOT/envs/timelens/bin/python}
RUN_TAG=${RUN_TAG:-timelens8b_dm_repro_$(date +%Y%m%d_%H%M%S)}
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
OFFLOAD_BASE=${OFFLOAD_BASE:-$SCRATCH_ROOT/offload/$RUN_TAG}

SMOKE_DATASET=${SMOKE_DATASET:-activitynet-timelens}
SMOKE_MAX_EXAMPLES=${SMOKE_MAX_EXAMPLES:-3}
SUBMIT_SMOKE=${SUBMIT_SMOKE:-0}

DATASETS=${DATASETS:-charades-timelens,activitynet-timelens,qvhighlights-timelens}
CHARADES_CHUNKS=${CHARADES_CHUNKS:-6}
ACTIVITYNET_CHUNKS=${ACTIVITYNET_CHUNKS:-6}
QVHIGHLIGHTS_CHUNKS=${QVHIGHLIGHTS_CHUNKS:-6}
MAX_CONCURRENT_PER_DATASET=${MAX_CONCURRENT_PER_DATASET:-1}

mkdir -p "$RUN_DIR" "$LOG_DIR" "$OFFLOAD_BASE"

if [ ! -s "$MODEL_PATH_FILE" ]; then
    echo "Missing TimeLens-8B model path file: $MODEL_PATH_FILE" >&2
    echo "Run scripts/kaya/prepare_kaya_timelens.sh with MODEL_ID=TencentARC/TimeLens-8B first." >&2
    exit 1
fi

for required in charades-timelens.json activitynet-timelens.json qvhighlights-timelens.json; do
    if [ ! -s "$DATA_ROOT/$required" ]; then
        echo "Missing benchmark annotation: $DATA_ROOT/$required" >&2
        exit 1
    fi
done

submit_smoke() {
    sbatch --parsable \
        --job-name tl8b_dm_smoke \
        --export "ALL,SCRATCH_ROOT=$SCRATCH_ROOT,DATA_ROOT=$DATA_ROOT,RUN_DIR=$RUN_DIR,DATASET=$SMOKE_DATASET,MIN_TOKENS=$MIN_TOKENS,TOTAL_TOKENS=$TOTAL_TOKENS,FPS=$FPS,DTYPE=$DTYPE,ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION,SDPA_BACKEND=$SDPA_BACKEND,USE_CACHE=$USE_CACHE,FORCE_REPEAT_KV_FOR_SDPA=$FORCE_REPEAT_KV_FOR_SDPA,NUM_WORKERS=$NUM_WORKERS,PIN_MEMORY=$PIN_MEMORY,MODEL_PATH_FILE=$MODEL_PATH_FILE,MAX_EXAMPLES=$SMOKE_MAX_EXAMPLES,COMPUTE_METRICS=0,DEVICE=$DEVICE,MAX_GPU_MEMORY_GIB=$MAX_GPU_MEMORY_GIB,CPU_OFFLOAD_GIB=$CPU_OFFLOAD_GIB,OFFLOAD_STATE_DICT=$OFFLOAD_STATE_DICT,OFFLOAD_FOLDER=$OFFLOAD_BASE/smoke" \
        "$REPO_ROOT/scripts/kaya/run_timelens_device_map_smoke.slurm"
}

submit_dataset_chunks() {
    local dataset=$1
    local chunk_count=$2
    local dependency=${3:-}
    local opts=(
        --parsable
        --job-name "tl8bdm_${dataset%%-*}"
        --array "0-$((chunk_count - 1))%$MAX_CONCURRENT_PER_DATASET"
        --export "ALL,SCRATCH_ROOT=$SCRATCH_ROOT,DATA_ROOT=$DATA_ROOT,RUN_DIR=$RUN_DIR,DATASET=$dataset,CHUNK=$chunk_count,MIN_TOKENS=$MIN_TOKENS,TOTAL_TOKENS=$TOTAL_TOKENS,FPS=$FPS,DTYPE=$DTYPE,ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION,SDPA_BACKEND=$SDPA_BACKEND,USE_CACHE=$USE_CACHE,FORCE_REPEAT_KV_FOR_SDPA=$FORCE_REPEAT_KV_FOR_SDPA,NUM_WORKERS=$NUM_WORKERS,PIN_MEMORY=$PIN_MEMORY,MODEL_PATH_FILE=$MODEL_PATH_FILE,MAX_EXAMPLES=,COMPUTE_METRICS=0,DEVICE=$DEVICE,MAX_GPU_MEMORY_GIB=$MAX_GPU_MEMORY_GIB,CPU_OFFLOAD_GIB=$CPU_OFFLOAD_GIB,OFFLOAD_STATE_DICT=$OFFLOAD_STATE_DICT,OFFLOAD_FOLDER=$OFFLOAD_BASE/$dataset"
    )
    if [ -n "$dependency" ]; then
        opts+=(--dependency "$dependency")
    fi
    sbatch "${opts[@]}" "$REPO_ROOT/scripts/kaya/run_timelens_device_map_chunk.slurm"
}

smoke_job=""
chunk_dependency=""
if [ "$SUBMIT_SMOKE" = "1" ]; then
    smoke_job=$(submit_smoke)
    chunk_dependency="afterok:$smoke_job"
    echo "smoke_job=$smoke_job"
fi

chunk_jobs=()
if [[ ",$DATASETS," == *",charades-timelens,"* ]]; then
    chunk_jobs+=("$(submit_dataset_chunks charades-timelens "$CHARADES_CHUNKS" "$chunk_dependency")")
fi
if [[ ",$DATASETS," == *",activitynet-timelens,"* ]]; then
    chunk_jobs+=("$(submit_dataset_chunks activitynet-timelens "$ACTIVITYNET_CHUNKS" "$chunk_dependency")")
fi
if [[ ",$DATASETS," == *",qvhighlights-timelens,"* ]]; then
    chunk_jobs+=("$(submit_dataset_chunks qvhighlights-timelens "$QVHIGHLIGHTS_CHUNKS" "$chunk_dependency")")
fi

if [ "${#chunk_jobs[@]}" -eq 0 ]; then
    echo "No datasets selected" >&2
    exit 1
fi

all_chunks=$(IFS=:; echo "${chunk_jobs[*]}")
aggregate_script="$RUN_DIR/aggregate_and_analyze.sh"
cat > "$aggregate_script" <<EOF
#!/bin/bash
set -euo pipefail
export RUN_DIR="$RUN_DIR"
export DATASETS="$DATASETS"
export SCRATCH_ROOT="$SCRATCH_ROOT"
export PYTHON_BIN="$PYTHON_BIN"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:\${PYTHONPATH:-}"
bash "$REPO_ROOT/scripts/kaya/aggregate_kaya_eval.sh"
"$PYTHON_BIN" "$REPO_ROOT/evaluation/analyze_timelens_predictions.py" \\
  --pred-run-dir "$RUN_DIR" \\
  --data-root "$DATA_ROOT" \\
  --datasets "$DATASETS" \\
  --baseline-run-dir "$SCRATCH_ROOT/outputs/qwen25vl7b_official_tp4_slowproc_20260322_154420" \\
  --output-dir "$RUN_DIR/analysis" \\
  --copy-video-examples 24 \\
  --demo-pairs 12
EOF
chmod +x "$aggregate_script"

aggregate_job=$(
    sbatch --parsable \
        --job-name tl8bdm_aggregate \
        --partition work \
        --cpus-per-task 4 \
        --mem 24G \
        --time 02:00:00 \
        --output "$LOG_DIR/tl8bdm_aggregate_%j.out" \
        --error "$LOG_DIR/tl8bdm_aggregate_%j.err" \
        --dependency "afterok:$all_chunks" \
        "$aggregate_script"
)

cat > "$RUN_DIR/submission_summary.txt" <<EOF
run_dir=$RUN_DIR
model_path=$(cat "$MODEL_PATH_FILE")
datasets=$DATASETS
official_input=min_tokens=$MIN_TOKENS,total_tokens=$TOTAL_TOKENS,fps=$FPS
runtime=device=$DEVICE,max_gpu_memory_gib=$MAX_GPU_MEMORY_GIB,cpu_offload_gib=$CPU_OFFLOAD_GIB,offload_state_dict=$OFFLOAD_STATE_DICT,dtype=$DTYPE,attn_implementation=$ATTN_IMPLEMENTATION,sdpa_backend=$SDPA_BACKEND,num_workers=$NUM_WORKERS,pin_memory=$PIN_MEMORY
chunks=charades:$CHARADES_CHUNKS,activitynet:$ACTIVITYNET_CHUNKS,qvhighlights:$QVHIGHLIGHTS_CHUNKS
max_concurrent_per_dataset=$MAX_CONCURRENT_PER_DATASET
smoke_job=$smoke_job
chunk_jobs=$all_chunks
aggregate_job=$aggregate_job
EOF

echo "chunk_jobs=$all_chunks"
echo "aggregate_job=$aggregate_job"
echo "run_dir=$RUN_DIR"
echo "summary=$RUN_DIR/submission_summary.txt"
