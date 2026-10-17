#!/bin/bash

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/group/pmc073/huwang/worktrees/TimeLens/timelens-search-wrapper}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/pmc073/huwang/timelens}
DATA_ROOT=${DATA_ROOT:-$SCRATCH_ROOT/data/TimeLens-Bench}
BASELINE_RUN_DIR=${BASELINE_RUN_DIR:-/scratch/pmc073/huwang/timelens/outputs/timelens8b_dm_repro_20260424_223712}
MODEL_PATH_FILE=${MODEL_PATH_FILE:-$SCRATCH_ROOT/models/timelens8b_snapshot_path.txt}
PYTHON_BIN=${PYTHON_BIN:-$SCRATCH_ROOT/envs/timelens/bin/python}
RUN_TAG=${RUN_TAG:-timelens8b_search_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$SCRATCH_ROOT/outputs/$RUN_TAG}
LOG_DIR=${LOG_DIR:-$SCRATCH_ROOT/logs}

MIN_TOKENS=${MIN_TOKENS:-64}
TOTAL_TOKENS=${TOTAL_TOKENS:-14336}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
FPS=${FPS:-2}
ZOOM_FPS=${ZOOM_FPS:-4}
DENSE_ZOOM_FPS=${DENSE_ZOOM_FPS:-8}
SEARCH_MODE=${SEARCH_MODE:-budgeted_graph}
COARSE_GRID_SIZE=${COARSE_GRID_SIZE:-9}
COARSE_WINDOW_FRACTION=${COARSE_WINDOW_FRACTION:-0.25}
REFINE_ROUNDS=${REFINE_ROUNDS:-2}
TOP_K_STAGE1=${TOP_K_STAGE1:-5}
TOP_K_REFINE=${TOP_K_REFINE:-5}
MIN_WINDOW_SECONDS=${MIN_WINDOW_SECONDS:-4.0}
TINY_EVENT_WINDOW_SECONDS=${TINY_EVENT_WINDOW_SECONDS:-8.0}
DIFFUSE_CONFIDENCE_THRESHOLD=${DIFFUSE_CONFIDENCE_THRESHOLD:-0.30}
BROAD_BASELINE_FRACTION=${BROAD_BASELINE_FRACTION:-0.60}
BROAD_REPLACE_THRESHOLD=${BROAD_REPLACE_THRESHOLD:-0.55}
MAX_TOTAL_CALLS=${MAX_TOTAL_CALLS:-15}
GRAPH_SEED_COUNT=${GRAPH_SEED_COUNT:-5}
GRAPH_BEAM_WIDTH=${GRAPH_BEAM_WIDTH:-3}
GRAPH_MERGE_IOU=${GRAPH_MERGE_IOU:-0.92}
GRAPH_SUPPORT_IOU=${GRAPH_SUPPORT_IOU:-0.55}
LOOKAGAIN_CONTROLLER=${LOOKAGAIN_CONTROLLER:-guided}
LOOKAGAIN_ANCHOR_PERIOD=${LOOKAGAIN_ANCHOR_PERIOD:-3}
CERTIFICATE_BOUNDARY_IOU=${CERTIFICATE_BOUNDARY_IOU:-0.55}
CERTIFICATE_SUPPORT_IOU=${CERTIFICATE_SUPPORT_IOU:-0.55}
CERTIFICATE_CONTRADICTION_WEIGHT=${CERTIFICATE_CONTRADICTION_WEIGHT:-0.45}
MAX_CONTROLLER_TRACE_STEPS=${MAX_CONTROLLER_TRACE_STEPS:-256}
CRITIC_MODE=${CRITIC_MODE:-self}
CRITIC_CALLS=${CRITIC_CALLS:-2}
CRITIC_FPS=${CRITIC_FPS:-4}
CRITIC_MIN_CONFIDENCE=${CRITIC_MIN_CONFIDENCE:-0.55}
DTYPE=${DTYPE:-auto}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
SDPA_BACKEND=${SDPA_BACKEND:-efficient}
USE_CACHE=${USE_CACHE:-auto}
FORCE_REPEAT_KV_FOR_SDPA=${FORCE_REPEAT_KV_FOR_SDPA:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
PIN_MEMORY=${PIN_MEMORY:-false}
SKIP_VIDEO_ERRORS=${SKIP_VIDEO_ERRORS:-false}
DEVICE=${DEVICE:-balanced}
MAX_GPU_MEMORY_GIB=${MAX_GPU_MEMORY_GIB:-14}
CPU_OFFLOAD_GIB=${CPU_OFFLOAD_GIB:-120}
OFFLOAD_STATE_DICT=${OFFLOAD_STATE_DICT:-true}
OFFLOAD_BASE=${OFFLOAD_BASE:-$SCRATCH_ROOT/offload/$RUN_TAG}

SMOKE_DATASET=${SMOKE_DATASET:-activitynet-timelens}
SMOKE_MAX_EXAMPLES=${SMOKE_MAX_EXAMPLES:-5}
SUBMIT_SMOKE=${SUBMIT_SMOKE:-0}
SUBMIT_FULL=${SUBMIT_FULL:-0}

DATASETS=${DATASETS:-activitynet-timelens}
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

check_dataset() {
    local dataset=$1
    local annotation="$DATA_ROOT/${dataset}.json"
    local baseline="$BASELINE_RUN_DIR/${dataset}.jsonl"
    if [ ! -s "$annotation" ]; then
        echo "Missing benchmark annotation: $annotation" >&2
        exit 1
    fi
    if [ ! -s "$baseline" ]; then
        echo "Missing baseline prediction file: $baseline" >&2
        exit 1
    fi
}

check_dataset "$SMOKE_DATASET"
IFS=',' read -ra REQUESTED_DATASETS <<< "$DATASETS"
for dataset in "${REQUESTED_DATASETS[@]}"; do
    check_dataset "$dataset"
done

common_export="SCRATCH_ROOT=$SCRATCH_ROOT,DATA_ROOT=$DATA_ROOT,BASELINE_RUN_DIR=$BASELINE_RUN_DIR,RUN_DIR=$RUN_DIR,MIN_TOKENS=$MIN_TOKENS,TOTAL_TOKENS=$TOTAL_TOKENS,MAX_NEW_TOKENS=$MAX_NEW_TOKENS,FPS=$FPS,ZOOM_FPS=$ZOOM_FPS,DENSE_ZOOM_FPS=$DENSE_ZOOM_FPS,SEARCH_MODE=$SEARCH_MODE,COARSE_GRID_SIZE=$COARSE_GRID_SIZE,COARSE_WINDOW_FRACTION=$COARSE_WINDOW_FRACTION,REFINE_ROUNDS=$REFINE_ROUNDS,TOP_K_STAGE1=$TOP_K_STAGE1,TOP_K_REFINE=$TOP_K_REFINE,MIN_WINDOW_SECONDS=$MIN_WINDOW_SECONDS,TINY_EVENT_WINDOW_SECONDS=$TINY_EVENT_WINDOW_SECONDS,DIFFUSE_CONFIDENCE_THRESHOLD=$DIFFUSE_CONFIDENCE_THRESHOLD,BROAD_BASELINE_FRACTION=$BROAD_BASELINE_FRACTION,BROAD_REPLACE_THRESHOLD=$BROAD_REPLACE_THRESHOLD,MAX_TOTAL_CALLS=$MAX_TOTAL_CALLS,GRAPH_SEED_COUNT=$GRAPH_SEED_COUNT,GRAPH_BEAM_WIDTH=$GRAPH_BEAM_WIDTH,GRAPH_MERGE_IOU=$GRAPH_MERGE_IOU,GRAPH_SUPPORT_IOU=$GRAPH_SUPPORT_IOU,LOOKAGAIN_CONTROLLER=$LOOKAGAIN_CONTROLLER,LOOKAGAIN_ANCHOR_PERIOD=$LOOKAGAIN_ANCHOR_PERIOD,CERTIFICATE_BOUNDARY_IOU=$CERTIFICATE_BOUNDARY_IOU,CERTIFICATE_SUPPORT_IOU=$CERTIFICATE_SUPPORT_IOU,CERTIFICATE_CONTRADICTION_WEIGHT=$CERTIFICATE_CONTRADICTION_WEIGHT,MAX_CONTROLLER_TRACE_STEPS=$MAX_CONTROLLER_TRACE_STEPS,CRITIC_MODE=$CRITIC_MODE,CRITIC_CALLS=$CRITIC_CALLS,CRITIC_FPS=$CRITIC_FPS,CRITIC_MIN_CONFIDENCE=$CRITIC_MIN_CONFIDENCE,DTYPE=$DTYPE,ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION,SDPA_BACKEND=$SDPA_BACKEND,USE_CACHE=$USE_CACHE,FORCE_REPEAT_KV_FOR_SDPA=$FORCE_REPEAT_KV_FOR_SDPA,NUM_WORKERS=$NUM_WORKERS,PIN_MEMORY=$PIN_MEMORY,SKIP_VIDEO_ERRORS=$SKIP_VIDEO_ERRORS,MODEL_PATH_FILE=$MODEL_PATH_FILE,DEVICE=$DEVICE,MAX_GPU_MEMORY_GIB=$MAX_GPU_MEMORY_GIB,CPU_OFFLOAD_GIB=$CPU_OFFLOAD_GIB,OFFLOAD_STATE_DICT=$OFFLOAD_STATE_DICT,REPO_ROOT=$REPO_ROOT"

submit_smoke() {
    sbatch --parsable \
        --job-name tl8b_search_smoke \
        --export "ALL,$common_export,DATASET=$SMOKE_DATASET,DATASETS=$SMOKE_DATASET,CHUNK=1,INDEX=0,MAX_EXAMPLES=$SMOKE_MAX_EXAMPLES,COMPUTE_METRICS=1,OFFLOAD_FOLDER=$OFFLOAD_BASE/smoke" \
        "$REPO_ROOT/scripts/kaya/run_timelens_search_chunk.slurm"
}

submit_dataset_chunks() {
    local dataset=$1
    local chunk_count=$2
    local dependency=${3:-}
    local opts=(
        --parsable
        --job-name "tl8bs_${dataset%%-*}"
        --array "0-$((chunk_count - 1))%$MAX_CONCURRENT_PER_DATASET"
        --export "ALL,$common_export,DATASET=$dataset,DATASETS=$dataset,CHUNK=$chunk_count,MAX_EXAMPLES=,COMPUTE_METRICS=0,OFFLOAD_FOLDER=$OFFLOAD_BASE/$dataset"
    )
    if [ -n "$dependency" ]; then
        opts+=(--dependency "$dependency")
    fi
    sbatch "${opts[@]}" "$REPO_ROOT/scripts/kaya/run_timelens_search_chunk.slurm"
}

smoke_job=""
if [ "$SUBMIT_SMOKE" = "1" ]; then
    smoke_job=$(submit_smoke)
    echo "smoke_job=$smoke_job"
fi

if [ "$SUBMIT_FULL" != "1" ]; then
    cat > "$RUN_DIR/submission_summary.txt" <<EOF
run_dir=$RUN_DIR
baseline_run_dir=$BASELINE_RUN_DIR
model_path=$(cat "$MODEL_PATH_FILE")
datasets=$DATASETS
search=mode=$SEARCH_MODE,controller=$LOOKAGAIN_CONTROLLER,anchor_period=$LOOKAGAIN_ANCHOR_PERIOD,max_total_calls=$MAX_TOTAL_CALLS,graph_seed_count=$GRAPH_SEED_COUNT,graph_beam_width=$GRAPH_BEAM_WIDTH,certificate_boundary_iou=$CERTIFICATE_BOUNDARY_IOU,certificate_support_iou=$CERTIFICATE_SUPPORT_IOU,certificate_contradiction_weight=$CERTIFICATE_CONTRADICTION_WEIGHT,critic_mode=$CRITIC_MODE,critic_calls=$CRITIC_CALLS,critic_fps=$CRITIC_FPS,critic_min_confidence=$CRITIC_MIN_CONFIDENCE,coarse_grid_size=$COARSE_GRID_SIZE,coarse_window_fraction=$COARSE_WINDOW_FRACTION,refine_rounds=$REFINE_ROUNDS,top_k_stage1=$TOP_K_STAGE1,top_k_refine=$TOP_K_REFINE,min_window_seconds=$MIN_WINDOW_SECONDS,tiny_event_window_seconds=$TINY_EVENT_WINDOW_SECONDS,fps=$FPS,zoom_fps=$ZOOM_FPS,dense_zoom_fps=$DENSE_ZOOM_FPS,max_new_tokens=$MAX_NEW_TOKENS
runtime=device=$DEVICE,max_gpu_memory_gib=$MAX_GPU_MEMORY_GIB,cpu_offload_gib=$CPU_OFFLOAD_GIB,offload_state_dict=$OFFLOAD_STATE_DICT,dtype=$DTYPE,attn_implementation=$ATTN_IMPLEMENTATION,sdpa_backend=$SDPA_BACKEND,num_workers=$NUM_WORKERS,pin_memory=$PIN_MEMORY
smoke_job=$smoke_job
submit_full=0
EOF
    echo "run_dir=$RUN_DIR"
    echo "summary=$RUN_DIR/submission_summary.txt"
    exit 0
fi

chunk_dependency=""
if [ -n "$smoke_job" ]; then
    chunk_dependency="afterok:$smoke_job"
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
aggregate_script="$RUN_DIR/aggregate_timelens_search.sh"
cat > "$aggregate_script" <<EOF
#!/bin/bash
set -euo pipefail
export RUN_DIR="$RUN_DIR"
export DATASETS="$DATASETS"
export PYTHON_BIN="$PYTHON_BIN"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:\${PYTHONPATH:-}"
IFS=',' read -ra DATASET_LIST <<< "\$DATASETS"
for dataset in "\${DATASET_LIST[@]}"; do
    output="\$RUN_DIR/\${dataset}.jsonl"
    shopt -s nullglob
    chunks=("\$RUN_DIR/\${dataset}"_*.jsonl)
    shopt -u nullglob
    if [ "\${#chunks[@]}" -eq 0 ]; then
        echo "No chunk outputs found for \$dataset" >&2
        exit 1
    fi
    cat "\${chunks[@]}" > "\$output"
    "\$PYTHON_BIN" evaluation/compute_metrics.py -f "\$output" | tee "\$RUN_DIR/\${dataset}.metrics.txt"
done
EOF
chmod +x "$aggregate_script"

aggregate_job=$(
    sbatch --parsable \
        --job-name tl8bs_aggregate \
        --partition work \
        --cpus-per-task 4 \
        --mem 24G \
        --time 02:00:00 \
        --output "$LOG_DIR/tl8bs_aggregate_%j.out" \
        --error "$LOG_DIR/tl8bs_aggregate_%j.err" \
        --dependency "afterok:$all_chunks" \
        "$aggregate_script"
)

cat > "$RUN_DIR/submission_summary.txt" <<EOF
run_dir=$RUN_DIR
baseline_run_dir=$BASELINE_RUN_DIR
model_path=$(cat "$MODEL_PATH_FILE")
datasets=$DATASETS
search=mode=$SEARCH_MODE,controller=$LOOKAGAIN_CONTROLLER,anchor_period=$LOOKAGAIN_ANCHOR_PERIOD,max_total_calls=$MAX_TOTAL_CALLS,graph_seed_count=$GRAPH_SEED_COUNT,graph_beam_width=$GRAPH_BEAM_WIDTH,certificate_boundary_iou=$CERTIFICATE_BOUNDARY_IOU,certificate_support_iou=$CERTIFICATE_SUPPORT_IOU,certificate_contradiction_weight=$CERTIFICATE_CONTRADICTION_WEIGHT,critic_mode=$CRITIC_MODE,critic_calls=$CRITIC_CALLS,critic_fps=$CRITIC_FPS,critic_min_confidence=$CRITIC_MIN_CONFIDENCE,coarse_grid_size=$COARSE_GRID_SIZE,coarse_window_fraction=$COARSE_WINDOW_FRACTION,refine_rounds=$REFINE_ROUNDS,top_k_stage1=$TOP_K_STAGE1,top_k_refine=$TOP_K_REFINE,min_window_seconds=$MIN_WINDOW_SECONDS,tiny_event_window_seconds=$TINY_EVENT_WINDOW_SECONDS,fps=$FPS,zoom_fps=$ZOOM_FPS,dense_zoom_fps=$DENSE_ZOOM_FPS,max_new_tokens=$MAX_NEW_TOKENS
runtime=device=$DEVICE,max_gpu_memory_gib=$MAX_GPU_MEMORY_GIB,cpu_offload_gib=$CPU_OFFLOAD_GIB,offload_state_dict=$OFFLOAD_STATE_DICT,dtype=$DTYPE,attn_implementation=$ATTN_IMPLEMENTATION,sdpa_backend=$SDPA_BACKEND,num_workers=$NUM_WORKERS,pin_memory=$PIN_MEMORY
chunks=charades:$CHARADES_CHUNKS,activitynet:$ACTIVITYNET_CHUNKS,qvhighlights:$QVHIGHLIGHTS_CHUNKS
max_concurrent_per_dataset=$MAX_CONCURRENT_PER_DATASET
smoke_job=$smoke_job
chunk_jobs=$all_chunks
aggregate_job=$aggregate_job
submit_full=1
EOF

echo "chunk_jobs=$all_chunks"
echo "aggregate_job=$aggregate_job"
echo "run_dir=$RUN_DIR"
echo "summary=$RUN_DIR/submission_summary.txt"
