#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/pmc073/huwang/timelens}
DATA_ROOT=${DATA_ROOT:-$SCRATCH_ROOT/data/TimeLens-Bench}
PRED_ROOT=${PRED_ROOT:-$SCRATCH_ROOT/outputs}
BASELINE_RUN_DIR=${BASELINE_RUN_DIR:-/scratch/pmc073/huwang/timelens/outputs/timelens8b_dm_repro_20260424_223712}
DATASETS=${DATASETS:-activitynet-timelens}
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
USE_CACHE=${USE_CACHE:-auto}
SDPA_BACKEND=${SDPA_BACKEND:-efficient}
FORCE_REPEAT_KV_FOR_SDPA=${FORCE_REPEAT_KV_FOR_SDPA:-1}
MAX_EXAMPLES=${MAX_EXAMPLES:-}
NUM_WORKERS=${NUM_WORKERS:-0}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
PIN_MEMORY=${PIN_MEMORY:-false}
SKIP_VIDEO_ERRORS=${SKIP_VIDEO_ERRORS:-false}
CHUNK=${CHUNK:-1}
INDEX=${INDEX:-0}
COMPUTE_METRICS=${COMPUTE_METRICS:-1}
MODEL_PATH_FILE=${MODEL_PATH_FILE:-$SCRATCH_ROOT/models/timelens8b_snapshot_path.txt}
DEVICE=${DEVICE:-balanced}
MAX_GPU_MEMORY_GIB=${MAX_GPU_MEMORY_GIB:-14}
CPU_OFFLOAD_GIB=${CPU_OFFLOAD_GIB:-120}
OFFLOAD_FOLDER=${OFFLOAD_FOLDER:-}
OFFLOAD_STATE_DICT=${OFFLOAD_STATE_DICT:-true}
TP_PLAN=${TP_PLAN:-none}
TP_NNODES=${TP_NNODES:-1}
TP_NODE_RANK=${TP_NODE_RANK:-${SLURM_PROCID:-${SLURM_NODEID:-0}}}
TP_MASTER_ADDR=${TP_MASTER_ADDR:-127.0.0.1}
TP_MASTER_PORT=${TP_MASTER_PORT:-$((10000 + (${SLURM_JOB_ID:-$$} % 50000)))}

if [ -n "${TP_NPROC_PER_NODE:-}" ]; then
    TP_NPROC_PER_NODE=${TP_NPROC_PER_NODE}
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    TP_NPROC_PER_NODE=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
    TP_NPROC_PER_NODE=1
fi

DEFAULT_PYTHON_BIN=$SCRATCH_ROOT/envs/timelens/bin/python
if [ -x "${PYTHON_BIN:-}" ]; then
    PYTHON_BIN=${PYTHON_BIN}
elif [ -x "$DEFAULT_PYTHON_BIN" ]; then
    PYTHON_BIN=$DEFAULT_PYTHON_BIN
else
    PYTHON_BIN=python
fi
TORCHRUN_BIN=$(dirname "$PYTHON_BIN")/torchrun
if [ ! -x "$TORCHRUN_BIN" ]; then
    TORCHRUN_BIN=torchrun
fi

if [ -z "${MODEL_PATH:-}" ]; then
    MODEL_PATH=$(<"$MODEL_PATH_FILE")
fi

mkdir -p "$PRED_ROOT" "$REPO_ROOT/data"
ln -sfn "$DATA_ROOT" "$REPO_ROOT/data/TimeLens-Bench"

if [ -n "${RUN_DIR:-}" ]; then
    run_dir=$RUN_DIR
else
    timestamp=${RUN_TAG:-timelens8b_search_$(date +%Y%m%d_%H%M%S)}
    run_dir="${PRED_ROOT}/${timestamp}"
fi
mkdir -p "$run_dir"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export FORCE_REPEAT_KV_FOR_SDPA

echo "Repo root: $REPO_ROOT"
echo "Data root: $DATA_ROOT"
echo "Run dir: $run_dir"
echo "Baseline run dir: $BASELINE_RUN_DIR"
echo "Datasets: $DATASETS"
echo "Model path: $MODEL_PATH"
echo "Python bin: $PYTHON_BIN"
echo "min_tokens=$MIN_TOKENS total_tokens=$TOTAL_TOKENS max_new_tokens=$MAX_NEW_TOKENS fps=$FPS zoom_fps=$ZOOM_FPS dense_zoom_fps=$DENSE_ZOOM_FPS"
echo "search_mode=$SEARCH_MODE coarse_grid_size=$COARSE_GRID_SIZE coarse_window_fraction=$COARSE_WINDOW_FRACTION"
echo "refine_rounds=$REFINE_ROUNDS top_k_stage1=$TOP_K_STAGE1 top_k_refine=$TOP_K_REFINE"
echo "max_total_calls=$MAX_TOTAL_CALLS graph_seed_count=$GRAPH_SEED_COUNT graph_beam_width=$GRAPH_BEAM_WIDTH"
echo "lookagain_controller=$LOOKAGAIN_CONTROLLER anchor_period=$LOOKAGAIN_ANCHOR_PERIOD certificate_boundary_iou=$CERTIFICATE_BOUNDARY_IOU certificate_support_iou=$CERTIFICATE_SUPPORT_IOU certificate_contradiction_weight=$CERTIFICATE_CONTRADICTION_WEIGHT"
echo "critic_mode=$CRITIC_MODE critic_calls=$CRITIC_CALLS critic_fps=$CRITIC_FPS critic_min_confidence=$CRITIC_MIN_CONFIDENCE"
echo "dtype=$DTYPE attn_implementation=$ATTN_IMPLEMENTATION sdpa_backend=$SDPA_BACKEND use_cache=$USE_CACHE"
echo "device=$DEVICE max_gpu_memory_gib=${MAX_GPU_MEMORY_GIB:-unset} cpu_offload_gib=${CPU_OFFLOAD_GIB:-unset}"
echo "chunk=$CHUNK index=$INDEX compute_metrics=$COMPUTE_METRICS"
if [ -n "$MAX_EXAMPLES" ]; then
    echo "max_examples=$MAX_EXAMPLES"
fi

IFS=',' read -ra DATASET_LIST <<< "$DATASETS"

for dataset in "${DATASET_LIST[@]}"; do
    baseline_jsonl="$BASELINE_RUN_DIR/${dataset}.jsonl"
    if [ ! -s "$baseline_jsonl" ]; then
        echo "Missing baseline JSONL for $dataset: $baseline_jsonl" >&2
        exit 1
    fi

    current_prefix="${run_dir}/${dataset}"
    should_finalize=1
    if [ "$TP_PLAN" = "auto" ] && [ "$TP_NODE_RANK" != "0" ]; then
        should_finalize=0
    fi

    args=(
        evaluation/eval_timelens_search.py
        --dataset "$dataset"
        --baseline_jsonl "$baseline_jsonl"
        --pred_path "$current_prefix"
        --model_path "$MODEL_PATH"
        --min_tokens "$MIN_TOKENS"
        --total_tokens "$TOTAL_TOKENS"
        --max_new_tokens "$MAX_NEW_TOKENS"
        --fps "$FPS"
        --zoom_fps "$ZOOM_FPS"
        --dense_zoom_fps "$DENSE_ZOOM_FPS"
        --dtype "$DTYPE"
        --attn_implementation "$ATTN_IMPLEMENTATION"
        --use_cache "$USE_CACHE"
        --sdpa_backend "$SDPA_BACKEND"
        --num_workers "$NUM_WORKERS"
        --prefetch_factor "$PREFETCH_FACTOR"
        --pin_memory "$PIN_MEMORY"
        --skip_video_errors "$SKIP_VIDEO_ERRORS"
        --device "$DEVICE"
        --tp_plan "$TP_PLAN"
        --chunk "$CHUNK"
        --index "$INDEX"
        --offload_state_dict "$OFFLOAD_STATE_DICT"
        --search_mode "$SEARCH_MODE"
        --coarse_grid_size "$COARSE_GRID_SIZE"
        --coarse_window_fraction "$COARSE_WINDOW_FRACTION"
        --refine_rounds "$REFINE_ROUNDS"
        --top_k_stage1 "$TOP_K_STAGE1"
        --top_k_refine "$TOP_K_REFINE"
        --min_window_seconds "$MIN_WINDOW_SECONDS"
        --tiny_event_window_seconds "$TINY_EVENT_WINDOW_SECONDS"
        --diffuse_confidence_threshold "$DIFFUSE_CONFIDENCE_THRESHOLD"
        --broad_baseline_fraction "$BROAD_BASELINE_FRACTION"
        --broad_replace_threshold "$BROAD_REPLACE_THRESHOLD"
        --max_total_calls "$MAX_TOTAL_CALLS"
        --graph_seed_count "$GRAPH_SEED_COUNT"
        --graph_beam_width "$GRAPH_BEAM_WIDTH"
        --graph_merge_iou "$GRAPH_MERGE_IOU"
        --graph_support_iou "$GRAPH_SUPPORT_IOU"
        --lookagain_controller "$LOOKAGAIN_CONTROLLER"
        --lookagain_anchor_period "$LOOKAGAIN_ANCHOR_PERIOD"
        --certificate_boundary_iou "$CERTIFICATE_BOUNDARY_IOU"
        --certificate_support_iou "$CERTIFICATE_SUPPORT_IOU"
        --certificate_contradiction_weight "$CERTIFICATE_CONTRADICTION_WEIGHT"
        --max_controller_trace_steps "$MAX_CONTROLLER_TRACE_STEPS"
        --critic_mode "$CRITIC_MODE"
        --critic_calls "$CRITIC_CALLS"
        --critic_fps "$CRITIC_FPS"
        --critic_min_confidence "$CRITIC_MIN_CONFIDENCE"
    )

    if [ -n "$MAX_EXAMPLES" ]; then
        args+=(--max_examples "$MAX_EXAMPLES")
    fi
    if [ -n "$MAX_GPU_MEMORY_GIB" ]; then
        args+=(--max_gpu_memory_gib "$MAX_GPU_MEMORY_GIB")
    fi
    if [ -n "$CPU_OFFLOAD_GIB" ]; then
        args+=(--cpu_offload_gib "$CPU_OFFLOAD_GIB")
    fi
    if [ -n "$OFFLOAD_FOLDER" ]; then
        current_offload_folder="${OFFLOAD_FOLDER}/${dataset}_chunk${CHUNK}_idx${INDEX}"
        mkdir -p "$current_offload_folder"
        args+=(--offload_folder "$current_offload_folder")
    fi

    if [ "$TP_PLAN" = "auto" ]; then
        "$TORCHRUN_BIN" \
            --nnodes "$TP_NNODES" \
            --node-rank "$TP_NODE_RANK" \
            --master-addr "$TP_MASTER_ADDR" \
            --master-port "$TP_MASTER_PORT" \
            --nproc-per-node "$TP_NPROC_PER_NODE" \
            "${args[@]}"
    else
        "$PYTHON_BIN" "${args[@]}"
    fi

    output_jsonl="${current_prefix}_${INDEX}.jsonl"
    if [ "$CHUNK" = "1" ] && [ "$INDEX" = "0" ] && [ "$should_finalize" = "1" ]; then
        mv "$output_jsonl" "${current_prefix}.jsonl"
        if [ "$COMPUTE_METRICS" = "1" ]; then
            "$PYTHON_BIN" evaluation/compute_metrics.py -f "${current_prefix}.jsonl" | tee "${current_prefix}.metrics.txt"
        fi
    elif [ "$COMPUTE_METRICS" = "1" ] && [ "$should_finalize" = "1" ]; then
        echo "Skipping metrics for chunked run; aggregate chunk outputs before scoring."
    fi
done

echo "$run_dir"
