#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/pmc073/huwang/timelens}
DATA_ROOT=${DATA_ROOT:-$SCRATCH_ROOT/data/TimeLens-Bench}
PRED_ROOT=${PRED_ROOT:-$SCRATCH_ROOT/outputs}
DATASETS=${DATASETS:-activitynet-timelens,qvhighlights-timelens}
MIN_TOKENS=${MIN_TOKENS:-64}
TOTAL_TOKENS=${TOTAL_TOKENS:-14336}
FPS=${FPS:-2}
DTYPE=${DTYPE:-auto}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-auto}
USE_CACHE=${USE_CACHE:-auto}
SDPA_BACKEND=${SDPA_BACKEND:-auto}
FORCE_REPEAT_KV_FOR_SDPA=${FORCE_REPEAT_KV_FOR_SDPA:-0}
MAX_EXAMPLES=${MAX_EXAMPLES:-}
NUM_WORKERS=${NUM_WORKERS:-0}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
PIN_MEMORY=${PIN_MEMORY:-false}
SKIP_VIDEO_ERRORS=${SKIP_VIDEO_ERRORS:-false}
CHUNK=${CHUNK:-1}
INDEX=${INDEX:-0}
COMPUTE_METRICS=${COMPUTE_METRICS:-1}
MODEL_PATH_FILE=${MODEL_PATH_FILE:-$SCRATCH_ROOT/models/qwen25vl7b_snapshot_path.txt}
DEVICE=${DEVICE:-auto}
MAX_GPU_MEMORY_GIB=${MAX_GPU_MEMORY_GIB:-}
CPU_OFFLOAD_GIB=${CPU_OFFLOAD_GIB:-}
OFFLOAD_FOLDER=${OFFLOAD_FOLDER:-}
OFFLOAD_STATE_DICT=${OFFLOAD_STATE_DICT:-auto}
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
    timestamp=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
    model_tag=$(basename "${MODEL_PATH%/}")
    run_dir="${PRED_ROOT}/${model_tag}_${timestamp}"
fi
mkdir -p "$run_dir"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "Repo root: $REPO_ROOT"
echo "Data root: $DATA_ROOT"
echo "Run dir: $run_dir"
echo "Datasets: $DATASETS"
echo "Model path: $MODEL_PATH"
echo "Python bin: $PYTHON_BIN"
echo "min_tokens=$MIN_TOKENS total_tokens=$TOTAL_TOKENS fps=$FPS"
echo "dtype=$DTYPE attn_implementation=$ATTN_IMPLEMENTATION"
echo "use_cache=$USE_CACHE"
echo "sdpa_backend=$SDPA_BACKEND"
echo "force_repeat_kv_for_sdpa=$FORCE_REPEAT_KV_FOR_SDPA"
echo "num_workers=$NUM_WORKERS prefetch_factor=$PREFETCH_FACTOR pin_memory=$PIN_MEMORY"
echo "skip_video_errors=$SKIP_VIDEO_ERRORS"
echo "device=$DEVICE max_gpu_memory_gib=${MAX_GPU_MEMORY_GIB:-unset} cpu_offload_gib=${CPU_OFFLOAD_GIB:-unset} offload_state_dict=$OFFLOAD_STATE_DICT"
if [ -n "$OFFLOAD_FOLDER" ]; then
    echo "offload_folder_base=$OFFLOAD_FOLDER"
fi
echo "tp_plan=$TP_PLAN tp_nnodes=$TP_NNODES tp_node_rank=$TP_NODE_RANK tp_nproc_per_node=$TP_NPROC_PER_NODE"
echo "chunk=$CHUNK index=$INDEX compute_metrics=$COMPUTE_METRICS"
if [ -n "$MAX_EXAMPLES" ]; then
    echo "max_examples=$MAX_EXAMPLES"
fi

IFS=',' read -ra DATASET_LIST <<< "$DATASETS"

for dataset in "${DATASET_LIST[@]}"; do
    current_prefix="${run_dir}/${dataset}"
    should_finalize=1
    if [ "$TP_PLAN" = "auto" ] && [ "$TP_NODE_RANK" != "0" ]; then
        should_finalize=0
    fi

    args=(
        evaluation/eval_dataloader.py
        --dataset "$dataset"
        --pred_path "$current_prefix"
        --model_path "$MODEL_PATH"
        --min_tokens "$MIN_TOKENS"
        --total_tokens "$TOTAL_TOKENS"
        --fps "$FPS"
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
