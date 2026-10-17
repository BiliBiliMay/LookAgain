#!/bin/bash
#SBATCH -J timelens_chunk
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH -o /scratch/pmc073/huwang/timelens/logs/timelens_chunk_%j.out
#SBATCH -e /scratch/pmc073/huwang/timelens/logs/timelens_chunk_%j.err

set -euo pipefail

REPO_ROOT=/group/pmc073/huwang/TimeLens
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/pmc073/huwang/timelens}
ENV_PREFIX=${ENV_PREFIX:-$SCRATCH_ROOT/envs/timelens}
DATA_ROOT=${DATA_ROOT:-$SCRATCH_ROOT/data/TimeLens-Bench}
RUN_DIR=${RUN_DIR:?RUN_DIR is required}
DATASET=${DATASET:?DATASET is required}
CHUNK=${CHUNK:?CHUNK is required}
INDEX=${INDEX:?INDEX is required}
MIN_TOKENS=${MIN_TOKENS:-32}
TOTAL_TOKENS=${TOTAL_TOKENS:-4096}
FPS=${FPS:-1}
DTYPE=${DTYPE:-auto}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-auto}
USE_CACHE=${USE_CACHE:-auto}
SDPA_BACKEND=${SDPA_BACKEND:-auto}
FORCE_REPEAT_KV_FOR_SDPA=${FORCE_REPEAT_KV_FOR_SDPA:-0}
NUM_WORKERS=${NUM_WORKERS:-0}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
PIN_MEMORY=${PIN_MEMORY:-false}
TP_PLAN=${TP_PLAN:-none}
TP_NPROC_PER_NODE=${TP_NPROC_PER_NODE:-2}
MODEL_PATH_FILE=${MODEL_PATH_FILE:-$SCRATCH_ROOT/models/qwen25vl7b_snapshot_path.txt}

export HF_HOME=${HF_HOME:-$SCRATCH_ROOT/hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$SCRATCH_ROOT/hf_cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$SCRATCH_ROOT/pip_cache}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

module load Anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

mkdir -p "$SCRATCH_ROOT/logs" "$RUN_DIR"
cd "$REPO_ROOT"

env \
    RUN_DIR="$RUN_DIR" \
    MODEL_PATH_FILE="$MODEL_PATH_FILE" \
    DATA_ROOT="$DATA_ROOT" \
    DATASETS="$DATASET" \
    MIN_TOKENS="$MIN_TOKENS" \
    TOTAL_TOKENS="$TOTAL_TOKENS" \
    FPS="$FPS" \
    DTYPE="$DTYPE" \
    ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
    USE_CACHE="$USE_CACHE" \
    SDPA_BACKEND="$SDPA_BACKEND" \
    FORCE_REPEAT_KV_FOR_SDPA="$FORCE_REPEAT_KV_FOR_SDPA" \
    NUM_WORKERS="$NUM_WORKERS" \
    PREFETCH_FACTOR="$PREFETCH_FACTOR" \
    PIN_MEMORY="$PIN_MEMORY" \
    TP_PLAN="$TP_PLAN" \
    TP_NPROC_PER_NODE="$TP_NPROC_PER_NODE" \
    CHUNK="$CHUNK" \
    INDEX="$INDEX" \
    COMPUTE_METRICS=0 \
    bash "$REPO_ROOT/scripts/kaya/run_kaya_eval.sh"
