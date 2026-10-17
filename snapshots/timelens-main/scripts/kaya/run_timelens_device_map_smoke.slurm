#!/bin/bash
#SBATCH -J timelens_dm_smoke
#SBATCH -p ondemand-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH -o /scratch/pmc073/huwang/timelens/logs/timelens_dm_smoke_%j.out
#SBATCH -e /scratch/pmc073/huwang/timelens/logs/timelens_dm_smoke_%j.err

set -euo pipefail

REPO_ROOT=/group/pmc073/huwang/TimeLens
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/pmc073/huwang/timelens}
DATA_ROOT=${DATA_ROOT:-$SCRATCH_ROOT/data/TimeLens-Bench}
RUN_DIR=${RUN_DIR:?RUN_DIR is required}
DATASET=${DATASET:?DATASET is required}
MIN_TOKENS=${MIN_TOKENS:-64}
TOTAL_TOKENS=${TOTAL_TOKENS:-14336}
FPS=${FPS:-2}
DTYPE=${DTYPE:-auto}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
USE_CACHE=${USE_CACHE:-auto}
SDPA_BACKEND=${SDPA_BACKEND:-efficient}
FORCE_REPEAT_KV_FOR_SDPA=${FORCE_REPEAT_KV_FOR_SDPA:-1}
MAX_EXAMPLES=${MAX_EXAMPLES:-3}
NUM_WORKERS=${NUM_WORKERS:-0}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
PIN_MEMORY=${PIN_MEMORY:-false}
COMPUTE_METRICS=${COMPUTE_METRICS:-0}
MODEL_PATH_FILE=${MODEL_PATH_FILE:-$SCRATCH_ROOT/models/timelens8b_snapshot_path.txt}
DEVICE=${DEVICE:-balanced}
MAX_GPU_MEMORY_GIB=${MAX_GPU_MEMORY_GIB:-14}
CPU_OFFLOAD_GIB=${CPU_OFFLOAD_GIB:-120}
OFFLOAD_STATE_DICT=${OFFLOAD_STATE_DICT:-true}
OFFLOAD_FOLDER=${OFFLOAD_FOLDER:-$SCRATCH_ROOT/offload/${SLURM_JOB_ID}}

export HF_HOME=${HF_HOME:-$SCRATCH_ROOT/hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$SCRATCH_ROOT/hf_cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$SCRATCH_ROOT/pip_cache}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$SCRATCH_ROOT/logs" "$RUN_DIR" "$OFFLOAD_FOLDER"

cd "$REPO_ROOT"
export HF_HOME
export HF_HUB_CACHE
export PIP_CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF
export SCRATCH_ROOT
export DATA_ROOT
export RUN_DIR
export DATASETS="$DATASET"
export MIN_TOKENS
export TOTAL_TOKENS
export FPS
export DTYPE
export ATTN_IMPLEMENTATION
export USE_CACHE
export SDPA_BACKEND
export FORCE_REPEAT_KV_FOR_SDPA
export MAX_EXAMPLES
export NUM_WORKERS
export PREFETCH_FACTOR
export PIN_MEMORY
export COMPUTE_METRICS
export MODEL_PATH_FILE
export DEVICE
export MAX_GPU_MEMORY_GIB
export CPU_OFFLOAD_GIB
export OFFLOAD_STATE_DICT
export OFFLOAD_FOLDER
export TP_PLAN=none
export CHUNK=1
export INDEX=0

bash "$REPO_ROOT/scripts/kaya/run_kaya_eval.sh"
