#!/bin/bash
#SBATCH -J timelens_tp_chunk
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH -o /scratch/pmc073/huwang/timelens/logs/timelens_tp_chunk_%j.out
#SBATCH -e /scratch/pmc073/huwang/timelens/logs/timelens_tp_chunk_%j.err

set -euo pipefail

REPO_ROOT=/group/pmc073/huwang/TimeLens
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/pmc073/huwang/timelens}
DATA_ROOT=${DATA_ROOT:-$SCRATCH_ROOT/data/TimeLens-Bench}
RUN_DIR=${RUN_DIR:?RUN_DIR is required}
DATASET=${DATASET:?DATASET is required}
CHUNK=${CHUNK:?CHUNK is required}
INDEX=${INDEX:-${SLURM_ARRAY_TASK_ID:-}}
INDEX=${INDEX:?INDEX is required}
MIN_TOKENS=${MIN_TOKENS:-64}
TOTAL_TOKENS=${TOTAL_TOKENS:-14336}
FPS=${FPS:-2}
DTYPE=${DTYPE:-auto}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-auto}
USE_CACHE=${USE_CACHE:-auto}
SDPA_BACKEND=${SDPA_BACKEND:-auto}
FORCE_REPEAT_KV_FOR_SDPA=${FORCE_REPEAT_KV_FOR_SDPA:-0}
TP_NPROC_PER_NODE=${TP_NPROC_PER_NODE:-2}
MODEL_PATH_FILE=${MODEL_PATH_FILE:-$SCRATCH_ROOT/models/qwen25vl7b_snapshot_path.txt}
TP_MASTER_PORT=${TP_MASTER_PORT:-$((10000 + (${SLURM_JOB_ID:-$$} % 50000)))}
MAX_EXAMPLES=${MAX_EXAMPLES:-}
NUM_WORKERS=${NUM_WORKERS:-0}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
PIN_MEMORY=${PIN_MEMORY:-false}
COMPUTE_METRICS=${COMPUTE_METRICS:-0}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export HF_HOME=${HF_HOME:-$SCRATCH_ROOT/hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$SCRATCH_ROOT/hf_cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$SCRATCH_ROOT/pip_cache}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$SCRATCH_ROOT/logs" "$RUN_DIR"

srun --kill-on-bad-exit=1 --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -lc "
set -euo pipefail
cd '$REPO_ROOT'
export HF_HOME='$HF_HOME'
export HF_HUB_CACHE='$HF_HUB_CACHE'
export PIP_CACHE_DIR='$PIP_CACHE_DIR'
export PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF'
export SCRATCH_ROOT='$SCRATCH_ROOT'
export MODEL_PATH_FILE='$MODEL_PATH_FILE'
export DATA_ROOT='$DATA_ROOT'
export RUN_DIR='$RUN_DIR'
export DATASETS='$DATASET'
export MIN_TOKENS='$MIN_TOKENS'
export TOTAL_TOKENS='$TOTAL_TOKENS'
export FPS='$FPS'
export DTYPE='$DTYPE'
export ATTN_IMPLEMENTATION='$ATTN_IMPLEMENTATION'
export USE_CACHE='$USE_CACHE'
export SDPA_BACKEND='$SDPA_BACKEND'
export FORCE_REPEAT_KV_FOR_SDPA='$FORCE_REPEAT_KV_FOR_SDPA'
export NUM_WORKERS='$NUM_WORKERS'
export PREFETCH_FACTOR='$PREFETCH_FACTOR'
export PIN_MEMORY='$PIN_MEMORY'
export TP_PLAN='auto'
export TP_NPROC_PER_NODE='$TP_NPROC_PER_NODE'
export TP_NNODES='$SLURM_NNODES'
export TP_MASTER_ADDR='$MASTER_ADDR'
export TP_MASTER_PORT='$TP_MASTER_PORT'
export TP_NODE_RANK=\$SLURM_PROCID
export CHUNK='$CHUNK'
export INDEX='$INDEX'
export MAX_EXAMPLES='$MAX_EXAMPLES'
export COMPUTE_METRICS='$COMPUTE_METRICS'
bash '$REPO_ROOT/scripts/kaya/run_kaya_eval.sh'
"
