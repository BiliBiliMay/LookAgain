#!/bin/bash
#SBATCH --job-name=q3vl_transfer
#SBATCH --account=pmc073
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
set -euo pipefail
ROOT=/group/pmc073/huwang/experiments/qwen3vl_transfer_20260922
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export FORCE_REPEAT_KV_FOR_SDPA=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
PYTHON=/scratch/pmc073/huwang/timelens/envs/timelens/bin/python
cd "$ROOT/code"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
if [ "${SMOKE:-0}" = 1 ]; then
    for dataset in charades-timelens activitynet-timelens qvhighlights-timelens; do
        "$PYTHON" -m evaluation.eval_qwen_transfer --root "$ROOT" --dataset "$dataset" --smoke
    done
    "$PYTHON" "$ROOT/validate_outputs.py" --root "$ROOT" --smoke
else
    test -f "$ROOT/smoke/PASSED.json"
    "$PYTHON" -m evaluation.eval_qwen_transfer --root "$ROOT" --dataset "$DATASET" --chunk 120 --index "$SLURM_ARRAY_TASK_ID"
fi
