#!/bin/bash
#SBATCH --job-name=q3vl_aggregate
#SBATCH --account=pmc073
#SBATCH --partition=work
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
ROOT=/group/pmc073/huwang/experiments/qwen3vl_transfer_20260922
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
cd "$ROOT/code"
/scratch/pmc073/huwang/timelens/envs/timelens/bin/python "$ROOT/validate_outputs.py" --root "$ROOT"
