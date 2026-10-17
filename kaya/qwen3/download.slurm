#!/bin/bash
#SBATCH --job-name=q3vl_download
#SBATCH --account=pmc073
#SBATCH --partition=work
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
export EXPERIMENT_ROOT=/group/pmc073/huwang/experiments/qwen3vl_transfer_20260922
export HF_HOME=/group/pmc073/huwang/scratch_rescue_20260910/data/timelens/hf_home
export HF_HUB_DOWNLOAD_TIMEOUT=120
export PYTHONUNBUFFERED=1
/scratch/pmc073/huwang/timelens/envs/timelens/bin/python "$EXPERIMENT_ROOT/download_model.py"
