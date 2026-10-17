#!/bin/bash
#SBATCH -J qwen25vl_timelens
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH -o /scratch/pmc073/huwang/timelens/logs/qwen25vl_timelens_%j.out
#SBATCH -e /scratch/pmc073/huwang/timelens/logs/qwen25vl_timelens_%j.err

set -euo pipefail

REPO_ROOT=/group/pmc073/huwang/TimeLens
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/pmc073/huwang/timelens}
ENV_PREFIX=${ENV_PREFIX:-$SCRATCH_ROOT/envs/timelens}
DATA_ROOT=${DATA_ROOT:-$SCRATCH_ROOT/data/TimeLens-Bench}
PRED_ROOT=${PRED_ROOT:-$SCRATCH_ROOT/outputs}
DATASETS=${DATASETS:-activitynet-timelens,qvhighlights-timelens}
CONFIG_CANDIDATES=${CONFIG_CANDIDATES:-"64:14336:2 64:10240:2 64:8192:2 64:6144:2 64:14336:1 64:10240:1 64:8192:1 64:6144:1 32:4096:1 16:4096:1"}
SMOKE_MAX_EXAMPLES=${SMOKE_MAX_EXAMPLES:-8}
MODEL_PATH_FILE=${MODEL_PATH_FILE:-$SCRATCH_ROOT/models/qwen25vl7b_snapshot_path.txt}
RUN_TAG_BASE=${RUN_TAG_BASE:-qwen25vl7b_timelens_${SLURM_JOB_ID}}

export HF_HOME=${HF_HOME:-$SCRATCH_ROOT/hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$SCRATCH_ROOT/hf_cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$SCRATCH_ROOT/pip_cache}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

module load Anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

mkdir -p "$SCRATCH_ROOT/logs" "$PRED_ROOT"
cd "$REPO_ROOT"

summary_file="$PRED_ROOT/${RUN_TAG_BASE}_summary.txt"
printf 'job_id=%s\n' "$SLURM_JOB_ID" > "$summary_file"
printf 'model_path=%s\n' "$(<"$MODEL_PATH_FILE")" >> "$summary_file"
printf 'datasets=%s\n' "$DATASETS" >> "$summary_file"
printf 'config_candidates=%s\n' "$CONFIG_CANDIDATES" >> "$summary_file"

run_and_capture() {
    local log_file=$1
    shift

    set +e
    "$@" 2>&1 | tee "$log_file"
    local status=${PIPESTATUS[0]}
    set -e
    return "$status"
}

is_retryable_failure() {
    local log_file=$1

    rg -qi \
        'out of memory|cuda out of memory|flash.?attention|bf16|bfloat16|no available kernel|not implemented|cudnn|cublas|cutlass|sdpa' \
        "$log_file"
}

collect_counts_and_metrics() {
    local run_dir=$1

    python - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
expected = {
    "activitynet-timelens": 4500,
    "qvhighlights-timelens": 1541,
}

for dataset, expected_count in expected.items():
    jsonl_path = run_dir / f"{dataset}.jsonl"
    if not jsonl_path.exists():
        raise SystemExit(f"Missing prediction file: {jsonl_path}")

    count = 0
    with jsonl_path.open("r") as fin:
        for line in fin:
            if line.strip():
                item = json.loads(line)
                count += len(item)

    print(f"{dataset}_count={count}")
    if count != expected_count:
        raise SystemExit(
            f"Unexpected prediction count for {dataset}: {count} != {expected_count}"
        )

    log_path = run_dir / f"{dataset}.log"
    if not log_path.exists():
        raise SystemExit(f"Missing metric log: {log_path}")

    content = log_path.read_text().strip()
    print(f"{dataset}_metrics={content}")
PY
}

for candidate in $CONFIG_CANDIDATES; do
    IFS=':' read -r min_tokens total_tokens fps <<< "$candidate"
    echo "===== Trying min_tokens=$min_tokens total_tokens=$total_tokens fps=$fps =====" | tee -a "$summary_file"

    smoke_log="$SCRATCH_ROOT/logs/${RUN_TAG_BASE}_smoke_min${min_tokens}_tok${total_tokens}_fps${fps}.log"
    if run_and_capture \
        "$smoke_log" \
        env \
            RUN_TAG="${RUN_TAG_BASE}_smoke_min${min_tokens}_tok${total_tokens}_fps${fps}" \
            MODEL_PATH_FILE="$MODEL_PATH_FILE" \
            DATA_ROOT="$DATA_ROOT" \
            PRED_ROOT="$PRED_ROOT" \
            DATASETS="activitynet-timelens" \
            MIN_TOKENS="$min_tokens" \
            TOTAL_TOKENS="$total_tokens" \
            FPS="$fps" \
            MAX_EXAMPLES="$SMOKE_MAX_EXAMPLES" \
            bash "$REPO_ROOT/scripts/kaya/run_kaya_eval.sh"; then
        echo "Smoke test passed at min_tokens=$min_tokens total_tokens=$total_tokens fps=$fps" | tee -a "$summary_file"
    else
        if is_retryable_failure "$smoke_log"; then
            echo "Smoke test failed with retryable error at min_tokens=$min_tokens total_tokens=$total_tokens fps=$fps" | tee -a "$summary_file"
            continue
        fi
        echo "Smoke test failed with non-retryable error. See $smoke_log" | tee -a "$summary_file"
        exit 1
    fi

    full_log="$SCRATCH_ROOT/logs/${RUN_TAG_BASE}_full_min${min_tokens}_tok${total_tokens}_fps${fps}.log"
    if run_and_capture \
        "$full_log" \
        env \
            RUN_TAG="${RUN_TAG_BASE}_full_min${min_tokens}_tok${total_tokens}_fps${fps}" \
            MODEL_PATH_FILE="$MODEL_PATH_FILE" \
            DATA_ROOT="$DATA_ROOT" \
            PRED_ROOT="$PRED_ROOT" \
            DATASETS="$DATASETS" \
            MIN_TOKENS="$min_tokens" \
            TOTAL_TOKENS="$total_tokens" \
            FPS="$fps" \
            bash "$REPO_ROOT/scripts/kaya/run_kaya_eval.sh"; then
        run_dir=$(tail -n 1 "$full_log")
        collect_counts_and_metrics "$run_dir" | tee -a "$summary_file"
        printf 'chosen_min_tokens=%s\n' "$min_tokens" | tee -a "$summary_file"
        printf 'chosen_total_tokens=%s\n' "$total_tokens" | tee -a "$summary_file"
        printf 'chosen_fps=%s\n' "$fps" | tee -a "$summary_file"
        printf 'run_dir=%s\n' "$run_dir" | tee -a "$summary_file"
        exit 0
    fi

    if is_retryable_failure "$full_log"; then
        echo "Full evaluation failed with retryable error at min_tokens=$min_tokens total_tokens=$total_tokens fps=$fps" | tee -a "$summary_file"
        continue
    fi

    echo "Full evaluation failed with non-retryable error. See $full_log" | tee -a "$summary_file"
    exit 1
done

echo "Exhausted token ladder without a successful evaluation." | tee -a "$summary_file"
exit 1
