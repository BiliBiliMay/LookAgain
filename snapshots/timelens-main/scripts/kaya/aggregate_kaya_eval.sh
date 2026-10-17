#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_DIR=${RUN_DIR:?RUN_DIR is required}
DATASETS=${DATASETS:-activitynet-timelens,qvhighlights-timelens}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/pmc073/huwang/timelens}
DEFAULT_PYTHON_BIN=$SCRATCH_ROOT/envs/timelens/bin/python
if [ -x "${PYTHON_BIN:-}" ]; then
    PYTHON_BIN=${PYTHON_BIN}
elif [ -x "$DEFAULT_PYTHON_BIN" ]; then
    PYTHON_BIN=$DEFAULT_PYTHON_BIN
else
    PYTHON_BIN=python
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

IFS=',' read -ra DATASET_LIST <<< "$DATASETS"

for dataset in "${DATASET_LIST[@]}"; do
    chunk_files=("$RUN_DIR"/"${dataset}"_*.jsonl)
    if [ ! -e "${chunk_files[0]}" ]; then
        echo "No chunk files found for $dataset in $RUN_DIR" >&2
        exit 1
    fi

    cat "$RUN_DIR"/"${dataset}"_*.jsonl > "$RUN_DIR/${dataset}.jsonl"
    "$PYTHON_BIN" evaluation/compute_metrics.py -f "$RUN_DIR/${dataset}.jsonl" | tee "$RUN_DIR/${dataset}.metrics.txt"
done
