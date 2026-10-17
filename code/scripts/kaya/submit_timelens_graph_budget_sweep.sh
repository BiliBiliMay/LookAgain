#!/bin/bash

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/group/pmc073/huwang/worktrees/TimeLens/timelens-search-wrapper}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/pmc073/huwang/timelens}
OUTPUT_ROOT=${OUTPUT_ROOT:-$SCRATCH_ROOT/outputs}
PYTHON_BIN=${PYTHON_BIN:-$SCRATCH_ROOT/envs/timelens/bin/python}
SWEEP_TAG=${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}
BUDGETS=${BUDGETS:-5,10,15}
DATASETS=${DATASETS:-charades-timelens,activitynet-timelens,qvhighlights-timelens}
SEARCH_MODE=${SEARCH_MODE:-lookagain_astar}
LOOKAGAIN_CONTROLLER=${LOOKAGAIN_CONTROLLER:-guided}
LOOKAGAIN_ANCHOR_PERIOD=${LOOKAGAIN_ANCHOR_PERIOD:-3}

CHARADES_BASELINE=${CHARADES_BASELINE:-$OUTPUT_ROOT/timelens8b_dm_charades_redownload_20260427_231800/charades-timelens.jsonl}
ACTIVITYNET_BASELINE=${ACTIVITYNET_BASELINE:-$OUTPUT_ROOT/timelens8b_dm_repro_20260424_223712/activitynet-timelens.jsonl}
QVHIGHLIGHTS_BASELINE=${QVHIGHLIGHTS_BASELINE:-$OUTPUT_ROOT/timelens8b_dm_repro_20260424_223712/qvhighlights-timelens.jsonl}

COMBINED_BASELINE_DIR=${COMBINED_BASELINE_DIR:-$OUTPUT_ROOT/timelens8b_graph_sweep_baselines_$SWEEP_TAG}
BASELINE_RUN_DIR_B1=${BASELINE_RUN_DIR_B1:-$OUTPUT_ROOT/timelens8b_graph_full_b1_$SWEEP_TAG}
MANIFEST=${MANIFEST:-$OUTPUT_ROOT/timelens8b_graph_budget_sweep_$SWEEP_TAG.txt}

CHUNKS=${CHUNKS:-60}
MAX_CONCURRENT_PER_DATASET=${MAX_CONCURRENT_PER_DATASET:-1}
GRAPH_SEED_COUNT=${GRAPH_SEED_COUNT:-5}
GRAPH_BEAM_WIDTH=${GRAPH_BEAM_WIDTH:-3}
CRITIC_MODE=${CRITIC_MODE:-self}
CRITIC_CALLS=${CRITIC_CALLS:-2}
CRITIC_FPS=${CRITIC_FPS:-4}
CRITIC_MIN_CONFIDENCE=${CRITIC_MIN_CONFIDENCE:-0.55}

for baseline in "$CHARADES_BASELINE" "$ACTIVITYNET_BASELINE" "$QVHIGHLIGHTS_BASELINE"; do
    if [ ! -s "$baseline" ]; then
        echo "Missing baseline JSONL: $baseline" >&2
        exit 1
    fi
done

mkdir -p "$COMBINED_BASELINE_DIR" "$BASELINE_RUN_DIR_B1"
ln -sfn "$CHARADES_BASELINE" "$COMBINED_BASELINE_DIR/charades-timelens.jsonl"
ln -sfn "$ACTIVITYNET_BASELINE" "$COMBINED_BASELINE_DIR/activitynet-timelens.jsonl"
ln -sfn "$QVHIGHLIGHTS_BASELINE" "$COMBINED_BASELINE_DIR/qvhighlights-timelens.jsonl"

ln -sfn "$CHARADES_BASELINE" "$BASELINE_RUN_DIR_B1/charades-timelens.jsonl"
ln -sfn "$ACTIVITYNET_BASELINE" "$BASELINE_RUN_DIR_B1/activitynet-timelens.jsonl"
ln -sfn "$QVHIGHLIGHTS_BASELINE" "$BASELINE_RUN_DIR_B1/qvhighlights-timelens.jsonl"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

for dataset in charades-timelens activitynet-timelens qvhighlights-timelens; do
    "$PYTHON_BIN" evaluation/compute_metrics.py \
        -f "$BASELINE_RUN_DIR_B1/${dataset}.jsonl" \
        > "$BASELINE_RUN_DIR_B1/${dataset}.metrics.txt"
done

{
    echo "sweep_tag=$SWEEP_TAG"
    echo "created_at=$(date --iso-8601=seconds)"
    echo "repo_root=$REPO_ROOT"
    echo "combined_baseline_dir=$COMBINED_BASELINE_DIR"
    echo "budget_1_run_dir=$BASELINE_RUN_DIR_B1"
    echo "budgets=1,$BUDGETS"
    echo "search_mode=$SEARCH_MODE"
    echo "lookagain_controller=$LOOKAGAIN_CONTROLLER"
    echo "lookagain_anchor_period=$LOOKAGAIN_ANCHOR_PERIOD"
    echo "datasets=$DATASETS"
    echo "chunks_per_dataset=$CHUNKS"
    echo "max_concurrent_per_dataset=$MAX_CONCURRENT_PER_DATASET"
    echo "critic=$CRITIC_MODE/$CRITIC_CALLS"
} > "$MANIFEST"

IFS=',' read -ra BUDGET_LIST <<< "$BUDGETS"
for budget in "${BUDGET_LIST[@]}"; do
    if ! [[ "$budget" =~ ^[0-9]+$ ]] || [ "$budget" -lt 2 ]; then
        echo "Invalid GPU search budget: $budget" >&2
        exit 1
    fi
    run_tag="timelens8b_lookagain_${LOOKAGAIN_CONTROLLER}_full_b${budget}_${SWEEP_TAG}"
    echo "Submitting total-call budget $budget as $run_tag"
    submission=$(
        REPO_ROOT="$REPO_ROOT" \
        BASELINE_RUN_DIR="$COMBINED_BASELINE_DIR" \
        RUN_TAG="$run_tag" \
        DATASETS="$DATASETS" \
        SEARCH_MODE="$SEARCH_MODE" \
        LOOKAGAIN_CONTROLLER="$LOOKAGAIN_CONTROLLER" \
        LOOKAGAIN_ANCHOR_PERIOD="$LOOKAGAIN_ANCHOR_PERIOD" \
        MAX_TOTAL_CALLS="$budget" \
        GRAPH_SEED_COUNT="$GRAPH_SEED_COUNT" \
        GRAPH_BEAM_WIDTH="$GRAPH_BEAM_WIDTH" \
        CRITIC_MODE="$CRITIC_MODE" \
        CRITIC_CALLS="$CRITIC_CALLS" \
        CRITIC_FPS="$CRITIC_FPS" \
        CRITIC_MIN_CONFIDENCE="$CRITIC_MIN_CONFIDENCE" \
        CHARADES_CHUNKS="$CHUNKS" \
        ACTIVITYNET_CHUNKS="$CHUNKS" \
        QVHIGHLIGHTS_CHUNKS="$CHUNKS" \
        MAX_CONCURRENT_PER_DATASET="$MAX_CONCURRENT_PER_DATASET" \
        SUBMIT_SMOKE=0 \
        SUBMIT_FULL=1 \
        bash scripts/kaya/submit_timelens_search.sh
    )
    printf '%s\n' "$submission"
    {
        echo
        echo "[budget_$budget]"
        printf '%s\n' "$submission"
    } >> "$MANIFEST"
done

echo "manifest=$MANIFEST"
