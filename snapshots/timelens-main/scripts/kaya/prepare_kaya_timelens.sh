#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/pmc073/huwang/timelens}
ENV_PREFIX=${ENV_PREFIX:-$SCRATCH_ROOT/envs/timelens}
DATA_ROOT=${DATA_ROOT:-$SCRATCH_ROOT/data/TimeLens-Bench}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}
MODEL_PATH_FILE=${MODEL_PATH_FILE:-$SCRATCH_ROOT/models/qwen25vl7b_snapshot_path.txt}

export HF_HOME=${HF_HOME:-$SCRATCH_ROOT/hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$SCRATCH_ROOT/hf_cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$SCRATCH_ROOT/pip_cache}
export DATA_ROOT

mkdir -p \
    "$SCRATCH_ROOT" \
    "$SCRATCH_ROOT/envs" \
    "$SCRATCH_ROOT/models" \
    "$SCRATCH_ROOT/data" \
    "$SCRATCH_ROOT/logs" \
    "$SCRATCH_ROOT/outputs" \
    "$HF_HOME" \
    "$HF_HUB_CACHE" \
    "$PIP_CACHE_DIR"

module load Anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$ENV_PREFIX" ]; then
    conda create -y -p "$ENV_PREFIX" python=3.11
fi

conda activate "$ENV_PREFIX"

python -m pip install --upgrade pip
python -m pip install \
    -r "$REPO_ROOT/requirements.txt" \
    -f https://download.pytorch.org/whl/cu124
python -m pip install "huggingface_hub[cli]"

mkdir -p "$REPO_ROOT/data"
ln -sfn "$DATA_ROOT" "$REPO_ROOT/data/TimeLens-Bench"

model_snapshot=$(
    hf download "$MODEL_ID" --quiet
)
printf '%s\n' "$model_snapshot" > "$MODEL_PATH_FILE"
echo "Model snapshot path: $model_snapshot"

hf download TencentARC/TimeLens-Bench \
    activitynet-timelens.json \
    charades-timelens.json \
    qvhighlights-timelens.json \
    video_shards/activitynet/activitynet_shard_01.tar.gz \
    video_shards/activitynet/activitynet_shard_02.tar.gz \
    video_shards/activitynet/activitynet_shard_03.tar.gz \
    video_shards/activitynet/activitynet_shard_04.tar.gz \
    video_shards/activitynet/activitynet_shard_05.tar.gz \
    video_shards/activitynet/activitynet_shard_06.tar.gz \
    video_shards/charades/charades_shard_01.tar.gz \
    video_shards/qvhighlights/qvhighlights_shard_01.tar.gz \
    video_shards/qvhighlights/qvhighlights_shard_02.tar.gz \
    video_shards/qvhighlights/qvhighlights_shard_03.tar.gz \
    --repo-type dataset \
    --local-dir "$DATA_ROOT"

mkdir -p "$DATA_ROOT/videos/activitynet" "$DATA_ROOT/videos/charades" "$DATA_ROOT/videos/qvhighlights"
if [ -d "$DATA_ROOT/video_shards/activitynet" ]; then
    find "$DATA_ROOT/video_shards/activitynet" -name "*.tar.gz" -print0 | \
        xargs -0 -P 4 -I {} tar -xzf {} -C "$DATA_ROOT/videos/activitynet"
fi
if [ -d "$DATA_ROOT/video_shards/charades" ]; then
    find "$DATA_ROOT/video_shards/charades" -name "*.tar.gz" -print0 | \
        xargs -0 -P 4 -I {} tar -xzf {} --strip-components=1 -C "$DATA_ROOT/videos/charades"
fi
if [ -d "$DATA_ROOT/video_shards/qvhighlights" ]; then
    find "$DATA_ROOT/video_shards/qvhighlights" -name "*.tar.gz" -print0 | \
        xargs -0 -P 4 -I {} tar -xzf {} -C "$DATA_ROOT/videos/qvhighlights"
fi

python - <<'PY'
import os
from pathlib import Path

data_root = Path(os.environ["DATA_ROOT"])
for name in ("activitynet", "charades", "qvhighlights"):
    video_dir = data_root / "videos" / name
    count = sum(1 for _ in video_dir.rglob("*.mp4"))
    print(f"{name}_mp4_count={count}")
PY
