import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ['EXPERIMENT_ROOT'])
revision = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
path = snapshot_download(
    'Qwen/Qwen3-VL-8B-Instruct', revision=revision,
    cache_dir='/group/pmc073/huwang/scratch_rescue_20260910/data/timelens/hf_cache',
    max_workers=4,
)
index = json.loads((Path(path) / 'model.safetensors.index.json').read_text())
for shard in set(index['weight_map'].values()):
    assert (Path(path) / shard).stat().st_size > 0, shard
(root / 'model_path.txt').write_text(path + '\n')
(root / 'model_manifest.json').write_text(json.dumps({
    'model_id': 'Qwen/Qwen3-VL-8B-Instruct', 'revision': revision,
    'snapshot': path, 'shards': sorted(set(index['weight_map'].values())),
}, indent=2) + '\n')
print('MODEL_READY', path, flush=True)
