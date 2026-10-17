"""Submit once; preserve job IDs immediately after each successful submission."""
import json
from pathlib import Path
import subprocess

root = Path('/group/pmc073/huwang/experiments/qwen3vl_transfer_20260922')
state_path = root / 'submission.json'
assert not state_path.exists(), 'Already submitted; inspect submission.json before any resubmission.'
state = {'root': str(root), 'download_job': '8322',
         'model_id': 'Qwen/Qwen3-VL-8B-Instruct',
         'revision': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
         'conditions': ['baseline', 'lookagain_b15', 'fixed_search_b15'],
         'chunks_per_dataset': 120, 'max_concurrent_per_dataset': 2, 'jobs': {}}


def submit(name, arguments, script):
    command = ['sbatch', '--parsable', '--job-name=' + name,
               '--output=' + str(root / 'logs' / (name + '_%A_%a.out')),
               '--error=' + str(root / 'logs' / (name + '_%A_%a.err')),
               '--kill-on-invalid-dep=yes', *arguments, str(root / script)]
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    assert job.isdigit(), job
    state['jobs'][name] = {'id': job, 'command': command}
    state_path.write_text(json.dumps(state, indent=2) + '\n')
    print(name, job, flush=True)
    return job


model = json.loads((root / 'model_manifest.json').read_text())
assert model['revision'] == state['revision']
assert (root / 'model_path.txt').read_text().strip() == model['snapshot']
for shard in model['shards']:
    assert (Path(model['snapshot']) / shard).stat().st_size > 0
# The completed download is already verified; its job may have aged out of
# Slurm's live dependency table. Do not depend on that historical job ID.
smoke = submit('q3vl_smoke', ['--partition=ondemand-gpu', '--time=04:00:00',
                            '--export=ALL,SMOKE=1'], 'run_chunk.slurm')
arrays = []
for dataset in ('charades-timelens', 'activitynet-timelens', 'qvhighlights-timelens'):
    arrays.append(submit('q3vl_' + dataset.split('-')[0],
                         ['--array=0-119%2', '--dependency=afterok:' + smoke,
                          '--export=ALL,SMOKE=0,DATASET=' + dataset], 'run_chunk.slurm'))
submit('q3vl_aggregate', ['--dependency=afterok:' + ':'.join(arrays)], 'aggregate.slurm')
