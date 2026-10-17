# LookAgain: Budgeted Test-Time Search for Video Temporal Grounding

Source code accompanying Hunter Wang's honours dissertation at the University of Western Australia. LookAgain searches temporal crops with a frozen video language model and verifies a selected interval against the one-shot prediction.

This release was retrieved from Kaya on 17 October 2026. It includes the frozen TimeLens search implementation, the Qwen3-VL transfer experiment, fixed-window consensus control, tests, and original cluster launch scripts. The files in `code/` match the saved Qwen3 experiment's twelve-file SHA-256 manifest. Its search implementation is identical to the frozen TimeLens experiment version.

## Contents

| Path | Contents |
| --- | --- |
| `code/evaluation/timelens_search.py` | Temporal hypothesis graph, search, candidate selection and verification decisions |
| `code/evaluation/timelens_runner.py` | Model loading, video crops, prompts and verifier calls |
| `code/evaluation/eval_timelens_search.py` | TimeLens evaluation using saved one-shot predictions |
| `code/evaluation/eval_qwen_transfer.py` | Paired Qwen3 baseline, LookAgain and fixed-search evaluation |
| `code/evaluation/qwen_transfer_control.py` | Fixed multiscale consensus baseline |
| `code/tests/` | Existing model-free regression tests |
| `code/timelens/`, `code/training/`, `code/train_scripts/` | Dataset loaders and inherited TimeLens inference/training support |
| `code/scripts/`, `kaya/qwen3/` | Original shell, Slurm, download, submission and validation scripts |
| `snapshots/timelens-frozen/` | Complete frozen TimeLens source snapshot used for the search experiments |
| `snapshots/timelens-worktree/`, `snapshots/timelens-main/` | Additional TimeLens development source retrieved from Kaya; use the frozen snapshot for thesis experiments |
| `provenance/kaya_sources.json` | Remote source locations and verified file hashes for all four source trees |
| `code_manifest.json`, `model_manifest.json` | Original Qwen3 experiment source and checkpoint records |

The original source files and licence headers are preserved. Historical README files describe their state when written; they are not current experiment-status reports. Model weights, videos, annotation datasets, prediction outputs, environments, caches and logs are supplied separately by their respective sources or retained in the research archive.

## Setup

The Qwen3 experiment used Python 3.11, PyTorch 2.6.0 with CUDA 12.4, Transformers 4.57.1, and two NVIDIA V100 GPUs with CPU offloading. Use a suitable Linux/CUDA environment for inference. The model-free tests also run without GPU libraries.

```bash
git clone https://github.com/BiliBiliMay/LookAgain.git
cd LookAgain
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
```

Obtain [TimeLens-Bench](https://huggingface.co/datasets/TencentARC/TimeLens-Bench) through its original distribution and arrange it as follows, or symlink that directory into `code/data/TimeLens-Bench`:

```text
code/data/TimeLens-Bench/
  charades-timelens.json
  activitynet-timelens.json
  qvhighlights-timelens.json
  videos/charades/
  videos/activitynet/
  videos/qvhighlights/
```

The expected evaluation set has 3,363 Charades-STA, 4,500 ActivityNet Captions and 1,541 QVHighlights video-query pairs. The dataset loader normalises query whitespace and trailing periods.

Download the checkpoint required for the experiment:

| Model | Thesis checkpoint revision |
| --- | --- |
| [TencentARC/TimeLens-8B](https://huggingface.co/TencentARC/TimeLens-8B) | `657ea9b5abefc2f9e34aaead144e4440c938f664` |
| [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |

## Verify the source and run tests

From the repository root:

```bash
python3 tools/verify_sources.py
cd code
python3 -m unittest discover -s tests -t . -v
```

All 17 existing tests passed when this release was prepared. They cover interval handling, graph merging, call budgets, verifier order disagreement and the fixed-search control. Python syntax and shell syntax were checked across the release. GPU inference was not rerun for publication of this source snapshot.

## Qwen3 transfer evaluation

Write the absolute local path of the pinned Qwen3 checkpoint to `model_path.txt` in the repository root. The harness expects to run from `code/`, with `--root` pointing to the repository root:

```bash
cd code
export FORCE_REPEAT_KV_FOR_SDPA=1
python -m evaluation.eval_qwen_transfer \
  --root .. --dataset charades-timelens --chunk 120 --index 0
```

The command processes one chunk. Repeat indices 0 through 119 for each of `charades-timelens`, `activitynet-timelens` and `qvhighlights-timelens`. Each chunk evaluates the same queries under the one-call baseline, LookAgain with a 15-call cap and fixed-search with a 15-call cap. Outputs are checkpointed under `outputs/`. The harness retains the original V100 memory and offload settings.

For the original duration-selected execution check, use `--smoke` in place of `--chunk` and `--index`. After all chunks have completed, validate and aggregate from `code/`:

```bash
PYTHONPATH=. python ../kaya/qwen3/validate_outputs.py --root ..
```

## TimeLens search evaluation

Generate one-shot predictions using the original evaluator described in `code/README.md`, then pass that saved baseline to the search evaluator. For the guided search used in the thesis:

```bash
cd code
python -m evaluation.eval_timelens_search \
  --dataset charades-timelens \
  --baseline_jsonl /absolute/path/to/charades-timelens.jsonl \
  --pred_path /absolute/path/to/output/charades-timelens \
  --model_path /absolute/path/to/pinned/TimeLens-8B \
  --search_mode lookagain_astar \
  --lookagain_controller guided --lookagain_anchor_period 3 \
  --max_total_calls 15 --critic_mode self --critic_calls 2 \
  --min_tokens 64 --total_tokens 14336 --max_new_tokens 512 \
  --fps 2 --zoom_fps 4 --critic_fps 4 \
  --dtype auto --attn_implementation sdpa --sdpa_backend efficient \
  --device balanced --max_gpu_memory_gib 14 --cpu_offload_gib 120
```

Set `--max_total_calls 25` for the 25-call configuration. The stored baseline prediction must come from the same checkpoint, query set and evaluation settings. The archived Kaya scripts record additional environment and scheduling settings. Their absolute paths, Slurm account and resource requests must be adapted to the execution environment before submission.

## Attribution and licence

This project builds on [TencentARC/TimeLens](https://github.com/TencentARC/TimeLens). The upstream README is retained at `code/README.md`. Upstream training files are included to preserve the retrieved source; LookAgain's thesis experiments use frozen model weights.

The original `LICENSE`, `NOTICE`, and per-file licence headers are retained. TimeLens redistribution and use are subject to those terms, including its academic-use and territorial conditions. Third-party components retain their original licences. Dataset and model access remain subject to their respective terms.
