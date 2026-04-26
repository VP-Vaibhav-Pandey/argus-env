# Training code · the exact pipeline that produced the 3 runs

This folder contains the real code we ran to produce **seed 0**, **seed 1**, and the **no-metacognition ablation**.
Same training entrypoint, three different launcher scripts and configs.

## Files

| file | purpose |
|---|---|
| [`run_seed0.sh`](run_seed0.sh) | Seed 0 launcher · full v3.4 stack · produced **+7.0 pp** |
| [`run_seed1.sh`](run_seed1.sh) | Seed 1 launcher · same stack, `--seed 1` · produced **−5.0 pp** (Type H discovered) |
| [`run_no_meta.sh`](run_no_meta.sh) | Ablation launcher · drops `--capability-map`, `--causal-attribution`, `--per-cluster-detectors`, `--proposer-validator`, `--adversarial-proposer`, `--most-learnable-cluster`, `--active-defense` · produced **+2.5 pp** with 4× thinner defenses |
| [`run_self_improve_v3.py`](run_self_improve_v3.py) | The actual training entrypoint (~940 lines) · all three runs invoke this |
| [`collect_training_data.py`](collect_training_data.py) | Post-run forensic collector · writes the snapshot artifacts |
| [`configs/seed0_config.json`](configs/seed0_config.json) | Exact runtime config that was applied during seed 0 |
| [`configs/seed1_config.json`](configs/seed1_config.json) | Exact runtime config for seed 1 |
| [`configs/no_meta_config.json`](configs/no_meta_config.json) | Exact runtime config for the ablation |

## How the three runs differ

All three run on the same warmstart adapter (`outputs_warmstart/adapters`),
the same Qwen-2.5-1.5B-Instruct base, the same external context source,
and the same 15-episode × 64-step budget. They differ only in:

| flag | seed 0 | seed 1 | no-meta |
|---|---|---|---|
| `--seed` | 0 (default) | 1 | 0 |
| `--capability-map` | ✓ | ✓ | ✗ |
| `--causal-attribution` | ✓ | ✓ | ✗ |
| `--per-cluster-detectors` | ✓ | ✓ | ✗ |
| `--proposer-validator` | ✓ | ✓ | ✗ |
| `--adversarial-proposer` | ✓ | ✓ | ✗ |
| `--most-learnable-cluster` | ✓ | ✓ | ✗ |
| `--active-defense` | ✓ | ✓ | ✗ |
| `--ercv-rollback` | ✓ | ✓ | ✓ |
| `--soft-replay-decay` | ✓ | ✓ | ✓ |
| `--epiplexity-weight` | ✓ | ✓ | ✓ |
| `--spsi` | ✓ | ✓ | ✓ |

The shared baseline (ERCV + replay-decay + epiplexity-weight + SPSI) is the floor.
The metacognition stack on top of that is what surfaces Types F, G, H during training.

## Reproduce

These scripts assume:
- A CUDA GPU (the runs were done on RTX 5070 Ti Laptop · ~5 h each)
- A `.venv` with `requirements-env.txt` installed (plus `torch`, `transformers`, `peft`, `trl`, `datasets`)
- A pre-trained warmstart adapter at `outputs_warmstart/adapters` (or pass `--warmstart-adapter <your-path>`)

For a GPU-free / one-click reproduction at smaller scale, use the
[Colab notebook](../notebooks/argus_colab.ipynb) — same env, smaller params, ~8 min on a free T4.

## What gets saved

Each run writes into `outputs_v34_<name>/seed<N>/`:

```
config.json                    # the runtime config (exact flags applied)
pre_eval.json / post_eval.json # external GSM8K eval (n=100)
episodes.jsonl                 # per-episode reward / skill / lie firings
lie_taxonomy.jsonl             # per-episode global + per-cluster firings
epiplexity.jsonl               # information per token, sample-by-sample
spsi.jsonl                     # self-play stability index (5-component composite)
self_explanation_ledger.jsonl  # structured causal events
adapters/                      # final LoRA weights
```

The forensic printers in [`../scripts/`](../scripts/)
(`show_v34_result.py`, `show_v34_seed1_result.py`, `show_v34_no_meta_result.py`)
read these artifacts and produce the screenshot-ready summaries that appear in §06.5
of the [live writeup](https://vaibhav-pandeyy-argus-self-learning-env.hf.space/).
