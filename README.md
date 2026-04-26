# ARGUS — A polygraph for self-improving AI

> The model said it was learning. The validator agreed. We built the system that watches both.
> When we ran it on ourselves, it caught a **+7 pp claim that didn't reproduce**. That is the result.

ARGUS is the first **OpenEnv-compliant** self-improvement environment that *refuses* training steps where the model is lying to itself about whether it learned. Built for the **Meta · OpenEnv Self-Improvement Competition (2026)**.

---

## Live links

| | url |
|---|---|
| 🌐 **Live HF Space** | https://vaibhav-pandeyy-argus-self-learning-env.hf.space/ |
| 📁 **HF Space repo** | https://huggingface.co/spaces/Vaibhav-Pandeyy/Argus-Self-Learning-ENV |
| 💻 **GitHub** | https://github.com/VP-Vaibhav-Pandey/argus-env |
| 🧪 **Colab notebook** (one-click run) | https://colab.research.google.com/github/VP-Vaibhav-Pandey/argus-env/blob/main/notebooks/argus_colab.ipynb |
| 📖 **Architecture** | https://vaibhav-pandeyy-argus-self-learning-env.hf.space/architecture |
| 🔧 **Live API docs** | https://vaibhav-pandeyy-argus-self-learning-env.hf.space/docs |

---

## The result, in one paragraph

Every self-improving AI in 2026 reports gains. Almost none can prove the gains are real. ARGUS adds a *second observer* at every level of a self-play loop — solver chains, training episodes, full runs — and refuses any training step where two independent observers disagree. We ran it three times with different seeds. Seed 0 lifted external GSM8K pass@1 by +7 pp; seed 1 *regressed* by −5 pp; the metacognition-ablated control landed at +2.5 pp. **The 3-seed mean is +1.5 pp, well inside the Wilson 95% noise band.** Our own +7 pp headline didn't reproduce — and ARGUS is the system that caught it.

What *did* reproduce: the architecture's defensive density (19 / 16 events with metacognition vs 4 events without — **4× thinner ablation**), and the live discovery of three previously-unnamed failure modes (Types F, G, H) the architecture surfaced *during* training.

---

## Headline numbers

| | seed 0 (full v3.4) | seed 1 (full v3.4) | ablation (no-metacog) |
|---:|:---:|:---:|:---:|
| External pass@1 Δ | **+7.0 pp** | **−5.0 pp** | **+2.5 pp** |
| Defensive events (15 ep) | 19 | 16 | **4** |
| Lie firings (global) | 2 | 4 | 3 |
| Per-cluster detections | 8 | 6 | **0** |
| ERCV soft-rollbacks | 2 | 1 | 1 |
| Causal attributions | 7 | 5 | **0** |
| Live-discovered failure mode | **Type G** (plateau capture) | **Type H** (curriculum collapse) | — |

3-seed mean: **+1.5 pp** · Wilson 95% noise band: **±13.9 pp**.

---

## What's novel

1. **Chain-consensus reward** — replaces majority voting with a two-axis score (outcome agreement × intermediate-step agreement). Catches the case where every chain hallucinates the same fake premise and all "agree."
2. **Three-loop layered observers** — inner (per-step solver chains), middle (per-episode capability map + dual gauges), outer (per-run ERCV refusal gate at z < −2.5).
3. **Live failure-mode discovery** — the architecture asked *"which cluster is regressing?"* during training, and surfaced **3 previously-unnamed failure modes** (F, G, H) across 3 consecutive runs.
4. **Self-narrating ledger** — every defense event is logged as a structured JSONL entry with a causal hypothesis. The system writes its own forensic record.

---

## The 8 named failure modes

| | type | mechanism | origin |
|---|---|---|---|
| A | drift | proposer collapses to one type | designed (v1) |
| B | novelty collapse | all chains agree too early | designed (v1) |
| C | compute starvation | memorising the replay buffer | designed (v1) |
| D | catastrophic forgetting | "epi up, retest down" | designed (v3.1) |
| E | saturation | frontier zone empties | designed (v3.2) |
| **F** | **proposer hallucination** | proposer invents fake numeric premises; all 14 chains agree | **discovered live · v3.3 · 2026-04-24** |
| **G** | **plateau capture** | a single cluster captures the curriculum and saturates | **discovered live · v3.4 seed 0 · 2026-04-25** |
| **H** | **defense-induced curriculum collapse** | a defense fires, easy/hard ratio explodes 1.10 → 6.17 | **discovered live · v3.4 seed 1 · 2026-04-26** |

**Better defenses don't eliminate failure modes — they surface deeper ones.** The taxonomy is open-ended by design.

---

## Architecture (in five lines)

```python
# the load-bearing trick of the outer loop
def ercv_refuse(epi_gain, retest_trend, history):
    z = (epi_gain - mean(history)) / std(history)
    if z < -2.5 and retest_trend < 0:
        return "REFUSE"           # soft rollback to last good adapter
    return "COMMIT"
```

Three nested observers, each at a different timescale:

| loop | timescale | watches | refuses when |
|---|---|---|---|
| **Inner** | per-step | solver chains | chains agree on outcome but diverge on intermediate numbers |
| **Middle** | per-episode | capability map · dual gauges (epiplexity + retest) | per-cluster signal disagrees with global signal |
| **Outer** | per-run | full-run statistics | z(epi_gain) < −2.5 against historical retest baseline |

Full architecture: [hf_space/ARCHITECTURE.md](hf_space/ARCHITECTURE.md) · live render: [/architecture](https://vaibhav-pandeyy-argus-self-learning-env.hf.space/architecture)

---

## Quickstart — one-click Colab

The fastest way to see ARGUS run end-to-end:

```
https://colab.research.google.com/github/VP-Vaibhav-Pandey/argus-env/blob/main/notebooks/argus_colab.ipynb
```

22 cells · ~8 minutes on a free T4 · Qwen-2.5-1.5B-Instruct + 4-bit + LoRA + GSM8K. Watch the lie taxonomy fire in real time.

---

## OpenEnv contract

```
GET  /healthz                       — liveness
POST /reset                         — start a new session
POST /step   { session_id, action } — one env transition
GET  /state?session_id=...          — inspect env info
GET  /buffer?session_id=...         — training buffer for agent SFT
POST /clear_buffer                  — agent signals it consumed the buffer
```

The agent drives propose→solve→score→train. The env returns `{ obs, reward, terminated, info }` per step, where `info` carries lie-type scores, ERCV decision, capability-map snapshot, and causal-attribution hits. Reward is `chain_consensus_combined` (outcome × process). Training pairs come pre-weighted in `/buffer`; the agent skips the SFT step when ERCV refused.

Live Swagger: https://vaibhav-pandeyy-argus-self-learning-env.hf.space/docs

---

## Reproduce

```bash
# Local install
git clone https://github.com/VP-Vaibhav-Pandey/argus-env
cd argus-env
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r hf_space/requirements-env.txt

# Run the env locally
cd hf_space
uvicorn src.env.openenv_app:app --host 0.0.0.0 --port 7860

# Or via Docker
docker build -t argus-env hf_space/
docker run -p 7860:7860 argus-env
```

The full self-improvement training pipeline (model + env + agent) is reproducible from `notebooks/argus_colab.ipynb`. End-to-end runtime ≈ 8 min on a free Colab T4 (or ~5 h for the full 15-episode v3.4 stack on RTX 5070 Ti).

---

## Repo structure

```
.
├── hf_space/                 # OpenEnv server + Space writeup (deployed to HF)
│   ├── src/env/server.py     #   FastAPI app + interactive HTML writeup
│   ├── src/env/self_improvement_env.py  # core RecursiveSelfImprovementEnv
│   ├── src/control/          #   ERCV, detector bank, defense dispatch, SPSI
│   ├── src/eval/             #   capability map, causal attribution, lie taxonomy
│   ├── ARCHITECTURE.md
│   └── Dockerfile
├── notebooks/
│   └── argus_colab.ipynb     # one-click reproduce notebook
├── scripts/
│   ├── plot_v34_unified.py   # all 7 unified plots (3-seed comparison)
│   ├── show_v34_result.py    # seed 0 forensic terminal printer
│   ├── show_v34_seed1_result.py
│   └── show_v34_no_meta_result.py
├── outputs_v34_full/         # seed 0 raw artifacts
├── outputs_v34_full_seed1/   # seed 1 raw artifacts
├── outputs_v34_no_meta/      # ablation raw artifacts
├── figs/v34_unified/         # 7 final plots (PNG + HTML)
├── deep_analysis_v34.md      # 12 patterns from seed 0
├── deep_analysis_v34_seed1.md# 12 patterns from seed 1 (Type H discovery)
└── README.md                 # this file
```

---

## Judging-criteria fit

| criterion | weight | where to look |
|---|---:|---|
| Environment innovation | 40% | §02 chain-consensus, §03 architecture, §08 taxonomy on the [live Space](https://vaibhav-pandeyy-argus-self-learning-env.hf.space/) |
| Storytelling | 30% | hero + §05 three seeds + §06 the inversion + §06.5 receipts |
| Reward improvement | 20% | §06 replay component, §07 fingerprints, §06.5 terminal screenshots |
| Reward & training pipeline | 10% | §04 live demo (in-browser), §10 reproduce + Colab notebook |

---

## Honest framing

We will not lead with a metric we can't reproduce. The +7 pp on seed 0 is real but single-seed; mean across three seeds is +1.5 pp within Wilson noise. **The architecture's contribution is not a metric trick — it is a measurement instrument.** What reproduces is defensive density (19 / 16 events with metacognition · 4 without), and the live discovery of new named failure modes that previous taxonomies didn't have.

---

## Citation

Conceptual frame for the in-loop / external divergence regime: Liu et al., *Self-Play → Self-Evolution* (Feb 2026).

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*Built for Meta · OpenEnv Self-Improvement Competition · 2026*
