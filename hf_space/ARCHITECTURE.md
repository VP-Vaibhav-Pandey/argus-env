# ARGUS — a self-improvement environment that catches AI lying to itself

**ARGUS** is the v3.4 architecture: a three-loop control system whose
many sensors continuously watch for self-deception in a self-play
self-improvement loop, and refuse any training step where two
independent gauges of learnable information disagree.

The name comes from Greek myth — Argus Panoptes, the many-eyed
watchman. In the myth Argus was *lulled to sleep by deception*; in
v3.3 (before this architecture) our system was lulled in exactly the
same way — silently amplifying proposer hallucinations into 25.5%
of total compute. ARGUS is the watchman who doesn't sleep.

This document describes the architecture: every module's purpose,
what it reads, what it writes, and which control loop it
participates in. Everything in `src/control/` and `src/eval/` is
either a *sensor*, a *decider*, or an *actuator* — there are no
free-floating features.

## Control objective (one sentence)

Maintain monotonic learnable-information gain on real word-problem
distributions, while refusing self-play episodes that fail
cross-validation between two independent gauges of "did we learn."

The two gauges are **epiplexity** (NLL_pre vs NLL_post on new SFT
data, Liu et al. 2026) and **retest_trend** (chain-consensus delta
on past problems, our work). Disagreement is the safety signal.

## ARGUS at a glance

| | |
|---|---|
| **Tagline** | The self-improvement environment that catches AI lying to itself. |
| **Headline mechanism** | Cross-validating two independent gauges of learnable info (epi_gauge + retest_gauge) — the only way to refuse a self-play step. |
| **Discovery** | Type F — proposer hallucination cascade. Found live in v3.3 long when ARGUS's predecessor lulled itself into amplifying false-premise problems to 25.5% of compute. ARGUS catches this at the inner loop (`proposer_validator`) AND at the outer loop (`curriculum_planner` refuses to target broken clusters). |
| **Single-scalar health** | SPSI (Self-Play Stability Index): composite scalar that goes negative when latent collapse is forming. |
| **Loops** | Outer (run, hours) — what should I learn?  Middle (episode, minutes) — did I actually learn?  Inner (step, seconds) — is this attempt valid? |

## Three loops at three timescales

```
┌─ OUTER LOOP — RUN (hours) ─────────────────────────────────────┐
│ "what should I learn?"                                         │
│                                                                │
│   capability_map  ──▶  curriculum_planner  ──▶  proposer seed  │
│   (sensor)             (decider)                                │
│                            ▲                                   │
│                            │                                   │
│                       capacity_planner                          │
│                       (solver_k, mnt growth)                    │
└────────────────────────────┬───────────────────────────────────┘
                             │
┌─ MIDDLE LOOP — EPISODE (minutes) ──┴───────────────────────────┐
│ "did I actually learn?"                                        │
│                                                                │
│   epi_gauge  ┐                                                 │
│              ├─▶ detector_bank ─▶ causal_attributor            │
│   retest     ┘    (sensor +       (sensor)                     │
│   gauge           per-cluster)         │                       │
│                       │                ▼                       │
│                       ▼          defense_dispatch              │
│                   ercv_module    (actuator)                    │
│                   (decider +     ─ soft replay decay           │
│                    actuator)     ─ external context inject     │
│                                  ─ solver temp bump            │
│                                  ─ frontier bump               │
└────────────────────────────┬───────────────────────────────────┘
                             │
┌─ INNER LOOP — STEP (seconds) ──┴───────────────────────────────┐
│ "is this attempt valid?"                                       │
│                                                                │
│   proposer ─▶ proposer_validator ─▶ solver (k chains)          │
│               (NEW, Type-F gate)        │                      │
│                                         ▼                      │
│                              chain_consensus_rubric            │
│                              (sensor → frontier zone)          │
│                                         │                      │
│                                         ▼                      │
│                                   ewcs_weighter                │
│                                   (SFT pair weight)            │
└────────────────────────────────────────────────────────────────┘

CROSS-CUTTING:
  spsi          — single composite scalar over all signals
  ledger        — every detector fire, every defense, every attribution
  adversarial_proposer — proposer-side reward shaping (factuality
                          + difficulty + frontier-fit)
```

## Module map

### `src/eval/` — sensors

| Module | Reads | Writes | Used in |
|---|---|---|---|
| `chain_consensus_rubric` | k solver chains | per-problem score, frontier label | inner |
| `epiplexity` | NLL_pre, NLL_post | epi_gain (per-token) | middle |
| `capability_map` | problem hidden states | cluster assignments + per-cluster stats | outer |
| `causal_attribution` | training pairs + regressions | hypothesis (cluster X→Y) | middle |
| `lie_taxonomy` | env_info + sft_stats + epi_gain | global Type A–E scores | middle |
| **`proposer_validator`** *(new)* | proposer text | accept/reject + kind | inner |

### `src/control/` — deciders + actuators

| Module | Reads | Writes | Loop | New in v3.4 |
|---|---|---|---|---|
| `curriculum_planner` | per-cluster stats | target_cluster_id + reason | outer | ✅ |
| `detector_bank` | env_info + cluster stats | global + per-cluster firings | middle | ✅ |
| `ercv` | epi_gain + retest_trend + history | rolled_back, severity, zscore | middle | ✅ (extracted) |
| `defense_dispatch` | typed firing | typed action (decay/temp/etc) | middle | ✅ |
| `spsi` | full episode log | composite scalar + label | cross | ✅ |

### `src/agent/` — actor

| Module | Reads | Writes | New in v3.4 |
|---|---|---|---|
| `self_improve_agent` | env obs | proposer/solver outputs + SFT loss | — |
| **`adversarial_proposer`** | proposer_buffer | reweighted buffer + breakdowns | ✅ |

## Data contract (the test for "does this module belong?")

Anything not in this contract doesn't belong. Things that fail it
today and should be removed in a future cleanup:

- `proposer_buffer_size` field — produces 0 every episode, no consumer.
- Standalone `skill_level` metric — not a planner input, not a defense
  input. Noisy decoration.
- Duplicate frontier counters (`n_solve_frontier` vs `frontier_steps`).

## ARGUS flag matrix (v3.4 implementation)

All flags default OFF, so prior runs (v3.3 long, v3.1b, v1) reproduce
exactly. Flip flags ON to enable ARGUS-aligned behavior.

| Flag | What it changes | Bug it fixes |
|---|---|---|
| `--most-learnable-cluster` | Replace `weakest_cluster` heuristic with eligibility-gated planner (skips oversized + regressing clusters) | cluster-2 amplification (25.5% wasted compute in v3.3 long) |
| `--per-cluster-detectors` | Run Type D per cluster; fires on cluster-local regressions | 6 of 15 eps had real cluster regressions but no global firing |
| `--soft-replay-decay` | Type-C action becomes "keep latest 50%" instead of full clear | -23pp ep9 skill cliff in v3.3 long |
| `--replay-keep-fraction` | Tunes the decay fraction (default 0.5) | — |
| `--proposer-validator` | Drop proposer pairs with demonstrably false numeric premises | Type F (proposer hallucinations) |
| `--adversarial-proposer` | Reweight proposer SFT by factuality × difficulty × frontier-fit | proposer not trained adversarially (was just sampled) |
| `--spsi` | Compute composite stability index per episode | no single-scalar health gauge for judges |

Recommended "ARGUS full" config (all flags on):
```
--most-learnable-cluster --per-cluster-detectors --soft-replay-decay
--proposer-validator --adversarial-proposer --spsi
--active-defense --capability-map --causal-attribution
--ercv-rollback --ercv-soft-rollback --ercv-zscore
--epiplexity-weight --capacity-growth 0.05
```

## Self-Play Stability Index (SPSI) — definition

Single composite scalar in roughly [-1, +1]:

```
SPSI = + 0.30 · epi_signal           (saturate epi_gain in [0, 0.04])
       + 0.20 · frontier_signal      (saturate frontier_share in [0.05, 0.25])
       - 0.15 · retest_decay         (saturate -retest_trend in [0, 0.08])
       - 0.15 · detector_load        (max(global_lie_score, per_cluster_d_pressure))
       - 0.20 · wasted_share         (share of compute on broken clusters)
```

`wasted_share` is the load-bearing innovation:

```
wasted_share = sum(share(c) for c in clusters
                   if mean_reward(c) < 0.50
                   and n_problems(c) >= 5
                   and last_retest_delta(c) < +0.05)
```

The third gate (last_retest_delta < +0.05) distinguishes structurally
broken clusters (the cluster-2 hallucination cohort) from healthy
weak-but-improving clusters early in training.

Validated retroactively on the v3.3 long run:
- ep1: SPSI=+0.473 (healthy, early training)
- ep5: SPSI=+0.047 (consolidation, Type D fire caught)
- eps 9–15: wasted_share rises monotonically 0.07 → 0.26 as cluster 2 grows

## ERCV — the load-bearing innovation

ERCV cross-validates two independent gauges of learnable-info:

```
if epi_gain > 0 AND retest_trend < -trigger:
    rolled_back = True
    severity = min(1, |retest_trend| / max(eps, |epi_gain|))
    weights = (1 - severity) · post + severity · pre
```

This is the only mechanism in the system that can *refuse* a
self-play step. Liu et al. 2026 propose epiplexity as the gauge; we
add the second gauge (retest from chain-consensus) and a refusal
mechanism. The disagreement IS the safety signal.

The module supports two modes:
- **fixed_threshold** (default) — `retest_trend < -trigger`
- **z-score** (long-horizon) — self-calibrating after warmup

And two rollback styles:
- **hard** — full revert to ref_weights
- **soft** (default) — interpolate by severity

## Type F — discovered failure mode

The v3.3 long run revealed a 6th failure mode that was not in the
A-E taxonomy: **the proposer can ship problems with arithmetically
false premises** that pass syntactic gates and chain-consensus
verification. Chain-consensus verifies *inter-chain agreement*, not
*ground truth*; k chains can hallucinate the same continuation given
a false premise.

The capability map silently caught it (cluster 2 isolated 209
hallucinated problems = 25.5% of total compute), but the lie
taxonomy never fired because all signals at the run level looked
healthy. The fix is two-layered:

1. **`proposer_validator`** at the inner loop — refuses problems
   with demonstrably false numeric claims at generation time.
2. **`curriculum_planner.most_learnable_cluster`** at the outer
   loop — refuses to target oversized or regressing clusters, so
   even if some hallucinations slip through, they don't compound.

Both are flag-gated in v3.4 so the v3.3 long behavior reproduces
when the flags are off.

## What's NOT in ARGUS (deliberately)

- More detector types beyond A–F. The system has more diagnosis than
  it can act on; per-cluster decomposition is more leverage than yet
  another global signal.
- New mechanisms beyond what fixes a documented v3.3 bug. Each new
  module exists because the data showed a gap.
- A new domain. The math/GSM8K substrate has enough sunk evidence
  that switching now loses more than it gains.
