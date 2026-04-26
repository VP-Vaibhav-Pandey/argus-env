"""V3 Recursive Self-Improvement loop — paper-aligned (Liu et al. 2026).

This driver extends run_self_improve.py with three mechanisms drawn
directly from Liu et al. 2026 "Self-Play Only Evolves When
Self-Synthetic Pipeline Ensures Learnable Information Gain":

  1. §3.3 Proactive Information Seeking. The env is configured with
     an external_context_pool of GSM8K-train problem STATEMENTS used
     ONLY as conditioning seeds for the proposer. The proposer must
     generate a *variation* (different numbers, harder twist). Gold
     labels are NEVER fed to the training loop — we use the statements
     as context, not supervision. This breaks the closed-loop bound
     that caused v2 regression.

  2. §3.2 Capacity Growth. solver_k and solver_mnt grow per episode.
     As the loop exposes more learnable structure, inference-time
     budget scales to absorb it.

  3. §3.1 Asymmetric Co-evolution (strong→weak verifier sync). The
     env's verifier_strength>0 makes the effective frontier lower bound
     slide up with skill_level, keeping the verifier's acceptance bar
     climbing with the solver.

Diagnostic (not training signal): after each episode, we snapshot solver
weights and measure epiplexity (bounded MDL via prequential coding,
paper's eq. 7). The resulting learnable-info-per-token curve is the
paper-aligned judge-facing artifact: monotonic growth = genuine
self-evolution, flat = closed-loop collapse.

Usage:
    python scripts/run_self_improve_v3.py \\
        --warmstart-adapter outputs_warmstart/adapters \\
        --episodes 5 --steps-per-episode 64 \\
        --eval-n 50 --output-dir outputs_self_improve_v3

v1 is reproducible by running this script with
  --external-context-rate 0 --verifier-strength 0 --capacity-growth 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _load_external_context_pool(path: str | None, limit: int) -> list[str]:
    """Load GSM8K-train problem texts to use as proposer seeds.

    We load ONLY the `question` field. Gold `answer` field is never
    read here. If path is None, we fall back to datasets.load_dataset.
    """
    questions: list[str] = []
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    q = row.get("question") or row.get("problem")
                    if isinstance(q, str) and q:
                        questions.append(q)
                except json.JSONDecodeError:
                    continue
        return questions[:limit]
    # Fallback: datasets
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="train")
        questions = [row["question"] for row in ds][:limit]
    except Exception as e:  # noqa: BLE001
        print(f"  WARN: no external context loaded ({e}) — proposer will be free-form")
    return questions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmstart-adapter", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps-per-episode", type=int, default=64)
    ap.add_argument("--sft-trigger", type=int, default=24)
    ap.add_argument("--solver-k", type=int, default=8,
                    help="Initial solver_k at episode 1.")
    ap.add_argument("--solver-mnt", type=int, default=256,
                    help="Initial solver max_new_tokens at episode 1.")
    ap.add_argument("--capacity-growth", type=float, default=0.15,
                    help="Per-episode multiplicative growth of k and mnt "
                         "(Liu §3.2). 0 disables.")
    ap.add_argument("--solver-k-max", type=int, default=16)
    ap.add_argument("--solver-mnt-max", type=int, default=384)
    ap.add_argument("--external-context-path", default=None,
                    help="JSONL of GSM8K-train {question, ...}. "
                         "If missing, tries datasets.load_dataset.")
    ap.add_argument("--external-context-limit", type=int, default=512,
                    help="Max number of seed problems loaded.")
    ap.add_argument("--external-context-rate", type=float, default=0.5,
                    help="Fraction of propose obs that carry a seed "
                         "(Liu §3.3). 0 disables.")
    ap.add_argument("--verifier-strength", type=float, default=0.2,
                    help="Strong→weak verifier sync coefficient (Liu §3.1). "
                         "Effective frontier_low += this*skill_level. "
                         "0 disables.")
    ap.add_argument("--blind-spot-rate", type=float, default=0.5,
                    help="v3.1 NOVEL. Probability a blind-spot problem "
                         "(retest regression) is used as proposer seed "
                         "instead of uniform external pool. Fuses our "
                         "consequence-tracking with Liu §3.3. 0 disables.")
    ap.add_argument("--epiplexity-weight", action="store_true",
                    help="v3.1 NOVEL. Weight each SFT pair by pre-update "
                         "per-token NLL (advantage = learnable-info "
                         "content). Fuses our consensus-weighted SFT "
                         "with Liu's bounded-MDL framework.")
    ap.add_argument("--ercv-rollback", action="store_true",
                    help="v3.1 NOVEL. Epiplexity-Retest Cross-Validation: "
                         "if epi_gain>0 but recent retest_trend < "
                         "-ercv-trigger, revert to pre-episode weights. "
                         "Guards against closed-loop overfitting.")
    ap.add_argument("--ercv-trigger", type=float, default=0.05,
                    help="Retest trend threshold for ERCV rollback.")
    ap.add_argument("--ercv-min-retests", type=int, default=4,
                    help="Min number of retests before ERCV can fire "
                         "(hysteresis against noisy single retest).")
    ap.add_argument("--ercv-hysteresis", type=int, default=1,
                    help="Require this many consecutive episodes of "
                         "divergence before ERCV rollback fires.")
    # ---- long-horizon ERCV extensions (Roadmap: A + C from README §long-training) ----
    ap.add_argument("--ercv-zscore", action="store_true",
                    help="Self-calibrating threshold. Instead of comparing "
                         "retest_trend to a fixed --ercv-trigger, compare "
                         "its z-score (using running mean/std across past "
                         "episodes) to --ercv-zscore-threshold. Kicks in "
                         "after ercv-zscore-warmup episodes of history.")
    ap.add_argument("--ercv-zscore-threshold", type=float, default=-2.5,
                    help="Trigger when (retest_trend - running_mean) / "
                         "running_std < this value. Default -2.5 ~= "
                         "1-in-100 tail event under gaussian noise.")
    ap.add_argument("--ercv-zscore-warmup", type=int, default=4,
                    help="Episodes of history required before z-score "
                         "is used; fall back to fixed trigger until then.")
    ap.add_argument("--ercv-soft-rollback", action="store_true",
                    help="Instead of fully reverting solver weights on "
                         "rollback, interpolate: new = (1-s)*post + s*pre "
                         "with s = severity in [0,1]. Severity = "
                         "min(1, |retest_trend| / max(eps, |epi_gain|)).")
    ap.add_argument("--active-defense", action="store_true",
                    help="When the lie-taxonomy classifier fires at/above "
                         "the firing threshold, take a type-specific "
                         "corrective action. Type A → force external "
                         "context for next ep; Type B → raise solver "
                         "temperature; Type C → clear replay memory; "
                         "Type D → trigger soft rollback; Type E → "
                         "bump frontier_steps target.")
    ap.add_argument("--lie-history-window", type=int, default=3,
                    help="Rolling window of past signal values passed "
                         "to the lie classifier for predictive trend "
                         "detection.")
    # ---- metacognition -----
    ap.add_argument("--capability-map", action="store_true",
                    help="Enable capability-map self-model: cluster "
                         "solved problems via solver hidden-state "
                         "embeddings, track per-cluster skill, use "
                         "weakest cluster for proposer seeding.")
    ap.add_argument("--capability-map-k", type=int, default=8,
                    help="Number of concept clusters in the "
                         "capability map.")
    ap.add_argument("--causal-attribution", action="store_true",
                    help="When Type D fires, identify which recent "
                         "training pairs most likely caused the "
                         "regressions (embedding similarity).")
    # ---- v3.4 architecture flags (all default OFF — prior runs reproduce) ----
    ap.add_argument("--most-learnable-cluster", action="store_true",
                    help="v3.4. Replace weakest-cluster heuristic with "
                         "eligibility-gated learnability planner. Skips "
                         "clusters that are oversized or regressing — "
                         "fixes the cluster-2 hallucination amplification "
                         "observed in v3.3 long.")
    ap.add_argument("--per-cluster-detectors", action="store_true",
                    help="v3.4. Run Type-D detection per cluster (in "
                         "addition to global). Catches sub-threshold "
                         "cluster-local regressions (6 of 15 eps had "
                         "real regressions but no global firing in "
                         "v3.3 long).")
    ap.add_argument("--soft-replay-decay", action="store_true",
                    help="v3.4. Replace destructive clear_replay defense "
                         "with graceful decay (keep latest 50%%). "
                         "Prevents the -23pp ep9 skill cliff observed "
                         "in v3.3 long.")
    ap.add_argument("--replay-keep-fraction", type=float, default=0.5,
                    help="Fraction of replay memory to retain when "
                         "soft_replay_decay defense fires.")
    ap.add_argument("--proposer-validator", action="store_true",
                    help="v3.4. Run numeric-premise validation on "
                         "proposer output. Rejects problems with "
                         "demonstrably false arithmetic claims (kills "
                         "Type F at the source).")
    ap.add_argument("--adversarial-proposer", action="store_true",
                    help="v3.4. Reweight proposer SFT pairs by "
                         "factuality * difficulty * frontier-fit. "
                         "Demoted hallucinated proposals from full "
                         "weight to ~5%% (asymmetric co-evolution).")
    ap.add_argument("--spsi", action="store_true",
                    help="v3.4. Compute Self-Play Stability Index per "
                         "episode and log to spsi.jsonl.")
    ap.add_argument("--frontier-low", type=float, default=0.4,
                    help="v1-restored default.")
    ap.add_argument("--frontier-high", type=float, default=0.8,
                    help="v1-restored default.")
    ap.add_argument("--eval-n", type=int, default=50)
    ap.add_argument("--eval-mnt", type=int, default=256)
    ap.add_argument("--output-dir", default="outputs_self_improve_v3")
    ap.add_argument("--epiplexity-samples", type=int, default=48,
                    help="Buffer items used for epiplexity estimate.")
    args = ap.parse_args()
    set_seed(args.seed)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch
    from safetensors.torch import load_file
    from src.env.self_improvement_env import RecursiveSelfImprovementEnv
    from src.agent.self_improve_agent import SelfImproveAgent
    from src.train.model_ops import generate_with_phi, load_base_with_two_adapters
    from src.eval.benchmarks import run_gsm8k
    from src.eval.epiplexity import measure_epiplexity, snapshot_solver_weights
    from src.eval.lie_taxonomy import classify_episode as lie_classify_episode
    from src.eval.capability_map import CapabilityMap
    from src.eval.causal_attribution import attribute_regression
    # v3.4 architecture modules — three-loop controller. Flags below
    # gate which path is active so prior v3.x results reproduce.
    from src.control.curriculum_planner import most_learnable_cluster
    from src.control.detector_bank import classify as detector_classify
    from src.control.defense_dispatch import (
        apply_defense, apply_pending, PendingNextEpisode,
        DECAY_REPLAY_MEMORY, CLEAR_REPLAY_MEMORY,
    )
    from src.control.spsi import compute_spsi
    from src.eval.proposer_validator import validate_problem
    from src.agent.adversarial_proposer import (
        reweight_buffer as adversarial_reweight_buffer,
        summarize_breakdowns as adversarial_summarize,
    )

    out = Path(args.output_dir) / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    print("Loading base + adapters ...")
    bundle = load_base_with_two_adapters()

    if args.warmstart_adapter:
        ws = Path(args.warmstart_adapter)
        cand = ws / "solver" / "adapter_model.safetensors"
        if not cand.exists():
            cand = ws / "adapter_model.safetensors"
        if cand.exists():
            sd = load_file(str(cand))
            live = {n: p for n, p in bundle.model.named_parameters()
                    if "solver" in n and "lora" in n}
            matched = 0
            for k, v in sd.items():
                target = k.replace("lora_A.weight", "lora_A.solver.weight") \
                          .replace("lora_B.weight", "lora_B.solver.weight")
                tk = target if target in live else k
                if tk in live:
                    with torch.no_grad():
                        live[tk].copy_(v.to(live[tk].device, dtype=live[tk].dtype))
                    matched += 1
            print(f"  warmstart: {matched}/{len(live)} tensors")
    else:
        print("  COLD START — no warmstart")

    # v3: load external context pool (Liu §3.3)
    external_pool: list[str] = []
    if args.external_context_rate > 0:
        external_pool = _load_external_context_pool(
            args.external_context_path, args.external_context_limit,
        )
        print(f"  external context pool: {len(external_pool)} seeds loaded")

    # Build env + agent with v3 knobs
    env = RecursiveSelfImprovementEnv(
        rng_seed=args.seed,
        solver_k=args.solver_k,
        max_steps=args.steps_per_episode * 2,
        frontier_low=args.frontier_low,
        frontier_high=args.frontier_high,
        external_context_pool=external_pool or None,
        external_context_rate=args.external_context_rate,
        verifier_strength=args.verifier_strength,
        blind_spot_seed_rate=args.blind_spot_rate,
    )
    agent = SelfImproveAgent(
        bundle=bundle, solver_mnt=args.solver_mnt,
    )

    # Metacognition: install a capability map if requested
    if args.capability_map:
        cmap = CapabilityMap(k=args.capability_map_k)
        env.install_capability_map(cmap)
        print(f"  [metacognition] capability map installed (k={args.capability_map_k})")
        # v3.4: when --most-learnable-cluster is on, route the env's
        # weakest_cluster() lookup (used by env._pick_seed and ._info)
        # through the curriculum planner. The env consults
        # cmap.weakest_cluster() directly, so we override the method on
        # the instance — no env-level changes required, and v3.3
        # behavior reproduces when the flag is off.
        if args.most_learnable_cluster:
            _orig_weakest = cmap.weakest_cluster
            def _planner_weakest(min_size: int = 3) -> int | None:
                decision = most_learnable_cluster(
                    cmap.get_stats(), min_size=min_size,
                )
                if decision.target_cluster is not None:
                    return decision.target_cluster
                # Planner found nothing eligible — defer to legacy.
                return _orig_weakest(min_size=min_size)
            cmap.weakest_cluster = _planner_weakest
            print(f"  [metacognition] curriculum_planner overrides weakest_cluster")
    else:
        cmap = None

    # Self-explanation ledger (structured event log)
    explanation_ledger: list[dict] = []
    def ledger_write(event_type: str, because: dict, action: str | None = None) -> None:
        explanation_ledger.append({
            "episode": len(episode_logs) + 1,
            "event_type": event_type,
            "because": because,
            "action_taken": action,
        })

    def _solver_generate(prompt: str) -> str:
        bundle.model.set_adapter("solver")
        text, _ = generate_with_phi(
            bundle, prompt, max_new_tokens=args.eval_mnt, temperature=0.8,
        )
        return text

    def _eval(tag: str) -> dict:
        if args.eval_n <= 0:
            return {}
        t0 = time.time()
        res = run_gsm8k(_solver_generate, n_problems=args.eval_n, ks=(1, 2))
        dt = time.time() - t0
        payload = {
            "tag": tag, "benchmark": res.benchmark,
            "n_problems": res.n_problems, "passk": res.passk,
            "eval_seconds": dt,
        }
        (out / f"{tag}_eval.json").write_text(json.dumps(payload, indent=2))
        print(f"  [{tag}] pass@1={res.passk.get(1,0):.3f}  "
              f"pass@2={res.passk.get(2,0):.3f}  ({dt:.0f}s)")
        return payload

    pre = _eval("pre")

    episode_logs = []
    epiplexity_logs = []
    lie_logs = []
    # ERCV hysteresis: count consecutive divergent episodes
    ercv_divergence_streak = 0
    # ERCV long-horizon: running history of retest_trend across episodes
    retest_trend_history: list[float] = []
    # Lie-taxonomy Type A trend tracking
    prev_proposer_entropy: float | None = None
    last_sft_stats: dict = {}
    # Rolling signal history for PREDICTIVE lie-trend detection.
    # Maps signal-name → list of per-episode values. Classifier consumes
    # the last `args.lie_history_window` entries.
    lie_signal_history: dict[str, list[float]] = {
        "proposer_entropy": [],
        "chain_len_std_mean": [],
        "consequence_trend": [],
        "epi_gain": [],
    }
    # Active-defense state for the NEXT episode. v3.4 refactor: state
    # carrier object owned by src.control.defense_dispatch; the legacy
    # inline scalars are kept implicit inside this object.
    pending_defense = PendingNextEpisode()
    # SPSI per-episode log (v3.4). Empty when --spsi off.
    spsi_logs: list[dict] = []
    # v3.4 per-episode counters for validator/adversarial proposer.
    ep_validator_stats: dict = {"accepted": 0, "rejected": 0, "kinds": {}}
    ep_adversarial_summary: dict = {}

    def _filtered_proposer_buf(prop_buf: list) -> list:
        """Apply --proposer-validator post-hoc filter, dropping
        proposer pairs whose problem text fails numeric-premise
        validation. Updates ep_validator_stats in place."""
        if not args.proposer_validator or not prop_buf:
            ep_validator_stats["accepted"] += len(prop_buf)
            return list(prop_buf)
        kept = []
        for entry in prop_buf:
            problem_text = entry[1] if len(entry) >= 2 else ""
            v = validate_problem(str(problem_text))
            if v.ok:
                kept.append(entry)
                ep_validator_stats["accepted"] += 1
            else:
                ep_validator_stats["rejected"] += 1
                kind = v.kind or "unknown"
                ep_validator_stats["kinds"][kind] = ep_validator_stats["kinds"].get(kind, 0) + 1
        return kept

    def _reweighted_proposer_buf(prop_buf: list) -> list:
        """Apply --adversarial-proposer reweighting: pairs with
        non-factual or low-difficulty problems get demoted to ~5% of
        original weight. Stores per-call summary in
        ep_adversarial_summary (last call wins per episode)."""
        if not args.adversarial_proposer or not prop_buf:
            return list(prop_buf)
        new_buf, breakdowns = adversarial_reweight_buffer(
            prop_buf,
            target_steps=4,
            frontier_low=args.frontier_low,
            frontier_high=args.frontier_high,
        )
        nonlocal_summary = adversarial_summarize(breakdowns)
        ep_adversarial_summary.update(nonlocal_summary)
        return new_buf

    def _learn(buf: list, prop_buf: list, replay: list) -> dict:
        """Filter → reweight → agent.learn pipeline for v3.4."""
        prop_buf = _filtered_proposer_buf(prop_buf)
        prop_buf = _reweighted_proposer_buf(prop_buf)
        return agent.learn(
            buf, proposer_buffer=prop_buf, replay=replay,
            replay_mix_ratio=0.25, epochs=1,
            epiplexity_weight=args.epiplexity_weight,
        )
    for ep in range(args.episodes):
        # v3.4: reset per-episode validator + adversarial-proposer counters
        ep_validator_stats = {"accepted": 0, "rejected": 0, "kinds": {}}
        ep_adversarial_summary = {}
        # Apply any active-defense adjustments set by previous episode
        if args.active_defense:
            changes = apply_pending(pending_defense, env, agent)
            for ch in changes:
                print(f"    [defense] {ch}")
        # v3: capacity growth (Liu §3.2)
        if args.capacity_growth > 0:
            mult = (1.0 + args.capacity_growth) ** ep
            env.solver_k = min(
                args.solver_k_max,
                max(args.solver_k, int(round(args.solver_k * mult))),
            )
            agent.solver_mnt = min(
                args.solver_mnt_max,
                max(args.solver_mnt, int(round(args.solver_mnt * mult))),
            )
        print(f"\n=== Episode {ep+1}/{args.episodes} "
              f"(solver_k={env.solver_k}, solver_mnt={agent.solver_mnt}) ===")

        # Snapshot solver weights BEFORE this episode's SFT (for epiplexity)
        ref_weights = snapshot_solver_weights(bundle)

        t_ep = time.time()
        obs, info = env.reset(seed=args.seed + ep * 101)
        step_rewards = []
        # Track training pairs produced this episode (for causal attribution).
        episode_training_pairs: list[dict] = []
        # Track regressed retests this episode (negative consequence_delta).
        episode_regressed: list[dict] = []
        for step in range(args.steps_per_episode * 2):
            # Reset per-iteration embedding — avoids stale embedding
            # from a previous iteration leaking into this step's capture.
            emb_this_step = None
            action = agent.act(obs)
            # Metacognition: embed the problem BEFORE env.step processes
            # the solve, so the env can tag the solve with a cluster id.
            if args.capability_map and obs.get("mode") == "solve" and obs.get("problem"):
                try:
                    emb_this_step = agent.embed_problem(obs["problem"])
                    env.annotate_last_solve(emb_this_step)
                except Exception as _e:  # noqa: BLE001
                    emb_this_step = None
            obs_next, reward, term, trunc, step_info = env.step(action)
            step_rewards.append(reward)
            # After a solve step, capture the observation into the
            # per-episode training pair list (for causal attribution
            # on Type D). Only frontier-zone pairs actually enter SFT.
            if (args.capability_map and step_info.get("zone") == "frontier"
                    and env.state.training_buffer):
                last_pair = env.state.training_buffer[-1]
                problem_text = last_pair[0]
                cluster_id = step_info.get("assigned_cluster")
                episode_training_pairs.append({
                    "problem": problem_text,
                    "embedding": emb_this_step,
                    "cluster_id": cluster_id,
                    "reward": last_pair[2],
                })
            # If this was a retest with negative delta, record for
            # causal attribution.
            if (args.capability_map and step_info.get("is_retest")
                    and step_info.get("consequence_delta") is not None
                    and step_info["consequence_delta"] < -0.02):
                episode_regressed.append({
                    "problem": obs.get("problem", ""),
                    "embedding": emb_this_step,
                    "cluster_id": step_info.get("assigned_cluster"),
                    "delta": step_info["consequence_delta"],
                })
            if step_info["buffer_size"] >= args.sft_trigger:
                buf = env.get_training_buffer()
                prop_buf = env.get_proposer_buffer()
                replay = env.get_replay_memory(max_items=64)
                sft_stats = _learn(buf, prop_buf, replay)
                last_sft_stats = sft_stats
                env.clear_training_buffer()
                env.clear_proposer_buffer()
                print(
                    f"    SFT: solver={sft_stats['n_solver']} "
                    f"(replay_mixed={sft_stats['n_replay_mixed']}) "
                    f"proposer={sft_stats['n_proposer']}  "
                    f"loss_s={sft_stats['loss_solver']:.3f} "
                    f"loss_p={sft_stats['loss_proposer']:.3f}"
                )
            if trunc or term:
                break
            obs = obs_next

        # End-of-episode SFT drain, same as v1/v2
        remaining = env.get_training_buffer()
        remaining_prop = env.get_proposer_buffer()
        final_sft = {
            "n_solver": 0, "n_proposer": 0, "loss_solver": 0.0,
            "loss_proposer": 0.0, "n_replay_mixed": 0,
        }
        if remaining or remaining_prop:
            replay = env.get_replay_memory(max_items=64)
            final_sft = _learn(remaining, remaining_prop, replay)
            last_sft_stats = final_sft
            env.clear_training_buffer()
            env.clear_proposer_buffer()
            print(
                f"    final SFT: solver={final_sft['n_solver']} "
                f"proposer={final_sft['n_proposer']} "
                f"loss_s={final_sft['loss_solver']:.3f} "
                f"loss_p={final_sft['loss_proposer']:.3f}"
            )

        # v3: epiplexity diagnostic on the episode's replay buffer
        # (captures all frontier pairs generated this episode)
        epi_gain = 0.0
        try:
            epi = measure_epiplexity(
                bundle=bundle,
                buffer=list(env.state.replay_memory)[-args.epiplexity_samples:],
                ref_state_dict=ref_weights,
                episode=ep + 1,
                max_samples=args.epiplexity_samples,
            )
            epi_gain = epi.learnable_info_per_token
            epiplexity_logs.append(epi.to_dict())
            print(f"    epiplexity: Δnll_per_tok={epi_gain:+.4f} "
                  f"(nats, n_samples={epi.n_samples})")
        except Exception as e:  # noqa: BLE001
            print(f"    epiplexity: skipped ({e})")

        # v3.1 NOVEL: Epiplexity-Retest Cross-Validation (ERCV).
        # If epiplexity says "we learned" (epi_gain > 0) but retests say
        # "we forgot" (recent retest trend < -trigger), the update is
        # overfitting to the closed loop — revert to pre-episode weights.
        # Hysteresis: require min_retests retests AND sustained
        # divergence across ercv_hysteresis consecutive episodes to
        # avoid firing on small-sample noise.
        ercv_rolled_back = False
        ercv_rollback_severity = 0.0
        ercv_zscore_val: float | None = None
        info_now = env._info()
        retest_trend = info_now.get("consequence_trend", 0.0)
        n_retests = info_now.get("n_retests", 0)

        # v3.1 long-horizon extension A: z-score ERCV (self-calibrating)
        # Falls back to fixed trigger until `ercv_zscore_warmup` episodes
        # of retest_trend history are accumulated. Past that, we judge
        # divergence by statistical tail (z < threshold), not by a
        # hand-tuned constant — this scales to arbitrary run lengths.
        if args.ercv_zscore and len(retest_trend_history) >= args.ercv_zscore_warmup:
            import numpy as _np
            hist = _np.asarray(retest_trend_history, dtype=_np.float64)
            mu = float(hist.mean())
            sigma = float(hist.std(ddof=1)) if len(hist) >= 2 else 0.0
            sigma_safe = max(sigma, 1e-4)
            ercv_zscore_val = (retest_trend - mu) / sigma_safe
            is_divergent = (
                epi_gain > 0.0
                and ercv_zscore_val < args.ercv_zscore_threshold
                and n_retests >= args.ercv_min_retests
            )
        else:
            is_divergent = (
                epi_gain > 0.0
                and retest_trend < -args.ercv_trigger
                and n_retests >= args.ercv_min_retests
            )

        if is_divergent:
            ercv_divergence_streak += 1
        else:
            ercv_divergence_streak = 0

        if (args.ercv_rollback
                and ref_weights is not None
                and ercv_divergence_streak >= args.ercv_hysteresis):
            # v3.1 long-horizon extension C: soft rollback (weight interpolation).
            # severity scales the revert: hard rollback = 1.0 (full pre-weights),
            # mild disagreement yields a softer pull-back that preserves most
            # of the episode's useful gradient while correcting drift.
            if args.ercv_soft_rollback:
                denom = max(abs(epi_gain), 1e-6)
                ercv_rollback_severity = max(
                    0.0, min(1.0, abs(retest_trend) / denom)
                )
            else:
                ercv_rollback_severity = 1.0
            trigger_desc = (
                f"zscore={ercv_zscore_val:+.2f} < "
                f"{args.ercv_zscore_threshold:+.2f}"
                if ercv_zscore_val is not None
                else f"trend={retest_trend:+.3f} < {-args.ercv_trigger:+.3f}"
            )
            print(f"    ERCV ROLLBACK: epi_gain={epi_gain:+.3f} "
                  f"{trigger_desc} (n_retests={n_retests}, "
                  f"streak={ercv_divergence_streak}, "
                  f"severity={ercv_rollback_severity:.2f}) — "
                  f"{'interpolating' if args.ercv_soft_rollback else 'reverting'}")
            with torch.no_grad():
                live = {n: p for n, p in bundle.model.named_parameters()
                        if "solver" in n and "lora" in n}
                s = ercv_rollback_severity
                for n, p in live.items():
                    if n in ref_weights:
                        ref_p = ref_weights[n].to(p.device, dtype=p.dtype)
                        # (1-s)*post + s*pre — s=1 is full revert (legacy)
                        p.copy_((1.0 - s) * p + s * ref_p)
            ercv_rolled_back = True
            ercv_divergence_streak = 0

        # Append to history for next episode's z-score calculation
        # (always appended, even on rollback — the observed trend IS
        # part of the process noise we're calibrating against).
        retest_trend_history.append(retest_trend)
        # Release snapshot
        ref_weights = None

        # Append to signal history BEFORE classification (so history
        # passed to classifier includes this episode's value).
        # Predictive detectors look at the most recent window.
        lie_signal_history["proposer_entropy"].append(info_now.get("proposer_entropy", 0.0))
        lie_signal_history["chain_len_std_mean"].append(info_now.get("chain_len_std_mean", 0.0))
        lie_signal_history["consequence_trend"].append(info_now.get("consequence_trend", 0.0))
        lie_signal_history["epi_gain"].append(epi_gain)

        # Lie-taxonomy classification (novel — first typed detection of
        # AI self-deception in self-play, with predictive trend signals).
        # v3.4: when --per-cluster-detectors is on, runs detector_bank
        # which adds cluster-local Type D detection on top of the global
        # classifier. Unified result has compatible interface.
        window = args.lie_history_window
        history_slice = {k: v[-window:] for k, v in lie_signal_history.items()}
        per_cluster_firings: list = []
        unified_cluster: int | None = None
        if args.per_cluster_detectors:
            bank = detector_classify(
                env_info=info_now,
                sft_stats=last_sft_stats,
                epi_gain=epi_gain,
                capability_map_stats=info_now.get("capability_map_stats"),
                prev_proposer_entropy=prev_proposer_entropy,
                history=history_slice,
            )
            lie_result = bank.global_classification
            per_cluster_firings = [f.to_dict() for f in bank.per_cluster_firings]
            unified_cluster = bank.unified_cluster
            # Surface unified firing — propagate cluster-local firings
            # into the lie_result interface used downstream.
            unified_firing = bank.any_firing
            unified_type = bank.unified_type
            unified_score = bank.unified_score
            unified_explanation = bank.unified_explanation
            unified_defense = bank.defense_action
        else:
            lie_result = lie_classify_episode(
                env_info=info_now, sft_stats=last_sft_stats,
                epi_gain=epi_gain, prev_proposer_entropy=prev_proposer_entropy,
                history=history_slice,
            )
            unified_firing = lie_result.any_firing
            unified_type = lie_result.strongest if unified_firing else None
            unified_score = lie_result.strongest_score
            unified_explanation = lie_result.explanation
            unified_defense = lie_result.defense_action
        prev_proposer_entropy = info_now.get("proposer_entropy", 0.0)
        lie_record = {
            "episode": ep + 1,
            **lie_result.to_dict(),
            "per_cluster_firings": per_cluster_firings,
            "unified_cluster": unified_cluster,
        }
        lie_logs.append(lie_record)
        if unified_firing:
            cluster_tag = f" cluster={unified_cluster}" if unified_cluster is not None else ""
            print(f"    LIE-TAXONOMY [{lie_result.phase}]:{cluster_tag} {unified_explanation}")
        else:
            print(f"    LIE-TAXONOMY [{lie_result.phase}]: no firing  "
                  f"(A={lie_result.types['A']:.2f} B={lie_result.types['B']:.2f} "
                  f"C={lie_result.types['C']:.2f} D={lie_result.types['D']:.2f} "
                  f"E={lie_result.types['E']:.2f})")
            if per_cluster_firings:
                pcf_str = ", ".join(f"C{f['cluster_id']}:{f['score']:.2f}" for f in per_cluster_firings)
                print(f"      per-cluster D: {pcf_str}")

        # Metacognition: causal attribution when Type D fires.
        # v3.4: triggers also on per-cluster Type D firings (broader gate;
        # the original gate fired only once in 15 episodes of v3.3 long
        # despite multiple cluster-local regressions).
        causal_hypothesis = None
        d_fired = (
            (lie_result.strongest == "D" and lie_result.any_firing)
            or (unified_type == "D" and unified_firing)
        )
        if (args.causal_attribution and args.capability_map and d_fired):
            # Filter: only include items where we successfully captured an embedding
            reg_clean = [r for r in episode_regressed if r.get("embedding") is not None]
            tp_clean = [t for t in episode_training_pairs if t.get("embedding") is not None]
            if reg_clean and tp_clean:
                causal_hypothesis = attribute_regression(reg_clean, tp_clean, top_k=3)
                print(f"    CAUSAL ATTRIBUTION: {causal_hypothesis['summary']}")
                ledger_write(
                    "causal_attribution",
                    because={
                        "trigger": "Type D fired",
                        "n_regressed": causal_hypothesis["n_regressed"],
                        "implicated_clusters": causal_hypothesis["implicated_clusters"],
                        "worst_regression_cluster": (
                            causal_hypothesis["worst_regression"]["cluster_id"]
                            if causal_hypothesis["worst_regression"] else None
                        ),
                    },
                    action="hypothesis logged",
                )

        # Metacognition: ledger write for lie firing
        # v3.4: writes for unified firing (global OR cluster-local).
        if unified_firing:
            ledger_write(
                "lie_fired",
                because={
                    "type": unified_type,
                    "score": unified_score,
                    "phase": lie_result.phase,
                    "explanation": unified_explanation,
                    "cluster": unified_cluster,
                },
                action=unified_defense if args.active_defense else None,
            )

        # Log capability map summary each episode.
        # v3.4: when --most-learnable-cluster is on, the planner replaces
        # the "lowest-reward" heuristic with eligibility-gated selection
        # that skips oversized or regressing clusters (kills cluster-2
        # amplification observed in v3.3 long).
        if args.capability_map and env.capability_map is not None:
            map_stats = env.capability_map.get_stats()
            # The planner override (installed at startup) makes
            # weakest_cluster() route through the planner when
            # --most-learnable-cluster is on. We surface the reason
            # for the ledger separately.
            target_cluster = env.capability_map.weakest_cluster(min_size=3)
            if args.most_learnable_cluster:
                decision = most_learnable_cluster(map_stats)
                planner_reason = decision.reason
                print(f"    CAPABILITY MAP: {env.capability_map.summary_line()}"
                      f"  target=C{target_cluster}  ({planner_reason})")
            else:
                planner_reason = "legacy weakest_cluster"
                print(f"    CAPABILITY MAP: {env.capability_map.summary_line()}"
                      f"  weakest={target_cluster}")
            ledger_write(
                "capability_map_snapshot",
                because={
                    "weakest_cluster": target_cluster,
                    "planner_reason": planner_reason,
                    "summary": env.capability_map.summary_line(),
                    "stats": map_stats,
                },
            )

        # Active-defense: dispatch typed action via defense_dispatch.
        # v3.4: clear_replay_memory is upgraded to a graceful decay
        # (keep latest fraction) when --soft-replay-decay is set —
        # prevents the ep9 skill cliff observed in v3.3 long.
        if args.active_defense and unified_firing and unified_defense:
            record = apply_defense(
                action=unified_defense,
                env=env, agent=agent,
                pending=pending_defense,
                use_decay_for_c=args.soft_replay_decay,
                decay_keep_fraction=args.replay_keep_fraction,
            )
            print(f"    [defense->{unified_type}] {record.detail}")
            ledger_write(
                "defense_action",
                because={"action": record.action, "applied": record.applied},
                action=record.detail,
            )

        final_info = env._info()
        dt = time.time() - t_ep
        ep_log = {
            "episode": ep + 1,
            "steps": len(step_rewards),
            "mean_reward": float(np.mean(step_rewards)) if step_rewards else 0.0,
            "n_propose_accepted": final_info["n_propose_accepted"],
            "n_propose_rejected": final_info["n_propose_rejected"],
            "n_solve_frontier": final_info["n_solve_frontier"],
            "n_solve_easy": final_info["n_solve_easy"],
            "n_solve_hard": final_info["n_solve_hard"],
            "skill_level": final_info["skill_level"],
            "frontier_steps": final_info["frontier_steps"],
            "replay_memory_size": final_info.get("replay_memory_size", 0),
            "proposer_buffer_size": final_info.get("proposer_buffer_size", 0),
            "final_loss_solver": final_sft.get("loss_solver", 0.0),
            "final_loss_proposer": final_sft.get("loss_proposer", 0.0),
            "solver_k_used": env.solver_k,
            "solver_mnt_used": agent.solver_mnt,
            "n_blind_spots": final_info.get("n_blind_spots", 0),
            "consequence_trend": final_info.get("consequence_trend", 0.0),
            "epiplexity_gain": epi_gain,
            "ercv_rolled_back": ercv_rolled_back,
            "ercv_rollback_severity": ercv_rollback_severity,
            "ercv_zscore": ercv_zscore_val,
            # Lie-taxonomy (novel): scores + strongest (global classifier)
            "lie_types": lie_result.types,
            "lie_strongest": lie_result.strongest,
            "lie_strongest_score": lie_result.strongest_score,
            "lie_any_firing": lie_result.any_firing,
            # v3.4: unified firing surface (global OR per-cluster).
            # When --per-cluster-detectors is off, these mirror lie_*.
            # When on, unified_* may fire on cluster-local D even when
            # the global classifier did not.
            "unified_type": unified_type,
            "unified_score": unified_score,
            "unified_firing": unified_firing,
            "unified_cluster": unified_cluster,
            "per_cluster_firings": per_cluster_firings,
            "easy_hard_ratio": info_now.get("easy_hard_ratio", 0.0),
            "proposer_entropy": info_now.get("proposer_entropy", 0.0),
            "proposer_unique_ratio": info_now.get("proposer_unique_ratio", 0.0),
            "chain_len_std_mean": info_now.get("chain_len_std_mean", 0.0),
            "process_over_outcome": info_now.get("process_over_outcome", 0.0),
            "pre_nll_new": last_sft_stats.get("pre_nll_new", 0.0),
            "pre_nll_replay": last_sft_stats.get("pre_nll_replay", 0.0),
            # Metacognition: capability map snapshot
            "capability_map_stats": info_now.get("capability_map_stats", []),
            "weakest_cluster": info_now.get("weakest_cluster"),
            "causal_hypothesis": (
                {"summary": causal_hypothesis["summary"],
                 "implicated_clusters": causal_hypothesis["implicated_clusters"]}
                if causal_hypothesis else None
            ),
            # v3.4: validator + adversarial-proposer summaries
            "validator_stats": dict(ep_validator_stats),
            "adversarial_proposer": dict(ep_adversarial_summary),
            "seconds": dt,
        }
        episode_logs.append(ep_log)
        # v3.4: SPSI single-scalar health index (composite of epi/frontier
        # vs forget/collapse/wasted-share). The wasted_share term is the
        # load-bearing predictor of cluster-2-style amplification.
        if args.spsi:
            spsi_breakdown = compute_spsi(ep_log)
            spsi_logs.append({"episode": ep + 1, **spsi_breakdown.to_dict(),
                              "label": spsi_breakdown.label()})
            print(f"    SPSI: {spsi_breakdown.spsi:+.3f} ({spsi_breakdown.label()}) "
                  f"waste={spsi_breakdown.wasted_share:.2f} "
                  f"forget={spsi_breakdown.retest_decay:.2f}")
        (out / "episodes.jsonl").write_text(
            "\n".join(json.dumps(e) for e in episode_logs)
        )
        (out / "epiplexity.jsonl").write_text(
            "\n".join(json.dumps(e) for e in epiplexity_logs)
        )
        (out / "lie_taxonomy.jsonl").write_text(
            "\n".join(json.dumps(e) for e in lie_logs)
        )
        (out / "self_explanation_ledger.jsonl").write_text(
            "\n".join(json.dumps(e, default=str) for e in explanation_ledger)
        )
        if args.spsi:
            (out / "spsi.jsonl").write_text(
                "\n".join(json.dumps(e) for e in spsi_logs)
            )
        print(f"  Ep {ep+1}: reward_mean={ep_log['mean_reward']:.3f}  "
              f"frontier={ep_log['n_solve_frontier']}  "
              f"easy={ep_log['n_solve_easy']}  "
              f"skill={ep_log['skill_level']:.3f}")

    post = _eval("post")
    try:
        bundle.model.save_pretrained(str(out / "adapters"))
        print(f"  saved adapters to {out / 'adapters'}")
    except Exception as e:  # noqa: BLE001
        print(f"  save warning: {e}")

    if pre and post:
        d1 = post["passk"].get(1, 0) - pre["passk"].get(1, 0)
        d2 = post["passk"].get(2, 0) - pre["passk"].get(2, 0)
        print(f"\nSelf-improvement delta: pass@1 {d1:+.3f}  pass@2 {d2:+.3f}")

    # Summarize learnable-info curve
    if epiplexity_logs:
        deltas = [e["learnable_info_per_token"] for e in epiplexity_logs]
        cum = float(np.sum(deltas))
        print(f"Cumulative learnable-info Δ across {len(deltas)} episodes: "
              f"{cum:+.4f} nats/token  (per-ep: "
              f"{', '.join(f'{d:+.3f}' for d in deltas)})")


if __name__ == "__main__":
    main()
