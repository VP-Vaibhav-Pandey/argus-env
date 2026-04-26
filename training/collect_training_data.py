"""Collect training run artifacts into logs/snapshots/<name>/.

After a training run finishes, this script consolidates everything
worth keeping (logs, jsonl streams, evals, plots, config) into one
folder under logs/snapshots/. That folder becomes the durable
archive for the run — README.md and submission docs reference it.

Usage:
    python scripts/collect_training_data.py --run outputs_v34_full --name v34_full

What it copies:
  outputs_<run>/seed*/episodes.jsonl
  outputs_<run>/seed*/lie_taxonomy.jsonl
  outputs_<run>/seed*/spsi.jsonl                       (if --spsi was on)
  outputs_<run>/seed*/self_explanation_ledger.jsonl
  outputs_<run>/seed*/epiplexity.jsonl
  outputs_<run>/seed*/pre_eval.json, post_eval.json
  outputs_<run>/seed*/config.json
  outputs_<run>/seed*/plot_*.png                       (if plotting was run)
  logs/training/<run>_*.log                            (latest matching)
  logs/screenshots/<name>/                             (whole subdir if exists)

Plus a SUMMARY.md the script generates: pre/post pass@k, run length,
SPSI trajectory min/max, lie firings count, per-cluster firings count,
defense actions count, ERCV rollback count.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _summarize(seed_dir: Path) -> dict:
    """Roll up per-run summary statistics into one dict."""
    eps = _read_jsonl(seed_dir / "episodes.jsonl")
    spsi = _read_jsonl(seed_dir / "spsi.jsonl")
    ledger = _read_jsonl(seed_dir / "self_explanation_ledger.jsonl")
    pre = _read_json(seed_dir / "pre_eval.json")
    post = _read_json(seed_dir / "post_eval.json")
    cfg = _read_json(seed_dir / "config.json")

    n_lie_fires_global = sum(1 for e in eps if e.get("lie_any_firing"))
    n_lie_fires_unified = sum(1 for e in eps if e.get("unified_firing"))
    pcf_total = sum(len(e.get("per_cluster_firings", [])) for e in eps)
    n_ercv = sum(1 for e in eps if e.get("ercv_rolled_back"))
    n_causal = sum(1 for e in ledger if e.get("event_type") == "causal_attribution")
    n_defense = sum(1 for e in ledger if e.get("event_type") == "defense_action")

    spsi_vals = [b.get("spsi") for b in spsi if b.get("spsi") is not None]
    spsi_min = min(spsi_vals) if spsi_vals else None
    spsi_max = max(spsi_vals) if spsi_vals else None
    spsi_final = spsi_vals[-1] if spsi_vals else None

    pre1 = pre.get("passk", {}).get("1")
    pre2 = pre.get("passk", {}).get("2")
    post1 = post.get("passk", {}).get("1")
    post2 = post.get("passk", {}).get("2")
    delta1 = (post1 - pre1) if (pre1 is not None and post1 is not None) else None
    delta2 = (post2 - pre2) if (pre2 is not None and post2 is not None) else None

    total_seconds = sum(float(e.get("seconds", 0.0)) for e in eps)

    return {
        "n_episodes": len(eps),
        "total_hours": round(total_seconds / 3600.0, 3),
        "pass_at_1_pre": pre1,
        "pass_at_1_post": post1,
        "pass_at_1_delta": delta1,
        "pass_at_2_pre": pre2,
        "pass_at_2_post": post2,
        "pass_at_2_delta": delta2,
        "lie_fires_global": n_lie_fires_global,
        "lie_fires_unified": n_lie_fires_unified,
        "per_cluster_firings_total": pcf_total,
        "ercv_rollbacks": n_ercv,
        "causal_attributions": n_causal,
        "defense_actions": n_defense,
        "spsi_min": spsi_min,
        "spsi_max": spsi_max,
        "spsi_final": spsi_final,
        "config": cfg,
    }


def _write_summary_md(snapshot_dir: Path, name: str, summary: dict, source: Path) -> None:
    cfg = summary.get("config", {})
    flags = " ".join(
        f"--{k.replace('_', '-')}"
        for k in ["most_learnable_cluster", "per_cluster_detectors",
                  "soft_replay_decay", "proposer_validator",
                  "adversarial_proposer", "spsi", "active_defense",
                  "capability_map", "causal_attribution",
                  "ercv_rollback", "ercv_soft_rollback", "ercv_zscore",
                  "epiplexity_weight"]
        if cfg.get(k)
    )
    lines = [
        f"# {name} — run summary",
        "",
        f"Source dir: `{source}`",
        f"Snapshot created: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Headline",
        "",
        f"- **pass@1**: {summary['pass_at_1_pre']} → {summary['pass_at_1_post']} "
        f"(Δ = {summary['pass_at_1_delta']:+.3f})"
        if summary['pass_at_1_delta'] is not None
        else f"- pass@1: pre={summary['pass_at_1_pre']} post={summary['pass_at_1_post']}",
        f"- **pass@2**: {summary['pass_at_2_pre']} → {summary['pass_at_2_post']} "
        f"(Δ = {summary['pass_at_2_delta']:+.3f})"
        if summary['pass_at_2_delta'] is not None
        else f"- pass@2: pre={summary['pass_at_2_pre']} post={summary['pass_at_2_post']}",
        f"- **Episodes**: {summary['n_episodes']}",
        f"- **Total runtime**: {summary['total_hours']:.2f} h",
        "",
        "## Mechanism activity",
        "",
        f"- Global lie firings: **{summary['lie_fires_global']}**",
        f"- Unified firings (incl. per-cluster): **{summary['lie_fires_unified']}**",
        f"- Per-cluster D firings (total events): **{summary['per_cluster_firings_total']}**",
        f"- ERCV rollbacks: **{summary['ercv_rollbacks']}**",
        f"- Causal attributions: **{summary['causal_attributions']}**",
        f"- Defense actions: **{summary['defense_actions']}**",
        "",
        "## SPSI trajectory",
        "",
        f"- min: {summary['spsi_min']}",
        f"- max: {summary['spsi_max']}",
        f"- final: {summary['spsi_final']}",
        "",
        "## Active flags",
        "",
        f"```\n{flags}\n```",
        "",
        "## Files in this snapshot",
        "",
    ]
    for p in sorted(snapshot_dir.iterdir()):
        if p.is_file():
            sz = p.stat().st_size
            lines.append(f"- `{p.name}` ({sz} bytes)")
        elif p.is_dir():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            lines.append(f"- `{p.name}/` ({n} files)")

    (snapshot_dir / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )


def collect(run_dir: Path, name: str, repo_root: Path) -> Path:
    """Copy all run artifacts into logs/snapshots/<name>/."""
    snapshot_dir = repo_root / "logs" / "snapshots" / name
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Copy each seed's artifacts
    for seed_dir in sorted(run_dir.glob("seed*")):
        if not seed_dir.is_dir():
            continue
        target = snapshot_dir / seed_dir.name
        target.mkdir(exist_ok=True)
        # Pick up jsonl streams + json evals + config + plots
        for pattern in [
            "*.jsonl", "*.json", "plot_*.png",
        ]:
            for f in seed_dir.glob(pattern):
                shutil.copy2(f, target / f.name)
        # The adapters dir is large — include as a pointer file, not
        # the weights themselves (weights stay in outputs_*).
        adapters_dir = seed_dir / "adapters"
        if adapters_dir.is_dir():
            (target / "ADAPTERS_LOCATION.txt").write_text(
                f"Adapter weights live at:\n  {adapters_dir.resolve()}\n",
                encoding="utf-8",
            )

    # Copy the most recent training log matching this name
    log_dir = repo_root / "logs" / "training"
    if log_dir.exists():
        matches = sorted(log_dir.glob(f"{name}_*.log"))
        if matches:
            shutil.copy2(matches[-1], snapshot_dir / matches[-1].name)

    # Copy the screenshots subdir if it exists
    shots_dir = repo_root / "logs" / "screenshots" / name
    if shots_dir.is_dir():
        target_shots = snapshot_dir / "screenshots"
        if target_shots.exists():
            shutil.rmtree(target_shots)
        shutil.copytree(shots_dir, target_shots)

    # Build summary across all seeds (use the first if just one)
    seed_dirs = sorted(run_dir.glob("seed*"))
    if seed_dirs:
        summary = _summarize(seed_dirs[0])
        _write_summary_md(snapshot_dir, name, summary, run_dir)
        # Also write summary.json for programmatic consumption
        (snapshot_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8",
        )

    return snapshot_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="Path to outputs_<run>/ directory")
    ap.add_argument("--name", required=True,
                    help="Name for the snapshot folder under logs/snapshots/")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    if not run_dir.exists():
        print(f"ERROR: {run_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    snapshot_dir = collect(run_dir, args.name, repo_root)
    print(f"Snapshot written: {snapshot_dir}")
    if (snapshot_dir / "SUMMARY.md").exists():
        print(f"  -> {snapshot_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
