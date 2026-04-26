"""ARGUS v3.4 result printer — screenshot-ready terminal output.

Reads the seed-0 snapshot in logs/snapshots/v34_full/seed0/ and prints
a clean summary suitable for a screenshot in the writeup.

Usage:
    .venv/Scripts/python.exe scripts/show_v34_result.py
"""
import json
import sys
from pathlib import Path

# Force UTF-8 stdout so box-drawing characters render on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "logs" / "snapshots" / "v34_full" / "seed0"

pre = json.loads((SNAP / "pre_eval.json").read_text())
post = json.loads((SNAP / "post_eval.json").read_text())
summary = json.loads((SNAP / ".." / "summary.json").read_text())
eps = [json.loads(l) for l in (SNAP / "episodes.jsonl").read_text().splitlines()]
lies = [json.loads(l) for l in (SNAP / "lie_taxonomy.jsonl").read_text().splitlines()]
spsi = [json.loads(l) for l in (SNAP / "spsi.jsonl").read_text().splitlines()]


def hr(c="─", n=78):
    return c * n


def box_line(text, width=78):
    pad = width - len(text) - 2
    return f"│ {text}{' ' * pad}│"


print()
print("┌" + hr("─", 76) + "┐")
print(box_line("ARGUS  ·  Adversarial Refusal-Gated Unified Self-improvement"))
print(box_line("v3.4 full stack  ·  seed 0  ·  GSM8K eval  ·  n=100"))
print("└" + hr("─", 76) + "┘")
print()

p1d = post["passk"]["1"] - pre["passk"]["1"]
p2d = post["passk"]["2"] - pre["passk"]["2"]

print("  HEADLINE")
print("  " + hr("─", 60))
print(f"    pass@1   {pre['passk']['1']:.3f}  →  {post['passk']['1']:.3f}     Δ = {p1d:+.3f}")
print(f"    pass@2   {pre['passk']['2']:.3f}  →  {post['passk']['2']:.3f}     Δ = {p2d:+.3f}")
print(f"    skill    {eps[0]['skill_level']:.3f}  →  {eps[-1]['skill_level']:.3f}     Δ = {eps[-1]['skill_level']-eps[0]['skill_level']:+.3f}")
print(f"    runtime  {summary['total_hours']:.2f} h    episodes {summary['n_episodes']}")
print()

print("  DEFENSIVE INTERVENTIONS  (architecture caught these — would have")
print("  committed bad steps in a naive trainer)")
print("  " + hr("─", 60))
print(f"    Lie firings (global Type D)         {summary['lie_fires_global']}")
print(f"    Lie firings (per-cluster)            {summary['per_cluster_firings_total']}")
print(f"    ERCV soft-rollbacks                  {summary['ercv_rollbacks']}")
print(f"    Causal attributions logged           {summary['causal_attributions']}")
print(f"    Defense actions dispatched           {summary['defense_actions']}")
print()

print("  PER-EPISODE TIMELINE")
print("  " + hr("─", 74))
print("   ep │ reward │ skill │ frontier │  lie  │ ERCV  │ SPSI    │ phase       ")
print("   ───┼────────┼───────┼──────────┼───────┼───────┼─────────┼─────────────")
for e, l, s in zip(eps, lies, spsi):
    fire = "  ─  "
    if e["unified_firing"]:
        fire = " D!  "
    elif e["per_cluster_firings"]:
        fire = " d·  "
    rb = "  ─  "
    z = e.get("ercv_zscore")
    if e["ercv_rolled_back"]:
        rb = f"{z:+.2f}"
    elif z is not None:
        rb = f"{z:+.2f}"
    label = s["label"][:11]
    print(f"   {e['episode']:>2} │  {e['mean_reward']:.2f}  │ {e['skill_level']:.3f} │    {e['n_solve_frontier']:>2}    │ {fire} │ {rb} │ {s['spsi']:+.3f}  │ {label}")
print()

print("  CAPABILITY MAP  (final state)")
print("  " + hr("─", 60))
final_stats = eps[-1]["capability_map_stats"]
total = sum(c["n_problems"] for c in final_stats)
for c in final_stats:
    share = c["n_problems"] / total * 100
    bar = "█" * int(share / 2)
    print(f"    C{c['cluster_id']}  n={c['n_problems']:>3}  r={c['mean_reward']:.2f}  share={share:>5.1f}%  {bar}")
print(f"    total: {total} problems, 10 clusters self-discovered")
print()

print("  KEY FINDINGS  (from deep_analysis_v34.md)")
print("  " + hr("─", 70))
print("    1.  Type G discovered live  — cluster 5 grew 2→110 with reward")
print("        stuck at 0.56-0.59; planner kept picking it as 'weakest'")
print("        ► new failure mode: plateau capture (6th in taxonomy)")
print()
print("    2.  ERCV z-score caught 2 stealth failures")
print("        ep9:  epi=+0.030  z=-2.82  retest=-0.068  ► refused")
print("        ep13: epi=+0.015  z=-3.05  retest=-0.104  ► refused")
print("        ► positive epi gain alone would have committed both")
print()
print("    3.  Causal attribution: 5/7 perfect cluster pick")
print("        eps 7,8,9,11,13 nailed worst-regression cluster from")
print("        embedding similarity over recent training pairs")
print()
print("    4.  Validator + adversarial-proposer converged on danger eps")
print("        weight_kept fraction <0.80 exactly at eps 9, 11, 13, 15")
print("        — independent defenses agreeing without explicit coupling")
print()
print("    5.  Consolidation signature: pass@1 +7pp, pass@2 +1pp")
print("        ARGUS optimises reliability, not capability expansion")
print("        — by design (lie taxonomy penalises Type D forgetting)")
print()
print(hr("─", 78))
print("  full forensic writeup:  deep_analysis_v34.md  (12 patterns)")
print("  raw artifacts:          logs/snapshots/v34_full/seed0/")
print(hr("─", 78))
print()
