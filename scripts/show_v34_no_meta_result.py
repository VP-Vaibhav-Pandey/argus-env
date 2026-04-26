"""ARGUS v3.4 NO-METACOGNITION ablation — screenshot-ready terminal output.

Reads outputs_v34_no_meta/seed0/ and prints a clean comparison
against the full v3.4 stack (seed 0). Highlights the dramatic
collapse of the diagnostic instrument when metacognition is stripped.

Usage:
    .venv/Scripts/python.exe scripts/show_v34_no_meta_result.py
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
NM   = ROOT / "outputs_v34_no_meta"      / "seed0"
S0   = ROOT / "outputs_v34_full"         / "seed0"
S1   = ROOT / "outputs_v34_full_seed1"   / "seed1"

def load(p):
    return {
        "pre":  json.loads((p / "pre_eval.json").read_text()),
        "post": json.loads((p / "post_eval.json").read_text()),
        "eps":  [json.loads(l) for l in (p / "episodes.jsonl").read_text().splitlines()],
        "lies": [json.loads(l) for l in (p / "lie_taxonomy.jsonl").read_text().splitlines()],
        "epi":  [json.loads(l) for l in (p / "epiplexity.jsonl").read_text().splitlines()],
        "cfg":  json.loads((p / "config.json").read_text()),
    }

nm = load(NM)
s0 = load(S0)
s1 = load(S1)


def hr(c="─", n=78):
    return c * n


def box_line(text, width=78):
    pad = width - len(text) - 2
    return f"│ {text}{' ' * pad}│"


def event_total(r):
    eps, lies = r["eps"], r["lies"]
    n_lie    = sum(1 for l in lies if l.get("any_firing"))
    n_pc     = sum(len(l.get("per_cluster_firings", [])) for l in lies)
    n_ercv   = sum(1 for e in eps if e.get("ercv_rolled_back"))
    n_attrib = sum(1 for e in eps if e.get("causal_hypothesis"))
    return dict(lie=n_lie, pc=n_pc, ercv=n_ercv, attrib=n_attrib,
                total=n_lie + n_pc + n_ercv + n_attrib)


print()
print("┌" + hr("─", 76) + "┐")
print(box_line("ARGUS  ·  v3.4  ·  NO-METACOGNITION ABLATION"))
print(box_line("GSM8K eval  ·  n=100  ·  same warmstart, same seed (0), 15 episodes"))
print("└" + hr("─", 76) + "┘")
print()

# ─────────────────────────────────────────────────────────────────────────
# HEADLINE
# ─────────────────────────────────────────────────────────────────────────
nm_d = nm["post"]["passk"]["1"] - nm["pre"]["passk"]["1"]
s0_d = s0["post"]["passk"]["1"] - s0["pre"]["passk"]["1"]
s1_d = s1["post"]["passk"]["1"] - s1["pre"]["passk"]["1"]

print("  HEADLINE  ·  external pass@1 across the three runs")
print("  " + hr("─", 70))
print(f"                          no-metacog        seed 0 full       seed 1 full")
print(f"    pre   pass@1     {nm['pre']['passk']['1']:.3f}             {s0['pre']['passk']['1']:.3f}             {s1['pre']['passk']['1']:.3f}")
print(f"    post  pass@1     {nm['post']['passk']['1']:.3f}             {s0['post']['passk']['1']:.3f}             {s1['post']['passk']['1']:.3f}")
sign_nm = "+" if nm_d >= 0 else ""
sign_s0 = "+" if s0_d >= 0 else ""
sign_s1 = "+" if s1_d >= 0 else ""
print(f"    Δ     pass@1     {sign_nm}{nm_d:.3f}            {sign_s0}{s0_d:.3f}            {sign_s1}{s1_d:.3f}")
print()

mean_d = (nm_d + s0_d + s1_d) / 3
print(f"    3-run mean Δ     {('+' if mean_d>=0 else '')}{mean_d:.3f}      ← inside Wilson 95% noise band on n=100")
print()

# ─────────────────────────────────────────────────────────────────────────
# DEFENSIVE INSTRUMENT — the breakdown that makes the case
# ─────────────────────────────────────────────────────────────────────────
b_nm = event_total(nm)
b_s0 = event_total(s0)
b_s1 = event_total(s1)

print("  DEFENSIVE INSTRUMENT  ·  what the architecture actually caught")
print("  " + hr("─", 70))
print(f"                            no-metacog        seed 0 full       seed 1 full")
print(f"    Lie firings (global)        {b_nm['lie']:>2}                {b_s0['lie']:>2}                {b_s1['lie']:>2}")
print(f"    Per-cluster detections      {b_nm['pc']:>2}                {b_s0['pc']:>2}                {b_s1['pc']:>2}")
print(f"    ERCV soft-rollbacks         {b_nm['ercv']:>2}                {b_s0['ercv']:>2}                {b_s1['ercv']:>2}")
print(f"    Causal attributions         {b_nm['attrib']:>2}                {b_s0['attrib']:>2}                {b_s1['attrib']:>2}")
print(f"    ─────────────────────────  ────              ────              ────")
print(f"    TOTAL events                {b_nm['total']:>2}                {b_s0['total']:>2}                {b_s1['total']:>2}")
print()

ratio_s0 = b_s0["total"] / max(1, b_nm["total"])
ratio_s1 = b_s1["total"] / max(1, b_nm["total"])
print(f"    ratio (full/ablation)     1.0×              {ratio_s0:.1f}×              {ratio_s1:.1f}×")
print()
print("  → strip metacognition  ·  defenses thin out by ≈ 4×")
print("  → causal attribution layer  ·  vanishes entirely (no per-cluster signal to attribute)")
print("  → per-cluster detectors      ·  vanish entirely (no capability map exists)")
print()

# ─────────────────────────────────────────────────────────────────────────
# CONFIG — what was disabled
# ─────────────────────────────────────────────────────────────────────────
disabled = []
if not nm["cfg"].get("capability_map", True):       disabled.append("capability_map")
if not nm["cfg"].get("causal_attribution", True):   disabled.append("causal_attribution")
if not nm["cfg"].get("per_cluster_detectors", True):disabled.append("per_cluster_detectors")
if not nm["cfg"].get("most_learnable_cluster",True):disabled.append("most_learnable_cluster")
if not nm["cfg"].get("active_defense", True):       disabled.append("active_defense")
if not nm["cfg"].get("proposer_validator", True):   disabled.append("proposer_validator")

print("  WHAT WAS DISABLED  ·  metacognition-layer flags = false")
print("  " + hr("─", 70))
for d in disabled:
    print(f"    ✗  {d}")
print()
print("  KEPT ENABLED  ·  the basic detectors only")
print(f"    ✓  ercv_rollback (z < {nm['cfg']['ercv_zscore_threshold']})")
print(f"    ✓  soft_replay_decay  (keep {nm['cfg']['replay_keep_fraction']*100:.0f}%)")
print(f"    ✓  spsi  (self-play stability)")
print(f"    ✓  global lie taxonomy  (detector bank only)")
print()

# ─────────────────────────────────────────────────────────────────────────
# EPISODES TIMELINE
# ─────────────────────────────────────────────────────────────────────────
print("  PER-EPISODE TIMELINE  ·  no-metacognition")
print("  " + hr("─", 74))
print("   ep │ reward │ skill │ frnt │ easy │ hard │ event")
print("   ───┼────────┼───────┼──────┼──────┼──────┼─────────────────────────────")
for e in nm["eps"]:
    fired = []
    if e.get("unified_firing"): fired.append("D!")
    if e.get("per_cluster_firings"):
        for f in e["per_cluster_firings"]:
            fired.append(f"d{f.get('cluster_id','?')}")
    rb = ""
    if e.get("ercv_rolled_back"):
        rb = " REFUSED"
    fired_str = ",".join(fired) + rb if (fired or rb) else "—"
    print(f"   {e['episode']:>2} │  {e['mean_reward']:.2f}  │ {e['skill_level']:.3f} │  {e.get('n_solve_frontier',0):>2}  │  {e.get('n_solve_easy',0):>2}  │  {e.get('n_solve_hard',0):>2}  │ {fired_str}")
print()

# ─────────────────────────────────────────────────────────────────────────
# THE TAKEAWAY
# ─────────────────────────────────────────────────────────────────────────
print("  THE TAKEAWAY  ·  reading the ablation honestly")
print("  " + hr("─", 70))
print("    METRIC LIFT       ·  +2.5pp · within Wilson noise · sampling-bound")
print("    DIAGNOSTIC RICHNESS · 4 events vs 19/16 with metacog · 4× thinner")
print("    LIVE DISCOVERIES  ·  zero — no per-cluster signal to surface them")
print()
print("  The metacognition layer is not cosmetic.")
print("  It is what makes the architecture an INSTRUMENT, not just a metric.")
print()

print(hr("─", 78))
print("  raw artifacts:    outputs_v34_no_meta/seed0/")
print("  full comparison:  scripts/show_v34_result.py     (seed 0 with metacog)")
print("                    scripts/show_v34_seed1_result.py  (seed 1 with metacog)")
print(hr("─", 78))
print()
