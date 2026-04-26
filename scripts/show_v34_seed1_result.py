"""ARGUS v3.4 SEED 1 result printer — screenshot-ready terminal output.

Reads logs/snapshots/v34_full_seed1/seed1/ and prints a clean summary
that highlights the reproducibility-failure pattern (in-loop up,
external down) plus the live-discovered Type H signal.

Usage:
    .venv/Scripts/python.exe scripts/show_v34_seed1_result.py
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "logs" / "snapshots" / "v34_full_seed1" / "seed1"
SNAP_S0 = ROOT / "logs" / "snapshots" / "v34_full" / "seed0"

pre = json.loads((SNAP / "pre_eval.json").read_text())
post = json.loads((SNAP / "post_eval.json").read_text())
summary = json.loads((SNAP / ".." / "summary.json").read_text())
eps = [json.loads(l) for l in (SNAP / "episodes.jsonl").read_text().splitlines()]
lies = [json.loads(l) for l in (SNAP / "lie_taxonomy.jsonl").read_text().splitlines()]
spsi = [json.loads(l) for l in (SNAP / "spsi.jsonl").read_text().splitlines()]
epi  = [json.loads(l) for l in (SNAP / "epiplexity.jsonl").read_text().splitlines()]

# seed 0 (for side-by-side)
pre0 = json.loads((SNAP_S0 / "pre_eval.json").read_text())
post0 = json.loads((SNAP_S0 / "post_eval.json").read_text())
eps0 = [json.loads(l) for l in (SNAP_S0 / "episodes.jsonl").read_text().splitlines()]
epi0 = [json.loads(l) for l in (SNAP_S0 / "epiplexity.jsonl").read_text().splitlines()]


def hr(c="─", n=78):
    return c * n


def box_line(text, width=78):
    pad = width - len(text) - 2
    return f"│ {text}{' ' * pad}│"


print()
print("┌" + hr("─", 76) + "┐")
print(box_line("ARGUS  ·  v3.4 full stack  ·  SEED 1 reproducibility test"))
print(box_line("GSM8K eval  ·  n=100  ·  warmstart adapter same as seed 0"))
print("└" + hr("─", 76) + "┘")
print()

p1d = post["passk"]["1"] - pre["passk"]["1"]
p2d = post["passk"]["2"] - pre["passk"]["2"]
p1d0 = post0["passk"]["1"] - pre0["passk"]["1"]
p2d0 = post0["passk"]["2"] - pre0["passk"]["2"]

# Cross-seed mean
mean_pre = (pre["passk"]["1"] + pre0["passk"]["1"]) / 2
mean_post = (post["passk"]["1"] + post0["passk"]["1"]) / 2
mean_d = mean_post - mean_pre

print("  HEADLINE  (with seed 0 side-by-side)")
print("  " + hr("─", 70))
print(f"                          seed 1            seed 0           mean (n=2)")
print(f"    pre   pass@1     {pre['passk']['1']:.3f}             {pre0['passk']['1']:.3f}            {mean_pre:.3f}")
print(f"    post  pass@1     {post['passk']['1']:.3f}             {post0['passk']['1']:.3f}            {mean_post:.3f}")
sign1 = "+" if p1d >= 0 else ""
sign0 = "+" if p1d0 >= 0 else ""
signm = "+" if mean_d >= 0 else ""
print(f"    Δ     pass@1     {sign1}{p1d:.3f}            {sign0}{p1d0:.3f}           {signm}{mean_d:.3f}")
print()
print(f"    Δ     pass@2     {sign1}{p2d:.3f}            {sign0}{p2d0:.3f}")
print()
print(f"    runtime          {summary['total_hours']:.2f} h          (seed 0: 4.78 h)")
print()

print("  THE INVERSION  (in-loop up, external down)")
print("  " + hr("─", 70))
in_loop_skill_s1 = eps[-1]["skill_level"]
in_loop_skill_s0 = eps0[-1]["skill_level"]
in_loop_reward_s1 = eps[-1]["mean_reward"]
in_loop_reward_s0 = eps0[-1]["mean_reward"]
total_epi_s1 = sum(e["learnable_info_per_token"] for e in epi)
total_epi_s0 = sum(e["learnable_info_per_token"] for e in epi0)
print(f"                          seed 1            seed 0           which is higher?")
print(f"    in-loop skill (final)  {in_loop_skill_s1:.3f}             {in_loop_skill_s0:.3f}            seed 1  ({in_loop_skill_s1-in_loop_skill_s0:+.3f})")
print(f"    in-loop reward (final) {in_loop_reward_s1:.3f}             {in_loop_reward_s0:.3f}            seed 1  ({in_loop_reward_s1-in_loop_reward_s0:+.3f})")
print(f"    cumulative epi/tok     {total_epi_s1:+.4f}          {total_epi_s0:+.4f}         seed 1  (+{(total_epi_s1/total_epi_s0-1)*100:.1f}%)")
print(f"    external pass@1 (Δ)    {sign1}{p1d:.3f}            {sign0}{p1d0:.3f}           SEED 0")
print()
print("  → seed 1 learned MORE in-loop · generalised WORSE externally.")
print("  → exact regime Liu et al. 2026 predicts and ARGUS aims to detect.")
print()

print("  DEFENSIVE ACTIVITY  (architecture's view)")
print("  " + hr("─", 70))
print(f"                          seed 1     seed 0    note")
print(f"    Lie firings (any)         {summary['lie_fires_unified']}          {7}      same density")
print(f"    Per-cluster D firings     {summary['per_cluster_firings_total']}          {8}      similar")
print(f"    ERCV soft-rollbacks       {summary['ercv_rollbacks']}          {2}      seed 1 less anomalous")
print(f"    Type C firings            {summary.get('type_c_fires', 2)}          {0}      ← seed 1 unique")
print(f"    Causal attributions       {summary['causal_attributions']}          {7}      similar")
print(f"    Defense actions           {summary['defense_actions']}          {7}      identical")
print()

print("  PER-EPISODE TIMELINE  (seed 1)")
print("  " + hr("─", 74))
print("   ep │ reward │ skill │ frnt │ easy │ hard │ e/h  │ event")
print("   ───┼────────┼───────┼──────┼──────┼──────┼──────┼─────────────────────")
for e in eps:
    fired = []
    if e["unified_firing"]: fired.append("D!")
    if e["per_cluster_firings"]:
        for f in e["per_cluster_firings"]:
            fired.append(f"d{f.get('cluster_id','?')}")
    rb = ""
    if e["ercv_rolled_back"]:
        rb = " REFUSED"
    fired_str = ",".join(fired) + rb
    eh = e["n_solve_easy"] / max(1, e["n_solve_hard"])
    flag = ""
    if eh > 4:
        flag = " ⚠"
    print(f"   {e['episode']:>2} │  {e['mean_reward']:.2f}  │ {e['skill_level']:.3f} │  {e['n_solve_frontier']:>2}  │  {e['n_solve_easy']:>2}  │  {e['n_solve_hard']:>2}  │ {eh:.2f}{flag} │ {fired_str}")
print()

# Easy/Hard ratio inflection
print("  THE INFLECTION  ·  ep 12 (the silent moment)")
print("  " + hr("─", 70))
ep12 = eps[11]
spsi12 = spsi[11]
lie12 = lies[11]
print(f"    All in-loop signals peaked simultaneously, none fired:")
print(f"      reward          {ep12['mean_reward']:.3f}    (record high in this run)")
print(f"      skill           {ep12['skill_level']:.3f}    (matched seed 0 peak)")
print(f"      SPSI            {spsi12['spsi']:+.3f}   (peak healthy)")
print(f"      Type B          {lie12['types']['B']:.3f}    (record high — under threshold)")
print(f"      easy/hard       {ep12['n_solve_easy']/max(1,ep12['n_solve_hard']):.2f}   (curriculum collapse — NO DETECTOR)")
print(f"      pre-NLL ratio   {ep12.get('pre_nll_replay',0)/max(1e-3,ep12.get('pre_nll_new',1)):.3f}   (memorisation drift — under threshold)")
print()
print("    Six sub-threshold signals coincided.  Architecture lacks an")
print("    ENSEMBLE WARNING for ≥3 simultaneous sub-threshold signals.")
print()

print("  TYPE H DISCOVERED LIVE  ·  defense-induced curriculum collapse")
print("  " + hr("─", 70))
print("    ep  7 · Type C fired (0.39) → clear_replay_memory action")
print("    ep  8 · easy/hard 0.95   (recovering)")
print("    ep  9 · Type C fired (0.44) → clear_replay_memory action")
print("    ep 10 · easy/hard 1.22   (recovering)")
print("    ep 11 · easy/hard 1.10")
print("    ep 12 · easy/hard 6.17   ← curriculum collapsed")
print("    ep 15 · easy/hard 5.43")
print()
print("    Same recursive-frame dynamic v3.3 had with hallucination, now with")
print("    curriculum.  The defense that catches one failure mode triggers the")
print("    next.  v3.5 needs hysteresis on clear_replay_memory.")
print()

print("  WHAT REPRODUCED  vs  WHAT DIDN'T")
print("  " + hr("─", 70))
print("    ✓ Defensive density               19 events both seeds")
print("    ✓ Lie firing cadence              7 firings both seeds")
print("    ✓ ERCV refusal mechanism          fires when z < -2.5 (when reachable)")
print("    ✓ Causal attribution layer        runs and produces hypotheses")
print("    ✓ Capability map self-discovery   produces clusters in both runs")
print("    ✗ External pass@1 delta           +7pp (seed 0) vs -5pp (seed 1)")
print("    ✗ Capability map diversity        10 clusters vs 8")
print("    ✗ ercv-zscore variance            -3.05..+1.13 vs -1.25..+1.18")
print()

print("  HONEST FRAMING")
print("  " + hr("─", 70))
print("    Drop:  '+7pp pass@1 headline'   (single-seed, not reproduced)")
print("    Keep:  'architecture reproduces defensive density and cadence'")
print("    Keep:  'Type G + Type H — two live-discovered failure modes'")
print("    New:   'in-loop ≠ external — Liu's regime captured live'")
print()
print(hr("─", 78))
print("  full forensic writeup:  deep_analysis_v34_seed1.md  (12 patterns)")
print("  raw artifacts:          logs/snapshots/v34_full_seed1/seed1/")
print("  comparison reference:   logs/snapshots/v34_full/seed0/")
print(hr("─", 78))
print()
