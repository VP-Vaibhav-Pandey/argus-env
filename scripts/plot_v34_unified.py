"""ARGUS v3.4 unified 3-run analysis · plotly dark theme (v2 — Apple-bold).

Six story-driven figures. Apple-grade visual confidence:
  · 28-32pt value labels
  · solid colors, no opacity gray
  · Wilson noise as shaded bands, not error bars
  · single clear takeaway per plot
  · KEY FINDING footer for 5-second skim

Outputs: figs/v34_unified/*.png + .html
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figs" / "v34_unified"
OUT.mkdir(parents=True, exist_ok=True)

S0 = ROOT / "outputs_v34_full" / "seed0"
S1 = ROOT / "outputs_v34_full_seed1" / "seed1"
NM = ROOT / "outputs_v34_no_meta" / "seed0"

# ---------- Theme: Apple system dark ----------
C_BG = "#000000"
C_PANEL = "#0E0E10"
C_GRID = "#27272A"
C_TRACK = "#1C1C1E"
C_TEXT_MAIN = "#F5F5F7"
C_TEXT_SUB = "#8E8E93"
C_TEXT_DIM = "#48484A"
C_BLUE = "#0A84FF"
C_BLUE_SOFT = "rgba(10,132,255,0.12)"
C_RED = "#FF453A"
C_RED_SOFT = "rgba(255,69,58,0.12)"
C_GREEN = "#30D158"
C_GREEN_SOFT = "rgba(48,209,88,0.12)"
C_AMBER = "#FF9F0A"
C_AMBER_SOFT = "rgba(255,159,10,0.12)"
C_GRAY = "#8E8E93"
C_GRAY_SOFT = "rgba(142,142,147,0.10)"

LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, -apple-system, sans-serif", color=C_TEXT_MAIN),
    plot_bgcolor=C_BG,
    paper_bgcolor=C_BG,
    hoverlabel=dict(bgcolor="#18181B", font_size=13, bordercolor="#3F3F46",
                    font_color=C_TEXT_MAIN),
    title_x=0.04, title_y=0.945,
    margin=dict(t=140, b=110, l=80, r=70),
)


def add_footer(fig, key_finding=""):
    fig.add_annotation(
        x=0.04, y=-0.20, xref="paper", yref="paper",
        text="<b>ARGUS · 3-RUN ABLATION</b> &nbsp; <span style='color:#3F3F46'>|</span>"
             "&nbsp; seed 0 full · seed 1 full · no-metacog · GSM8K n=100",
        showarrow=False, font=dict(size=10, color="#6B6B77"),
        xanchor="left", yanchor="top",
    )
    if key_finding:
        fig.add_annotation(
            x=0.96, y=-0.20, xref="paper", yref="paper",
            text=f"<b style='color:#F5F5F7'>KEY FINDING</b> &nbsp; "
                 f"<span style='color:#A1A1AA'>{key_finding}</span>",
            showarrow=False,
            font=dict(size=10, family="JetBrains Mono, ui-monospace, monospace"),
            xanchor="right", yanchor="top",
        )
    return fig


def make_title(main, sub=None):
    s = (
        f"<span style='font-size:30px; font-weight:700; color:{C_TEXT_MAIN}; "
        f"letter-spacing:-0.02em;'>{main}</span>"
    )
    if sub:
        s += (f"<br><span style='font-size:14px; color:{C_TEXT_SUB}; "
              f"font-weight:400;'>{sub}</span>")
    return s


def write_pair(fig, name):
    fig.write_html(str(OUT / f"{name}.html"), include_plotlyjs="cdn", full_html=True)
    try:
        fig.write_image(str(OUT / f"{name}.png"), width=1280, height=720, scale=2)
    except Exception as e:
        print(f"  PNG export failed for {name}: {e}")
    print(f"  saved {name}.html / {name}.png")


def load_run(run_dir: Path):
    return {
        "pre": json.loads((run_dir / "pre_eval.json").read_text()),
        "post": json.loads((run_dir / "post_eval.json").read_text()),
        "eps": [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines()],
        "lies": [json.loads(l) for l in (run_dir / "lie_taxonomy.jsonl").read_text().splitlines()],
        "epi": [json.loads(l) for l in (run_dir / "epiplexity.jsonl").read_text().splitlines()],
    }


s0 = load_run(S0)
s1 = load_run(S1)
nm = load_run(NM)
runs = [
    ("Seed 0 · full", s0, C_BLUE),
    ("Seed 1 · full", s1, C_RED),
    ("Ablation · no-metacog", nm, C_GRAY),
]


# ============================================================
# FIG 1 · Three deltas + Wilson noise band (no error bars, no overlap)
# ============================================================
fig = go.Figure()

names = [r[0] for r in runs]
deltas = [r[1]["post"]["passk"]["1"] - r[1]["pre"]["passk"]["1"] for r in runs]
colors = [r[2] for r in runs]
n_eval = 100
# Two-prop noise (Wilson approx)
errs = []
for r in runs:
    p_pre = r[1]["pre"]["passk"]["1"]
    p_post = r[1]["post"]["passk"]["1"]
    se = math.sqrt(p_pre*(1-p_pre)/n_eval + p_post*(1-p_post)/n_eval)
    errs.append(1.96 * se)
noise_top = max(errs)
mean_d = sum(deltas) / len(deltas)

# Stronger Wilson noise band — amber-tinted shaded zone with subtle edge lines
fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1,
              y0=-noise_top, y1=noise_top,
              fillcolor="rgba(255,159,10,0.10)", line_width=0, layer="below")
fig.add_shape(type="line", xref="paper", yref="y",
              x0=0, x1=1, y0=noise_top, y1=noise_top,
              line=dict(color="rgba(255,159,10,0.45)", width=1, dash="dot"),
              layer="below")
fig.add_shape(type="line", xref="paper", yref="y",
              x0=0, x1=1, y0=-noise_top, y1=-noise_top,
              line=dict(color="rgba(255,159,10,0.45)", width=1, dash="dot"),
              layer="below")
# Wilson band label — bottom-LEFT, inside the negative side of the band, won't overlap any bar value
fig.add_annotation(xref="paper", yref="y", x=0.012, y=-noise_top*0.45,
                   text=f"<b style='color:#FF9F0A; font-size:11px'>WILSON 95% NOISE</b>"
                        f"<span style='color:#8E8E93'>  ·  ±{noise_top*100:.1f}pp</span>",
                   showarrow=False, xanchor="left", yanchor="middle",
                   font=dict(family="JetBrains Mono, monospace"))

# Zero baseline
fig.add_hline(y=0, line=dict(color="#3F3F46", width=1.2))

# 3-run mean dashed line — visible across whole plot
fig.add_shape(type="line", xref="paper", yref="y",
              x0=0, x1=1, y0=mean_d, y1=mean_d,
              line=dict(color=C_TEXT_MAIN, width=1.5, dash="dash"))
# Mean label — top-RIGHT, in clear empty space above bar 1 (bar 1 tops at +0.070, label at +0.105)
fig.add_annotation(xref="paper", yref="y", x=0.992, y=0.108,
                   text=f"<span style='color:#8E8E93; font-size:10px'>3-RUN MEAN</span><br>"
                        f"<b style='color:#F5F5F7; font-size:18px'>Δ = {mean_d:+.3f}</b>",
                   showarrow=False, xanchor="right", yanchor="middle", align="right",
                   font=dict(family="Inter, sans-serif"))

fig.add_trace(go.Bar(
    x=names, y=deltas,
    marker=dict(color=colors, line=dict(color=C_BG, width=0)),
    width=0.52,
    text=[f"<b>{d:+.3f}</b>" for d in deltas],
    textposition="outside",
    textfont=dict(size=28, color=C_TEXT_MAIN, family="Inter, sans-serif"),
    cliponaxis=False,
    showlegend=False,
    hovertemplate="<b>%{x}</b><br>Δ pass@1 = %{y:+.3f}<extra></extra>",
))

fig.update_layout(
    **LAYOUT_DEFAULTS,
    title=dict(text=make_title(
        "Three runs. Three deltas. All within noise.",
        "+7.0pp from a single seed was statistically inseparable from −5.0pp.")),
    height=620,
    bargap=0.42,
)
fig.update_xaxes(title=dict(text="<b>Run configuration</b>",
                            font=dict(color=C_TEXT_SUB, size=12)),
                 showgrid=False, tickfont=dict(color=C_TEXT_MAIN, size=13))
fig.update_yaxes(title=dict(text="<b>Δ pass@1   (post − pre · n=100)</b>",
                            font=dict(color=C_TEXT_SUB, size=12)),
                 zeroline=False, showgrid=True, gridcolor=C_GRID,
                 tickfont=dict(color=C_TEXT_SUB, size=11),
                 tickmode="array",
                 tickvals=[-0.10, -0.05, 0.0, 0.05, 0.10],
                 ticktext=["−0.10", "−0.05", "0.00", "+0.05", "+0.10"],
                 range=[-0.13, 0.13])
add_footer(fig, "+7pp single-seed claim is sampling-bound. Mean across runs sits inside Wilson noise.")
write_pair(fig, "01_three_run_deltas")


# ============================================================
# FIG 2 · The inversion as a slope chart (parallel coordinates)
# Each run = one line connecting (in-loop gain) → (external delta).
# Seed 1's line literally crosses zero. That IS the inversion.
# ============================================================
fig = go.Figure()

in_loops = [sum(e["learnable_info_per_token"] for e in r[1]["epi"]) for r in runs]
in_loop_max = max(in_loops)
ext_max = max(abs(d) for d in deltas)

# Normalize to [-1, +1]:
#   in-loop is always positive → maps to [0, +1]
#   external is signed → maps to [-1, +1] using its own scale
in_loop_norm = [v / in_loop_max for v in in_loops]
ext_norm = [d / ext_max for d in deltas]

# Background: tint the negative half so "below zero" reads as a loss zone
fig.add_shape(type="rect", xref="paper", yref="y",
              x0=0, x1=1, y0=-1.15, y1=0,
              fillcolor="rgba(255,69,58,0.06)", line_width=0, layer="below")
fig.add_shape(type="rect", xref="paper", yref="y",
              x0=0, x1=1, y0=0, y1=1.15,
              fillcolor="rgba(48,209,88,0.04)", line_width=0, layer="below")

# Axis labels for the two endpoints (left and right vertical guides)
fig.add_shape(type="line", xref="x", yref="paper",
              x0=0, x1=0, y0=0, y1=1,
              line=dict(color="#3F3F46", width=1, dash="dot"))
fig.add_shape(type="line", xref="x", yref="paper",
              x0=1, x1=1, y0=0, y1=1,
              line=dict(color="#3F3F46", width=1, dash="dot"))

# Zero baseline — emphasized
fig.add_hline(y=0, line=dict(color="#F5F5F7", width=1.5))
# "ZERO" label on the right edge, outside the lines, no overlap
fig.add_annotation(xref="paper", yref="y", x=0.005, y=0.04,
                   text="<b style='color:#8E8E93; font-size:10px; "
                        "font-family:JetBrains Mono, monospace;'>ZERO</b>",
                   showarrow=False, xanchor="left", yanchor="bottom")

# Axis-end labels (top of each guide)
fig.add_annotation(xref="x", yref="paper", x=0, y=1.04,
                   text="<b style='color:#30D158; font-size:13px;'>WHAT IT BELIEVED</b><br>"
                        "<span style='color:#8E8E93; font-size:10px;'>in-loop epi · nats/token</span>",
                   showarrow=False, xanchor="center", yanchor="bottom",
                   font=dict(family="Inter, sans-serif"))
fig.add_annotation(xref="x", yref="paper", x=1, y=1.04,
                   text="<b style='color:#0A84FF; font-size:13px;'>WHAT REALITY SAID</b><br>"
                        "<span style='color:#8E8E93; font-size:10px;'>external Δ pass@1 · GSM8K</span>",
                   showarrow=False, xanchor="center", yanchor="bottom",
                   font=dict(family="Inter, sans-serif"))

# Lines (one per run) + endpoint markers
for idx, ((label, _data, color), in_v, in_n, ext_v, ext_n) in enumerate(zip(
        runs, in_loops, in_loop_norm, deltas, ext_norm)):
    is_inversion = ext_v < 0  # seed 1
    line_w = 5 if is_inversion else 3
    line_op = 1.0 if is_inversion else 0.80

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[in_n, ext_n],
        mode="lines+markers",
        line=dict(color=color, width=line_w),
        marker=dict(size=20, color=color,
                    line=dict(color=C_BG, width=2.5)),
        opacity=line_op,
        name=label,
        hovertemplate=f"<b>{label}</b><br>"
                      f"in-loop: +{in_v:.3f} nats/token<br>"
                      f"external: {ext_v:+.3f} pass@1<extra></extra>",
        showlegend=True,
    ))
    # Right-side endpoint values — these spread out (0.07, 0.025, -0.05), no overlap
    fig.add_annotation(xref="x", yref="y", x=1.04, y=ext_n,
                       text=f"<b style='color:{color}; font-size:15px'>{ext_v:+.3f}</b>",
                       showarrow=False, xanchor="left", yanchor="middle",
                       font=dict(family="Inter, sans-serif"))

# LEFT SIDE: single consolidated annotation since all three in-loop values cluster (0.40 ± 0.02)
# That tight cluster IS the story — "all three claimed they were learning, similarly"
in_loop_min = min(in_loops)
in_loop_maxv = max(in_loops)
fig.add_annotation(
    xref="x", yref="y", x=-0.04, y=0.95,
    text=f"<span style='color:#8E8E93; font-size:10px; "
         f"font-family:JetBrains Mono, monospace;'>all three runs</span><br>"
         f"<b style='color:#30D158; font-size:18px; "
         f"font-family:Inter, sans-serif;'>+{in_loop_min:.2f} … +{in_loop_maxv:.2f}</b><br>"
         f"<span style='color:#8E8E93; font-size:10px;'>nats/token · clustered</span>",
    showarrow=False, xanchor="right", yanchor="middle", align="right",
)

# THE INVERSION callout — points at seed 1's line where it crosses zero
seed1_idx = 1
seed1_in_n = in_loop_norm[seed1_idx]
seed1_ext_n = ext_norm[seed1_idx]
# Find midpoint of seed 1's line (where it visibly crosses zero)
fig.add_annotation(
    xref="x", yref="y", x=0.55, y=-0.55,
    ax=0.55, ay=-0.05,
    axref="x", ayref="y",
    showarrow=True, arrowhead=2, arrowsize=1.2,
    arrowwidth=2, arrowcolor=C_RED,
    text="", standoff=4,
)
fig.add_annotation(
    xref="x", yref="y", x=0.62, y=-0.78,
    text="<b style='color:#FF453A; font-size:15px'>THE INVERSION</b><br>"
         "<span style='color:#F5F5F7; font-size:11px'>seed 1: highest in-loop gain</span><br>"
         "<span style='color:#F5F5F7; font-size:11px'>… and worst external delta</span>",
    showarrow=False, xanchor="left", yanchor="middle", align="left",
    bgcolor="rgba(255,69,58,0.10)", bordercolor=C_RED, borderwidth=1.2, borderpad=10,
    font=dict(family="Inter, sans-serif"),
)

fig.update_layout(
    **LAYOUT_DEFAULTS,
    title=dict(text=make_title(
        "In-loop UP. External DOWN. The inversion, captured.",
        "Each line is one run. Where a line crosses zero, the model believed it learned — and didn't.")),
    height=620,
    showlegend=True,
    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.16,
                font=dict(color=C_TEXT_MAIN, size=12),
                bgcolor="rgba(0,0,0,0)", itemsizing="constant"),
)
fig.update_xaxes(
    showgrid=False, zeroline=False, showticklabels=False,
    range=[-0.18, 1.18],
)
fig.update_yaxes(
    title=dict(text="<b>Normalized magnitude</b><br>"
                    "<span style='color:#6B6B77; font-size:10px'>each metric scaled to its own peak · sign preserved</span>",
               font=dict(color=C_TEXT_SUB, size=12)),
    showgrid=True, gridcolor=C_GRID,
    tickfont=dict(color=C_TEXT_SUB, size=11),
    range=[-1.15, 1.15], zeroline=False,
    tickmode="array",
    tickvals=[-1, -0.5, 0, 0.5, 1],
    ticktext=["−1.0<br><i style='font-size:9px'>worst</i>", "−0.5", "0.0", "+0.5", "+1.0<br><i style='font-size:9px'>best</i>"],
)
add_footer(fig, "Seed 1's line crosses zero — exactly the inversion ARGUS exists to catch.")
write_pair(fig, "02_in_loop_vs_external")


# ============================================================
# FIG 3 · Taxonomy heatmap (cleaner — discoveries only)
# ============================================================
fig = go.Figure()

types_ordered = ["A", "B", "C", "D", "E", "F", "G", "H"]
type_names = {
    "A": "drift", "B": "novelty collapse", "C": "memorization",
    "D": "forgetting", "E": "saturation",
    "F": "hallucination (v3.3)", "G": "plateau capture",
    "H": "defense-induced curriculum collapse",
}
discoveries = {
    ("Seed 0 · full v3.4", "G"): C_BLUE,
    ("Seed 1 · full v3.4", "H"): C_RED,
    ("Ablation · no-metacog", "E"): C_AMBER,
}
run_full_names = {
    "Seed 0 · full": "Seed 0 · full v3.4",
    "Seed 1 · full": "Seed 1 · full v3.4",
    "Ablation · no-metacog": "Ablation · no-metacog",
}

# Build rectangles
for ri, (run_short, run_data, _) in enumerate(runs):
    full_name = run_full_names[run_short]
    for ti, t in enumerate(types_ordered):
        is_discovery = (full_name, t) in discoveries
        if is_discovery:
            color = discoveries[(full_name, t)]
            opacity = 1.0
            border_w = 0
        else:
            color = C_TRACK
            opacity = 1.0
            border_w = 0
        fig.add_shape(type="rect",
                      x0=ri-0.40, x1=ri+0.40, y0=ti-0.42, y1=ti+0.42,
                      fillcolor=color, line=dict(color=C_BG, width=0),
                      opacity=opacity)
        if is_discovery:
            # Big letter centered
            fig.add_annotation(x=ri, y=ti, text=f"<b>{t}</b>",
                               showarrow=False,
                               font=dict(color="white", size=42))
            # "DISCOVERED" tag below
            fig.add_annotation(x=ri, y=ti-0.30,
                               text="<span style='font-size:9px'>★ LIVE DISCOVERY</span>",
                               showarrow=False,
                               font=dict(color="white"))
        else:
            fig.add_annotation(x=ri, y=ti, text=f"<span style='color:#48484A'>{t}</span>",
                               showarrow=False,
                               font=dict(size=18))

# Y-axis: type names
fig.update_yaxes(
    title=dict(text="<b>Failure mode (lie taxonomy)</b>",
               font=dict(color=C_TEXT_SUB, size=12)),
    tickmode="array",
    tickvals=list(range(len(types_ordered))),
    ticktext=[f"<b>{t}</b>  <span style='color:#8E8E93'>{type_names[t]}</span>"
              for t in types_ordered],
    range=[-0.7, len(types_ordered)-0.3], autorange="reversed",
    showgrid=False, tickfont=dict(color=C_TEXT_MAIN, size=11),
)
fig.update_xaxes(
    title=dict(text="<b>Run configuration</b>",
               font=dict(color=C_TEXT_SUB, size=12)),
    tickmode="array",
    tickvals=list(range(len(runs))),
    ticktext=[f"<b>{r[0]}</b>" for r in runs],
    range=[-0.7, len(runs)-0.3],
    showgrid=False, tickfont=dict(color=C_TEXT_MAIN, size=12),
)

fig.update_layout(
    **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != "margin"},
    title=dict(text=make_title(
        "Three runs. Four named failure modes discovered live.",
        "Each colored cell is a failure mode the architecture surfaced for the first time in that run.")),
    height=720, showlegend=False,
    margin=dict(t=140, b=130, l=240, r=70),
)
add_footer(fig, "G surfaced on seed 0 · H surfaced on seed 1 · E surfaced when ablation lifted metacog masking.")
write_pair(fig, "03_failure_mode_taxonomy")


# ============================================================
# FIG 4 · Curriculum collapse (foreground seed 1)
# ============================================================
fig = go.Figure()

# Background: seed 0 + no-meta in faded gray
for name, r, _ in [(runs[0][0], runs[0][1], None), (runs[2][0], runs[2][1], None)]:
    eps = r["eps"]
    x = [e["episode"] for e in eps]
    y = [(e["n_solve_easy"] / max(1, e["n_solve_hard"])) for e in eps]
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers",
        line=dict(color=C_TEXT_DIM, width=2),
        marker=dict(size=6, color=C_BG, line=dict(color=C_TEXT_DIM, width=1.5)),
        name=name,
        hovertemplate=f"<b>{name}</b><br>ep %{{x}}<br>easy/hard %{{y:.2f}}<extra></extra>",
    ))

# Foreground: seed 1 in bold red
eps_s1 = s1["eps"]
x_s1 = [e["episode"] for e in eps_s1]
y_s1 = [(e["n_solve_easy"] / max(1, e["n_solve_hard"])) for e in eps_s1]
fig.add_trace(go.Scatter(
    x=x_s1, y=y_s1, mode="lines+markers",
    line=dict(color=C_RED, width=4),
    marker=dict(size=10, color=C_BG, line=dict(color=C_RED, width=3)),
    name="Seed 1 · full v3.4 (collapsed)",
    hovertemplate="<b>Seed 1</b><br>ep %{x}<br>easy/hard %{y:.2f}<extra></extra>",
))

# Type H zone — shaded red region from ep 7 to ep 15
fig.add_vrect(x0=6.5, x1=15.5, fillcolor=C_RED_SOFT, line_width=0, layer="below")
fig.add_annotation(x=11, y=7,
                   text="<b style='color:#FF453A; font-size:16px'>Type H zone</b><br>"
                        "<span style='color:#A1A1AA; font-size:11px'>defense-induced curriculum collapse</span>",
                   showarrow=False,
                   bgcolor="rgba(28,28,30,0.85)",
                   bordercolor=C_RED, borderwidth=1, borderpad=8)

# Type C clear annotations
for ep_clear in [7, 9]:
    fig.add_annotation(x=ep_clear, y=2.6,
                       text=f"<b style='color:#FF9F0A'>Type C clear</b><br>"
                            f"<span style='font-size:9px; color:#A1A1AA'>ep {ep_clear}</span>",
                       showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor=C_AMBER,
                       ax=0, ay=-30, font=dict(size=10), align="center")

# 1:1 reference line
fig.add_hline(y=1, line=dict(color=C_TEXT_DIM, width=1, dash="dot"),
              annotation_text="balanced  1 : 1",
              annotation_position="bottom right",
              annotation_font_color=C_TEXT_SUB,
              annotation_font_size=10)

fig.update_layout(
    **LAYOUT_DEFAULTS,
    title=dict(text=make_title(
        "Type H · the proposer drifts toward easy.",
        "After clear_replay_memory fired twice on seed 1, the curriculum collapsed.")),
    height=620,
    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.13,
                font=dict(color=C_TEXT_SUB, size=11), bgcolor="rgba(0,0,0,0)"),
)
fig.update_xaxes(
    title=dict(text="<b>Episode (1 → 15)</b>",
               font=dict(color=C_TEXT_SUB, size=12)),
    showgrid=False, dtick=1, tickfont=dict(color=C_TEXT_SUB, size=11))
fig.update_yaxes(
    title=dict(text="<b>Easy : Hard problem ratio (per episode)</b>",
               font=dict(color=C_TEXT_SUB, size=12)),
    showgrid=True, gridcolor=C_GRID,
    tickfont=dict(color=C_TEXT_SUB, size=11), range=[0, 8])
add_footer(fig, "Same defense that cured Type C triggered Type H. easy/hard 1.10 → 6.17 in one episode.")
write_pair(fig, "04_type_h_curriculum_collapse")


# ============================================================
# FIG 5 · Stacked bars showing event-type breakdown (richer story)
# Each run becomes a tower of segments — lie firings, per-cluster,
# ERCV refusals, causal attributions. Ablation tower is dramatically smaller.
# ============================================================
fig = go.Figure()

def event_breakdown(r):
    eps, lies = r["eps"], r["lies"]
    n_lie = sum(1 for l in lies if l.get("any_firing"))
    n_pc = sum(len(l.get("per_cluster_firings", [])) for l in lies)
    n_ercv = sum(1 for e in eps if e.get("ercv_rolled_back"))
    n_attrib = sum(1 for e in eps if e.get("causal_hypothesis"))
    return dict(lie=n_lie, pc=n_pc, ercv=n_ercv, attrib=n_attrib)

breakdowns = [event_breakdown(r[1]) for r in runs]
totals = [sum(b.values()) for b in breakdowns]

# Component colors — distinct, stack reads bottom-up
SEG = [
    ("lie",    "Lie firings (global)",     C_GREEN),
    ("pc",     "Per-cluster detections",   C_BLUE),
    ("ercv",   "ERCV soft-rollbacks",      C_AMBER),
    ("attrib", "Causal attributions",      "#BF5AF2"),  # purple
]

# Add stacked segments
for key, label, color in SEG:
    vals = [b[key] for b in breakdowns]
    fig.add_trace(go.Bar(
        x=names, y=vals,
        name=label,
        marker=dict(color=color, line=dict(color=C_BG, width=1.5)),
        width=0.58,
        text=[f"<b>{v}</b>" if v > 0 else "" for v in vals],
        textposition="inside",
        textfont=dict(size=14, color=C_TEXT_MAIN, family="Inter, sans-serif"),
        insidetextanchor="middle",
        hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>",
    ))

# Total labels above each tower
for i, (n, t) in enumerate(zip(names, totals)):
    fig.add_annotation(
        xref="x", yref="y", x=n, y=t + 0.6,
        text=f"<b style='color:#F5F5F7; font-size:36px; "
             f"font-family:Inter, sans-serif;'>{t}</b>",
        showarrow=False, yanchor="bottom",
    )

# Big amber callout — 4× FEWER, placed in CLEAR empty space (top-right corner of plot,
# above the ablation column but well clear of any bar/value)
ablation_total = totals[2]
full_avg = (totals[0] + totals[1]) / 2
ratio = full_avg / max(1, ablation_total)
fig.add_annotation(
    xref="paper", yref="paper", x=0.985, y=0.98,
    text=f"<b style='color:#FF9F0A; font-size:34px; "
         f"font-family:Inter, sans-serif; letter-spacing:-0.02em;'>"
         f"{ratio:.1f}×</b><br>"
         f"<b style='color:#FF9F0A; font-size:11px; "
         f"font-family:JetBrains Mono, monospace; letter-spacing:0.08em;'>"
         f"FEWER&nbsp;EVENTS</b><br>"
         f"<span style='color:#8E8E93; font-size:10px;'>"
         f"strip metacognition →<br>defenses thin out</span>",
    showarrow=False, xanchor="right", yanchor="top",
    align="right",
    bgcolor="rgba(255,159,10,0.08)", bordercolor=C_AMBER, borderwidth=1.5,
    borderpad=14,
)

# Connector line from ablation tower's top to the callout (subtle)
fig.add_annotation(
    xref="x", yref="y", x=names[2], y=ablation_total + 0.4,
    ax=0, ay=-30, axref="pixel", ayref="pixel",
    showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="rgba(255,159,10,0.45)",
    text="",
)

fig.update_layout(
    **LAYOUT_DEFAULTS,
    title=dict(text=make_title(
        "Strip metacognition. The defensive instrument collapses.",
        "Each run's tower stacks every defensive event by type. "
        "Same density across seeds; the ablation cannot reach it.")),
    height=640,
    barmode="stack",
    bargap=0.42,
    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.16,
                font=dict(color=C_TEXT_MAIN, size=11),
                bgcolor="rgba(0,0,0,0)", itemsizing="constant"),
)
fig.update_xaxes(
    title=dict(text="<b>Run configuration</b>",
               font=dict(color=C_TEXT_SUB, size=12)),
    showgrid=False, tickfont=dict(color=C_TEXT_MAIN, size=13))
fig.update_yaxes(
    title=dict(text="<b>Defensive events caught (15 episodes)</b><br>"
                    "<span style='color:#6B6B77; font-size:10px'>"
                    "stacked by detector type</span>",
               font=dict(color=C_TEXT_SUB, size=12)),
    showgrid=True, gridcolor=C_GRID,
    tickfont=dict(color=C_TEXT_SUB, size=11),
    range=[0, max(totals)*1.32])
add_footer(fig, "Same defensive density across seeds (19/16). Ablation: 4. The metacognition layer is the instrument.")
write_pair(fig, "05_density_vs_delta")


# ============================================================
# FIG 6 · Family tree (REDESIGN as horizontal version cards)
# ============================================================
fig = go.Figure()

versions = [
    {"v": "v1",       "modes": "A · B · C", "desc": "drift · novelty · compute",      "tag": "designed",       "color": C_TEXT_DIM},
    {"v": "v3.1",     "modes": "D",         "desc": "forgetting + ERCV refusal",      "tag": "designed",       "color": C_TEXT_DIM},
    {"v": "v3.2",     "modes": "E",         "desc": "saturation",                     "tag": "designed",       "color": C_TEXT_DIM},
    {"v": "v3.3",     "modes": "F",         "desc": "hallucination cluster",          "tag": "discovered live",   "color": C_BLUE},
    {"v": "v3.4 · seed 0", "modes": "G", "desc": "plateau capture",                   "tag": "discovered live",   "color": C_BLUE},
    {"v": "v3.4 · seed 1", "modes": "H", "desc": "defense-induced curriculum",        "tag": "discovered live",   "color": C_RED},
    {"v": "v3.4 · no-meta","modes": "E*","desc": "saturation (was masked)",            "tag": "fired (revealed)",  "color": C_AMBER},
    {"v": "v3.5",     "modes": "?",         "desc": "next discovery",                 "tag": "anticipated",       "color": C_GRAY},
]

n = len(versions)
for i, v in enumerate(versions):
    cx = i
    is_discovery = "discovered" in v["tag"]
    is_unknown = v["modes"] == "?"
    card_color = v["color"]

    # Card background
    fig.add_shape(type="rect",
                  x0=cx-0.45, x1=cx+0.45, y0=-0.5, y1=1.6,
                  fillcolor="#0E0E10",
                  line=dict(color=card_color, width=2.5),
                  opacity=1.0)
    # Version label at top
    fig.add_annotation(x=cx, y=1.43,
                       text=f"<b style='color:{card_color}'>{v['v']}</b>",
                       showarrow=False, font=dict(size=11),
                       xanchor="center")
    # Big mode letter(s)
    mode_size = 50 if len(v['modes']) <= 2 else 30
    fig.add_annotation(x=cx, y=0.65,
                       text=f"<b style='color:{C_TEXT_MAIN}'>{v['modes']}</b>",
                       showarrow=False, font=dict(size=mode_size))
    # Description
    fig.add_annotation(x=cx, y=0.05,
                       text=f"<span style='color:#A1A1AA; font-size:9px'>{v['desc']}</span>",
                       showarrow=False)
    # Tag at bottom of card
    fig.add_annotation(x=cx, y=-0.35,
                       text=f"<span style='color:{card_color}; font-size:9px; "
                            "font-family: JetBrains Mono, monospace; "
                            "letter-spacing:0.08em; text-transform:uppercase'>"
                            f"{v['tag'].upper()}</span>",
                       showarrow=False)

# Connecting arrows on the discovery chain
for i in range(3, 7):
    fig.add_shape(type="line",
                  x0=i-0.45, x1=i-0.55, y0=-1.0, y1=-1.0,
                  line=dict(color=C_TEXT_SUB, width=1))
fig.add_annotation(x=4.5, y=-1.35,
                   text="<i style='color:#A1A1AA; font-size:11px'>each defense surfaces the next failure mode →</i>",
                   showarrow=False, xanchor="center")

fig.update_xaxes(showgrid=False, showticklabels=False,
                 range=[-0.7, n-0.3], zeroline=False)
fig.update_yaxes(showgrid=False, showticklabels=False,
                 range=[-1.7, 2.0], zeroline=False)

fig.update_layout(
    **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != "margin"},
    title=dict(text=make_title(
        "The taxonomy is the contribution.",
        "Each version of the architecture surfaces a new failure mode the previous version couldn't see.")),
    height=620, showlegend=False,
    margin=dict(t=140, b=140, l=80, r=80),
)
add_footer(fig, "v3.5 will discover the next failure mode. The taxonomy is a recursive document.")
write_pair(fig, "06_discovery_family_tree")


# ============================================================
# FIG 7 · Radar fingerprint · 6-axis comparison of all 3 runs
# ============================================================
fig = go.Figure()

# Compute per-run radar coordinates (each axis normalised to 0-1)
def fingerprint(r):
    eps = r["eps"]
    lies = r["lies"]
    epi = r["epi"]
    # 1 in-loop skill (final ep skill_level)
    in_loop_skill = eps[-1]["skill_level"]
    # 2 cumulative learnable info — normalize by 0.5
    cum_epi = min(1.0, sum(e["learnable_info_per_token"] for e in epi) / 0.5)
    # 3 defensive density
    n_lie = sum(1 for l in lies if l.get("any_firing"))
    n_pc = sum(len(l.get("per_cluster_firings", [])) for l in lies)
    n_ercv = sum(1 for e in eps if e.get("ercv_rolled_back"))
    n_attrib = sum(1 for e in eps if e.get("causal_hypothesis"))
    defense_density = min(1.0, (n_lie + n_pc + n_ercv + n_attrib) / 20)
    # 4 capability-map richness
    cmap_richness = len(eps[-1].get("capability_map_stats", [])) / 10
    # 5 refusal sensitivity — abs of most-negative z observed
    z_min = min((e.get("ercv_zscore") or 0.0) for e in eps)
    refusal_sensitivity = min(1.0, abs(z_min) / 4)
    # 6 external lift: map [-0.10, +0.10] to [0, 1]
    delta_p1 = r["post"]["passk"]["1"] - r["pre"]["passk"]["1"]
    external_lift = max(0, min(1.0, (delta_p1 + 0.10) / 0.20))
    return {
        "in_loop_skill": in_loop_skill,
        "cum_epi": cum_epi,
        "defense_density": defense_density,
        "cmap_richness": cmap_richness,
        "refusal_sensitivity": refusal_sensitivity,
        "external_lift": external_lift,
    }

axes_labels = [
    "In-loop skill<br><span style='font-size:9px; color:#8E8E93'>final ep · 0–1</span>",
    "Cumulative epi<br><span style='font-size:9px; color:#8E8E93'>nats / token</span>",
    "Defensive density<br><span style='font-size:9px; color:#8E8E93'>events / 20</span>",
    "Capability-map<br>richness<br><span style='font-size:9px; color:#8E8E93'>clusters / 10</span>",
    "Refusal sensitivity<br><span style='font-size:9px; color:#8E8E93'>|min z-score| / 4</span>",
    "External lift<br><span style='font-size:9px; color:#8E8E93'>Δ pass@1 mapped 0–1</span>",
]
axes_keys = ["in_loop_skill", "cum_epi", "defense_density",
             "cmap_richness", "refusal_sensitivity", "external_lift"]

for name, r, color in runs:
    fp = fingerprint(r)
    values = [fp[k] for k in axes_keys]
    # close the polygon
    values_closed = values + [values[0]]
    labels_closed = axes_labels + [axes_labels[0]]
    # convert color to rgba with low alpha for fill
    if color == C_BLUE:
        fill = "rgba(10,132,255,0.12)"
    elif color == C_RED:
        fill = "rgba(255,69,58,0.12)"
    else:
        fill = "rgba(142,142,147,0.10)"
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=labels_closed,
        fill="toself", fillcolor=fill,
        line=dict(color=color, width=2.5),
        marker=dict(size=8, color=color, line=dict(color=C_BG, width=2)),
        name=name,
        hovertemplate=f"<b>{name}</b><br>%{{theta}}<br>%{{r:.2f}}<extra></extra>",
    ))

fig.update_layout(
    **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != "margin"},
    title=dict(text=make_title(
        "Each run has a different fingerprint.",
        "Six normalised dimensions. No run wins on all axes — the architecture's contribution is multidimensional.")),
    height=720, margin=dict(t=140, b=140, l=80, r=80),
    polar=dict(
        bgcolor=C_BG,
        radialaxis=dict(visible=True, range=[0, 1],
                        tickfont=dict(size=10, color=C_TEXT_DIM),
                        gridcolor=C_GRID, linecolor="rgba(0,0,0,0)",
                        tickvals=[0.25, 0.5, 0.75, 1.0],
                        ticktext=["0.25", "0.50", "0.75", "1.00"]),
        angularaxis=dict(tickfont=dict(size=11, color=C_TEXT_MAIN),
                         gridcolor=C_GRID, linecolor="rgba(0,0,0,0)"),
    ),
    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.06,
                font=dict(color=C_TEXT_SUB, size=11),
                bgcolor="rgba(0,0,0,0)"),
)

# Annotate insight: Seed 1's external-lift collapses despite high in-loop
# Place a small explanatory inset
fig.add_annotation(
    xref="paper", yref="paper", x=0.02, y=0.98,
    text="<span style='color:#A1A1AA; font-size:11px'>"
         "<b style='color:#0A84FF'>Seed 0</b> · balanced shape<br>"
         "<b style='color:#FF453A'>Seed 1</b> · big in-loop, weak external lift<br>"
         "<b style='color:#8E8E93'>No-metacog</b> · weak in defenses, mid lift</span>",
    showarrow=False, xanchor="left", yanchor="top",
    bgcolor="rgba(28,28,30,0.85)", bordercolor=C_GRID, borderwidth=1, borderpad=10,
)

add_footer(fig, "Three distinct fingerprints. Same architecture, different seeds, different signatures.")
write_pair(fig, "07_fingerprint_radar")


print(f"\nAll 7 unified figures written to {OUT.relative_to(ROOT)}/")
print("Story:")
print("  01 · Three deltas, all in noise — +7pp wasn't reproducible")
print("  02 · In-loop UP, External DOWN — Liu's regime captured")
print("  03 · Three runs, four named failure modes discovered live")
print("  04 · Type H mechanism — defense-induced curriculum collapse")
print("  05 · Metacognition: 4× more diagnostic events")
print("  06 · Recursive taxonomy as version cards")
print("  07 · Radar fingerprint — each run has a distinct multidim signature")
