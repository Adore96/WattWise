"""
Conceptual figures for the report (system, tariff + day profile, DP stage
grid, GA loop). Result plots stay in the notebooks; this script only draws
the explanatory figures, in the same palette the notebooks use.

Usage (from the repo root):
    python scripts/make_report_figures.py
Writes figures/report_*.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
DATA = ROOT / "data" / "profiles_30min.csv"

# ---- shared palette (same hex codes as notebooks/dp_solver.ipynb)
DP = "#2d5c86"  # DP / SoC
GA = "#1a7f5a"  # GA
SOLAR = "#e0a400"  # solar, greedy
LOAD = "#bbbbbb"  # load
DISCH = "#c0562a"  # battery discharge
CHARGE = "#2d8a5f"  # battery charge
INK = "#333333"
BAND_BG = {"Off-Peak": "#e8f1fb", "Day": "#fbf6e8", "Peak": "#fbe8e8"}
BAND_FG = {"Off-Peak": "#6f9fd8", "Day": "#e0b84a", "Peak": "#d0605a"}
RATE = {"Off-Peak": 33, "Day": 47, "Peak": 106}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "savefig.dpi": 200})


def box(ax, xy, w, h, text, fc, ec=INK, fs=10, bold=False, tc=INK, lw=1.2, r=0.04):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p, q, color=INK, lw=1.6, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=color,
                                 lw=lw, linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


# ------------------------------------------------------------------ 1. system
def system_diagram():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 10);
    ax.set_ylim(0, 5.2);
    ax.axis("off")

    box(ax, (0.2, 3.5), 2.4, 1.3, "Solar PV\n$g_t$ (kWh / slot)", "#fdf3d6", ec=SOLAR, bold=False)
    box(ax, (0.2, 0.4), 2.4, 1.5, "Battery\nC ∈ {0, 2, 4, 7} kWh\n2.2 kW in / 2.5 kW out\nη = 0.91 each way",
        "#e7f0f8", ec=DP, fs=9)
    box(ax, (3.8, 1.9), 2.4, 1.5, "Hybrid inverter\n+ dispatch decision\n$a_t$ every 30 min",
        "#ffffff", ec=INK, bold=True, lw=1.8)
    box(ax, (7.4, 3.5), 2.4, 1.3, "House load\n$d_t$ (kWh / slot)", "#f1f1f1", ec="#888")
    box(ax, (7.4, 0.4), 2.4, 1.5, "CEB grid\nToU tariff $p_t$\nimport only (no export)", "#fbe8e8", ec=BAND_FG["Peak"], fs=9)

    arrow(ax, (2.6, 4.0), (3.9, 3.3), color=SOLAR, lw=2.2)
    arrow(ax, (2.6, 1.35), (3.8, 2.25), color=CHARGE, lw=2.0, style="<|-|>")
    arrow(ax, (6.2, 3.1), (7.4, 3.95), color=INK, lw=2.2)
    arrow(ax, (7.4, 1.25), (6.2, 2.2), color=BAND_FG["Peak"], lw=2.2)

    ax.text(2.95, 3.95, "solar", color=SOLAR, fontsize=9, rotation=-28)
    ax.text(2.55, 2.0, "charge (−) /\ndischarge (+)", color=CHARGE, fontsize=8.5, ha="left")
    ax.text(6.55, 3.75, "serve load", color=INK, fontsize=9, rotation=35)
    ax.text(6.25, 0.95, "import\n$grid_t$ = max(0, $d_t − g_t − a_t$)", color=BAND_FG["Peak"],
            fontsize=8.5, ha="center", va="center")

    # tariff strip under the title
    x0, wtot = 3.4, 3.2
    segs = [("Off-Peak", 5.5), ("Day", 13), ("Peak", 4), ("Off-Peak", 1.5)]
    ax.text(5.0, 5.05, "Goal: choose $a_t$ for all 48 slots to minimise  Σ $p_t$·$grid_t$",
            ha="center", fontsize=10.5, fontweight="bold")
    x = x0
    for band, hrs in segs:
        w = wtot * hrs / 24
        ax.add_patch(Rectangle((x, 4.35), w, 0.3, fc=BAND_FG[band], ec="white", lw=0.8))
        x += w
    ax.text(x0 + wtot / 2, 4.2, "00:00 → 24:00   Rs 33 / 47 / 106 per kWh", ha="center", va="top", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG / "report_system_diagram.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 2. tariff + profile
def tariff_profile(date="2026-09-11"):
    df = pd.read_csv(DATA)
    d = df[df["date"] == date].sort_values("slot_start").reset_index(drop=True)
    x = np.arange(len(d))

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 3.2], "hspace": 0.08})
    # tariff step
    for t, band in enumerate(d["band"]):
        a0.add_patch(Rectangle((t - 0.5, 0), 1, RATE[band], fc=BAND_FG[band], ec="none"))
    a0.set_ylim(0, 125);
    a0.set_ylabel("Rs/kWh")
    a0.set_yticks([0, 50, 100])
    for band, xc in [("Off-Peak", 5), ("Day", 23.5), ("Peak", 40.5)]:
        a0.text(xc, RATE[band] + 4, f"{band}  Rs {RATE[band]}", ha="center", fontsize=8.5)
    a0.spines[["top", "right"]].set_visible(False)

    for t, band in enumerate(d["band"]):
        a1.axvspan(t - 0.5, t + 0.5, color=BAND_BG[band], zorder=0, lw=0)
    a1.bar(x, d["consumption_kWh"], color=LOAD, width=0.8, label="load $d_t$")
    a1.plot(x, d["production_kWh"], color=SOLAR, lw=2, label="solar $g_t$")
    net = np.maximum(0, d["consumption_kWh"] - d["production_kWh"])
    a1.step(x, net, where="mid", color=INK, lw=1, ls="--", label="net load max(0, $d_t − g_t$)")
    a1.set_ylabel("kWh per 30-min slot")
    a1.set_xticks(range(0, 49, 6))
    a1.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
    a1.set_xlim(-0.5, 47.5)
    a1.legend(loc="upper left", fontsize=8.5, frameon=False)
    a1.spines[["top", "right"]].set_visible(False)
    a1.set_xlabel(f"time of day ({date}, real household logs)")
    fig.savefig(FIG / "report_tariff_profile.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 3. DP stage grid
def dp_grid():
    T, N = 9, 5  # schematic: 9 stages, 5 SoC levels
    bands = ["Off-Peak"] * 3 + ["Day"] * 3 + ["Peak"] * 3
    path = [0, 1, 2, 3, 3, 3, 3, 2, 1, 0]  # SoC index before slot 0 .. after slot T-1

    fig, ax = plt.subplots(figsize=(9, 4.4))
    for t in range(T):
        ax.add_patch(Rectangle((t + 0.5, -0.6), 1, N - 0.2 + 0.6, fc=BAND_BG[bands[t]], ec="none", zorder=0))
        ax.text(t + 1, N - 0.45, f"{bands[t]}\nRs {RATE[bands[t]]}", ha="center", fontsize=7.5, color="#555")

    # light edges: every feasible move from each state (|k| <= 1 here)
    for t in range(T):
        for i in range(N):
            for k in (-1, 0, 1):
                j = i + k
                if 0 <= j < N:
                    ax.plot([t + 0.5, t + 1.5], [i, j], color="#c9c9c9", lw=0.7, zorder=1)
    # nodes
    for t in range(T + 1):
        ax.scatter([t + 0.5] * N, range(N), s=36, color="white", edgecolor="#777", zorder=3)
    # optimal path
    xs = [t + 0.5 for t in range(T + 1)]
    ax.plot(xs, path, color=DP, lw=3, zorder=4)
    ax.scatter(xs, path, s=60, color=DP, zorder=5)
    for t in range(T):
        k = path[t + 1] - path[t]
        lab = "charge" if k > 0 else ("discharge" if k < 0 else "hold")
        col = CHARGE if k > 0 else (DISCH if k < 0 else "#666")
        ax.text(t + 1, -0.42, lab, ha="center", fontsize=7.5, color=col, fontweight="bold")

    # backward-induction arrow
    arrow(ax, (T + 0.4, -1.05), (0.6, -1.05), color=DISCH, lw=1.8)
    ax.text((T + 1) / 2, -1.35, "backward pass: compute $V_t(s)$ from $t = T$ (where $V_T = 0$) down to $t = 0$",
            ha="center", fontsize=8.5, color=DISCH)
    arrow(ax, (0.6, -1.85), (T + 0.4, -1.85), color=DP, lw=1.8)
    ax.text((T + 1) / 2, -2.15, "forward pass: follow the stored best move from the start state (reserve floor)",
            ha="center", fontsize=8.5, color=DP)

    ax.set_yticks(range(N))
    ax.set_yticklabels(["0.2·C\n(floor)", "", "SoC levels\nstep δ", "", "C\n(full)"], fontsize=8)
    ax.set_xticks([t + 0.5 for t in range(T + 1)])
    ax.set_xticklabels([f"t={t}" for t in range(T)] + ["T"], fontsize=8)
    ax.set_xlim(0.2, T + 0.8);
    ax.set_ylim(-2.4, N + 0.1)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.set_title("$V_t(s) = \\min_k \\; [\\, p_t \\cdot \\max(0,\\, d_t - g_t - a(k)) + V_{t+1}(s - k\\delta) \\,]$",
                 fontsize=11)
    fig.savefig(FIG / "report_dp_stage_grid.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 4. GA loop
def ga_loop():
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.set_xlim(0, 10);
    ax.set_ylim(0, 6.2);
    ax.axis("off")

    # chromosome strip
    rng = np.random.default_rng(3)
    genes = np.r_[np.zeros(10), -rng.uniform(0.1, 0.9, 6), rng.uniform(0, 0.3, 4) * 0,
    -rng.uniform(0.05, 0.4, 12), rng.uniform(0.1, 0.5, 4), rng.uniform(0.05, 0.12, 8),
    np.zeros(4)][:48]
    x0, w = 0.6, 8.8 / 48
    for t, g in enumerate(genes):
        c = CHARGE if g < 0 else (DISCH if g > 0 else "#e6e6e6")
        ax.add_patch(Rectangle((x0 + t * w, 5.35), w * 0.92, 0.45, fc=c, ec="none"))
    ax.text(5, 6.0, "One individual = a full-day schedule: 48 genes $a_0 … a_{47}$ (kWh, house side)",
            ha="center", fontsize=9.5, fontweight="bold")
    ax.text(x0, 5.2, "green = charge (−)   orange = discharge (+)   grey = hold", fontsize=8, va="top", color="#555")

    bw, bh = 2.6, 0.95
    nodes = {
        "init": (0.3, 3.55, "1. Initial population (100)\nrandom + greedy seed\n+ do-nothing seed", "#eef6f1"),
        "eval": (3.7, 3.55, "2. Evaluate fitness\nbill + 10⁵ × violation\n(shared simulate())", "#ffffff"),
        "sel": (7.1, 3.55, "3. Tournament\nselection (k = 3)", "#eef6f1"),
        "cx": (7.1, 1.85, "4. Blend crossover\n(rate 0.8)", "#eef6f1"),
        "mut": (3.7, 1.85, "5. Gaussian mutation\n(rate 0.1, σ = 0.2)\n+ clip genes to bounds", "#eef6f1"),
        "elite": (0.3, 1.85, "6. Elitism: carry best 2\n→ next generation", "#eef6f1"),
    }
    for key, (x, y, txt, fc) in nodes.items():
        box(ax, (x, y), bw, bh, txt, fc, ec=GA, fs=8.4, bold=(key == "eval"))

    def mid(key, side):
        x, y, *_ = nodes[key]
        return {"r": (x + bw, y + bh / 2), "l": (x, y + bh / 2), "t": (x + bw / 2, y + bh),
                "b": (x + bw / 2, y)}[side]

    arrow(ax, mid("init", "r"), mid("eval", "l"), color=GA)
    arrow(ax, mid("eval", "r"), mid("sel", "l"), color=GA)
    arrow(ax, mid("sel", "b"), mid("cx", "t"), color=GA)
    arrow(ax, mid("cx", "l"), mid("mut", "r"), color=GA)
    arrow(ax, mid("mut", "l"), mid("elite", "r"), color=GA)
    arrow(ax, (2.9, 2.8), (3.7, 3.55), color=GA, lw=2.0)
    ax.text(3.0, 3.3, "repeat", color=GA, fontsize=8.5, rotation=43, ha="center")
    box(ax, (3.7, 0.2), bw, bh, "Stop after 200 generations\n→ best schedule\n(replayed by simulate())",
        "#e7f0f8", ec=DP, fs=8.4)
    arrow(ax, mid("elite", "b"), (3.7, 0.2 + bh / 2), color=DP, ls=":", rad=0.3)

    fig.savefig(FIG / "report_ga_loop.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    system_diagram()
    tariff_profile()
    dp_grid()
    ga_loop()
    print("wrote figures/report_{system_diagram,tariff_profile,dp_stage_grid,ga_loop}.png")


if __name__ == "__main__":
    main()
