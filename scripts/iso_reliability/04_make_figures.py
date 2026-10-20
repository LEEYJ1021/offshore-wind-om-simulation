#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures_v14.py -- corrected publication figures (strictly black / white / gray) for
"Iso-reliability benchmarking of maintenance policies for capacity-constrained fleets" (C&OR).

Usage
    python make_figures_v14.py --indir outputs_v11 --outdir figures_v14 [--dpi 600] [--tiff]

Changes vs v13
  panels : letters of axes in the same row (same top edge) are placed at the same height
  FigA4  : zero-span bars drawn as a thick baseline stub and labelled "n\u2020" (no more overlapping
           "(span 0)" text); the dagger is explained in the xlabel
  Fig1   : wider BAY/BAYe cell so the text no longer touches the box edge

Changes vs v12
  panels : panel letters clear the TOP of long y-labels (Fig2(c), Fig5(c)/(d))
  Fig4   : x-labels shortened (q50 goes to the caption)
  FigA1  : S0 note "EAT hidden behind ORA (identical in S0)"
  FigA3  : tie footnote wrapped to two lines

Changes vs v11 (kept)
  Fig1  : label column no longer hidden by boxes; clean input bus arrow
  Fig2  : x-limits padded, two-line category labels, panel letters auto-placed, (c) label/legend fixed
  Fig3  : ST/L/ST x L text moved to xlabel (no overlap with legend), legend + note at bottom
  Fig4  : y tick labels restored (sharey bug), x-limits padded
  Fig5  : legend moved out of data region, clipped markers drawn cleanly, tighter vertical layout
  FigA1/A2 : all sub-panels lettered; FigA3 : tied optimum (identical policy) flagged with ringed marker
  FigA4 : (c) panel letter mis-placement fixed, plain log tick labels, notes moved to xlabel

Caption reminders
  Fig4   : all panels are q50.
  Fig5   : off-scale markers are drawn at the panel edge; the true value is printed next to them.
  FigA4b : \u2020 = span 0, i.e. a single grid point lies within the 1 bn band.
"""
import os, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--indir", default="outputs_v11")
ap.add_argument("--outdir", default="figures_v14")
ap.add_argument("--dpi", type=int, default=600)
ap.add_argument("--tiff", action="store_true")
args, _unknown = ap.parse_known_args()
os.makedirs(args.outdir, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.8, "axes.titlesize": 8.2,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.5, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "axes.unicode_minus": True,
})
MINUS = "\u2212"

# ----------------------------------------------------------------------------- data
def rd(name):
    p = os.path.join(args.indir, name)
    if not os.path.exists(p):
        print(f"[warn] missing {p}")
        return None
    d = pd.read_csv(p)
    for c in ("valid_iso", "narrow_range", "covers_zero"):
        if c in d.columns:
            d[c] = d[c].astype(str).str.lower().eq("true")
    return d

Y1, Y2, Y3, Y4 = rd("Y1_core_means.csv"), rd("Y2_core_contrasts.csv"), rd("Y3_recovery.csv"), rd("Y4_attribution.csv")
Y5, Y6b, Y7 = rd("Y5_misspec.csv"), rd("Y6b_cost_joint.csv"), rd("Y7_info_ladder.csv")
Y9, Y10, Y11 = rd("Y9_capacity_sweep.csv"), rd("Y10_scale.csv"), rd("Y11_tuning_budget.csv")
Y13, Y14, Y15 = rd("Y13_propB.csv"), rd("Y14_opt_flatness.csv"), rd("Y15_runtime.csv")

SC = ["S0 homogeneous", "S1 hetero", "S2 hetero+fast-wear", "S3 strong-hetero"]
SCT = {"S0 homogeneous": "S0: homogeneous", "S1 hetero": "S1: heterogeneous",
       "S2 hetero+fast-wear": "S2: het., faster wear", "S3 strong-hetero": "S3: strong het."}
SS = {"S1 hetero": "S1", "S2 hetero+fast-wear": "S2", "S3 strong-hetero": "S3", "S0 homogeneous": "S0"}

PS = {
    "TBM":    dict(m="s", f=0,  s=1.0, lab="TBM (uniform calendar)"),
    "EAT":    dict(m="o", f=0,  s=1.0, lab="EAT (effective age, nominal)"),
    "IDX":    dict(m="s", f=0,  s=1.0, lab="IDX (cost-weighted index)"),
    "IDXL":   dict(m="s", f=1,  s=1.0, lab="IDXL (index + learned wear)"),
    "PS":     dict(m="h", f=1,  s=1.15, lab="PS (policy search)"),
    "LTBM":   dict(m="^", f=1,  s=1.1, lab="LTBM (learned calendar)"),
    "BAY":    dict(m="v", f=1,  s=1.1, lab="BAY (Bayes, fixed prior)"),
    "BAYe":   dict(m="D", f=1,  s=0.9, lab="BAYe (empirical Bayes)"),
    "ORA":    dict(m="*", f=1,  s=1.7, lab="ORA (oracle bound)"),
    "SEN0.3": dict(m="X", f=.5, s=1.1, lab=r"SEN, iid noise $\sigma$=0.3"),
    "SEN1.0": dict(m="X", f=.5, s=1.1, lab=r"SEN, iid noise $\sigma$=1.0"),
    "SENc0.3": dict(m="P", f=.5, s=1.1, lab=r"SEN, AR(1) noise $\sigma$=0.3"),
    "SENc0.6": dict(m="P", f=.5, s=1.1, lab=r"SEN, AR(1) noise $\sigma$=0.6"),
    "SENb0.3": dict(m="p", f=.5, s=1.1, lab=r"SEN, biased, $\sigma$=0.3"),
}
FILL = {0: "white", .5: "0.62", 1: "black"}
MS = 4.6

# ----------------------------------------------------------------------------- helpers
def panels(fig, items, dy=0.006):
    """Place bold panel letters at the top-left of each axes, measured AFTER layout so that
    they never collide with y tick labels, the TOP of y labels, or titles.
    [v14] axes that share the same top edge (same row) get their letters at the same height.
    items: list of (ax, text, with_ylabels)"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    pos = []
    for ax, txt, with_y in items:
        bb = ax.get_window_extent(r)
        xs, ys = [bb.x0], [bb.y1]
        if with_y:
            for t in ax.get_yticklabels():
                if t.get_visible() and t.get_text():
                    xs.append(t.get_window_extent(r).x0)
            if ax.yaxis.label.get_text():
                lb = ax.yaxis.label.get_window_extent(r)
                xs.append(lb.x0)
                ys.append(lb.y1)                                  # y-label top
        if ax.title.get_text():
            ys.append(ax.title.get_window_extent(r).y1)
        pos.append([min(xs), max(ys), txt, bb.y1])
    for p in pos:                                                 # [v14] same row -> same letter height
        p[1] = max(q[1] for q in pos if abs(q[3] - p[3]) < 2)
    for x, y, txt, _ in pos:
        fig.text(x / W, y / H + dy, txt, fontweight="bold", fontsize=8.8, ha="left", va="bottom")

def pad_xlim(ax, los, his, incl0=True, pad=0.09):
    lo, hi = np.nanmin(los), np.nanmax(his)
    if incl0:
        lo, hi = min(lo, 0), max(hi, 0)
    p = pad * (hi - lo if hi > lo else 1.0)
    ax.set_xlim(lo - p, hi + p)

def ci_bar(ax, y, lo, hi, horizontal=True, cap=0.11, lw=0.8):
    if horizontal:
        ax.plot([lo, hi], [y, y], color="k", lw=lw, solid_capstyle="butt")
        ax.plot([lo, lo], [y - cap, y + cap], color="k", lw=lw)
        ax.plot([hi, hi], [y - cap, y + cap], color="k", lw=lw)
    else:
        ax.plot([y, y], [lo, hi], color="k", lw=lw, solid_capstyle="butt")
        ax.plot([y - cap, y + cap], [lo, lo], color="k", lw=lw)
        ax.plot([y - cap, y + cap], [hi, hi], color="k", lw=lw)

def marker(ax, x, y, st, filled=None):
    f = st["f"] if filled is None else filled
    ax.plot([x], [y], ls="none", marker=st["m"], ms=MS * st["s"], mfc=FILL[f], mec="black", mew=0.8, zorder=3)

def forest(ax, ys, mu, lo, hi, styles):
    for y, m, l, h, st in zip(ys, mu, lo, hi, styles):
        if np.isnan(m):
            continue
        ci_bar(ax, y, l, h)
        marker(ax, m, y, st)

def sg(v, nd=1):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.{nd}f}".replace("-", MINUS)

def plain_log_ticks(ax, axis, ticks):
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    a.set_minor_locator(NullLocator())

def save(fig, name):
    fig.savefig(os.path.join(args.outdir, name + ".pdf"))
    fig.savefig(os.path.join(args.outdir, name + ".png"), dpi=args.dpi)
    if args.tiff:
        fig.savefig(os.path.join(args.outdir, name + ".tiff"), dpi=args.dpi, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("saved", name)

def safe(fn):
    try:
        fn()
    except Exception as e:
        import traceback
        print(f"[FAILED] {fn.__name__}: {e!r}")
        traceback.print_exc()

# ----------------------------------------------------------------------------- Fig. 1
def fig1():
    fig, ax = plt.subplots(figsize=(7.48, 3.6))
    ax.set_xlim(0, 10.3); ax.set_ylim(0, 4.65); ax.axis("off")

    def box(x, y, w, h, txt, fc="white", fs=6.6, lw=0.9, bold_first=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.09", fc=fc, ec="k", lw=lw))
        lines = txt.split("\n")
        if bold_first:
            ax.text(x + w / 2, y + h - 0.2, lines[0], ha="center", va="center", fontsize=fs + 0.4, fontweight="bold")
            ax.text(x + w / 2, y + (h - 0.34) / 2, "\n".join(lines[1:]), ha="center", va="center", fontsize=fs, linespacing=1.35)
        else:
            ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, linespacing=1.35)

    def arrow(p0, p1):
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", lw=0.9, color="k", shrinkA=0, shrinkB=0, mutation_scale=8))

    ax.text(1.4, 4.5, "Inputs", ha="center", fontsize=7.4, fontweight="bold")
    box(0.05, 3.2, 2.7, 1.15, "Fleet and degradation\n50 turbines, 11 component types\nWeibull wear, latent heterogeneity\nKijima type II imperfect repair", fc="0.94")
    box(0.05, 1.75, 2.7, 1.15, "Access model\n4-state sea-state Markov chain\nCTV / SOV limits, daily task cap\npriority-ordered task queue", fc="0.94")
    box(0.05, 0.3, 2.7, 1.15, "Evaluation design\n2-yr burn-in, 3-yr window\ncommon random numbers\ndisjoint training / test seeds", fc="0.94")
    ax.text(5.15, 4.5, "Computational protocol", ha="center", fontsize=7.4, fontweight="bold")
    box(3.55, 3.2, 3.2, 1.15, "Algorithm 1\nIso-reliability operating points\nfrontier interpolation, local-slope correction\ncost: |grid| \u00d7 training seeds \u00d7 one simulation", bold_first=True)
    box(3.55, 1.75, 3.2, 1.15, "Algorithm 2\nFrequency matching\nfixed-point calibration of calendar intervals\n\u2264 7 iterations \u00d7 10 replicates", bold_first=True)
    box(3.55, 0.3, 3.2, 1.15, "Algorithm 3\nAttribution and inference\n2 \u00d7 2 factorial contrasts, paired bootstrap\nO(B R) per contrast, B = 20,000, R = 50", bold_first=True)
    # single input bus -> Algorithm 1
    for yc in (3.775, 2.325, 0.875):
        ax.plot([2.78, 3.15], [yc, yc], color="k", lw=0.9)
    ax.plot([3.15, 3.15], [0.875, 3.775], color="k", lw=0.9)
    arrow((3.15, 3.775), (3.52, 3.775))
    arrow((5.15, 3.2), (5.15, 2.92)); arrow((5.15, 1.75), (5.15, 1.47))
    # policy design
    PX, PW = 7.45, 2.8
    ax.text(PX + PW / 2, 4.5, "Policy design", ha="center", fontsize=7.4, fontweight="bold")
    ax.add_patch(FancyBboxPatch((PX, 0.3), PW, 4.05, boxstyle="round,pad=0.015,rounding_size=0.09", fc="0.94", ec="k", lw=0.9))
    ax.text(PX + PW / 2, 4.05, "2 \u00d7 2 attribution design", ha="center", va="center", fontsize=7.0, fontweight="bold")
    cx1, cx2, cw = 8.98, 9.82, 0.8                                   # [v14] wider cells
    ax.text((cx1 + cx2) / 2, 3.68, "Learning of latent wear", ha="center", fontsize=6.4, style="italic")
    ax.text(cx1, 3.38, "none", ha="center", fontsize=6.4); ax.text(cx2, 3.38, "Bayesian", ha="center", fontsize=6.4)
    ax.text(PX + 0.13, 2.9, "Calendar", ha="left", va="center", fontsize=6.4, style="italic")
    ax.text(PX + 0.13, 2.12, "State\ntracking", ha="left", va="center", fontsize=6.4, style="italic", linespacing=1.1)
    for (cx, cy), t in {(cx1, 2.9): "TBM", (cx2, 2.9): "LTBM", (cx1, 2.12): "EAT", (cx2, 2.12): "BAY/BAYe"}.items():
        ax.add_patch(Rectangle((cx - cw / 2, cy - 0.24), cw, 0.48, fc="white", ec="k", lw=0.9))
        ax.text(cx, cy, t, ha="center", va="center", fontsize=5.5 if len(t) > 5 else 6.3, fontweight="bold")
    ax.plot([PX + 0.15, PX + PW - 0.15], [1.62, 1.62], color="k", lw=0.5)
    ax.text(PX + PW / 2, 1.35, "Reference policies", ha="center", fontsize=6.4, fontweight="bold")
    ax.text(PX + PW / 2, 0.88, "TBMm: frequency-matched calendar\nORA: oracle bound (not implementable)\nIDX, IDXL, PS, SEN: baselines",
            ha="center", va="center", fontsize=5.9, linespacing=1.4)
    arrow((PX - 0.03, 2.325), (6.82, 2.325))
    ax.text((PX + 6.82) / 2, 2.46, "policies", ha="center", fontsize=5.8, style="italic")
    save(fig, "Fig1_protocol")

# ----------------------------------------------------------------------------- Fig. 2
def fig2():
    d2 = Y2[(Y2.tag == "iso_vs_TBM") & (Y2.metric == "cost")]
    arms = ["EAT", "LTBM", "BAY", "BAYe", "ORA"]
    fig = plt.figure(figsize=(7.48, 5.3))
    gs = fig.add_gridspec(1, 4, left=0.23, right=0.995, top=0.93, bottom=0.60, wspace=0.14)
    gs2 = fig.add_gridspec(1, 2, left=0.13, right=0.995, top=0.38, bottom=0.17, wspace=0.34, width_ratios=[1.25, 1])
    ys = np.arange(len(arms)); top_axes = []
    for i, sc in enumerate(SC):
        ax = fig.add_subplot(gs[0, i]); top_axes.append(ax)
        d = d2[(d2.scenario == sc) & (d2.op == "q50")].set_index("arm").reindex(arms)
        forest(ax, ys, d["diff"].values, d.lo.values, d.hi.values, [PS[a] for a in arms])
        ax.axvline(0, ls="--", lw=0.6, color="k")
        pad_xlim(ax, d.lo.values, d.hi.values)
        ax.set_ylim(len(arms) - 0.45, -0.65)
        ax.set_yticks(ys); ax.set_yticklabels([PS[a]["lab"] for a in arms] if i == 0 else [])
        ax.tick_params(axis="y", length=0 if i else 2.4)
        ax.set_title(SCT[sc], pad=3); ax.set_xlabel(r"$\Delta$ cost vs. TBM (bn KRW)")
        ax.grid(axis="x", lw=0.3, color="0.85"); ax.set_axisbelow(True)
        if sc == "S0 homogeneous":      # ORA coincides with EAT in the homogeneous fleet
            ra = d.loc["ORA"]
            ax.annotate(r"ORA $\equiv$ EAT", xy=(ra["hi"], 4), xytext=(6, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=6, style="italic")
    # (b) oracle recovery
    axb = fig.add_subplot(gs2[0, 0])
    rec = Y3[Y3.op == "q50"]
    pols = ["EAT", "LTBM", "BAYe"]; scs = SC[1:]; w = 0.26; x = np.arange(3)
    sty = {"EAT": ("white", "////"), "LTBM": ("0.72", ""), "BAYe": ("0.28", "")}
    labs = {"EAT": "EAT (nominal)", "LTBM": "LTBM (learned calendar)", "BAYe": "BAYe (empirical Bayes)"}
    for k, p in enumerate(pols):
        q = rec[rec.policy == p].set_index("scenario").reindex(scs)
        axb.bar(x + (k - 1) * w, q.oracle_recovery, w, color=sty[p][0], hatch=sty[p][1], edgecolor="k", linewidth=0.6, label=labs[p])
        axb.errorbar(x + (k - 1) * w, q.oracle_recovery, yerr=[q.oracle_recovery - q.lo, q.hi - q.oracle_recovery],
                     fmt="none", ecolor="k", elinewidth=0.7, capsize=1.8)
    axb.axhline(0, color="k", lw=0.6); axb.axhline(1, color="k", lw=0.6, ls=":")
    axb.set_xticks(x); axb.set_xticklabels(["S1\nheterogeneous", "S2\nhet., faster wear", "S3\nstrong het."])
    axb.set_ylim(-0.2, 1.1); axb.set_ylabel("Oracle benefit recovered")
    axb.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24), handlelength=1.6, columnspacing=1.2)
    # (c) stability across reliability levels
    axc = fig.add_subplot(gs2[0, 1])
    ops = ["q25", "q50", "q75"]; mk = {"S1 hetero": "o", "S2 hetero+fast-wear": "s", "S3 strong-hetero": "^"}
    for sc in SC[1:]:
        for a, ls, col in (("BAYe", "-", "k"), ("LTBM", "--", "0.45")):
            dd = d2[(d2.scenario == sc) & (d2.arm == a)].set_index("op").reindex(ops)
            axc.plot(range(3), dd["diff"].values, ls=ls, color=col, lw=0.8, marker=mk[sc], ms=4.2,
                     mfc=("k" if a == "BAYe" else "white"), mec=col if a == "LTBM" else "k", mew=0.8)
    axc.axhline(0, ls="--", lw=0.6, color="k")
    axc.set_xticks(range(3)); axc.set_xticklabels(["q25", "q50", "q75"]); axc.set_xlim(-0.25, 2.25)
    axc.set_xlabel("Equal-reliability point (low \u2192 high failure level)")
    axc.set_ylabel(r"$\Delta$ cost vs. TBM (bn KRW)", labelpad=2)
    h = [Line2D([], [], color="k", ls="-", marker="D", ms=3.8, mfc="k", label="BAYe"),
         Line2D([], [], color="0.45", ls="--", marker="D", ms=3.8, mfc="white", mec="0.45", label="LTBM")] + \
        [Line2D([], [], color="k", ls="none", marker=m_, ms=4.2, mfc="white", label=SS[s_]) for s_, m_ in mk.items()]
    axc.legend(handles=h, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.30), columnspacing=0.9, handletextpad=0.3)
    axc.grid(axis="y", lw=0.3, color="0.85"); axc.set_axisbelow(True)
    panels(fig, [(top_axes[0], "(a)", True), (axb, "(b)", True), (axc, "(c)", True)])
    save(fig, "Fig2_main_results")

# ----------------------------------------------------------------------------- Fig. 3
def fig3():
    fig = plt.figure(figsize=(7.48, 5.7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.12, 1], hspace=1.05, wspace=0.55, left=0.09, right=0.99, top=0.95, bottom=0.2)
    ax = fig.add_subplot(gs[0, :])
    series = [("opt_vs_TBM", "EAT", "TBM", "EAT $-$ TBM  (apparent advantage)", "o", 0),
              ("opt_vs_TBM", "TBMm", "TBM", "TBMm $-$ TBM  (frequency + component differentiation)", "s", .5),
              ("opt_vs_TBMm", "EAT", "TBMm", "EAT $-$ TBMm  (residual trigger effect)", "D", 1)]
    for k, (tag, arm, vs, lab, m, f) in enumerate(series):
        d = Y2[(Y2.tag == tag) & (Y2.arm == arm) & (Y2.vs == vs) & (Y2.metric == "crit") & (Y2.op == "opt")].drop_duplicates("scenario").set_index("scenario").reindex(SC)
        xx = np.arange(4) + (k - 1) * 0.24
        for xi, lo, hi in zip(xx, d.lo, d.hi):
            ci_bar(ax, xi, lo, hi, horizontal=False, cap=0.05)
        ax.plot(xx, d["diff"], ls="none", marker=m, ms=5.0, mfc=FILL[f], mec="k", mew=0.8, label=lab, zorder=3)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xticks(range(4)); ax.set_xticklabels([SCT[s] for s in SC])
    ax.set_ylabel(r"$\Delta$ critical failures over 3 years")
    ax.grid(axis="y", lw=0.3, color="0.85"); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=6.8)
    y1 = Y1[Y1.op == "q50"].set_index(["scenario", "fam"])["cost"]
    bot = []
    for i, sc in enumerate(SC):
        a = fig.add_subplot(gs[1, i]); bot.append(a)
        nol = [y1[(sc, "TBM")], y1[(sc, "EAT")]]; lrn = [y1[(sc, "LTBM")], y1[(sc, "BAY")]]
        a.plot([0, 1], nol, ls="--", color="k", lw=0.9, marker="o", ms=4.8, mfc="white", mec="k", mew=0.8, label="No learning")
        a.plot([0, 1], lrn, ls="-", color="k", lw=0.9, marker="s", ms=4.6, mfc="k", mec="k", mew=0.8, label="Bayesian learning")
        a.set_xlim(-0.35, 1.35); a.set_xticks([0, 1]); a.set_xticklabels(["Calendar", "State\ntracking"])
        a.set_title(SCT[sc], fontsize=7.6, pad=3)
        if i == 0:
            a.set_ylabel("Adjusted cost, q50 (bn KRW)")
        e = Y4[(Y4.scenario == sc) & (Y4.op == "q50") & (Y4.learner == "BAY")].set_index("term")["effect"]
        a.set_xlabel("ST {}   L {}   ST\u00d7L {}".format(sg(e.get("State-tracking", np.nan)), sg(e.get("Learning", np.nan)),
                                                         sg(e.get("State-trackingxLearning", np.nan))), fontsize=6.2, labelpad=4)
        a.grid(axis="y", lw=0.3, color="0.85"); a.set_axisbelow(True)
    hh = bot[0].get_legend_handles_labels()
    fig.legend(*hh, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.0, "ST: state-tracking effect; L: learning effect; ST\u00d7L: interaction (bn KRW, q50, learner = BAY)",
             ha="center", va="bottom", fontsize=6.0, style="italic")
    panels(fig, [(ax, "(a)", True), (bot[0], "(b)", True)])
    save(fig, "Fig3_attribution")

# ----------------------------------------------------------------------------- Fig. 4
def fig4():
    order = ["EAT", "IDX", "IDXL", "PS", "LTBM", "BAYe", "SEN1.0", "SENc0.6", "SENb0.3", "SENc0.3", "SEN0.3", "ORA"]
    fig, axs = plt.subplots(1, 3, figsize=(7.48, 4.3), sharey=True,
                            gridspec_kw=dict(wspace=0.08, left=0.26, right=0.82, top=0.93, bottom=0.15))
    ys = np.arange(len(order))
    for j, (ax, sc) in enumerate(zip(axs, SC[1:])):
        d = Y7[Y7.scenario == sc].set_index("policy").reindex(order)
        ax.axhspan(5.5, 10.5, facecolor="0.9", edgecolor="none", zorder=0)
        for yy in (1.5, 5.5, 10.5):
            ax.axhline(yy, color="0.35", lw=0.5, ls=":", zorder=1)
        forest(ax, ys, d.diff_vs_TBM.values, d.lo.values, d.hi.values, [PS[o] for o in order])
        ax.axvline(0, ls="--", lw=0.6, color="k")
        ax.axvline(d.loc["BAYe", "diff_vs_TBM"], ls="-.", lw=0.6, color="0.3")
        pad_xlim(ax, d.lo.values, d.hi.values)
        ax.set_ylim(len(order) - 0.45, -0.65)
        ax.set_title(SCT[sc], pad=3)
        ax.set_xlabel(r"$\Delta$ cost vs. TBM (bn KRW)")      # ", q50" goes to the caption
        ax.grid(axis="x", lw=0.3, color="0.85"); ax.set_axisbelow(True)
        if j > 0:
            ax.tick_params(axis="y", length=0, labelleft=False)     # NOT set_yticklabels([]) -> would blank shared axis
    axs[0].set_yticks(ys); axs[0].set_yticklabels([PS[o]["lab"] for o in order])
    for yc, t in ((0.5, "no wear\ninformation"), (3.5, "learns wear from\nfailure histories"),
                  (8.0, "degradation sensor\n(optimistic upper\nbound)"), (11.0, "oracle")):
        axs[2].text(1.03, yc, t, transform=axs[2].get_yaxis_transform(), ha="left", va="center", fontsize=6.2, style="italic", linespacing=1.15)
    h = [Line2D([], [], color="0.3", ls="-.", lw=0.8, label="BAYe estimate"),
         Line2D([], [], color="k", ls="--", lw=0.8, label="TBM (zero)")]
    fig.legend(handles=h, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.54, 0.0))
    panels(fig, [(axs[0], "(a)", True), (axs[1], "(b)", False), (axs[2], "(c)", False)])
    save(fig, "Fig4_information_ladder")

# ----------------------------------------------------------------------------- Fig. 5
SETLAB = {"baseline": "Baseline (correct model)", "shape_low (beta=2.0)": r"Weibull shape $\beta$=2.0",
          "shape_high (beta=3.2)": r"Weibull shape $\beta$=3.2", "repair_wear_accel (+5%*RF)": "Repair-induced wear, +5% RF",
          "repair_wear_accel (+10%*RF)": "Repair-induced wear, +10% RF", "repair_wear_accel (+25%*RF)": "Repair-induced wear, +25% RF",
          "renewal_on_replace": "Wear redrawn at replacement", "time_drift (+60%)": "Wear drift, +60%",
          "bimodal_heterog (20% weak)": "Bimodal heterogeneity", "heavy_tail t3": r"Heavy-tailed heterogeneity ($t_3$)"}

def fig5():
    fig = plt.figure(figsize=(7.48, 7.1))
    gs = fig.add_gridspec(1, 2, left=0.20, right=0.99, top=0.965, bottom=0.55, wspace=0.07)
    gs2 = fig.add_gridspec(1, 2, left=0.11, right=0.99, top=0.37, bottom=0.17, wspace=0.42)
    d5 = Y5[Y5.vs == "TBM"]
    order = list(dict.fromkeys(d5.setting)); ys = np.arange(len(order)); top_axes = []
    for j, sc in enumerate(["S1 hetero", "S3 strong-hetero"]):
        ax = fig.add_subplot(gs[0, j]); top_axes.append(ax); d = d5[d5.scenario == sc]
        valid = d.drop_duplicates("setting").set_index("setting")["valid_iso"].reindex(order)
        vv = d[d.valid_iso & d.policy.isin(["BAYe", "LTBM"])]
        pad = 0.1 * (vv.hi.max() - vv.lo.min())
        xl = (min(vv.lo.min(), 0) - pad, max(vv.hi.max(), 0) + pad)
        rng = xl[1] - xl[0]
        for i, s_ in enumerate(order):
            if not valid[s_]:
                ax.axhspan(i - 0.5, i + 0.5, facecolor="0.95", edgecolor="0.55", hatch="////", lw=0, zorder=0)
            for off, p in ((-0.17, "LTBM"), (0.17, "BAYe")):
                r = d[(d.setting == s_) & (d.policy == p)]
                if r.empty:
                    continue
                r = r.iloc[0]; mu, lo, hi = r["diff"], r.lo, r.hi
                f = 1 if valid[s_] else 0
                if mu > xl[1] - 0.04 * rng or mu < xl[0] + 0.04 * rng:      # off-scale: marker at edge + value, no CI
                    xe = xl[1] - 0.05 * rng if mu > 0 else xl[0] + 0.05 * rng
                    marker(ax, xe, i + off, dict(PS[p]), filled=f)
                    ax.text(xe - 0.025 * rng if mu > 0 else xe + 0.025 * rng, i + off, f"{mu:+.0f} \u2192" if mu > 0 else f"\u2190 {mu:+.0f}",
                            ha="right" if mu > 0 else "left", va="center", fontsize=5.6)
                    continue
                lo_, hi_ = np.clip([lo, hi], xl[0] + 0.01 * rng, xl[1] - 0.04 * rng)
                ci_bar(ax, i + off, lo_, hi_, cap=0.08)
                marker(ax, mu, i + off, dict(PS[p]), filled=f)
        ax.axvline(0, ls="--", lw=0.6, color="k")
        ax.set_xlim(xl); ax.set_ylim(len(order) - 0.45, -0.6)
        ax.set_yticks(ys); ax.set_yticklabels([SETLAB.get(s_, s_) for s_ in order] if j == 0 else [])
        ax.tick_params(axis="y", length=0 if j else 2.4)
        ax.set_title(SCT[sc], fontsize=7.8, pad=3)
        ax.set_xlabel(r"$\Delta$ cost vs. TBM (bn KRW, q50)")
        ax.grid(axis="x", lw=0.3, color="0.85"); ax.set_axisbelow(True)
    hh = [Line2D([], [], ls="none", marker="D", ms=4.2, mfc="k", mec="k", label="BAYe"),
          Line2D([], [], ls="none", marker="^", ms=4.8, mfc="k", mec="k", label="LTBM"),
          Line2D([], [], ls="none", marker="D", ms=4.2, mfc="w", mec="k", label="frame not applicable (open, hatched row)")]
    fig.legend(handles=hh, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.6, 0.455))
    # (c) capacity
    axc = fig.add_subplot(gs2[0, 0])
    ser = [("S1 hetero", "ctv", "o", "-"), ("S3 strong-hetero", "ctv", "s", "-"),
           ("S1 hetero", "adaptive", "o", "--"), ("S3 strong-hetero", "adaptive", "s", "--")]
    for sc, mo, m, ls in ser:
        d = Y9[(Y9.scenario == sc) & (Y9["mode"] == mo)].sort_values("cap")
        axc.plot(d.cap, d.cost_BAYe, ls=ls, color="k", lw=0.7, zorder=2)
        for c_, mu, lo, hi in zip(d.cap, d.cost_BAYe, d.cost_BAYe_lo, d.cost_BAYe_hi):
            ci_bar(axc, c_, lo, hi, horizontal=False, cap=c_ * 0.06, lw=0.7)
            sig = (hi < 0) or (lo > 0)
            axc.plot([c_], [mu], ls="none", marker=m, ms=3.9, mfc="k" if sig else "w", mec="k", mew=0.75, zorder=3)
    axc.axhline(0, ls="--", lw=0.6, color="k"); axc.axvline(4, ls=":", lw=0.7, color="0.35")
    axc.text(4.2, 0.97, "default\ncap = 4", transform=axc.get_xaxis_transform(), fontsize=5.8, va="top", ha="left", style="italic")
    axc.set_xscale("log"); caps = sorted(Y9.cap.unique()); plain_log_ticks(axc, "x", [float(c) for c in caps])
    axc.set_xlabel("CTV daily task capacity"); axc.set_ylabel("BAYe $-$ TBM total cost (bn KRW)")
    axc.grid(axis="y", lw=0.3, color="0.85"); axc.set_axisbelow(True)
    h = [Line2D([], [], color="k", ls="-", marker="o", ms=3.9, mfc="k", label="S1, CTV only"),
         Line2D([], [], color="k", ls="-", marker="s", ms=3.9, mfc="k", label="S3, CTV only"),
         Line2D([], [], color="k", ls="--", marker="o", ms=3.9, mfc="k", label="S1, CTV + SOV"),
         Line2D([], [], color="k", ls="--", marker="s", ms=3.9, mfc="k", label="S3, CTV + SOV"),
         Line2D([], [], color="k", ls="none", marker="o", ms=3.9, mfc="w", label="95% CI includes 0 (open)")]
    axc.legend(handles=h, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.27), columnspacing=1.0, handletextpad=0.4)
    # (d) scale
    axd = fig.add_subplot(gs2[0, 1])
    for sc, m in (("S1 hetero", "o"), ("S3 strong-hetero", "s")):
        d = Y10[(Y10.scenario == sc) & (Y10.policy == "BAYe")]
        for caps_, ls, filled in (("scaled", "-", True), ("fixed", "--", False)):
            q = d[(d.caps == caps_) & (~d.narrow_range)].sort_values("N")
            if q.empty:
                continue
            fac = (q.pct / q["diff"].replace(0, np.nan)).values
            axd.plot(q.N, q.pct, ls=ls, color="k", lw=0.7, zorder=2)
            for n_, mu, lo, hi in zip(q.N, q.pct, q.lo.values * fac, q.hi.values * fac):
                if np.isnan(lo):
                    continue
                ci_bar(axd, n_, lo, hi, horizontal=False, cap=n_ * 0.05, lw=0.7)
                axd.plot([n_], [mu], ls="none", marker=m, ms=3.9, mfc="k" if filled else "w", mec="k", mew=0.75, zorder=3)
    axd.axhline(0, ls="--", lw=0.6, color="k")
    axd.set_xscale("log"); plain_log_ticks(axd, "x", [float(n) for n in sorted(Y10.N.unique())])
    axd.set_xlabel("Number of turbines N"); axd.set_ylabel("BAYe vs. TBM (% of TBM cost)")
    axd.grid(axis="y", lw=0.3, color="0.85"); axd.set_axisbelow(True)
    h = [Line2D([], [], color="k", ls="-", marker="o", ms=3.9, mfc="k", label="S1, capacity scaled with N"),
         Line2D([], [], color="k", ls="-", marker="s", ms=3.9, mfc="k", label="S3, capacity scaled with N"),
         Line2D([], [], color="k", ls="--", marker="o", ms=3.9, mfc="w", label="S1, capacity fixed (valid N only)"),
         Line2D([], [], color="k", ls="--", marker="s", ms=3.9, mfc="w", label="S3, capacity fixed (valid N only)")]
    axd.legend(handles=h, frameon=False, ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.27), handletextpad=0.4)
    panels(fig, [(top_axes[0], "(a)", True), (top_axes[1], "(b)", False), (axc, "(c)", True), (axd, "(d)", True)])
    save(fig, "Fig5_robustness_boundaries")

# ----------------------------------------------------------------------------- Appendix
def figA1():
    fig, axs = plt.subplots(1, 4, figsize=(7.48, 3.2), gridspec_kw=dict(wspace=0.42, bottom=0.27))
    pols = ["TBM", "EAT", "LTBM", "BAYe", "ORA"]
    LS = {"TBM": ("-", "0.45"), "EAT": ("--", "k"), "LTBM": ("-.", "k"), "BAYe": (":", "k"), "ORA": ("-", "k")}
    for ax, sc in zip(axs, SC):
        for p in pols:
            d = Y1[(Y1.scenario == sc) & (Y1.fam == p) & Y1.op.isin(["q25", "q50", "q75"])].sort_values("crit")
            ls, col = LS[p]; st = PS[p]
            ax.plot(d.crit, d.cost, ls=ls, color=col, lw=0.75)
            ax.plot(d.crit, d.cost, ls="none", marker=st["m"], ms=3.6 * st["s"] * (0.85 if p == "ORA" else 1), mfc=FILL[st["f"]], mec=col, mew=0.7)
        ax.set_title(SCT[sc], fontsize=7.4, pad=3); ax.set_xlabel("Critical failures (3 yr)")
        ax.grid(lw=0.3, color="0.88"); ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    axs[0].set_ylabel("Adjusted cost (bn KRW)")
    # S0: ORA is identical to EAT, so the EAT curve sits underneath the ORA curve
    axs[0].text(0.97, 0.97, "EAT hidden behind ORA\n(identical in S0)", transform=axs[0].transAxes,
                ha="right", va="top", fontsize=5.6, style="italic", linespacing=1.15)
    h = [Line2D([], [], ls=LS[p][0], color=LS[p][1], lw=0.9, marker=PS[p]["m"], ms=3.8, mfc=FILL[PS[p]["f"]], mec=LS[p][1], label=PS[p]["lab"]) for p in pols]
    fig.legend(handles=h, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.02), columnspacing=1.0, handletextpad=0.4, fontsize=6.4)
    panels(fig, [(a, "(" + "abcd"[i] + ")", i == 0) for i, a in enumerate(axs)])
    save(fig, "FigA1_cost_reliability_view")

def figA2():
    fig, axs = plt.subplots(1, 3, figsize=(7.48, 2.9), sharey=True, gridspec_kw=dict(wspace=0.12, bottom=0.27))
    pairs = [("BAYe-TBM", "-", "k", 1.1, "BAYe $-$ TBM"), ("BAYe-LTBM", "--", "k", 1.1, "BAYe $-$ LTBM"),
             ("BAYe-IDXL", "-.", "0.45", 1.1, "BAYe $-$ IDXL"), ("EAT-TBM", ":", "k", 1.4, "EAT $-$ TBM")]
    for j, (ax, sc) in enumerate(zip(axs, SC[1:])):
        for pr, ls, col, lw, lab in pairs:
            v = np.sort(Y6b[(Y6b.scenario == sc) & (Y6b.pair == pr)]["diff"].values)
            if len(v):
                ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", ls=ls, color=col, lw=lw, label=lab)
        ax.axvline(0, ls="--", lw=0.6, color="k"); ax.set_title(SCT[sc], pad=3)
        ax.set_xlabel(r"$\Delta$ cost (bn KRW), random re-weighting"); ax.grid(lw=0.3, color="0.88"); ax.set_axisbelow(True)
        if j > 0:
            ax.tick_params(axis="y", labelleft=False)
    axs[0].set_ylabel("Empirical CDF over 400 cost draws")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02), handlelength=3.2)
    panels(fig, [(a, "(" + "abc"[i] + ")", i == 0) for i, a in enumerate(axs)])
    save(fig, "FigA2_cost_reweighting")

def figA3():
    fams = ["TBM3", "EAT2", "BAYe2"]; scs = ["S1 hetero", "S3 strong-hetero"]
    fig, axs = plt.subplots(1, 2, figsize=(7.48, 3.2), gridspec_kw=dict(wspace=0.75, left=0.14, right=0.99, bottom=0.25))

    def is_tie(r):
        """multi-parameter search selected the same policy as one-parameter tuning (CRN -> identical outcome)"""
        try:
            return bool(np.isclose(r.cost_rich, r.cost_one_param, rtol=0, atol=1e-9)
                        and np.isclose(r.crit_rich, r.crit_one, rtol=0, atol=1e-9)
                        and abs(r.hi - r.lo) < 1e-9)
        except Exception:
            return False

    # ---- (a) multi- vs one-parameter tuning
    ax = axs[0]; ys, labs, rows = [], [], []; y = 0
    for sc in scs:
        for f_ in fams:
            r = Y11[(Y11.scenario == sc) & (Y11.family == f_)]
            if r.empty:
                y += 1; continue
            rows.append(r.iloc[0]); ys.append(y); labs.append(f"{SS[sc]}: {f_}"); y += 1
        y += 0.6
    st = {"TBM3": PS["TBM"], "EAT2": PS["EAT"], "BAYe2": PS["BAYe"]}
    any_tie = False
    for yy, r in zip(ys, rows):
        if is_tie(r):
            any_tie = True
            marker(ax, r["diff"], yy, st[r.family])
            ax.plot([r["diff"]], [yy], ls="none", marker="o", ms=MS * 2.0, mfc="none", mec="k", mew=0.7, zorder=4)
            ax.annotate("tie: same optimum\nas one-param.", xy=(r["diff"], yy), xytext=(10, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=5.8, style="italic", linespacing=1.15)
        else:
            ci_bar(ax, yy, r.lo, r.hi)
            marker(ax, r["diff"], yy, st[r.family])
    ax.axvline(0, ls="--", lw=0.6, color="k"); ax.set_yticks(ys); ax.set_yticklabels(labs); ax.invert_yaxis()
    lo_all = [r.lo for r in rows]; hi_all = [r.hi for r in rows]
    pad_xlim(ax, lo_all, hi_all, pad=0.08)
    xl = ax.get_xlim(); ax.set_xlim(xl[0], xl[1] + 0.22 * (xl[1] - xl[0]))   # room for the tie annotation
    ax.set_xlabel(r"$\Delta$ cost: multi-param. $-$ one-param. tuning (bn KRW)")
    ax.grid(axis="x", lw=0.3, color="0.88"); ax.set_axisbelow(True)
    if any_tie:
        ax.text(0.0, -0.30, "ringed marker: both searches chose the identical\npolicy (\u0394 = 0 by construction)",
                transform=ax.transAxes, ha="left", va="top", fontsize=5.8, style="italic", linespacing=1.25)
    la = ax

    # ---- (b) richly tuned policies against each other
    ax = axs[1]; ys, labs, rows = [], [], []; y = 0
    for sc in scs:
        for p_ in ["BAYe2 - TBM3", "BAYe2 - EAT2", "EAT2 - TBM3"]:
            r = Y11[(Y11.scenario == sc) & (Y11.family == p_)]
            if r.empty:
                y += 1; continue
            rows.append(r.iloc[0]); ys.append(y); labs.append(f"{SS[sc]}: {p_.replace(' - ', ' $-$ ')}"); y += 1
        y += 0.6
    neutral = dict(m="o", f=1, s=0.9)
    forest(ax, ys, [r["diff"] for r in rows], [r.lo for r in rows], [r.hi for r in rows], [neutral] * len(rows))
    ax.axvline(0, ls="--", lw=0.6, color="k"); ax.set_yticks(ys); ax.set_yticklabels(labs); ax.invert_yaxis()
    pad_xlim(ax, [r.lo for r in rows], [r.hi for r in rows], pad=0.06)
    ax.set_xlabel(r"$\Delta$ cost between richly tuned policies (bn KRW)")
    ax.grid(axis="x", lw=0.3, color="0.88"); ax.set_axisbelow(True)
    panels(fig, [(la, "(a)", True), (ax, "(b)", True)])
    save(fig, "FigA3_tuning_budget")

def figA4():
    fig = plt.figure(figsize=(7.48, 3.6))
    gs = fig.add_gridspec(1, 3, wspace=0.62, width_ratios=[1, 1, 1.3], left=0.08, right=0.99, top=0.9, bottom=0.27)
    ax = fig.add_subplot(gs[0, 0]); a_ax = ax
    for p, m, f in (("TBM", "s", 0), ("EAT", "o", 0), ("BAYe", "D", 1)):
        ax.plot(Y15.N, Y15[p], color="k", lw=0.8, ls={"TBM": "-", "EAT": "--", "BAYe": "-"}[p], marker=m, ms=4, mfc=FILL[f], mec="k", mew=0.8, label=p)
    n0 = Y15.N.iloc[0]; y0 = Y15["BAYe"].iloc[0]
    ax.plot(Y15.N, y0 * Y15.N / n0, color="0.4", ls=":", lw=0.8, label=r"$\propto N$ (reference)")
    ax.set_xscale("log"); ax.set_yscale("log")
    plain_log_ticks(ax, "x", [float(n) for n in Y15.N])
    allv = np.concatenate([Y15[p].values for p in ("TBM", "EAT", "BAYe")])
    plain_log_ticks(ax, "y", [t for t in (0.1, 0.2, 0.5, 1, 2, 5) if allv.min() * 0.7 <= t <= max(allv.max(), y0 * Y15.N.iloc[-1] / n0) * 1.4])
    ax.set_xlabel("Number of turbines N"); ax.set_ylabel("Seconds per simulation (1 process)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handletextpad=0.4, columnspacing=0.9)
    ax.grid(which="major", lw=0.25, color="0.9")
    ax = fig.add_subplot(gs[0, 1]); b_ax = ax
    scs = ["S1 hetero", "S2 hetero+fast-wear", "S3 strong-hetero"]; w = 0.34; x = np.arange(3)
    for k, (p, fc, ht) in enumerate((("BAY", "white", "////"), ("BAYe", "0.3", ""))):
        q = Y14[Y14.policy == p].set_index("scenario").reindex(scs)
        ax.bar(x + (k - 0.5) * w, q.crit_span_in_band, w, color=fc, hatch=ht, edgecolor="k", lw=0.6, label=p)
        for xi, v, n in zip(x + (k - 0.5) * w, q.crit_span_in_band, q.n_grid_within_1bn):
            if not np.isfinite(v) or not np.isfinite(n):
                continue
            if v > 0:
                ax.text(xi, v + 2, f"{int(n)}", ha="center", va="bottom", fontsize=5.8)
            else:   # [v14] zero span: visible baseline stub + dagger (short label, no overlap)
                ax.plot([xi - w / 2, xi + w / 2], [0, 0], color="k", lw=1.8, solid_capstyle="butt", zorder=3)
                ax.text(xi, 2, f"{int(n)}\u2020", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(x); ax.set_xticklabels([SS[s] for s in scs]); ax.set_ylabel("Failure range within 1 bn of cost optimum")
    ax.set_xlabel("numbers above bars: grid points within 1 bn\n\u2020 span 0 = single grid point", fontsize=6.0, style="italic", labelpad=3)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, handletextpad=0.4)
    ax = fig.add_subplot(gs[0, 2]); c_ax = ax
    ys, labs, rows = [], [], []; y = 0
    for sc in ["S1 hetero", "S3 strong-hetero"]:
        for pol_ in ["TBM", "EAT", "LTBM", "BAYe"]:
            for r in Y13[(Y13.scenario == sc) & (Y13.policy == pol_)].itertuples():
                rows.append(r); ys.append(y); labs.append(f"{SS[sc]} {pol_} {r.displaced_target.replace(' failures', '').replace('-', MINUS)}"); y += 1
        y += 0.7
    for yy, r in zip(ys, rows):
        ci_bar(ax, yy, r.lo, r.hi, cap=0.18)
        ax.plot([r.adj_cost_diff], [yy], ls="none", marker="o", ms=3.8, mfc="k" if not r.covers_zero else "w", mec="k", mew=0.75, zorder=3)
    ax.axvline(0, ls="--", lw=0.6, color="k"); ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=5.8); ax.invert_yaxis()
    ax.set_xlabel("Slope-corrected cost bias (bn KRW)"); ax.grid(axis="x", lw=0.3, color="0.88"); ax.set_axisbelow(True)
    ax.text(0.5, -0.30, "row label: scenario, policy, displaced failure target\nopen marker: 95% CI includes 0",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.8, style="italic", linespacing=1.3)
    panels(fig, [(a_ax, "(a)", True), (b_ax, "(b)", True), (c_ax, "(c)", True)])
    save(fig, "FigA4_computation_diagnostics")

if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, figA1, figA2, figA3, figA4):
        safe(f)
    print("done ->", os.path.abspath(args.outdir))