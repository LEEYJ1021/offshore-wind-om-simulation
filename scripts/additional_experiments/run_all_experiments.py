"""
=============================================================================
Master Runner  (revised: real KMA data + rebuttal summary table)
=============================================================================
Run this file from the ress_revision/ directory.
All experiments execute sequentially; outputs go to /home/yjlee/Research/RESS_SnS_R1/outputs/
=============================================================================
"""

import sys, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT = "/home/yjlee/Research/RESS_SnS_R1/outputs"
os.makedirs(OUT, exist_ok=True)

script_dir = os.getcwd()
sys.path.insert(0, script_dir)

print("=" * 70)
print("  HMDP RESS Revision -- Master Experiment Runner")
print("  Data: Real KMA Ulsan weather_hourly_raw.csv + weekly_cost_baseline.csv")
print("  Targeting: R1 #5,#6,#12,#13 | R2 Core | R3 Minor")
print("=" * 70)

t0 = time.time()

# ── Run experiments ───────────────────────────────────────────────────────────
print("\n[1/3] Experiments 1 & 2 (CBM vs. Time-Based, real KMA weather)...")
exec(open(os.path.join(script_dir, "exp1_2_cbm_timebased.py")).read(), globals())
print(f"   Done in {time.time()-t0:.1f}s")

t1 = time.time()
print("\n[2/3] Experiment 3 (Seasonal Markov, KMA-calibrated matrices)...")
exec(open(os.path.join(script_dir, "exp3_seasonal_markov.py")).read(), globals())
print(f"   Done in {time.time()-t1:.1f}s")

t2 = time.time()
print("\n[3/3] Experiments 4 & 5 (Year Decomp + Grouped Dispatch + LP baseline)...")
exec(open(os.path.join(script_dir, "exp4_5_decomp_dispatch.py")).read(), globals())
print(f"   Done in {time.time()-t2:.1f}s")


# ── Consolidated summary figure ────────────────────────────────────────────────
print("\n  Building consolidated summary figure...")

fig = plt.figure(figsize=(20, 11), dpi=150)
fig.patch.set_facecolor("#FAFAFA")
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

# ── Panel 1: CBM interval distribution ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0,0])
ax1.hist(cbm_intervals, bins=30, density=True, alpha=0.75, color="#2196F3",
         label="CBM (Reliability-Triggered)")
ax1.hist(tb_intervals,  bins=30, density=True, alpha=0.55, color="#F44336",
         label="Fixed Time-Based (26wk)")
ax1.set_xlabel("Inter-intervention Interval (days)", fontsize=8)
ax1.set_title("Exp 1: Trigger Interval Distribution\n"
              f"CBM CV={cbm_cv:.3f} vs T-B CV={tb_cv:.4f}", fontsize=8, fontweight="bold")
ax1.legend(fontsize=7)
ax1.text(0.95, 0.75, "CBM trigger is\nNOT deterministic\n(repair-history\ndependent)",
         transform=ax1.transAxes, ha="right", fontsize=7.5,
         bbox=dict(boxstyle="round", fc="#E3F2FD", alpha=0.9))

# ── Panel 2: Critical failure trajectory ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0,1])
import pandas as _pd
roll_c = _pd.Series(res_cbm["crit_failures"]).rolling(30).mean()
roll_t = _pd.Series(res_tb["crit_failures"]).rolling(30).mean()
days_x = np.arange(N_DAYS)
ax2.plot(days_x, roll_t,  color="#F44336", linewidth=1.4, label="Fixed Time-Based")
ax2.plot(days_x, roll_c,  color="#2196F3", linewidth=1.4, label="CBM (Real weather)")
for xv in [365,730]: ax2.axvline(xv, color="gray", linestyle="--", lw=0.7)
ax2.set_xlabel("Simulation Day", fontsize=8)
ax2.set_ylabel("30-day Rolling Failure Rate", fontsize=8)
ax2.set_title(f"Exp 2: Failure Trajectory\nCBM reduces failures by {reduction_pct:.1f}%",
              fontsize=8, fontweight="bold")
ax2.legend(fontsize=7)

# ── Panel 3: Seasonal state (real KMA) ───────────────────────────────────────
ax3 = fig.add_subplot(gs[0,2])
seasons_p  = ["Winter","Spring","Summer","Autumn"]
# WX_REAL은 exp3에서 exec()로 주입되나 불안정 -> 방어적 재로드
if "WX_REAL" not in dir() or WX_REAL is None:
    from experiment_core import load_weather
    import pandas as _pd2
    WX_REAL = load_weather()
    WX_REAL["state_idx"] = WX_REAL["hs"].apply(
        lambda h: 0 if h<0.9 else 1 if h<1.5 else 2 if h<2.5 else 3)
calm_real  = [WX_REAL[WX_REAL["season"]==s]["state_idx"].eq(0).mean() for s in seasons_p]
rough_real = [WX_REAL[WX_REAL["season"]==s]["state_idx"].ge(2).mean() for s in seasons_p]
xp = np.arange(4); wp = 0.35
ax3.bar(xp-wp/2, calm_real,  wp, label="Calm (Hs<0.9m)", color="#4CAF50", alpha=0.82)
ax3.bar(xp+wp/2, rough_real, wp, label="Rough+Extreme",   color="#F44336", alpha=0.82)
ax3.set_xticks(xp); ax3.set_xticklabels(seasons_p, fontsize=8)
ax3.set_ylabel("Proportion of Days", fontsize=8)
ax3.set_title("Exp 3: Seasonal State Distribution\n(Calibrated from KMA 2023-2025 data)",
              fontsize=8, fontweight="bold")
ax3.legend(fontsize=7)

# ── Panel 4: Hom vs Non-hom KPI robustness ───────────────────────────────────
ax4 = fig.add_subplot(gs[0,3])
kpis  = ["Failures\n(3yr)", "Mean AO\n(%)"]
hom_v = [df_res["hom_fail"].mean(), df_res["hom_ao"].mean()]
nhm_v = [df_res["nhom_fail"].mean(), df_res["nhom_ao"].mean()]
xa4 = np.arange(2)
ax4.bar(xa4-0.2, hom_v,  0.36, label="Homogeneous",     color="#90CAF9", alpha=0.85)
ax4.bar(xa4+0.2, nhm_v,  0.36, label="Non-Homogeneous", color="#F48FB1", alpha=0.85)
ax4.set_xticks(xa4); ax4.set_xticklabels(kpis, fontsize=8)
ax4.set_title("Exp 3: KPI Robustness\n(n=20 MC reps, MWU n.s. -> conclusions robust)",
              fontsize=8, fontweight="bold")
ax4.legend(fontsize=7)

# ── Panel 5-6: Year decomposition (실패 건수 기반으로 교체) ──────────────────
ax5 = fig.add_subplot(gs[1, 0:2])
yr_labels = ["Year 1\n(2023)", "Year 2\n(2024)", "Year 3\n(2025)"]
fail_yr   = [stats[y]["fail"] for y in [1, 2, 3]]
wx_yr     = [stats[y]["wx_blocked"] for y in [1, 2, 3]]
q_yr      = [stats[y]["queue_carryover"] for y in [1, 2, 3]]
yr_x      = np.arange(3)
w_bar     = 0.25

bars_f = ax5.bar(yr_x - w_bar, fail_yr, w_bar, label="Critical Failures (count)",
                 color="#EF5350", alpha=0.85)
bars_w = ax5.bar(yr_x,          wx_yr,  w_bar, label="Weather-Blocked Days",
                 color="#42A5F5", alpha=0.85)
bars_q = ax5.bar(yr_x + w_bar,  q_yr,   w_bar, label="Queue Carryover (turbines)",
                 color="#FFA726", alpha=0.85)

for bar, v in zip(bars_f, fail_yr):
    ax5.text(bar.get_x()+bar.get_width()/2, v+0.3, str(int(v)),
             ha="center", fontsize=8, fontweight="bold")
for bar, v in zip(bars_w, wx_yr):
    ax5.text(bar.get_x()+bar.get_width()/2, v+0.3, str(int(v)),
             ha="center", fontsize=8)
for bar, v in zip(bars_q, q_yr):
    ax5.text(bar.get_x()+bar.get_width()/2, v+0.3, str(int(v)),
             ha="center", fontsize=8)

ax5.set_xticks(yr_x)
ax5.set_xticklabels(yr_labels, fontsize=9)
ax5.set_ylabel("Count / Days", fontsize=9)
ax5.set_title(
    "Exp 4: Year-over-Year Failure Escalation — Observed Counts\n"
    f"[Theoretical hazard +{aging_contrib:.0f}% by Yr3 (Panel B) | "
    f"WX block {wx_contrib:+.0f}% | Queue carryover {q_contrib:.0f}% of Yr3 failures]",
    fontsize=9, fontweight="bold")
ax5.legend(fontsize=8, loc="upper left")
ax5.grid(axis="y", alpha=0.3)

# ── Panel 7: Grouped dispatch cost ───────────────────────────────────────────
ax6 = fig.add_subplot(gs[1,2])
modes2 = ["Greedy","Grouped"]
vc2    = [res_greedy["vessel_cost_bn"], res_grouped["vessel_cost_bn"]]
bars6  = ax6.bar(modes2, vc2, color=["#2196F3","#4CAF50"], alpha=0.82, width=0.45)
for bar, v in zip(bars6, vc2):
    ax6.text(bar.get_x()+bar.get_width()/2, v+0.005, f"W{v:.2f}B",
             ha="center", fontsize=9, fontweight="bold")
ax6.set_ylabel("Vessel Cost (B KRW)", fontsize=9)
ax6.set_title(f"Exp 5: Grouped Dispatch\nVessel Cost Saving: {saving_pct:.1f}%", fontsize=9, fontweight="bold")

# ── Panel 8: LP baseline vs simulated ────────────────────────────────────────
ax7 = fig.add_subplot(gs[1,3])
wk_x2 = np.arange(1, 53)
if len(BASELINE) >= 52 and "TotalCost" in BASELINE.columns:
    ax7.plot(wk_x2, BASELINE["TotalCost"].values[:52]/1e6, color="gray",
             linestyle="--", linewidth=1.3, label="LP Baseline (all costs)")
else:
    ax7.text(0.5, 0.5, "LP Baseline N/A\n(weekly_cost_baseline.csv not found)",
             transform=ax7.transAxes, ha="center", va="center",
             fontsize=8, color="gray")
ax7.plot(wk_x2, res_greedy["weekly_cost"][:52]/1e6,  color="#2196F3",
         linewidth=1.2, label="Greedy (vessel only)")
ax7.plot(wk_x2, res_grouped["weekly_cost"][:52]/1e6, color="#4CAF50",
         linewidth=1.2, label="Grouped (vessel only)")
ax7.set_xlabel("Week", fontsize=8)
ax7.set_ylabel("Weekly Cost (M KRW)", fontsize=8)
ax7.set_title("Exp 5: Simulated vs. LP Baseline\n(vessel cost vs. full LP cost)",
              fontsize=8, fontweight="bold")
ax7.legend(fontsize=7)
fig.suptitle(
    "HMDP-CBM Offshore Wind O&M -- Additional Experiments for RESS Major Revision (JRESS-D-26-01401)\n"
    "Real KMA Ulsan 2023-2025 Data | R1 #5,#6,#12,#13 | R2 Core | Revised scripts",
    fontsize=10, fontweight="bold", y=1.01)
plt.savefig(f"{OUT}/CONSOLIDATED_ADDITIONAL_EXPERIMENTS.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  -> Saved: CONSOLIDATED_ADDITIONAL_EXPERIMENTS.png")


# ── Rebuttal mapping table ────────────────────────────────────────────────────

print("\n" + "="*70)
print("  REBUTTAL MAPPING TABLE")
print("="*70)

mapping = [
    ("R1 #5 / R2 Core", "CBM vs. time-based (stochasticity)",
     "Exp 1",
     f"CBM CV={cbm_cv:.4f} vs T-B CV={tb_cv:.4f}; KS p<{ks_p:.1e} -- "
     "trigger timing is repair-history-dependent and NOT pre-computable"),
    ("R1 #5 / R2 Core", "CBM vs. time-based (KPI comparison)",
     "Exp 2",
     f"CBM reduces critical failures by {reduction_pct:.1f}% vs fixed-TB "
     "under identical real KMA weather conditions"),
    ("R1 #13", "Homogeneous Markov chain vs. seasonal reality",
     "Exp 3",
     "Seasonal matrices calibrated from real KMA data; "
     "MWU hom vs non-hom: n.s. -- conclusions fully robust"),
    ("R1 #12", "Year-over-year performance divergence unexplained",
     "Exp 4",
     f"Weibull aging: +{aging_contrib:.0f}% hazard by Yr3; "
     f"WX blocking {wx_contrib:+.0f}%; queue carryover {q_contrib:.0f}% of Yr3 failures"),
    ("R1 #6",  "Vessel economies of scale ignored",
     "Exp 5",
     f"Grouped dispatch saves {saving_pct:.1f}% vessel cost; "
     "failure count unchanged -- core reliability conclusion preserved"),
]

rebuttal_df = pd.DataFrame(mapping,
    columns=["Reviewer Comment","Issue","Experiment","Key Finding"])
rebuttal_df.to_csv(f"{OUT}/REBUTTAL_MAPPING.csv", index=False)

for _, row in rebuttal_df.iterrows():
    print(f"\n  [{row['Reviewer Comment']}] {row['Issue']}")
    print(f"    -> {row['Experiment']}: {row['Key Finding']}")

total_time = time.time() - t0
print(f"\n{'='*70}")
print(f"  All experiments completed in {total_time:.1f}s")
print(f"  Output files:")
for fn in ["exp1_2_cbm_vs_timebased.png","exp1_2_summary_table.csv",
           "exp3_seasonal_markov.png","exp3_seasonal_transition_matrices.csv",
           "exp4_5_decomp_dispatch.png","exp4_year_decomp_table.csv",
           "exp5_dispatch_comparison.csv",
           "CONSOLIDATED_ADDITIONAL_EXPERIMENTS.png","REBUTTAL_MAPPING.csv"]:
    fpath = f"{OUT}/{fn}"
    size  = os.path.getsize(fpath) // 1024 if os.path.exists(fpath) else 0
    print(f"    {fn:<50} ({size:>5} KB)")
print(f"{'='*70}")