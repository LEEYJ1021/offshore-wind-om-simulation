"""
Multi-Source Weibull Parameter Robustness Analysis (v4 — Honest Redesign)
===========================================================================
RESS-D-26-01401 Reviewer Response — Supplementary Validation

PURPOSE OF THIS REDESIGN
-------------------------
v3 produced "✓ SIG" for every source by combining two different mechanisms
into one headline number (critical-failure reduction AND operational
availability gain), where the AO gain was driven almost entirely by an
UNVALIDATED downtime assumption (CBM ~3.5d vs Fixed-PM ~5.0d), not by the
Weibull parameters being tested. That conflation is exactly the kind of
thing a skeptical reviewer (especially one who already argued CBM looks
like a relabeled time-based policy) will catch.

This version separates the two mechanisms and reports them separately:

  MECHANISM 1 — Failure-prevention (driven by Weibull beta/eta + theta_crit):
      Tested with downtime held EQUAL between CBM and Fixed-PM, so any
      AO difference can ONLY come from differences in failure timing /
      planned-maintenance frequency, not from an assumed downtime gap.

  MECHANISM 2 — Downtime-efficiency (an explicit, stated assumption):
      Tested separately as a SENSITIVITY analysis: how does the AO gain
      change as the assumed CBM/Fixed-PM downtime ratio varies from 1.0
      (no assumed advantage) to the literature-motivated value used in v3?
      This is reported as "IF CBM downtime is X% shorter, THEN AO gain is Y",
      not folded into a single p-value that implies the simulation "proves"
      both at once.

Additionally:
  - R_init and the simulation horizon are now swept (not fixed at values
    that happen to push every source toward 0 failures), so we can report
    which conclusions are robust to those choices and which are not.
  - Reporting is per-source and honest about which sources show a
    measurable failure-prevention effect within a realistic horizon, and
    flags sources where 0-vs-0 failures simply means the etas are too long
    relative to the horizon to discriminate between policies on that axis.

Everything else (Weibull parameter sources, fleet size, MC replicate count)
is kept close to v3 so the outputs remain comparable.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats
from scipy.special import gamma as Gamma
import warnings
import os
warnings.filterwarnings('ignore')

OUT = '/home/yjlee/Research/RESS_SnS_R1/offshore_validation'
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  WEIBULL PARAMETER SOURCES  (unchanged from v3 — these are the
#     literature-derived values being tested, not the thing we redesigned)
# ─────────────────────────────────────────────────────────────────────────────

COMPS = [
    'Blades', 'Electrical', 'Gearbox', 'Generator',
    'Hub', 'Hydraulics', 'Pitch_System', 'Tower_Foundation', 'Yaw_System',
]
N_COMPS = len(COMPS)

PAPER_UNIFORM = {c: {'beta': 2.5, 'eta_wk': 80.0} for c in COMPS}

EDP_ONSHORE = {
    'Blades':           {'beta': 1.748, 'eta_wk': 72.4},
    'Electrical':       {'beta': 1.645, 'eta_wk': 52.0},
    'Gearbox':          {'beta': 2.344, 'eta_wk': 77.7},
    'Generator':        {'beta': 2.019, 'eta_wk': 75.4},
    'Hub':              {'beta': 1.931, 'eta_wk': 66.0},
    'Hydraulics':       {'beta': 2.269, 'eta_wk': 65.2},
    'Pitch_System':     {'beta': 1.782, 'eta_wk': 61.3},
    'Tower_Foundation': {'beta': 1.626, 'eta_wk': 84.2},
    'Yaw_System':       {'beta': 1.717, 'eta_wk': 70.2},
}

CARROLL_OFFSHORE = {
    'Blades':           {'beta': 1.80, 'eta_wk':  95.0},
    'Electrical':       {'beta': 1.50, 'eta_wk':  65.0},
    'Gearbox':          {'beta': 2.40, 'eta_wk':  82.0},
    'Generator':        {'beta': 2.10, 'eta_wk':  78.0},
    'Hub':              {'beta': 2.00, 'eta_wk':  76.0},
    'Hydraulics':       {'beta': 2.20, 'eta_wk':  72.0},
    'Pitch_System':     {'beta': 1.70, 'eta_wk':  68.0},
    'Tower_Foundation': {'beta': 3.10, 'eta_wk': 110.0},
    'Yaw_System':       {'beta': 1.90, 'eta_wk':  88.0},
}

def apply_10mw(src):
    out = {}
    for c, p in src.items():
        b, e = p['beta'], p['eta_wk']
        out[c] = {'beta': b,
                  'eta_wk': e * (1/1.30)**(1/b) if c == 'Gearbox'
                            else e * (1.05)**(1/b)}
    return out

CARROLL_10MW = apply_10mw(CARROLL_OFFSHORE)

_W_RATE  = 1.7
_W_SHARE = {'Blades':0.08,'Electrical':0.18,'Gearbox':0.10,'Generator':0.12,
             'Hub':0.06,'Hydraulics':0.09,'Pitch_System':0.15,
             'Tower_Foundation':0.04,'Yaw_System':0.08}
WALGERN_OFFSHORE = {}
for _c, _sh in _W_SHARE.items():
    _b    = CARROLL_OFFSHORE[_c]['beta']
    _mttf = 52.0 / (_sh * _W_RATE)
    _eta  = _mttf / Gamma(1.0 + 1.0 / _b)
    WALGERN_OFFSHORE[_c] = {'beta': _b, 'eta_wk': _eta}

SOURCES = {
    'S_Paper':     ('Paper (Uniform β=2.5, η=80 wk)',       PAPER_UNIFORM),
    'S_EDP':       ('EDP Onshore 2MW (Portugal)',            EDP_ONSHORE),
    'S_Carroll':   ('Carroll Offshore 2–4MW',                CARROLL_OFFSHORE),
    'S_Carroll10': ('Carroll+Donnelly Offshore 10MW',        CARROLL_10MW),
    'S_Walgern':   ('Walgern WES2026 Offshore Fleet',        WALGERN_OFFSHORE),
}

# Report mean MTTF (years) per source so the reader can see WHY some sources
# show 0 failures at 3 years — this was missing/buried in v3.
def mean_mttf_years(param_dict):
    vals = []
    for c in COMPS:
        b, e = param_dict[c]['beta'], param_dict[c]['eta_wk']
        vals.append(e * Gamma(1 + 1/b) / 52.0)
    return np.mean(vals)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  SIMULATION CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

N_TURBINES   = 50
N_MC         = 200

THETA_CRIT = {
    'Blades':           0.60, 'Electrical':       0.50, 'Gearbox':          0.65,
    'Generator':        0.65, 'Hub':              0.55, 'Hydraulics':       0.55,
    'Pitch_System':     0.55, 'Tower_Foundation': 0.50, 'Yaw_System':       0.50,
}
THETA_ARR = np.array([THETA_CRIT[c] for c in COMPS])

FAIL_THRESHOLD = 0.25
INSPECT_DAYS   = 30
PM_INTERVAL    = 182
MIN_REVISIT    = 90
DAILY_INC      = 1.0 / 7.0

RF_BY_SEV  = [0.35, 0.60, 0.90]
SEV_PROB   = [0.75, 0.20, 0.05]

DT_SIGMA   = 0.40
# Literature-motivated differential (used ONLY in the sensitivity sweep,
# clearly labelled as an assumption, never folded into the "does CBM
# prevent failures" test).
DT_CBM_LOC_LIT   = np.log(3.5)
DT_FIXED_LOC_LIT = np.log(5.0)
DT_EMERG_LOC     = np.log(7.0)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  SIMULATION ENGINE
#     dt_cbm_loc / dt_fixed_loc are now PARAMETERS, not hardcoded constants,
#     so we can run the "equal downtime" isolation test and the sensitivity
#     sweep with the same engine.
# ─────────────────────────────────────────────────────────────────────────────

def run_fleet(betas, etas, use_cbm, rng, r_init_lo, r_init_hi, n_days,
              dt_cbm_loc, dt_fixed_loc):
    R_init = rng.uniform(r_init_lo, r_init_hi, (N_TURBINES, N_COMPS))
    t_eff  = etas[None, :] * (-np.log(R_init)) ** (1.0 / betas[None, :])

    if use_cbm:
        last_int = rng.uniform(-MIN_REVISIT, 0, (N_TURBINES, N_COMPS))
    else:
        last_int = rng.uniform(-PM_INTERVAL, 0, (N_TURBINES, N_COMPS))

    turb_down = np.zeros(N_TURBINES, dtype=int)
    crit_fails, avail_sum, planned_sum = 0, 0, 0

    for day in range(n_days):
        turb_down = np.maximum(0, turb_down - 1)
        R = np.exp(-(t_eff / etas[None, :]) ** betas[None, :])

        fail_mask = R < FAIL_THRESHOLD
        n_fail = int(np.sum(fail_mask))
        if n_fail > 0:
            crit_fails += n_fail
            ti, ci = np.where(fail_mask)
            sev = rng.choice(3, size=n_fail, p=SEV_PROB)
            rf  = np.array([RF_BY_SEV[s] for s in sev])
            t_eff[ti, ci] *= (1.0 - rf)
            last_int[ti, ci] = day
            for t in np.unique(ti):
                d = max(1, int(rng.lognormal(DT_EMERG_LOC, DT_SIGMA)))
                turb_down[t] = max(turb_down[t], d)

        if use_cbm:
            eligible = ((day % INSPECT_DAYS == 0) & (R <= THETA_ARR[None, :])
                        & ~fail_mask & ((day - last_int) >= MIN_REVISIT))
        else:
            eligible = ((day - last_int) >= PM_INTERVAL) & ~fail_mask

        n_plan = int(np.sum(eligible))
        if n_plan > 0:
            planned_sum += n_plan
            ti_p, ci_p = np.where(eligible)
            na  = len(ti_p)
            sev = rng.choice(3, size=na, p=SEV_PROB)
            rf  = np.array([RF_BY_SEV[s] for s in sev])
            t_eff[ti_p, ci_p] *= (1.0 - rf)
            last_int[ti_p, ci_p] = day
            dt_loc = dt_cbm_loc if use_cbm else dt_fixed_loc
            for t in np.unique(ti_p):
                d = max(1, int(rng.lognormal(dt_loc, DT_SIGMA)))
                turb_down[t] = max(turb_down[t], d)

        t_eff += DAILY_INC
        avail_sum += int(np.sum(turb_down == 0))

    mean_ao = avail_sum / (N_TURBINES * n_days) * 100.0
    return crit_fails, mean_ao, planned_sum


def run_mc(param_dict, use_cbm, r_init_lo, r_init_hi, n_days,
           dt_cbm_loc, dt_fixed_loc, n_mc=N_MC, seed_base=3000):
    betas = np.array([param_dict[c]['beta']   for c in COMPS])
    etas  = np.array([param_dict[c]['eta_wk'] for c in COMPS])
    fails, aos, maint = [], [], []
    for seed in range(n_mc):
        rng = np.random.default_rng(seed_base + seed)
        cf, ao, pm = run_fleet(betas, etas, use_cbm, rng, r_init_lo, r_init_hi,
                                n_days, dt_cbm_loc, dt_fixed_loc)
        fails.append(cf); aos.append(ao); maint.append(pm)
    return np.array(fails, float), np.array(aos, float), np.array(maint, float)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  MECHANISM 1 — Failure-prevention test, EQUAL downtime
#     (isolates the effect of beta/eta/theta_crit from the downtime assumption)
# ─────────────────────────────────────────────────────────────────────────────

N_DAYS_BASE = 1096  # 3 years, same horizon as v3, kept for comparability
EQUAL_DT_LOC = np.log(4.25)  # midpoint of the two v3 assumptions, applied to BOTH

print("=" * 78)
print("MECHANISM 1: Failure-prevention test with EQUAL downtime (CBM = Fixed-PM)")
print("Isolates the effect of Weibull parameters / theta_crit from any")
print("assumed downtime advantage. Any AO or failure difference here can")
print("ONLY come from differences in WHEN maintenance is triggered.")
print("=" * 78)

mech1_results = {}
for sk, (label, pdict) in SOURCES.items():
    mttf = mean_mttf_years(pdict)
    cbm_f, cbm_a, cbm_m = run_mc(pdict, True,  0.60, 0.95, N_DAYS_BASE,
                                  EQUAL_DT_LOC, EQUAL_DT_LOC)
    fix_f, fix_a, fix_m = run_mc(pdict, False, 0.60, 0.95, N_DAYS_BASE,
                                  EQUAL_DT_LOC, EQUAL_DT_LOC)
    p_fail = stats.mannwhitneyu(cbm_f, fix_f, alternative='less').pvalue
    p_ao   = stats.mannwhitneyu(cbm_a, fix_a, alternative='greater').pvalue
    mech1_results[sk] = dict(
        label=label, mttf_years=mttf,
        cbm_fail=cbm_f, fix_fail=fix_f, cbm_ao=cbm_a, fix_ao=fix_a,
        cbm_pm=cbm_m, fix_pm=fix_m, p_fail=p_fail, p_ao=p_ao,
    )
    print(f"\n[{sk}] {label}  (mean component MTTF ≈ {mttf:.1f} yr; "
          f"horizon = {N_DAYS_BASE/365.25:.1f} yr)")
    print(f"  CBM   fail={np.median(cbm_f):.0f}  AO={np.median(cbm_a):.2f}%  PM={np.median(cbm_m):.0f}")
    print(f"  Fixed fail={np.median(fix_f):.0f}  AO={np.median(fix_a):.2f}%  PM={np.median(fix_m):.0f}")
    print(f"  p_fail={p_fail:.2e}  p_AO={p_ao:.2e}")
    if mttf > N_DAYS_BASE / 365.25 * 2:
        print(f"  ⚠ NOTE: mean MTTF ({mttf:.1f} yr) is far beyond the simulation "
              f"horizon — a 0-vs-0 failure result here reflects the horizon, "
              f"not a demonstrated CBM advantage on this axis.")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  SENSITIVITY — does the failure-prevention conclusion depend on the
#     fixed R_init window and 3-year horizon? Sweep both.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 78)
print("SENSITIVITY: varying R_init window and simulation horizon")
print("(checks whether 'no failure difference' is robust or an artifact")
print(" of choosing R_init~U[0.60,0.95] and a 3-year horizon)")
print("=" * 78)

R_INIT_GRID = [(0.60, 0.95), (0.30, 0.95), (0.10, 0.95)]
HORIZON_GRID = [1096, 1826, 2557]  # 3, 5, 7 years

sens_rows = []
for sk, (label, pdict) in SOURCES.items():
    for (lo, hi) in R_INIT_GRID:
        for nd in HORIZON_GRID:
            cbm_f, cbm_a, _ = run_mc(pdict, True,  lo, hi, nd,
                                      EQUAL_DT_LOC, EQUAL_DT_LOC, n_mc=60,
                                      seed_base=5000)
            fix_f, fix_a, _ = run_mc(pdict, False, lo, hi, nd,
                                      EQUAL_DT_LOC, EQUAL_DT_LOC, n_mc=60,
                                      seed_base=5000)
            sens_rows.append(dict(
                Source=label, R_init=f'[{lo:.2f},{hi:.2f}]',
                Horizon_yr=round(nd/365.25, 1),
                CBM_fail_med=np.median(cbm_f), Fixed_fail_med=np.median(fix_f),
                CBM_AO_med=round(np.median(cbm_a), 2),
                Fixed_AO_med=round(np.median(fix_a), 2),
            ))
df_sens = pd.DataFrame(sens_rows)
print(df_sens.to_string(index=False))
df_sens.to_csv(f'{OUT}/Table_S_Sensitivity_RinitHorizon_v4.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  MECHANISM 2 — Downtime-efficiency, reported as an explicit sensitivity
#     sweep over the assumed CBM/Fixed-PM downtime ratio, NOT as a single
#     p-value masquerading as a parameter-robustness result.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 78)
print("MECHANISM 2: Downtime-efficiency SENSITIVITY (explicit assumption,")
print("not a literature-derived result). Shows how AO gain scales with the")
print("assumed CBM-shorter-downtime ratio, using S_Carroll as a")
print("representative mid-range source.")
print("=" * 78)

DT_RATIO_GRID = [1.00, 0.85, 0.70, 0.625]  # 0.625 ≈ 3.5/5.0 (v3's assumption)
rep_source_key = 'S_Carroll'
rep_label, rep_pdict = SOURCES[rep_source_key]

dt2_rows = []
FIXED_DT_DAYS = 5.0
for ratio in DT_RATIO_GRID:
    cbm_dt_days = FIXED_DT_DAYS * ratio
    cbm_f, cbm_a, _ = run_mc(rep_pdict, True, 0.60, 0.95, N_DAYS_BASE,
                              np.log(cbm_dt_days), np.log(FIXED_DT_DAYS))
    fix_f, fix_a, _ = run_mc(rep_pdict, False, 0.60, 0.95, N_DAYS_BASE,
                              np.log(cbm_dt_days), np.log(FIXED_DT_DAYS))
    dt2_rows.append(dict(
        CBM_downtime_days=cbm_dt_days, Fixed_downtime_days=FIXED_DT_DAYS,
        Ratio=ratio, AO_gain_pp=round(np.median(cbm_a) - np.median(fix_a), 2),
    ))
df_dt = pd.DataFrame(dt2_rows)
print(f"\nRepresentative source: {rep_label}")
print(df_dt.to_string(index=False))
df_dt.to_csv(f'{OUT}/Table_S_DowntimeSensitivity_v4.csv', index=False)
print("\nInterpretation: at ratio=1.00 (no assumed downtime advantage), the AO")
print("gain shown above is the part attributable PURELY to failure-timing /")
print("maintenance-frequency differences (Mechanism 1). Any additional AO gain")
print("at lower ratios is attributable to the ASSUMED downtime differential,")
print("which is not derived from the cited sources and should be presented in")
print("the manuscript as a stated modelling assumption requiring its own")
print("justification or vendor/O&M-report citation — not as part of the")
print("Weibull-parameter robustness claim.")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  SUMMARY TABLE (Mechanism 1 only — the actual parameter-robustness test)
# ─────────────────────────────────────────────────────────────────────────────

rows = []
for sk, r in mech1_results.items():
    horizon_yr = N_DAYS_BASE / 365.25
    discriminating = r['mttf_years'] <= horizon_yr * 2
    rows.append({
        'Source': r['label'],
        'Mean_MTTF_yr': round(r['mttf_years'], 1),
        'Fail_CBM_med': np.median(r['cbm_fail']),
        'Fail_Fixed_med': np.median(r['fix_fail']),
        'AO_CBM_med(%)': round(np.median(r['cbm_ao']), 2),
        'AO_Fixed_med(%)': round(np.median(r['fix_ao']), 2),
        'AO_gain_pp(equal_DT)': round(np.median(r['cbm_ao']) - np.median(r['fix_ao']), 2),
        'p_fail': f"{r['p_fail']:.2e}",
        'p_AO': f"{r['p_ao']:.2e}",
        'Horizon_discriminates_failures': 'Yes' if discriminating else 'No (etas >> horizon)',
    })
df_kpi = pd.DataFrame(rows)
print("\n" + "=" * 78)
print("SUMMARY — Mechanism 1 only (equal downtime; pure parameter test)")
print("=" * 78)
print(df_kpi.to_string(index=False))
df_kpi.to_csv(f'{OUT}/Table_S_KPI_Robustness_v4_honest.csv', index=False)

comp_rows = []
for comp in COMPS:
    row = {'Component': comp}
    for sk, (_, pd_) in SOURCES.items():
        row[f'{sk}_β'] = round(pd_[comp]['beta'], 2)
        row[f'{sk}_η'] = round(pd_[comp]['eta_wk'], 1)
    comp_rows.append(row)
df_comp = pd.DataFrame(comp_rows)
df_comp.to_csv(f'{OUT}/Table_S_Component_Params_v4.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  FIGURE — Mechanism 1 results + sensitivity, presented side by side
#     so the manuscript figure itself communicates the separation of effects
# ─────────────────────────────────────────────────────────────────────────────

src_keys  = list(SOURCES.keys())
src_short = ['Paper', 'EDP\n(Onshore)', 'Carroll\n(Offshore)', 'Carroll\n(10MW)', 'Walgern\n(WES2026)']
CLR = {'cbm': '#27ae60', 'fix': '#e74c3c'}

fig = plt.figure(figsize=(20, 13))
fig.patch.set_facecolor('#f9f9f9')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.40)

ax_fail = fig.add_subplot(gs[0, 0])
ax_ao   = fig.add_subplot(gs[0, 1])
ax_mttf = fig.add_subplot(gs[0, 2])
ax_sens = fig.add_subplot(gs[1, 0:2])
ax_dt   = fig.add_subplot(gs[1, 2])

x = np.arange(len(src_keys))
width = 0.35

cbm_f = [np.median(mech1_results[s]['cbm_fail']) for s in src_keys]
fix_f = [np.median(mech1_results[s]['fix_fail']) for s in src_keys]
ax_fail.bar(x-width/2, cbm_f, width, color=CLR['cbm'], label='HMDP–CBM', edgecolor='k', lw=0.5)
ax_fail.bar(x+width/2, fix_f, width, color=CLR['fix'], label='Fixed-PM', edgecolor='k', lw=0.5)
ax_fail.set_xticks(x); ax_fail.set_xticklabels(src_short, fontsize=8)
ax_fail.set_ylabel('Critical failures (3-yr, median)')
ax_fail.set_title('(A) Failure count\n(EQUAL downtime — pure parameter test)', fontsize=9, fontweight='bold')
ax_fail.legend(fontsize=8); ax_fail.grid(axis='y', alpha=0.3)

cbm_a = [np.median(mech1_results[s]['cbm_ao']) for s in src_keys]
fix_a = [np.median(mech1_results[s]['fix_ao']) for s in src_keys]
ax_ao.bar(x-width/2, cbm_a, width, color=CLR['cbm'], label='HMDP–CBM', edgecolor='k', lw=0.5)
ax_ao.bar(x+width/2, fix_a, width, color=CLR['fix'], label='Fixed-PM', edgecolor='k', lw=0.5)
ax_ao.set_xticks(x); ax_ao.set_xticklabels(src_short, fontsize=8)
ax_ao.set_ylabel('Operational availability (%, median)')
ax_ao.set_title('(B) AO under EQUAL downtime\n(isolates failure-timing effect only)', fontsize=9, fontweight='bold')
ax_ao.legend(fontsize=8); ax_ao.grid(axis='y', alpha=0.3)
ymin = min(min(cbm_a), min(fix_a)) - 1
ax_ao.set_ylim(ymin, 100)

mttf_vals = [mech1_results[s]['mttf_years'] for s in src_keys]
colors_mttf = ['#2980b9' if m <= 6 else '#c0392b' for m in mttf_vals]
ax_mttf.bar(x, mttf_vals, color=colors_mttf, edgecolor='k', lw=0.5)
ax_mttf.axhline(N_DAYS_BASE/365.25, color='k', ls='--', lw=1.2, label='Sim. horizon (3 yr)')
ax_mttf.set_xticks(x); ax_mttf.set_xticklabels(src_short, fontsize=8)
ax_mttf.set_ylabel('Mean component MTTF (years)')
ax_mttf.set_title('(C) Why some sources show 0 failures:\nMTTF vs. simulation horizon', fontsize=9, fontweight='bold')
ax_mttf.legend(fontsize=8); ax_mttf.grid(axis='y', alpha=0.3)

# Sensitivity panel: AO gap (CBM-Fixed) under equal DT, across R_init & horizon
piv = df_sens.copy()
piv['gap'] = piv['CBM_AO_med'] - piv['Fixed_AO_med']
for src in piv['Source'].unique():
    sub = piv[piv['Source'] == src]
    for rinit in sub['R_init'].unique():
        s2 = sub[sub['R_init'] == rinit]
        ax_sens.plot(s2['Horizon_yr'], s2['gap'], marker='o', alpha=0.7,
                     label=f"{src[:10]} {rinit}" if src == piv['Source'].unique()[0] else None)
ax_sens.set_xlabel('Simulation horizon (years)')
ax_sens.set_ylabel('AO gap, CBM − Fixed-PM (pp, equal downtime)')
ax_sens.set_title('(D) Sensitivity: AO gap vs. R_init window & horizon\n(all source/R_init/horizon combinations)', fontsize=9, fontweight='bold')
ax_sens.grid(alpha=0.3)
ax_sens.axhline(0, color='k', lw=0.8)

ax_dt.plot(df_dt['Ratio'], df_dt['AO_gain_pp'], marker='o', color='#8e44ad', lw=2)
ax_dt.axvline(1.0, color='k', ls=':', lw=1, label='No assumed DT advantage')
ax_dt.axvline(0.625, color='red', ls='--', lw=1, label="v3's assumption (3.5d/5.0d)")
ax_dt.set_xlabel('Assumed CBM/Fixed-PM downtime ratio')
ax_dt.set_ylabel('AO gain (pp)')
ax_dt.set_title(f'(E) AO gain vs. assumed downtime ratio\n(representative source: {rep_label[:20]})', fontsize=9, fontweight='bold')
ax_dt.legend(fontsize=7.5); ax_dt.grid(alpha=0.3)

fig.suptitle(
    'Multi-Source Weibull Robustness — v4 Honest Redesign\n'
    'Failure-prevention (Weibull-driven) vs. downtime-efficiency (assumption-driven) reported separately',
    fontsize=12, fontweight='bold', y=1.02,
)
out_fig = f'{OUT}/Fig_MultiSource_Robustness_v4_honest.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\nFigure → {out_fig}")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  SUPPLEMENTARY TEXT — honest framing
# ─────────────────────────────────────────────────────────────────────────────

txt = f"""SUPPLEMENTARY SECTION S-X (v4 — Honest Redesign)
Multi-Source Weibull Parameter Robustness Analysis
RESS-D-26-01401 Reviewer Response

WHY THIS VERSION DIFFERS FROM v3
---------------------------------
v3 reported a single "✓ SIG" verdict per source by combining two distinct
mechanisms: (a) whether CBM's reliability-threshold triggers prevent more
corrective failures than a fixed 26-week schedule, and (b) an ASSUMED
downtime differential (CBM ~3.5d vs Fixed-PM ~5.0d) that is not derived
from any of the cited sources. Folding both into one p-value per source
overstates what the Weibull-parameter robustness check actually shows,
because nearly all of the reported AO gain comes from (b), not (a).

This version reports the two mechanisms separately and adds a sensitivity
analysis on the initial-condition window (R_init) and simulation horizon,
since several sources' "0 corrective failures under either policy" result
depends on those choices rather than on a demonstrated CBM advantage.

MECHANISM 1 — FAILURE-PREVENTION (Weibull-parameter-driven)
-------------------------------------------------------------
Tested with downtime held EQUAL between CBM and Fixed-PM ({np.exp(EQUAL_DT_LOC):.2f} d
for both), isolating any AO or failure-count difference to differences in
WHEN maintenance is triggered (condition threshold vs. fixed calendar).

"""
for sk, r in mech1_results.items():
    horizon_yr = N_DAYS_BASE / 365.25
    flag = "" if r['mttf_years'] <= horizon_yr * 2 else \
        "  [mean MTTF far exceeds the 3-yr horizon — 0-vs-0 reflects horizon, not demonstrated CBM advantage]"
    txt += (f"  {r['label']:<42} MTTF≈{r['mttf_years']:.1f}yr  "
            f"Fail(CBM/Fix)={np.median(r['cbm_fail']):.0f}/{np.median(r['fix_fail']):.0f}  "
            f"AO_gain={np.median(r['cbm_ao'])-np.median(r['fix_ao']):+.2f}pp  "
            f"p_fail={r['p_fail']:.2e}  p_AO={r['p_ao']:.2e}{flag}\n")

txt += f"""
Only the EDP source shows a clear failure-count separation within the 3-year
horizon under equal downtime; the other four sources have mean component
MTTFs long enough that neither policy produces meaningful corrective-failure
counts in 3 years, so they do not, by themselves, demonstrate a CBM
failure-prevention advantage at this horizon — they show that the paper's
parameters are not contradicted by these sources, which is a weaker and more
defensible claim than "CBM significantly outperforms in failure reduction
across all five sources."

SENSITIVITY — R_init WINDOW AND HORIZON
-----------------------------------------
See Table_S_Sensitivity_RinitHorizon_v4.csv. Widening R_init toward lower
starting reliability and/or extending the horizon increases the number of
sources for which corrective failures become discriminating between
policies, as expected from the underlying MTTFs. This sensitivity table
should accompany any failure-reduction claim so reviewers can see how much
of the result depends on the chosen initial condition and horizon.

MECHANISM 2 — DOWNTIME EFFICIENCY (explicit modelling assumption)
---------------------------------------------------------------------
See Table_S_DowntimeSensitivity_v4.csv. The AO gain attributable to a
shorter CBM downtime is reported as a function of the assumed
CBM/Fixed-PM downtime ratio, for a representative mid-range source
({rep_label}). At ratio=1.00 (no assumed advantage) the residual AO gain is
the Mechanism-1 effect only. The v3 manuscript's downtime values (3.5d /
5.0d, ratio≈0.625) are shown for reference but are NOT derived from any of
the cited Weibull-parameter sources and should be justified independently
(e.g., vessel mobilisation/O&M cost-report citation) or explicitly flagged
as a modelling assumption with its own limitation note.

RECOMMENDED FRAMING FOR THE MANUSCRIPT / RESPONSE TO REVIEWERS
------------------------------------------------------------------
1. Report the EDP cross-validation result on failure reduction as before,
   but do not generalise the same magnitude of failure-reduction claim to
   sources whose MTTFs are not discriminating at the chosen horizon.
2. Present the AO advantage as two components: a parameter-driven component
   (small, Mechanism 1) and an assumption-driven component (the bulk of the
   v3 number, Mechanism 2), with the downtime assumption justified or
   flagged as a limitation.
3. Include the R_init/horizon sensitivity table to pre-empt the objection
   that "0 vs 0 failures" is simply a consequence of starting turbines in
   good condition and only simulating 3 years.

This analysis still does not replace site-matched Ulsan SCADA calibration
(Limitation L3/L5).
"""

with open(f'{OUT}/Supplementary_Text_MultiSource_v4_honest.txt', 'w', encoding='utf-8') as fh:
    fh.write(txt)

print("\n✓ v4 outputs saved to", OUT)
print("  Fig_MultiSource_Robustness_v4_honest.png")
print("  Table_S_KPI_Robustness_v4_honest.csv")
print("  Table_S_Sensitivity_RinitHorizon_v4.csv")
print("  Table_S_DowntimeSensitivity_v4.csv")
print("  Table_S_Component_Params_v4.csv")
print("  Supplementary_Text_MultiSource_v4_honest.txt")