# Iso-Reliability Benchmarking of Maintenance Policies for Capacity-Constrained Fleets
## Separating the Value of Triggers, State Tracking, and Learning — Offshore Wind O&M Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![R](https://img.shields.io/badge/R-4.3%2B-276DC3?logo=r)](https://www.r-project.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://www.python.org/)
[![Protocol](https://img.shields.io/badge/Protocol-iso--reliability%20v11-brightgreen)](scripts/iso_reliability/03_iso_reliability_benchmark.py)

---

## Overview

This repository is the reproducible companion to the manuscript

> **Iso-reliability benchmarking of maintenance policies for capacity-constrained fleets: separating the value of triggers, state tracking, and learning**
> Yong-Jae Lee, Yong-Hyuk Shin\*, Min-Gyu Sang\* (prepared for *Computers & Operations Research*).

Maintenance policies for capacity-constrained asset fleets are routinely compared at unequal maintenance frequency, tuning effort and information, so reported gains of condition-based triggers over calendar schedules cannot be attributed to a cause. We treat the comparison as the estimation of **iso-reliability slices of simulated cost–reliability frontiers** and provide a three-step computational protocol:

1. **Algorithm 1 — equal-reliability operating points.** Every policy is tuned on training seeds disjoint from the test seeds; operating points are found by frontier interpolation at common target failure levels (q25/q50/q75) and a local-slope correction removes residual reliability differences.
2. **Algorithm 2 — frequency matching.** A per-component calendar (TBMm) is calibrated by fixed-point iteration to the planned-task counts of a state-dependent policy, removing overall frequency and component differentiation as confounders.
3. **Algorithm 3 — attribution and inference.** A 2 × 2 factorial design crosses the trigger type (calendar vs. effective-age state tracking) with learning (none vs. Bayesian estimation of latent wear rates), with a paired bootstrap (B = 20,000) over 50 common-random-number replicates.

The protocol is applied to a **50-turbine, 11-component offshore fleet** with Kijima type II imperfect repair and a **weather-limited vessel queue** estimated from **1,012 days** of Korea Meteorological Administration (KMA) coastal data, with index (IDX/IDXL), policy-search (PS) and degradation-sensor (SEN) baselines and an oracle (ORA) bound.

### Headline findings (all conditional on the simulation model and assumed cost parameters)

| # | Finding | Evidence (q50, bn KRW unless stated) |
|---|---------|--------------------------------------|
| 1 | Frequency matching removes most of the apparent advantage of an effective-age trigger; the advantage vanishes or reverses under strong heterogeneity. | EAT−TBM: −57.8 (S0), −88.6 (S1) critical failures; EAT−TBMm: −16.6 (S0), −0.7 [−3.7, +2.1] (S1), **+22.3** [+18.2, +26.5] (S3). Without learning EAT vs. TBM cost: −9.1 (S0), −2.4 (S1), +0.7 n.s. (S2), **+2.9** (S3). |
| 2 | Information on latent wear (learned from failure histories or read from a sensor) accounts for most of the value: **≈ 3–9 % of cost at equal reliability**. | LTBM −11.2 / −10.2 / −27.8 (3.9 / 2.9 / 8.1 %) in S1–S3; BAY/BAYe −13.5 to −31.0 (3.8–9.1 %); ORA −20.0 to −41.2. |
| 3 | A **learned calendar** recovers 50–67 % of the oracle benefit; BAYe 69–75 %; sensors 69–101 %. | LTBM 0.50 / 0.51 / 0.67; BAYe 0.71 / 0.69 / 0.75 (S1–S3). |
| 4 | State tracking adds a smaller, complementary and **rule-dependent** gain. | ST×L interaction −1.1 / −2.0 / −2.9 (S1–S3); BAY−LTBM −4.7 / −3.2 / −2.9; IDXL beats BAYe by 8.0 in S3 but loses in S1–S2 (confounded: cost weighting vs. trigger form). |
| 5 | The learning advantage persists in **all 16 valid structural-misspecification settings** (BAYe and LTBM CI < 0), but falls to **42–52 %** of its size under bimodal heterogeneity. | Median retention 87 % (valid settings); EAT retains an advantage in only 6 of 16. |
| 6 | The iso-reliability frame **requires a stable task queue**: below a capacity threshold policy differences cannot appear and the frame fails. | CTV-only: BAYe−TBM CI < 0 from capacity 5 (S1: −4.8, S3: −21.1); mean queue under TBM 226 / 314 tasks at capacity 4 vs. 11 / 20 at capacity 8. |
| 7 | Conclusions are stable under cost re-pricing, tuning budget and fleet size (10–200 turbines, capacity scaled). | BAYe−TBM favourable and significant in 100 % of 400 joint cost re-weightings; scaled-capacity advantage −5.4 to −6.9 % (S1), −8.8 to −11.2 % (S3). |

---

## Relationship to the Earlier Study (Legacy Material)

This repository previously accompanied the *Reliability-Threshold Maintenance via Hierarchical MDP* study (Ulsan cluster, four-strategy S1–S4 factorial design, Proposition 1, multi-cell ETA accounting, weather-adaptive logistics). **All legacy scripts, data layout and results are retained unchanged** in the Repository Structure below. The present manuscript differs in four respects, and the two code bases should not be mixed:

| Aspect | Legacy RT–HMDP study | Iso-reliability study (this README) |
|--------|----------------------|--------------------------------------|
| Question | Is a reliability-threshold trigger distinguishable from a fixed interval? | How much of a trigger's value is due to frequency, component differentiation, state tracking, or information? |
| Fleet / components | 50 DTU 10 MW turbines, 18 component types | 50 turbines, **11 component types**, 3 criticality tiers |
| Heterogeneity | Uniform Weibull (β = 2.5, η = 80 wk) | Latent component-specific wear rates (σ = 0, 0.35, 0.60), κ scaling |
| Weather data | N = 4,383 days (2010–2022) | **N = 1,012 days** (daily series estimated from hourly KMA data) |
| Inference unit | Daily observations | **Replicates** (50 paired, common random numbers) |
| Entry points | `scripts/02_om_simulation.py` | `scripts/iso_reliability/03_iso_reliability_benchmark.py`, `04_make_figures.py` |

> **Recommendation:** tag the last legacy commit (e.g. `git tag v1-rt-hmdp`) before merging this README so the earlier documentation remains citable.

---

## Research Questions

| ID | Question | Where answered |
|----|----------|----------------|
| **Q1** | How much of the apparent advantage of an effective-age trigger over a uniform calendar disappears once frequency and component differentiation are matched? | T05, T06, Fig. 3a |
| **Q2** | How is the remaining value attributed to state tracking vs. learning of latent wear? | T07, T08, Fig. 2b, Fig. 3b |
| **Q3** | Where do index, policy-search and sensor baselines sit on the information ladder? | T09, Fig. 4 |
| **Q4** | How robust are the conclusions to structural misspecification, cost re-weighting, tuning budget and fleet size? | T10, T11, T14, T15, Fig. 5, Fig. A2–A3 |
| **Q5** | When does the iso-reliability frame stop being applicable? | T12, T13, Fig. 5c–d |

---

## Repository Structure

```
offshore-wind-om-simulation/
│
├── README.md
├── LICENSE
│
├── data/
│   ├── raw/
│   │   ├── weather_hourly_raw.csv          ← Raw hourly KMA weather input (place here)
│   │   └── weekly_cost_baseline.csv        ← Weekly O&M cost baseline (LP reference)
│   ├── processed/                          ← Auto-generated by 01_weather_preprocessing.R
│   │   ├── ulsan_hourly_weather_simple.csv
│   │   ├── ulsan_hourly_weather_detailed.csv
│   │   ├── ulsan_daily_weather_simple.csv
│   │   ├── ulsan_daily_weather_detailed.csv
│   │   ├── ulsan_weekly_weather_simple.csv
│   │   └── ulsan_weekly_weather_detailed.csv
│   └── outputs/                            ← Statistical tables from R
│       ├── Table1_Seasonal_Statistics.csv
│       ├── Table1b_Daily_Seasonal_Statistics.csv
│       ├── Table1c_Hourly_Seasonal_Statistics.csv
│       ├── Table1d_Hourly_TimeOfDay_Statistics.csv
│       ├── Table1e_Hourly_ByHour_Statistics.csv
│       ├── Table2_Weather_Transition_Matrix.csv
│       ├── Table3_Accessibility_Summary.csv
│       └── Table3b_Daily_Accessibility_Summary.csv
│
├── scripts/
│   ├── 01_weather_preprocessing.R          ← R: raw KMA data → processed datasets + figures
│   ├── 02_om_simulation.py                 ← Python: HMDP-Greedy O&M simulation (main); supports --diagnostic s1_rt
│   ├── additional_experiments/             ← Supplementary validation experiments
│   │   ├── run_all_experiments.py          ← Master runner (reads from data/raw/)
│   │   ├── experiment_core.py              ← Shared utilities, weather loader, fallback generator
│   │   ├── exp1_2_cbm_timebased.py         ← Exp 1–2: CBM trigger stochasticity & KPI comparison
│   │   ├── exp3_seasonal_markov.py         ← Exp 3: Seasonal vs. homogeneous Markov robustness
│   │   └── exp4_5_decomp_dispatch.py       ← Exp 4–5: Year decomposition & grouped dispatch
│   ├── edp_validation/                     ← EDP Open Data parameter validation
│   │   └── edp_scada_validation.py         ← Weibull MLE + ETA coefficient cross-validation
│   ├── multisource_validation/             ← Multi-source Weibull robustness analysis (NEW)
│   │   └── multisource_weibull_robustness_v4.py   ← Mechanism-separated robustness test (see below)
│   └── iso_reliability/                    ← Iso-reliability benchmarking protocol (NEW)
│       ├── README.md                       ← Run instructions, environment variables, runtime notes
│       ├── 03_iso_reliability_benchmark.py ← Protocol v11: Algorithms 1–3, core study, baselines,
│       │                                     robustness, capacity/scale, tuning budget, diagnostics
│       └── 04_make_figures.py              ← Publication figures v14 (black/white/gray; PDF+PNG[+TIFF])
│
└── results/
    ├── figures_R/                          ← PNG figures generated by R preprocessing
    │   ├── Fig01_Weather_Trends_Enhanced.png
    │   ├── Fig02_Seasonal_Distributions_Enhanced.png
    │   ├── Fig03_Comprehensive_Analysis_Fixed.png
    │   ├── Fig04_Transition_Matrix_Enhanced.png
    │   ├── Fig05_Monthly_Patterns_Enhanced.png
    │   ├── Fig06_Daily_CTV_Accessibility_Heatmap.png
    │   ├── Fig07_Hourly_WindWave_ByHour.png
    │   ├── Fig08_Hourly_CTV_ByHour_Season.png
    │   ├── Fig09_Hourly_Heatmap_Month_x_Hour.png
    │   └── Fig10_Hourly_TimeOfDay_Season.png
    ├── figures_PY/                         ← PNG figures generated by Python simulation
    │   ├── Fig01_Weather_Overview.png
    │   ├── Fig02_Seasonal_Accessibility.png
    │   ├── Fig03_ComponentCriticality_Weibull.png
    │   ├── Fig04_Strategy_Comparison.png
    │   ├── Fig05_Availability_Decomposition.png
    │   ├── Fig06_HMDP_LP_Integration.png
    │   ├── Fig07_CBM_vs_FixedPM.png
    │   ├── Fig08_Baseline_Comparison.png
    │   ├── Fig09_Scenario_Sensitivity.png
    │   ├── Fig10_Carbon_Pareto.png
    │   ├── Fig11_ETA_Derating_Analysis.png
    │   ├── Fig12_Feedback_Loop_Analysis.png
    │   ├── Fig13_Empirical_Validation.png
    │   └── Fig_ComponentIsolation_S1RT.png  ← Component-isolation diagnostic (NEW)
    ├── additional_experiments/             ← Outputs from supplementary validation experiments
    │   ├── CONSOLIDATED_ADDITIONAL_EXPERIMENTS.png
    │   ├── exp1_2_cbm_vs_timebased.png
    │   ├── exp1_2_summary_table.csv
    │   ├── exp3_seasonal_markov.png
    │   ├── exp3_seasonal_transition_matrices.csv
    │   ├── exp4_5_decomp_dispatch.png
    │   ├── exp4_year_decomp_table.csv
    │   └── exp5_dispatch_comparison.csv
    ├── edp_validation/                     ← Outputs from EDP parameter validation
    │   ├── weibull_parameters_comparison.csv
    │   ├── eta_coefficient_validation.csv
    │   ├── Fig_V1_weibull_fits.png
    │   ├── Fig_V2_eta_validation.png
    │   └── Fig_V3_beta_eta_sensitivity.png
    ├── multisource_validation/             ← Multi-source Weibull robustness outputs (NEW)
    │   ├── Table_S_KPI_Robustness_v4_honest.csv
    │   ├── Table_S_Sensitivity_RinitHorizon_v4.csv
    │   ├── Table_S_DowntimeSensitivity_v4.csv
    │   ├── Table_S_Component_Params_v4.csv
    │   ├── Supplementary_Text_MultiSource_v4_honest.txt
    │   └── Fig_MultiSource_Robustness_v4_honest.png
    ├── iso_reliability/                    ← Iso-reliability study outputs (NEW)
    │   ├── README.md                       ← Output index, raw-file (Y*) → curated-table (T*) mapping
    │   ├── outputs_v11/                    ← Raw outputs written by the protocol script (Y0–Y15 CSVs,
    │   │                                     V11_manuscript_numbers.txt, V11_manifest.json, draft figures)
    │   ├── tables/                         ← Curated, manuscript-ready tables
    │   │   ├── T01_simulation_parameters.csv
    │   │   ├── T02_policy_set.csv
    │   │   ├── T03_protocol_validity.csv
    │   │   ├── T04_iso_reliability_cost_vs_TBM.csv
    │   │   ├── T05_trigger_effect_decomposition.csv
    │   │   ├── T06_equivalence_heuristic_check_S0.csv
    │   │   ├── T07_attribution_2x2_q50.csv
    │   │   ├── T08_oracle_recovery_q50.csv
    │   │   ├── T09_information_ladder_q50.csv
    │   │   ├── T10_structural_misspecification_q50.csv
    │   │   ├── T11_cost_reweighting_400_draws.csv
    │   │   ├── T12_capacity_sweep_fixed_parameters.csv
    │   │   ├── T13_capacity_retuned_ctv_only.csv
    │   │   ├── T14_scale_experiment_q50.csv
    │   │   ├── T15_tuning_budget.csv
    │   │   ├── T16_slope_correction_bias.csv
    │   │   ├── T17_cost_optimum_flatness.csv
    │   │   ├── T18_runtime_scaling.csv
    │   │   ├── T19_policy_search_and_learning_accuracy.csv
    │   │   ├── T20_prior_misspecification_supplementary.csv
    │   │   └── T21_energy_output_q50.csv
    │   ├── figures/                        ← Publication figures (generated by 04_make_figures.py)
    │   │   ├── FIGURES.md                  ← Captions, panel guide, reading notes
    │   │   ├── Fig1_protocol.{pdf,png}
    │   │   ├── Fig2_main_results.{pdf,png}
    │   │   ├── Fig3_attribution.{pdf,png}
    │   │   ├── Fig4_information_ladder.{pdf,png}
    │   │   ├── Fig5_robustness_boundaries.{pdf,png}
    │   │   ├── FigA1_cost_reliability_view.{pdf,png}
    │   │   ├── FigA2_cost_reweighting.{pdf,png}
    │   │   ├── FigA3_tuning_budget.{pdf,png}
    │   │   └── FigA4_computation_diagnostics.{pdf,png}
    │   └── manifest/
    │       └── V11_manifest_curated.json   ← Seeds, grids, settings, assumptions, regression check
    ├── tables/                             ← KPI and statistical tables from Python simulation
    │   ├── kpis_annual.csv
    │   ├── bootstrap_ci.csv
    │   ├── delay_kpi.csv
    │   ├── cost_breakdown.csv
    │   ├── stat_tests.csv
    │   └── component_isolation.csv          ← S1+RT diagnostic results (NEW)
    └── csv/                                ← Full daily/event logs per strategy (1,096 rows each)
        ├── daily_S1_NoWeather.csv
        ├── daily_S2_WeatherCTV.csv
        ├── daily_S3_MultiPort.csv
        ├── daily_HMDP_CBM.csv
        ├── daily_S1RT_Diagnostic.csv         ← S1+RT diagnostic daily log (NEW)
        ├── completed_S1_NoWeather.csv
        ├── completed_S2_WeatherCTV.csv
        ├── completed_S3_MultiPort.csv
        ├── completed_HMDP_CBM.csv
        ├── feedback_S1_NoWeather.csv
        ├── feedback_S2_WeatherCTV.csv
        ├── feedback_S3_MultiPort.csv
        └── feedback_HMDP_CBM.csv
```

---

## Quick Start

```bash
# 1. (optional) weather preprocessing (R) – unchanged from the legacy pipeline
Rscript scripts/01_weather_preprocessing.R

# 2. smoke test of the whole protocol (a few minutes)
cd results/iso_reliability
V11_FAST=1 V11_HEADLESS=1 python ../../scripts/iso_reliability/03_iso_reliability_benchmark.py

# 3. full run (50 test replicates, 6 training seeds, B = 20,000; ~23 min on 23 worker processes)
NPROC=23 V11_HEADLESS=1 nohup python ../../scripts/iso_reliability/03_iso_reliability_benchmark.py > run_v11.log 2>&1 &

# 4. publication figures from the raw outputs
python ../../scripts/iso_reliability/04_make_figures.py --indir outputs_v11 --outdir figures --dpi 600 [--tiff]
```

**Dependencies:** `pip install numpy pandas scipy matplotlib` (Python ≥ 3.10; NumPy/SciPy/Matplotlib only, no GPU). Serif fonts: Times New Roman → Liberation Serif → STIX → DejaVu Serif fallback.

**Environment variables (protocol script):** `V11_FAST=1` (reduced design for installation checks), `V11_HEADLESS=1` (no plot windows), `NPROC` (worker processes; default = CPU count − 1), `CALIB_TARGET` (annual critical failures per turbine; adds an externally calibrated scenario S4).

**Weather input.** The protocol re-estimates the sea-state chain from `data/raw/weather_hourly_raw.csv` (path list `KMA_PATHS` at the top of the script — add the repository-relative path `../../data/raw/weather_hourly_raw.csv` for portability). If the file is absent, the script falls back to a published transition matrix and the numbers will **not** match the tables in this repository. The script prints the number of daily observations used (`1,012` in the reported run); check it against the manuscript.

---

## Simulation Model (summary)

| Component | Specification |
|-----------|---------------|
| Fleet | N = 50 turbines × C = 11 components (5 Critical, 5 Semi-critical, 1 Non-critical aggregate); daily time step |
| Degradation | Cumulative hazard H_ic(t) = λ_ic (t/η)^β, β = 2.5, η = 560 days; effective age accrues only while operating |
| Latent heterogeneity | λ_ic = m_ic^−β, m_ic = κ exp(σ z_ic), z ~ N(0,1); scenarios S0 (σ = 0, κ = 1), S1 (0.35, 1), S2 (0.35, 0.8), S3 (0.60, 1) |
| Repair | Severity Minor/Major/Replacement with p = 0.80/0.15/0.05; Kijima type II, t⁺ = (1 − RF) t⁻, RF = 0.35/0.55/0.90; durations 1/3/7 d; cost multipliers 1/3/8; λ_ic unchanged by repair in the base model |
| Access | Four-state Markov sea-state chain (Hs cut-offs 0.9/1.5/2.5 m); CTV: Hs ≤ 1.5 m & wind ≤ 10 m/s, cap 4 tasks/day; SOV: Hs ≤ 2.5 m & wind ≤ 15 m/s, cap 8 tasks/day; access modes *adaptive*, *CTV-only* (fixed daily charter) and *unconstrained* |
| Queue | Failed components first, otherwise criticality weight × (1 + 10 × weekly hazard); trigger time and execution time are distinct |
| Costs | Planned 25/10/2 and corrective 120/40/8 M KRW per task (× severity multiplier); sailing day 5 (CTV) / 30 (SOV) M KRW; lost output 14.4 M KRW per turbine-day at full derating |
| Evaluation | 730-day burn-in + 1,096-day measurement; total cost = planned + corrective + vessel + lost output (bn KRW) |

Full parameter list: [`results/iso_reliability/tables/T01_simulation_parameters.csv`](results/iso_reliability/tables/T01_simulation_parameters.csv). **All cost, repair and heterogeneity values are modelling assumptions of plausible order of magnitude, not field estimates; only the weather process is estimated from data.** Conclusions are therefore stated in relative terms.

## Policies and Information Ladder

All policies share the same queue, access, repair and cost model and differ only in how planned tasks are triggered ([`T02_policy_set.csv`](results/iso_reliability/tables/T02_policy_set.csv)).

| Information class | Policies |
|-------------------|----------|
| None | **TBM** (tuned uniform calendar), TBMm / TBMmB (frequency-matched, component-differentiated calendars; reference) |
| Effective age + nominal parameters | **EAT** (tier thresholds 0.72/0.65/0.55 + shift, 60-day minimum gap), **IDX** (cost-weighted hazard index) |
| Failure/survival history (Bayesian wear-rate learning, 13-point posterior grid) | **LTBM** (learned calendar), **BAY** (fixed prior), **BAYe** (empirical-Bayes prior), **IDXL** (index + learned wear), **PS** (cross-entropy policy search on BAYe's posterior) |
| Degradation indicator | **SEN** ×5 (iid noise σ = 0.3 / 1.0; AR(1) 0.8 with σ = 0.3 / 0.6; persistent bias) — an optimistic information bound |
| True wear rate | **ORA** (oracle; not implementable) |

Each policy has a **single** tuning parameter (interval, threshold shift, index level or intercept shift); a multi-parameter check is reported in T15.

## Protocol

| Step | Procedure | Cost (simulations) |
|------|-----------|--------------------|
| **Algorithm 1** | Tune on 6 training seeds (3000–3005) over each grid → monotone frontier → common range [k_lo, k_hi] → targets q25/q50/q75 → interpolate parameter → evaluate on 50 test seeds (1–50) → slope-adjusted cost C_adj = C − ŝ (K − K*) | Σ_π\|G_π\|·\|S_tr\| + \|Π\|(\|Q\|+1)\|S_te\| ≈ 4×10³ per scenario |
| **Algorithm 2** | TBM intervals T_c ← T_c · n_c / n′_c until total task count within 1.5 % and per-component within 5 % of the target policy (≤ 6 iterations, 10 calibration replicates) | ≤ 70 per calibration |
| **Algorithm 3** | Paired differences; 2 × 2 effects ST, L, ST×L (Eq. 7a–c); percentile bootstrap, one shared index matrix, B = 20,000, R = 50; Holm adjustment for released tables, **intervals are the primary report** | O(B·R) per contrast |

**Applicability criterion** (diagnostic, derived from an exploratory run, fixed before the final 50-replicate run — *not* a pre-registered rule): (V1) Spearman(tuning parameter, training-frontier failures) × expected sign ≥ 0.8 for TBM, EAT, ORA; (V2) ORA − TBM interval at q50 below zero. Settings failing it are reported in full and flagged, but excluded from summary counts. All settings are reported so that readers can apply alternative cut-offs.

---

## Key Results

### 1. Cost at equal reliability (q50; difference vs. tuned TBM, bn KRW; 95 % bootstrap interval)

| Policy | S0 homogeneous | S1 heterogeneous | S2 het., faster wear | S3 strong het. |
|--------|:--------------:|:----------------:|:--------------------:|:--------------:|
| EAT | −9.07 [−10.41, −7.73] | −2.43 [−3.71, −1.14] | +0.72 [−0.68, +2.11] | **+2.94** [+1.67, +4.23] |
| LTBM | +0.72 [−0.66, +2.14] | −11.16 [−12.70, −9.65] | −10.22 [−11.70, −8.79] | −27.75 [−29.20, −26.35] |
| BAY | −4.90 [−6.08, −3.69] | −15.84 [−17.24, −14.51] | −13.45 [−14.97, −11.98] | −30.63 [−32.20, −29.03] |
| BAYe | −7.39 [−8.48, −6.25] | −15.98 [−17.69, −14.28] | −13.82 [−15.30, −12.34] | −30.97 [−32.87, −29.07] |
| ORA | −9.07 [−10.41, −7.73] | −22.51 [−23.98, −21.06] | −19.97 [−21.44, −18.51] | −41.21 [−43.46, −39.00] |
| *TBM adjusted cost* | *258.9* | *287.3* | *349.4* | *340.6* |

In S0 the oracle coincides with EAT (no latent heterogeneity). The sign of the EAT contrast is not stable across reliability levels (S2: −1.5, +0.7, +1.2 at q25/q50/q75) — a reason to avoid single-point comparisons (T04).

### 2. What is left of the trigger advantage after frequency matching (cost-optimal parameters)

| Scenario | EAT − TBM (crit.) | EAT − TBMm (crit.) | EAT − TBMm (cost) |
|----------|:-----------------:|:------------------:|:-----------------:|
| S0 | −57.8 [−61.2, −54.5] | −16.6 [−19.8, −13.4] | −5.39 [−6.58, −4.21] |
| S1 | −88.6 [−92.5, −84.8] | −0.7 [−3.7, +2.1] | −1.18 [−2.43, +0.01] |
| S2 | −5.3 [−9.3, −1.0] | −1.3 [−6.2, +3.5] | −0.47 [−1.65, +0.68] |
| S3 | −23.4 [−27.5, −19.1] | **+22.3** [+18.2, +26.5] | **+4.72** [+3.59, +5.83] |

In S0, frequency matching alone explains 41.2 of 57.8 failures (71 %). A numerical check (T06) shows that even with deterministic repair and unconstrained access EAT keeps a small advantage over TBMm, so the "effective-age trigger = component-specific calendar" equivalence is used **only as a heuristic, not as a proposition**; the residual grows with stochastic repair (−14.3) and access constraints (−16.6), pointing to the trigger's response to repair severity.

### 3. 2 × 2 attribution (q50, bn KRW; negative favours the higher level)

| Scenario | State tracking (ST) | Learning (L) | Interaction ST×L |
|----------|:-------------------:|:------------:|:----------------:|
| S0 | −7.35 [−8.24, −6.44] | +2.45 [+1.40, +3.48] | +1.73 [+0.81, +2.62] |
| S1 | −3.55 [−4.41, −2.67] | −12.29 [−13.37, −11.23] | −1.12 [−2.18, −0.09] |
| S2 | −1.26 [−2.16, −0.38] | −12.20 [−13.44, −10.96] | −1.98 [−3.00, −0.93] |
| S3 | +0.03 [−0.81, +0.87] | −30.66 [−32.13, −29.19] | −2.91 [−3.77, −2.07] |

Learning dominates when heterogeneity exists and is a cost in a homogeneous fleet. A negative interaction means state tracking and learning are complementary: in S3, EAT is costlier than a calendar (+2.9) but BAY is 2.9 cheaper than LTBM. LTBM supplies 71 / 76 / 91 % of BAY's gain over TBM (S1–S3). The empirical-Bayes prior adds nothing at a correctly specified prior (BAYe − BAY: −0.1, −0.4, −0.3; all intervals include zero). Posterior-mean wear rates rank-correlate with the truth at Spearman ρ = 0.83 / 0.87 / 0.92 (BAY) and 0.80 / 0.86 / 0.91 (BAYe).

### 4. Information ladder (share of oracle benefit recovered, q50)

| Policy | S1 | S2 | S3 |
|--------|:--:|:--:|:--:|
| EAT / IDX (no wear information) | 0.11 / −0.09 | −0.04 / 0.00 | −0.07 / −0.11 |
| LTBM | 0.50 | 0.51 | 0.67 |
| IDXL | 0.47 | 0.58 | 0.95 |
| BAY / BAYe | 0.70 / 0.71 | 0.67 / 0.69 | 0.74 / 0.75 |
| PS | 0.72 | 0.69 | 0.74 |
| SEN (5 variants) | 0.78–1.01 | 0.69–0.91 | 0.85–0.99 |

PS is statistically indistinguishable from BAYe (−0.24, +0.08, +0.40 bn KRW) — expected, because PS receives BAYe's posterior as a feature and its training cost improved by only 0.4–1.0 %. It must be read as a policy-search refinement within the estimate-then-act class, **not as an independent learning baseline**. IDXL is 5.4 and 2.3 bn KRW costlier than BAYe in S1/S2 but 8.0 [−11.9, −4.2] cheaper in S3; the comparison is confounded (cost weighting and trigger form change together). Sensor policies are an optimistic bound (direct noisy reading of true wear, no sensing cost).

### 5. Robustness

* **Structural misspecification** (Weibull shape 2.0/3.2, repair-induced wear +5/+10/+25 % RF, wear redrawn at replacement, +60 % drift, bimodal and heavy-tailed heterogeneity; S1 and S3): in the **16 valid settings** BAYe and LTBM both have CI < 0 vs. TBM (16/16); EAT 6/16. BAYe − LTBM is below zero in 14/16 (CI includes zero for renewal-at-replacement and heavy tails in S3). Median retention 87 % (range 42–117 %); lowest under bimodal (52 % S1, 42 % S3) and heavy-tailed heterogeneity (81 % S1, 48 % S3) — the empirical-Bayes prior adapts parameters of the heterogeneity distribution, not its family. The two +25 % repair-wear settings are outside the frame (non-monotone frontiers; in S1 the oracle is costlier than the calendar, +52.2 bn KRW). The S3 +10 % setting is **borderline** (oracle monotonicity 0.89; an earlier smaller run flagged it at 0.79).
* **Cost re-weighting** (T11): ±50 % per component leaves signs of LTBM−TBM, BAYe−TBM, BAYe−LTBM unchanged; in 400 joint Latin-hypercube draws BAYe−TBM and LTBM−TBM are favourable and significant in 100 %, BAYe−LTBM in 100 % (S1, S2) and 94 % (S3); EAT−TBM in 66 / 21 / 0 %; BAYe vs. IDXL is cost-dependent (76 / 60 / 0 %). The magnitude is most sensitive to planned-maintenance cost (halving it reduces S1 BAYe−TBM from −16.0 to −7.6).
* **Tuning budget** (T15): tier-specific calendars (TBM3) change cost by −0.45 [−1.77, +0.87] (S1) and −0.85 (S3); BAYe2 remains 11.8 / 26.3 bn KRW cheaper than TBM3.
* **Slope-correction bias** (T16): 15 of 16 displaced-target checks include zero; maximum bias ≈ 1.4 bn KRW (0.4 % of cost). It is an adequate first-order approximation, presented as a diagnostic, not a proved property.
* **Cost-only tuning is unreliable** (T17): BAYe's cost optimum is flat — three grid points within 1 bn KRW span 165 (S1) / 126 (S3) critical failures — which is why equal-reliability contrasts are the primary analysis.

### 6. Where the frame holds: capacity and scale

| Check | Result |
|-------|--------|
| Fixed core parameters, CTV-only | BAYe−TBM CI includes zero for capacity ≤ 4; significant from 5 (S1 −4.8; S3 −21.1), plateau at 6–8 |
| Fixed core parameters, adaptive (CTV+SOV) | Significant from capacity 2 |
| Mean task queue under TBM, CTV-only | 226 (S1) / 314 (S3) at capacity 4 vs. 11 / 20 at capacity 8; at capacity 1 (S1) failed components spend ≈ 394,000 component-days under repair or waiting (65 % of all) |
| Re-tuned (30 test / 4 training seeds) | BAYe −10.7 (−3.7 %) in S1 at capacity 5; −24.5 (−7.2 %) in S3 at capacity 6; −14.7 / −37.2 at capacity 12. Below these, the common failure range collapses (Algorithm 1 fallback) and contrasts such as −219 / −918 are artefacts that are **not interpreted** (T13) |
| Fleet size 10–200, capacity scaled with N | BAYe advantage stable: −5.4, −6.9, −5.7, −6.2, −6.0 % (S1); −11.2, −10.8, −8.8, −10.6, −10.0 % (S3) |
| Fleet size, capacity fixed | Valid only for N = 10, 25; saturates at 100 and 200 |

Failure counts are non-monotone in capacity under saturation (a failed, waiting component cannot fail again; e.g. 736 / 1,138 / 1,125 at capacities 1/2/3, S1, TBM), so the capacity analysis uses cost, output, queue length and failed component-days. The boundary is interpreted as the **onset of an unstable task queue**, not as an independent effect of weather.

### 7. Computational cost

One 50-turbine simulation takes 0.22 s (TBM), 0.24 s (EAT), 0.69 s (BAYe) on one process; fitted time ∝ N^0.45 / N^0.47 / N^0.68 over N = 10–200 (T18). These exponents are below the linear scaling suggested by the O(T·N·C·K_g) operation count (K_g = 13), most plausibly because fixed per-day overheads dominate for small fleets — an observation about this implementation, not a complexity result. The complete study runs in **1,357 s** on 23 worker processes.

---

## Figures

Publication figures (black/white/gray, vector PDF + 600-dpi PNG, optional TIFF-LZW) are produced by `scripts/iso_reliability/04_make_figures.py` from the raw `outputs_v11/` files. See [`results/iso_reliability/figures/FIGURES.md`](results/iso_reliability/figures/FIGURES.md) for captions and reading notes.

| Figure | Content |
|--------|---------|
| **Fig. 1** `Fig1_protocol` | Inputs, Algorithms 1–3 and the 2 × 2 attribution design |
| **Fig. 2** `Fig2_main_results` | (a) cost vs. TBM at q50 by scenario; (b) oracle benefit recovered; (c) stability across q25/q50/q75 |
| **Fig. 3** `Fig3_attribution` | (a) EAT−TBM split into frequency/differentiation and residual trigger effect; (b) 2 × 2 interaction plots |
| **Fig. 4** `Fig4_information_ladder` | Forest plot of all policies at q50, ordered by information used |
| **Fig. 5** `Fig5_robustness_boundaries` | (a,b) structural misspecification; (c) capacity sweep; (d) fleet-size scaling |
| **Fig. A1–A4** | Cost–reliability view; cost re-weighting CDFs; tuning budget; run-time, optimum flatness and slope-correction diagnostics |

> **Naming note:** the protocol script also writes draft figures `Fig2_iso_cost` … `Fig6_capacity_scale` into `outputs_v11/`. These are superseded by the publication figures above.

---

## Outputs and Provenance

The protocol script writes raw files `Y0`–`Y15` to `results/iso_reliability/outputs_v11/`. Curated tables in `results/iso_reliability/tables/` are manuscript-ready extracts of those files (see mapping in [`results/iso_reliability/README.md`](results/iso_reliability/README.md)). All numbers are simulation results under the stated assumptions and carry 95 % percentile-bootstrap intervals over 50 paired replicates; the **replicate, not the day, is the unit of analysis**.

---

## Reproducibility

* Seeds — training 3000–3005; test 1–50; calibration 9000–9009; policy-search training uses the first three training seeds; extra random streams for misspecification and sensor noise are separate and do not alter the base streams.
* Common random numbers — weather path, initial states, failure and severity draws are shared across policies for a given seed, so contrasts are paired.
* Regression check — the engine reproduces the critical-failure counts of the earlier code version exactly (maximum absolute difference 0.0000, BAYe, q50, S1).
* `V11_FAST=1` runs the full flow with a reduced design (3 replicates, 1 training seed) for installation checks; reported numbers use the full mode.
* Software versions and run time are written to `outputs_v11/V11_manifest.json`; a curated copy is in `results/iso_reliability/manifest/`.

## Data Availability

Raw hourly KMA observations cannot be redistributed and are available from the corresponding author on reasonable request. The estimated weather-chain parameters, simulation code, configuration, seeds and result tables are provided here.

## Limitations

1. **Simulation evidence.** Cost, restoration-factor, duration, derating, heterogeneity, misspecification and sensor parameters are assumptions; absolute failure counts are not intended for site decisions.
2. **Well-specified learner in the base model.** The learner's likelihood and prior family match the data-generating process; misspecification is relaxed one aspect at a time (joint misspecification remains open).
3. **Single-parameter tuning** of each core policy (multi-parameter check in T15; grids are not supersets, so small differences of either sign reflect grid coarseness).
4. **Baselines.** PS shares BAYe's posterior (not independent); IDXL confounds cost weighting and trigger form; sensors are an optimistic bound; a rolling-horizon or end-to-end trained policy is a natural next baseline.
5. **Applicability criterion** is a post-hoc diagnostic; one setting (S3, +10 % repair-induced wear) lies close to the threshold.
6. **Supplementary blocks** — capacity analyses (30 test seeds), the 13-point posterior grid, and the prior-sensitivity experiments (T20) were run at an earlier stage of the study.
7. **Omitted:** opportunistic maintenance, spare-parts constraints, crew scheduling and weather seasonality.

## License

Released under the **MIT License** (see [LICENSE](LICENSE)). Raw KMA weather data are subject to KMA data-sharing terms and are not redistributed; EDP Open Data (legacy validation) is used under the EDP Open Data platform terms.
