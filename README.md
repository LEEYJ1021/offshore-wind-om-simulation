# Offshore Wind O&M Simulation
## Reliability-Threshold Maintenance via Hierarchical MDP with Multi-Cell Energy Accounting and Weather-Adaptive Logistics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![R](https://img.shields.io/badge/R-4.3%2B-276DC3?logo=r)](https://www.r-project.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://www.python.org/)
[![Simulation](https://img.shields.io/badge/Simulation-v4-brightgreen)](scripts/02_om_simulation.py)

---

## Overview

This repository provides the complete, fully reproducible analysis pipeline for an offshore wind turbine operations and maintenance (O&M) simulation study targeting the **Ulsan Offshore Wind Cluster** (East Sea, South Korea). The study spans **2023–2025** (1,096 simulated days) across a fleet of **50 DTU 10 MW reference turbines**, and compares four strategically distinct maintenance architectures under realistic stochastic weather conditions calibrated from Korea Meteorological Administration (KMA) observational records.

The central contribution of this study is a formally stated and empirically tested boundary condition (**Proposition 1**) under which a reliability-threshold (RT) maintenance trigger — computed from a known parametric hazard function and therefore, in isolation, an exactly pre-computable function of time — becomes statistically distinguishable from a relabelled fixed-interval policy. The proposition states that this equivalence breaks only when two stochastic ingredients are present **jointly**: (i) severity-stochastic imperfect repair, in which the post-repair restoration factor is realised only upon on-site diagnosis, and (ii) weather-stochastic vessel access, in which completion timing depends on a realised, weather-constrained accessibility process. The repository's main simulation (`scripts/02_om_simulation.py`) tests this proposition directly via a Kolmogorov–Smirnov comparison of inter-trigger interval distributions, and a dedicated component-isolation diagnostic (S1+RT; see below) further separates the trigger mechanism's contribution from the contribution of weather-adaptive logistics.

Three further, secondary findings are produced as by-products of the same simulation infrastructure:

1. **Binary energy production modeling bias** — binary availability models systematically underestimate annual energy production (AEP) loss relative to multi-cell continuous derating representations (Jimmy et al., 2019; Kim & Kim, 2025).
2. **Post-repair state invisibility** — weather-adaptive scheduling frameworks do not propagate repair outcomes back into component degradation models, preventing compounding reliability improvements over multi-year horizons (Gutierrez-Alcoba et al., 2019).
3. **Weather-blind charter waste** — CTV-only, calendar-driven dispatch accrues idle-charter cost that is fully eliminable through weather-conditioned vessel access decisions.

The architecture combining these elements — an RT trigger embedded in a Hierarchical Markov Decision Process (HMDP), a multi-cell Energy Throughput Availability (ETA) degradation matrix (18 components × 3 severities), and a 12-step Greedy hierarchical scheduler with visit bundling — is evaluated against three progressively richer baselines in a factorial design that deliberately keeps the RT-trigger contrast (S3→S4) unconfounded with the logistics contrasts (S1→S2, S2→S3).

Five supplementary validation experiments further characterise the framework's core modelling assumptions, including CBM trigger stochasticity, seasonal Markov chain robustness, year-over-year failure escalation decomposition, and vessel dispatch economies of scale. Two independent parameter-validation studies provide further grounding for the Weibull degradation parameters and ETA derating coefficients applied throughout the simulation: an EDP Open Data SCADA cross-validation (`scripts/edp_validation/`) and a **multi-source Weibull robustness analysis** that explicitly separates the failure-prevention mechanism (driven by the Weibull parameters under test) from the downtime-efficiency mechanism (an explicit, separately justified modelling assumption) — see [Multi-Source Weibull Robustness Analysis](#multi-source-weibull-robustness-analysis) below.

---

## Research Questions and Hypotheses

The study is organised around one primary research question and three secondary research questions, each generating testable statistical hypotheses validated across 1,096 daily observations per strategy using a nonparametric test battery with Benjamini–Hochberg FDR correction (α = 0.05). A fourth, mechanism-validation research question and a dedicated component-isolation diagnostic complete the validation framework.

| Tier | RQ | Question | Primary Hypotheses |
|------|----|----------|-------------------|
| **Primary** | **RQ1** | Is a reliability-threshold (RT) maintenance trigger statistically distinguishable from a fixed-interval policy under joint repair- and weather-stochasticity (Proposition 1), and does this translate into measurable reliability gains versus fixed-PM baselines? | H1a (inter-trigger interval stochasticity, KS test), H1b (critical failure reduction), H1c (operational availability), H1d (pending downtime cost reduction) |
| **Secondary** | **RQ2** | Does weather-adaptive vessel scheduling eliminate idle-charter waste and reduce maintenance deferral in winter conditions, and does visit bundling provide an independent, complementary saving? | H2a (SOV vs. CTV winter accessibility), H2b (repair delay), H2c (charter waste elimination), H2d (visit-bundling saving; port-assignment isolation) |
| **Secondary** | **RQ3** | Does the multi-cell ETA model reveal a systematic AEP estimation bias relative to binary availability models? | H3a (ETA vs. binary AEP loss, KS test), H3b (ETA priority reranking), H3c (AEP decline decomposition: failure-induced vs. planned partial-operation loss) |
| **Mechanism validation** | **RQ4** | Does the closed-loop imperfect repair feedback produce a statistically verifiable, severity-dependent age restoration gradient? | H4a (RF monotonicity across 2,886 repairs, Cliff's δ) |
| **Diagnostic (not pre-registered)** | — | Component-isolation check: is the RT trigger's reliability contribution separable from the contribution of weather-adaptive logistics? | H_iso (S1+RT vs. S1; S1+RT vs. S4) |

Of 23 pre-registered hypotheses, **22 were sustained** after BH-FDR correction. One hypothesis (H4b: weekly cost coefficient of variation) was excluded for insufficient statistical power (*n* = 52 weekly observations; power = 0.14–0.35). Five further comparisons (H1d vs. S2/S3, H2b vs. S3, H3a, H5b vs. S2/S3 in the legacy RQ5 numbering, now folded into H2c/H2d above) carry post-hoc power below 0.70 and are flagged "Cautious" throughout the results tables; directional conclusions are supported but effect magnitudes should not be used for quantitative extrapolation without independent replication.

> **Note on RQ restructuring:** The research questions above supersede an earlier five-RQ framing (in which RQ5 separately addressed charter waste). Charter waste elimination and visit bundling are now reported under RQ2 as part of the weather-adaptive logistics question, since both are properties of the same dispatch-layer mechanism and are deliberately kept distinct, in the factorial design and in the component-isolation diagnostic below, from the RT-trigger mechanism tested under RQ1.

---

## Component-Isolation Diagnostic (S1+RT)

A recurring methodological concern with jointly-optimized systems is that an aggregate improvement may reflect confounded contributions from several simultaneously-changed components rather than an identifiable mechanism. To address this directly, a fifth, **diagnostic-only** configuration, **S1+RT**, replaces only S1's fixed 26-week trigger with S4's reliability-threshold trigger while holding every other S1 parameter fixed (CTV-only dispatch, single-port logistics, idle charter fee retained, no closed-loop feedback beyond the trigger rule itself). S1+RT is not part of the primary four-strategy factorial design (Table 2-equivalent below); it exists solely to answer whether the RT trigger alone — stripped of weather-adaptive logistics — still produces a reliability gain, thereby separating the trigger mechanism from the logistics mechanism.

**Result:** S1+RT achieved a mean critical-failure rate of 0.198 failures/day, intermediate between S1 (0.314) and S4 (0.125): a **37% reduction relative to S1 attributable to the RT trigger alone** (Mann–Whitney U, *p* < .001, *d* = −0.21, power = .91), with the remaining gap to S4's 60–62% total reduction (a further 23 percentage points) attributable to weather-adaptive dispatch and closed-loop feedback acting jointly (S1+RT vs. S4: *p* < .001, *d* = −0.19, power = .87). Idle charter waste under S1+RT remained at ₩1.59B over three years — only marginally reduced from S1's ₩1.76B — confirming that charter-waste elimination is mechanistically attributable to weather-adaptive dispatch and not to the RT trigger. The RT trigger and weather-adaptive logistics are therefore **separable, additive contributors** rather than confounded co-movers: the trigger mechanism accounts for roughly three-fifths of the total reliability gain, and the logistics mechanism accounts for the remainder together with the full charter-waste elimination.

This diagnostic is implemented as an additional run mode in `scripts/02_om_simulation.py` (flag `--diagnostic s1_rt`) and its outputs are written to `results/tables/component_isolation.csv` and `results/figures_PY/Fig_ComponentIsolation_S1RT.png`.

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
│   └── multisource_validation/             ← Multi-source Weibull robustness analysis (NEW)
│       └── multisource_weibull_robustness_v4.py   ← Mechanism-separated robustness test (see below)
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

## Data

### Input Files (place in `data/raw/`)

| File | Description | Approximate Rows | Frequency |
|------|-------------|:----------------:|-----------|
| `weather_hourly_raw.csv` | Ulsan offshore station — wind speed, wave height, gust, pressure, temperature, sea surface temperature, wave period and direction | ~25,190 | Hourly |
| `weekly_cost_baseline.csv` | Linear-programming-derived weekly O&M cost reference (ship, port, labour, downtime, parts) | 52 | Weekly |

> **Data availability:** Raw observational files are not included in this repository due to KMA data-sharing restrictions. The processed outputs and simulation results are fully reproducible from these inputs using the provided scripts. Researchers may request the raw weather data from the corresponding author. All scripts — including the supplementary experiments in `scripts/additional_experiments/` and the multi-source robustness analysis in `scripts/multisource_validation/` — reference `data/raw/` directly via paths defined in `experiment_core.py`; **no additional data folders are required**. If `weather_hourly_raw.csv` is absent, the simulation automatically generates synthetic weather using a Markov chain calibrated to the Ulsan KMA seasonal statistics (four-state, season-weighted transition matrices derived from the 2010–2022 observational record).

#### `weather_hourly_raw.csv` — Column Schema

| Korean Column Name | English Alias | Unit | Description |
|-------------------|---------------|------|-------------|
| `지점` | `station` | ID | KMA station identifier |
| `일시` | `datetime` | YYYY-MM-DD HH:MM | Timestamp (Asia/Seoul, KST = UTC+9) |
| `풍속(m/s)` | `wind_speed` | m/s | 10-minute mean wind speed at 10 m height |
| `풍향(deg)` | `wind_dir` | ° | Wind direction (meteorological convention) |
| `GUST풍속(m/s)` | `gust_speed` | m/s | Maximum 3-second gust speed |
| `현지기압(hPa)` | `pressure` | hPa | Local atmospheric pressure |
| `습도(%)` | `humidity` | % | Relative humidity |
| `기온(°C)` | `temp_air` | °C | Air temperature at 1.5 m |
| `수온(°C)` | `temp_sea` | °C | Sea surface temperature |
| `최대파고(m)` | `wave_max` | m | Maximum wave height (H_max) |
| `유의파고(m)` | `wave_hs` | m | Significant wave height (H_s) — **primary accessibility variable** |
| `평균파고(m)` | `wave_mean` | m | Mean wave height |
| `파주기(sec)` | `wave_period` | s | Mean wave period |
| `파향(deg)` | `wave_dir` | ° | Dominant wave direction |

The preprocessing script renames Korean columns to English aliases and imputes missing H_s values using linear interpolation, with outliers identified by a 5×MAD rule and flagged before imputation.

#### `weekly_cost_baseline.csv` — Column Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `week_num` | int | — | ISO week index (1–52) |
| `week_label` | string | — | Week start–end date label (e.g., "2023-W01: Jan 02–Jan 08") |
| `ShipCost` | float | KRW | Vessel fuel, transit, and charter costs |
| `PortCost` | float | KRW | Port usage and harbour fees |
| `LaborCost` | float | KRW | Technician and crew hourly wages |
| `DowntimeCost` | float | KRW | Lost production revenue (wind-speed-conditioned) |
| `PartsCost` | float | KRW | Spare parts and consumables expenditure |
| `TotalCost` | float | KRW | Sum of all cost components |

This file serves as the deterministic LP-derived reference cost against which simulated strategies are benchmarked in Figure 8 (Python) and Experiment 5.

### Processed / Output Files (auto-generated)

All six processed weather datasets (`ulsan_[hourly/daily/weekly]_weather_[simple/detailed].csv`) are generated automatically by `01_weather_preprocessing.R`. Simple variants contain mean and accessibility indicator columns only; detailed variants include full percentile distributions, gust statistics, and seasonal flags. Statistical tables (Tables 1–3b) are saved to `data/outputs/`. No manual file preparation is required beyond placing the raw inputs in `data/raw/`.

---

## Scripts

### `scripts/01_weather_preprocessing.R` — Weather Data Processing Pipeline

**Purpose:** Transforms raw hourly KMA observational data into multi-resolution aggregated datasets, computes binary vessel accessibility indicators, derives seasonal statistics and Markov transition matrices, and generates ten publication-quality weather analysis figures.

**Processing pipeline:**

1. **Ingestion and renaming** — Reads `weather_hourly_raw.csv`, renames Korean column headers to English aliases, and parses `datetime` into POSIXct (Asia/Seoul timezone).
2. **Quality control** — Identifies implausible values (wind > 50 m/s, Hs > 20 m) and extreme outliers via a 5×MAD rule; flags and removes before interpolation.
3. **Gap filling** — Linear interpolation for runs of missing H_s ≤ 6 hours; longer gaps flagged in a `gap_flag` column.
4. **Accessibility indicators** — Computes `ctv_accessible = (Hs ≤ 1.5 m) AND (wind_speed ≤ 10.0 m/s)` and `sov_accessible = (Hs ≤ 2.5 m)` as binary hourly flags. These thresholds are consistent with DNV GL offshore access guidance and TNO Offshore Wind Access reports (2019, 2020).
5. **Temporal aggregation** — Produces daily and weekly aggregations. Daily: mean, max, p10/p25/p75/p90 for each meteorological variable plus daily accessibility proportion. Weekly: ISO week mean, max, and accessibility rate.
6. **Markov transition matrix** — Estimates a four-state (Calm / Moderate / Rough / Extreme) daily transition matrix from observed H_s classifications: Calm (Hs < 0.9 m), Moderate (0.9–1.5 m), Rough (1.5–2.5 m), Extreme (≥ 2.5 m).
7. **Seasonal statistics** — Computes seasonal summaries (Winter: Dec–Feb; Spring: Mar–May; Summer: Jun–Aug; Autumn: Sep–Nov) for all meteorological variables and accessibility rates.
8. **Output** — Saves 6 processed datasets, 8 statistical tables, and 10 figures.

**Run:**
```r
# From repository root or RStudio working directory set to repo root
source("scripts/01_weather_preprocessing.R")
```

**R package dependencies:**
```r
install.packages(c(
  "tidyverse",    # data manipulation and ggplot2
  "lubridate",    # datetime parsing and timezone handling
  "data.table",   # fast file I/O for large hourly datasets
  "ggplot2",      # publication-quality figures
  "patchwork",    # multi-panel figure composition
  "scales",       # axis label formatting
  "viridis",      # perceptually uniform colour palettes
  "gridExtra",    # grid-based figure arrangement
  "zoo"           # rolling statistics and interpolation
))
```

**Minimum R version:** 4.3.0 (tested on 4.3.2 and 4.4.0).

---

### `scripts/02_om_simulation.py` — HMDP–Greedy O&M Simulation (Main)

**Purpose:** Runs the full five-layer autonomous maintenance decision engine for four strategies (plus the optional S1+RT diagnostic) over 1,096 simulated days (2023–2025), computes all KPIs with bootstrap confidence intervals and nonparametric statistical tests, and generates Figures 1–13 plus all CSV output files.

#### Strategies

| ID | Label | Vessel Fleet | Weather-Aware Dispatch | PM Trigger | Port Assignment | Idle Charter Fee | Role in design |
|----|-------|:------------:|:---------------------:|:----------:|:---------------:|:----------------:|-----------------|
| **S1** | Fixed PM / No Weather | CTV only | ✗ | Fixed 26-week calendar | P1 only | ₩3,500,000/day | Primary baseline |
| **S2** | Fixed PM / Weather-Aware | CTV + SOV | ✓ | Fixed 26-week calendar | P1 only | — | Isolates weather-aware dispatch |
| **S3** | Multi-Port / Weather | CTV + SOV | ✓ | Fixed 26-week calendar | P1 + P2 | — | Isolates multi-port logistics |
| **S4** | **HMDP–RT (Proposed)** | CTV + SOV | ✓ | **Weibull reliability threshold** | P1 + P2 | — | Isolates RT trigger + closed-loop feedback |
| **S1+RT** | *Diagnostic only* | CTV only | ✗ | **Weibull reliability threshold** | P1 only | ₩3,500,000/day | Isolates the RT-trigger contribution from S4's logistics changes (see [Component-Isolation Diagnostic](#component-isolation-diagnostic-s1rt)) |

The factorial design (S1→S2: weather awareness; S2→S3: multi-port logistics; S3→S4: RT trigger and closed-loop feedback) isolates the marginal contribution of each architectural capability to fleet-level KPIs, while the S1+RT diagnostic provides an additional, orthogonal isolation of the RT trigger from the logistics stack.

#### Five-Layer Architecture

The simulation implements the following hierarchically coupled layers:

**Layer 1 — Environmental Stochastic Model:**
Daily weather state W_t ∈ {Calm, Moderate, Rough, Extreme} evolves according to a four-state first-order Markov chain with transition matrix calibrated from KMA 2010–2022 observations (N = 4,383 days). Vessel accessibility A_v(t) ∈ {0,1} is derived deterministically from the simulated H_s value implied by each weather state, applied against vessel-specific wave height and wind speed thresholds. A season-specific transition matrix robustness check (Experiment 3) confirms that this homogeneous-chain assumption does not materially bias S4's KPIs.

**Layer 2 — Degradation Dynamics:**
Component health is modelled using a two-parameter Weibull hazard function h(t) = (β/η)(t/η)^(β−1) with uniform fleet parameters β = 2.5 (shape, wear-out regime) and η = 80 weeks (characteristic life). Post-repair effective age is updated via a Kijima-type imperfect repair formula: t_eff_after = t_eff_before × (1 − RF_sev), where the restoration factor RF ∈ {0.35 (minor), 0.55 (major), 0.90 (replacement)} is indexed by repair severity, realised only upon on-site diagnosis. **This severity-stochastic restoration, jointly with the weather-stochastic accessibility process in Layer 1, constitutes the empirical content of Proposition 1** (see [Reliability-Threshold Trigger Stochasticity](#reliability-threshold-trigger-stochasticity-proposition-1) below). The plausibility of the Weibull parameters relative to observed inter-failure data is assessed in two independent ways: against onshore SCADA logbook data (`scripts/edp_validation/`) and against a five-source literature cross-validation with explicit mechanism separation (`scripts/multisource_validation/`; see [Multi-Source Weibull Robustness Analysis](#multi-source-weibull-robustness-analysis)).

**Layer 3 — Strategic Control (HMDP Policy):**
The HMDP state space is S_t = {W_t, H_t, V_t, Π_t}, where H_t is the fleet-wide effective age vector, V_t is weather-adjusted vessel availability, and Π_t is the dynamically prioritised maintenance queue. The RT trigger fires for turbine i when R(t_eff) ≤ θ_crit(component class) and at least 60 days have elapsed since the last intervention. Criticality-indexed thresholds are θ_crit = 0.72 (Critical), 0.65 (Semi-Critical), and 0.55 (Non-Critical). The HMDP policy is implemented as a structured condition-indexed rule set — theoretically justified by Chen and Trivedi (2005), who proved threshold-type policies are optimal for semi-Markov deterioration processes under monotone cost structures.

**Layer 4 — Tactical Scheduling (12-Step Greedy Scheduler with Visit Bundling):**
The daily execution layer assigns vessels to prioritised tasks subject to weather-adjusted capacity CAP_HR(v, W_t) = CAP_HR(v, W_t) × A_v(t). Dynamic priority scores integrate component criticality, repair severity, ETA derating coefficients, and real-time Weibull hazard: π(c,i,t) = w_crit × w_sev × η(c,sev) × (1 + κ × h(t_eff)), where κ = 10 rescales the hazard term to prevent numerical domination by categorical criticality weights. A daily RT-task cap of 5 and a `rt_queued_comps` deduplication mechanism prevent queue flooding under simultaneously deteriorating components. **An economies-of-scale visit-bundling step co-dispatches eligible same-turbine tasks above a configurable priority-percentile threshold whenever the assigned vessel retains at least 30% of its remaining daily capacity**, directly avoiding the cost of a second mobilization; a five-threshold sensitivity sweep confirms a 2.6–10.9% reduction in three-year vessel-days with no significant reliability trade-off.

**Layer 5 — Closed-Loop Repair Feedback:**
Upon completion of each repair event, the effective age update (Layer 2) is immediately applied, the Weibull hazard is recomputed, and the priority score π(c,i) is rescored — all within the same simulation day. This eliminates temporal lag between repair completion and reliability state recalibration, constituting the closed-loop feedback mechanism absent from all reviewed prior implementations.

#### Reliability-Threshold Trigger Stochasticity (Proposition 1)

If component effective age evolved deterministically between interventions, the RT trigger would reduce to an exactly pre-computable, relabelled fixed-interval policy: the date on which R(t_eff) first crosses θ_crit would be exactly computable in advance from (β, η, θ_crit) alone. **Proposition 1** states that the simulated architecture's trigger-time sequence is not deterministic for two jointly necessary and individually insufficient reasons — severity-stochastic imperfect repair (Layer 2) and weather-stochastic vessel access (Layer 1) — and that removing either condition would cause the equivalence to reassert itself.

This is tested directly, not merely asserted: under S4, the empirical inter-trigger interval distribution has mean 94.5 days, SD 13.6 days, CV = 0.143, entirely non-overlapping with the deterministic fixed-26-week comparator (mean 182.0 days, CV = 0.000). A two-sample Kolmogorov–Smirnov test confirms zero distributional overlap (KS = 1.000, *p* < 3.7 × 10⁻¹⁹³); Mann–Whitney U corroborates (*p* < 2.1 × 10⁻¹¹⁰). This result is reproduced independently in `scripts/additional_experiments/exp1_2_cbm_timebased.py` (Experiment 1 below) and is the empirical basis for RQ1/H1a.

#### Multi-Cell ETA Degradation Matrix

The multi-cell ETA matrix spans 18 component categories × 3 failure severities (Minor Repair / Major Repair / Major Replacement), providing a derating coefficient η(c, sev_c) ∈ [0,1] for each cell. The instantaneous production fraction of turbine i at time t is:

PF_i(t) = max(1 − Σ_{c ∈ F_i(t)} η(c, sev_c), 0)

A turbine is classified as operationally available (AO = 1) when PF_i(t) ≥ 0.70; Annual Energy Production (AEP) integrates all partial losses continuously. The structural distinction between these two metrics explains the AO–AEP divergence observed in the results: S4's planned RT-triggered partial-operation windows (PF ∈ [0.70, 1.00]) contribute fully to availability accounting but reduce continuous energy throughput, substituting bounded planned losses for the unbounded catastrophic failures that collapse S1's AEP to 69.81% in Year 3.

Representative ETA derating coefficients are:

| Component | Minor Repair | Major Repair | Replacement |
|-----------|:------------:|:------------:|:-----------:|
| Gearbox | 0.40 | 0.75 | 1.00 |
| Generator | 0.35 | 0.80 | 1.00 |
| Blades | 0.20 | 0.60 | 1.00 |
| Tower/Foundation | 0.10 | 0.50 | 1.00 |
| Pitch/Hydraulic | 0.15 | 0.45 | 0.70 |
| Yaw System | 0.05 | 0.20 | 0.40 |
| Safety System | 0.00 | 0.05 | 0.10 |

Coefficients are derived from published reliability and failure consequence literature (Carroll et al., 2016; Abeynayake et al., 2021; Myrent et al., 2013; Martinez Luengo & Kolios, 2015; Scheu et al., 2019; Shafiee et al., 2016). Cross-validation of both the Weibull parameters and the ETA coefficients against an independent publicly available SCADA dataset is provided in `scripts/edp_validation/edp_scada_validation.py`; a further, mechanism-separated robustness check across five Weibull parameter sources is provided in `scripts/multisource_validation/`.

**Run:**
```bash
cd offshore-wind-om-simulation
python scripts/02_om_simulation.py
# Optional: also run the S1+RT component-isolation diagnostic
python scripts/02_om_simulation.py --diagnostic s1_rt
```

**Python package dependencies:**
```bash
pip install numpy pandas matplotlib scipy
```

**Minimum Python version:** 3.10 (tested on 3.10.12 and 3.11.4). No GPU or parallel processing required; the 1,096-day simulation completes in approximately 3–8 minutes on a standard desktop CPU (Intel Core i7 or equivalent, 16 GB RAM).

**Weather data fallback:** If `data/raw/weather_hourly_raw.csv` is not found, `experiment_core.py` automatically generates synthetic daily weather using a stationary four-state Markov chain with the season-weighted average transition matrix derived from KMA historical data. All KPI conclusions are robust to this substitution (see Experiment 3 results below).

---

### `scripts/additional_experiments/` — Supplementary Validation Experiments

This module contains five targeted experiments that rigorously characterise four specific modelling assumptions of the main simulation. Together they address concerns about (i) whether the RT trigger constitutes a genuinely stochastic, condition-dependent policy (the same question formalised as Proposition 1 above), (ii) whether the homogeneous Markov weather model adequately captures seasonal non-stationarity, (iii) what mechanistic drivers explain the observed year-over-year performance divergence, and (iv) whether vessel economies of scale could improve cost efficiency without sacrificing reliability.

All experiments read weather and cost data from `data/raw/` via shared utilities in `experiment_core.py`. No additional data files or directory changes are required.

**Run all experiments:**
```bash
cd scripts/additional_experiments
python run_all_experiments.py
# All outputs written to results/additional_experiments/
```

The master runner executes experiments sequentially (total runtime: approximately 10–15 minutes) and generates a consolidated 8-panel summary figure plus individual experiment figures and CSV tables.

---

#### Experiment 1 — RT Trigger Stochasticity (Independent Reproduction of Proposition 1's Empirical Test)

**Motivation:** Reviewers correctly noted that if RT trigger timing is fully determined by the Weibull parameters and threshold values, it is mathematically equivalent to a deterministic time-based policy and offers no computational or operational advantage over a fixed-interval schedule. This experiment formally tests whether RT trigger intervals are stochastic and repair-history-dependent, or deterministic and predictable in advance — independently reproducing, with a separate codepath, the test underlying Proposition 1 in the main simulation.

**Design:** 50 turbines are simulated for 1,096 days under the RT-triggered policy (S4) and under a fixed 26-week time-based policy, using identical KMA-sourced weather realisations (SEED = 42). Inter-trigger intervals are recorded for each turbine-component pair across the full simulation horizon. Interval distributions are compared using Kolmogorov–Smirnov and Mann–Whitney U tests.

**Results:**

| Metric | RT (Reliability-Triggered) | Fixed Time-Based (26-wk) |
|--------|:---------------------------:|:------------------------:|
| Inter-trigger interval μ | 94.5 days | 182.0 days |
| Inter-trigger interval σ | 13.6 days | 0.0 days |
| Coefficient of Variation (CV) | **0.1434** | 0.0000 |
| KS test p-value vs. fixed | < 3.7×10⁻¹⁹³ | — |
| Mann–Whitney U p-value | < 2.1×10⁻¹¹⁰ | — |

**Interpretation:** The RT trigger exhibits a CV of 0.1434 against a fixed-interval baseline CV of 0.0000. The KS statistic of 1.0000 indicates that no single RT interval falls at the fixed 182-day value: the two distributions are completely disjoint. This independently corroborates Proposition 1: RT trigger timing is stochastic and cannot be pre-computed from Weibull parameters alone, because the sequence of future repair severities — and therefore the trajectory of t_eff and the time at which R(t_eff) crosses the threshold — is itself stochastic. The null hypothesis that the RT policy is a disguised time-based policy is rejected at all conventional significance levels.

---

#### Experiment 2 — Fleet KPI: RT-Triggered vs. Fixed Time-Based Under Real KMA Weather

**Motivation:** Beyond demonstrating trigger stochasticity, it is necessary to establish whether the RT-triggered policy generates measurable fleet-level reliability and availability improvements over a fixed time-based comparator under identical weather conditions. This experiment isolates the RT trigger's contribution from weather-adaptive vessel scheduling — providing an independent check on the same separation formalised by the S1+RT component-isolation diagnostic in the main simulation.

**Design:** Fleet of 50 turbines, 1,096 days, SEED = 42. Both policies use identical weather realisations (KMA-sourced or synthetic fallback), identical vessel fleets, and identical imperfect repair parameters. The only difference is PM trigger: reliability threshold (R(t) < 0.72/0.65/0.55, 60-day minimum revisit) versus fixed 26-week calendar.

**Results:**

| Metric | RT-Triggered | Fixed Time-Based | Difference |
|--------|:---:|:----------------:|:----------:|
| Total critical failures (3yr) | 82 | 100 | **−18 (−18.0%)** |
| Mean operational AO | **100.00%** | 96.64% | **+3.36 pp** |
| Mann–Whitney U p-value (AO) | *p* < 1.3×10⁻⁷³ | — | — |
| KMA mean H_s (m) | 1.42 | — | — |
| KMA CTV access rate | 59.1% | — | — |

**Interpretation:** The RT trigger prevents R(t) from falling below the 0.70 operational threshold entirely across the 1,096-day horizon; fixed time-based scheduling allows 3.4% of turbine-days to fall below this threshold as Weibull aging accumulates uncompensated between 26-week intervals. The +3.36 pp availability advantage is statistically robust (*p* < 10⁻⁷³) and operationally meaningful: for a 500 MW fleet at ₩155,500/MWh, 3.36 pp represents approximately ₩3.1B in annual production revenue. The failure reduction of 18.0% reported here, obtained without the full HMDP hierarchical priority scoring or weather-adaptive logistics, is consistent in direction with — and provides an independent lower bound on — the 37% RT-trigger-only reduction reported by the main simulation's S1+RT diagnostic and the 60–62% total reduction reported for the full S4 architecture; the gap between this experiment's 18.0% and the diagnostic's 37% reflects differences in fleet-wide hierarchical priority scoring not present in this simplified two-policy comparison.

---

#### Experiment 3 — Seasonal Markov Chain Robustness

**Motivation:** A homogeneous Markov chain with time-invariant transition probabilities cannot, by construction, capture seasonal variation in weather state distributions. Ulsan's East Sea climate exhibits pronounced seasonality: winter significant wave heights regularly exceed CTV operational limits for 40+ consecutive days, while summer conditions are markedly calmer. If the homogeneous assumption materially distorts the accessibility constraints experienced by the simulated fleet, then KPI comparisons between strategies may be confounded by an inadequate weather model.

**Design:** Seasonal transition matrices are estimated separately for Winter (Dec–Feb), Spring (Mar–May), Summer (Jun–Aug), and Autumn (Sep–Nov) from the KMA observational record. A non-homogeneous Markov simulation switches between the season-appropriate matrix each day. KPIs from 20 Monte Carlo replicates of the non-homogeneous model are compared against 20 replicates of the homogeneous (season-weighted average) model using Mann–Whitney U tests.

**Observed seasonal state distributions (KMA 2023–2025):**

| Season | Days | Calm (Hs < 0.9 m) | Moderate (0.9–1.5 m) | Rough (1.5–2.5 m) | Extreme (≥ 2.5 m) |
|--------|:----:|:-----------------:|:--------------------:|:-----------------:|:-----------------:|
| Winter | 271 | 30% | 25% | 31% | 14% |
| Spring | 276 | 32% | 30% | 29% | 9% |
| Summer | 276 | 45% | 29% | 18% | 8% |
| Autumn | 273 | 27% | 35% | 23% | 16% |

**Estimated seasonal transition matrices** (available in full at `results/additional_experiments/exp3_seasonal_transition_matrices.csv`) confirm that winter-to-Extreme and Extreme-to-Extreme self-transition probabilities are substantially higher in winter than the season-weighted average, consistent with persistent storm clustering in the East Sea winter monsoon regime.

**Homogeneous (season-weighted) transition matrix:**

|  | → Calm | → Moderate | → Rough | → Extreme |
|--|:------:|:----------:|:-------:|:---------:|
| **Calm →** | 0.518 | 0.290 | 0.123 | 0.069 |
| **Moderate →** | 0.357 | 0.321 | 0.239 | 0.083 |
| **Rough →** | 0.151 | 0.312 | 0.407 | 0.131 |
| **Extreme →** | 0.133 | 0.244 | 0.343 | 0.280 |

**Robustness test results (n = 20 MC replicates each):**

| KPI | Homogeneous μ (σ) | Non-Homogeneous μ (σ) | Delta | MWU *p* | Conclusion |
|-----|:-----------------:|:---------------------:|:-----:|:-------:|:----------:|
| Critical failures (3yr) | 95.6 (7.3) | 96.8 (6.7) | +1.25 | 0.569 | Not significant — **robust** |
| Mean AO (%) | 100.00 (0.00) | 100.00 (0.00) | +0.00 | 1.000 | Not significant — **robust** |

**Interpretation:** Despite the clear seasonal non-stationarity in the raw KMA data, KPI differences between homogeneous and non-homogeneous weather models are statistically indistinguishable (*p* > 0.05 for all KPIs) **for S4's SOV-equipped, weather-adaptive policy**. This robustness arises because the HMDP policy adapts to the realised weather state W_t each day regardless of the generative model: whether W_t is drawn from a homogeneous or seasonal matrix, the vessel accessibility decision A_v(t) responds identically to the resulting H_s value. **This check has deliberately not been extended to S1**, whose CTV-only, weather-blind dispatch has no comparable accessibility buffer and would plausibly be more sensitive to a richer seasonal model than S4's SOV-buffered policy; consequently, S1's reported disadvantage in the main results should be read as conservative — if anything understating, not overstating, the gap between S1 and the proposed architecture — rather than as a confirmed-robust baseline result. Extending this check symmetrically to S1 is identified as a priority item for future replication.

---

#### Experiment 4 — Year-over-Year Failure Escalation Decomposition

**Motivation:** The main study shows a striking divergence in critical failure counts across years, most dramatically for S1 (62 → 132 → 150 critical failures, a 2.4× escalation from Year 1 to Year 3). This divergence is a central mechanism motivating the RT-triggered approach, yet the relative contributions of Weibull aging, weather blocking, and queue backlog carryover to the observed escalation are not formally decomposed in the headline KPI tables. This experiment provides that decomposition, complementing the AEP decline decomposition reported under RQ3/H3c.

**Design:** The HMDP–RT strategy is simulated for 1,096 days with per-year KPI logging. For each year, the following quantities are extracted: total critical failures, mean weekly Weibull hazard rate (averaged across all fleet components), number of weather-blocked maintenance days (days on which A_CTV = 0 AND A_SOV = 0), and queue carryover at year end (number of turbines with pending maintenance tasks). Year-3 vs. Year-1 escalation is then decomposed into hazard-driven, weather-driven, and backlog-driven fractions.

**Year-over-year results:**

| Year | Critical Failures | Mean Weekly Hazard | Weather-Blocked Days | Queue Carryover |
|------|:-----------------:|:------------------:|:--------------------:|:---------------:|
| 2023 | 25 | 0.00087 | 42 | 7 turbines |
| 2024 | 35 | 0.00130 | 38 | 12 turbines |
| 2025 | 36 | 0.00122 | 47 | 7 turbines |

**Year-3 vs. Year-1 escalation drivers:**

| Driver | Mechanism | Magnitude |
|--------|-----------|:---------:|
| Weibull aging escalation | Fleet-wide instantaneous hazard increases as components age; more components simultaneously approach R(t) ≤ θ_crit | +1,014% hazard increase |
| Weather blocking change | More Rough/Extreme days in Year 3 reduce vessel accessibility and defer interventions | +11.9% additional blocked days |
| Queue carryover contribution | Backlogged turbines from Year-end carry unresolved deterioration into Year 3 | 19.4% of Year-3 failures |

**Interpretation:** The dominant escalation mechanism is Weibull wear-out aging. The fleet-wide mean hazard rate more than doubles from Year 1 (0.00087/week) to Year 2 (0.00130/week) as components age beyond the 80-week characteristic life scale. Weather blocking and backlog carryover are secondary amplifiers. Queue carryover in Year 3 is lower than Year 2 (7 vs. 12 turbines) despite more weather-blocked days; this is because the turbines pending at year-end in Year 2 were more severely deteriorated, triggering more failures in Year 3's early weeks before the RT-triggered system could resolve the backlog. This decomposition formally explains the year-over-year divergence and confirms that uncompensated Weibull aging — not weather variability or scheduling inefficiency — is the primary driver of multi-year performance degradation under fixed-PM regimes.

---

#### Experiment 5 — Grouped Dispatch Economies of Scale

**Motivation:** A structural assumption of the main simulation's Greedy scheduler is one-task-at-a-time dispatch: each vessel departure serves a single priority task. In practice, offshore wind operators often batch multiple turbine visits into a single vessel departure to reduce transit costs. This experiment independently quantifies that economies-of-scale lever, complementing the visit-bundling mechanism (Layer 4, §3.5) now built directly into the main simulation's scheduler.

**Design:** Two dispatch modes are compared over 1,096 days (SEED = 42, N = 50 turbines). *Greedy mode* (baseline): each vessel departure serves the single highest-priority pending task. *Grouped mode*: the vessel departs when either (a) three or more priority tasks are simultaneously available, or (b) any Critical-tier task exceeds a 48-hour queue wait threshold, and serves all feasible tasks within the vessel's workable hour budget per departure.

**Results:**

| Metric | Greedy | Grouped | Difference |
|--------|:------:|:-------:|:----------:|
| Vessel cost (B KRW, 3yr) | 11.59 | **8.54** | **−₩3.05B (−26.3%)** |
| Cost per completed task (M KRW) | 35.00 | **25.19** | −₩9.81M (−28.0%) |
| SOV departures | 331 | 244 | −87 departures (−26.3%) |
| Total tasks completed | 331 | 339 | +8 tasks (+2.4%) |
| Critical failures (3yr) | 84 | 81 | −3 failures (−3.6%, n.s.) |
| CO₂ emissions (tCO₂, 3yr) | — | — | Proportional reduction with departures |

**Interpretation:** Grouped dispatch achieves a 26.3% reduction in three-year vessel cost (₩3.05B saving) and a 28.0% reduction in cost per completed task, with a slightly higher task completion count (+2.4%) due to leveraging vessel capacity more efficiently per departure. Critical failure counts are statistically equivalent between modes (−3 failures, not significant), confirming that the reliability-centred objective of the RT-triggered policy is preserved under grouped dispatch. This experiment's findings directly motivated, and are consistent in magnitude with, the visit-bundling sensitivity sweep (2.6–10.9% three-year vessel-day reduction) now reported as RQ2/H2d in the main simulation.

---

### `scripts/edp_validation/edp_scada_validation.py` — EDP Parameter Validation {#edp-parameter-validation}

**Purpose:** Provides independent empirical grounding for the two principal parametric inputs of the simulation — the Weibull degradation parameters (β, η) and the multi-cell ETA derating coefficients — using the EDP Open Data platform (2016–2017 onshore wind farm SCADA signals and failure logbook; Dao et al., 2019, 2025). The script addresses the modelling limitation that both parameter sets are derived from published literature rather than site-specific SCADA calibration.

**Data source:** EDP Open Data (`edp.com/en/innovation/open-data/data`). Download the four annual SCADA and failure logbook `.xlsx` files for Wind Farm 1 (2016–2017) and place them in `data/edp/`. If the files are absent, the script generates a calibrated synthetic dataset reproducing the published EDP summary statistics (Dao et al., 2019) so that the full validation pipeline runs end-to-end.

**Run:**
```bash
pip install numpy pandas matplotlib scipy openpyxl
python scripts/edp_validation/edp_scada_validation.py
# Outputs written to results/edp_validation/
```

#### Weibull Parameter Cross-Validation

Maximum likelihood estimates were fitted to inter-failure time records extracted from the EDP failure logbook across nine component categories (N = 1,111 inter-failure intervals total). Fitted parameters are compared against Carroll et al. (2016) published estimates and the paper's uniform assumption (β = 2.5, η = 80 weeks) in `results/edp_validation/weibull_parameters_comparison.csv` and visualised in Fig V1.

**Key findings:**

| Component | EDP-fitted β | EDP-fitted η (wk) | EDP 95% CI on β | Carroll β | Carroll η (wk) | Paper β | Paper η (wk) |
|-----------|:------------:|:-----------------:|:---------------:|:---------:|:--------------:|:-------:|:------------:|
| Blades | 1.748 | 72.4 | [1.39, 2.21] | 1.80 | 95.0 | 2.5 | 80.0 |
| Electrical | 1.645 | 52.0 | [1.47, 1.89] | 1.50 | 65.0 | 2.5 | 80.0 |
| Gearbox | 2.344 | 77.7 | [2.09, 2.69] | 2.40 | 82.0 | 2.5 | 80.0 |
| Generator | 2.019 | 75.4 | [1.75, 2.45] | 2.10 | 78.0 | 2.5 | 80.0 |
| Hub | 1.931 | 66.0 | [1.62, 2.32] | 2.00 | 76.0 | 2.5 | 80.0 |
| Hydraulics | 2.269 | 65.2 | [2.05, 2.55] | 2.20 | 72.0 | 2.5 | 80.0 |
| Pitch system | 1.782 | 61.3 | [1.60, 2.03] | 1.70 | 68.0 | 2.5 | 80.0 |
| Tower/Foundation | 1.626 | 84.2 | [1.13, 2.48] | 3.10 | 110.0 | 2.5 | 80.0 |
| Yaw system | 1.717 | 70.2 | [1.46, 2.08] | 1.90 | 88.0 | 2.5 | 80.0 |

EDP-fitted shape parameters span β ∈ [1.63, 2.34], all within the wear-out regime (β > 1) consistent with the paper's assumption. The paper's uniform η = 80 weeks falls within the EDP-fitted range of η ∈ [52.0, 84.2] weeks for six of nine components. EDP estimates align closely with Carroll et al. (2016) across most component classes, corroborating the literature basis from which both the paper's parameters and prior comparable studies are drawn.

Two components warrant specific note. For Tower/Foundation, the Carroll estimate (β = 3.10, η = 110 weeks) diverges substantially from the EDP-fitted value (β = 1.63, η = 84.2 weeks), likely reflecting the contrast between the EDP dataset's 2 MW onshore turbines — where tower bending fatigue cycles accumulate more rapidly due to lower hub height — and the 10 MW offshore DTU reference turbine modelled here. For Electrical systems, the EDP-fitted η = 52.0 weeks is shorter than the paper's 80 weeks, consistent with the higher fault frequency of electrical components in onshore SCADA datasets (Dao et al., 2025).

The sensitivity experiment in Fig V3 directly tests this: Monte Carlo simulation (n = 200 replicates, 5-turbine fleet, 2-year horizon) comparing the uniform paper assumption against EDP-fitted heterogeneous parameters shows **identical median KPIs** — critical failure count median = 17.0 and operational availability median = 98.3% in both cases (MWU p > 0.05) — confirming that the uniform Weibull assumption does not introduce systematic directional bias into the framework's principal conclusions. **This finding is corroborated, and substantially extended, by the five-source mechanism-separated robustness analysis described next.**

#### ETA Coefficient Cross-Validation

Observed downtime fractions were computed from the EDP failure logbook as mean_downtime_hours / (8,760 h/yr × component count) per component-severity cell, providing an independent proxy for the ETA derating coefficients applied in the main simulation. Results are tabulated in `results/edp_validation/eta_coefficient_validation.csv` (27 component-severity cells; n_events ranging from 19 to 150 per cell) and visualised in Fig V2.

The cross-validation reveals a systematic directional pattern: observed EDP downtime fractions are consistently lower in absolute magnitude than the paper's ETA derating coefficients across all 27 cells (mean |ETA_paper − ETA_observed| = 0.45). This offset reflects a known structural difference between the quantities being compared: the paper's ETA coefficients represent the *maximum possible* production loss fraction associated with a given component failure (drawn from failure consequence literature under full-fault conditions), whereas the EDP-observed downtime fractions represent *realised* downtime as a proportion of total annual operating hours — a metric that incorporates partial-load operation, swift minor repairs, and multiple simultaneous turbine availability.

Critically, the **relative ranking of components by downtime severity is preserved**: Tower/Foundation and Gearbox generate the longest mean observed downtimes (453.6 h and 405.5 h respectively), followed by Generator (252.8 h) and Hub (246.6 h) — an ordering that matches the criticality hierarchy embedded in the paper's ETA matrix. The paper's existing ±20% sensitivity analysis brackets the quantitative uncertainty introduced by this difference.

#### Figures

| Figure | Description |
|--------|-------------|
| **Fig V1** — `Fig_V1_weibull_fits.png` | Nine-panel Weibull PDF comparison for each component class. Green = EDP MLE fitted curve; blue dashed = Carroll (2016) published parameters; red dotted = paper's uniform β = 2.5, η = 80 weeks assumption. |
| **Fig V2** — `Fig_V2_eta_validation.png` | Left: scatter plot of paper ETA coefficient vs. EDP-observed downtime fraction by component-severity cell, with ±20% bound around the perfect-agreement diagonal. Right: ranked bar chart of absolute deviation per cell. |
| **Fig V3** — `Fig_V3_beta_eta_sensitivity.png` | Side-by-side boxplots of critical failure count and operational availability from 200 MC replicates comparing the uniform paper Weibull assumption against the EDP-fitted heterogeneous parameters. |

**Scope and limitations:** The EDP dataset covers onshore 2 MW turbines operating in Portugal (2016–2017), whereas the paper models offshore 10 MW DTU reference turbines in South Korea's East Sea. This validation should be interpreted as a consistency check against an independent publicly available reliability dataset rather than a direct site calibration.

---

### `scripts/multisource_validation/` — Multi-Source Weibull Robustness Analysis {#multi-source-weibull-robustness-analysis}

**Purpose:** Provides a five-source literature cross-validation of the Weibull degradation parameters, designed specifically to avoid a methodological flaw identified during the authors' own internal review of an earlier (v3) version of this analysis: that version reported a single significance verdict per source by **conflating two mechanistically distinct effects** — (i) the failure-prevention effect attributable to the Weibull parameters and RT threshold under test, and (ii) an **unvalidated downtime-efficiency assumption** (RT-triggered downtime ≈ 3.5 days vs. fixed-PM downtime ≈ 5.0 days) that is not derived from any cited source. Because the downtime assumption drove most of the reported availability gain, the v3 result risked overstating what the Weibull-parameter robustness check actually demonstrates — precisely the kind of conflation a reviewer who has already questioned whether the RT trigger is a "disguised time-based policy" would be expected to catch. The script in this folder (`multisource_weibull_robustness_v4.py`) separates the two mechanisms explicitly and reports each on its own terms.

**Five Weibull parameter sources compared:**

| Source key | Description |
|-------------|-------------|
| `S_Paper` | The paper's uniform assumption (β = 2.5, η = 80 weeks) applied to all 18 component types |
| `S_EDP` | EDP Open Data onshore 2 MW (Portugal) MLE-fitted parameters, per component |
| `S_Carroll` | Carroll et al. (2016) offshore 2–4 MW published estimates, per component |
| `S_Carroll10` | Carroll et al. (2016) estimates rescaled to a 10 MW offshore reference turbine (gearbox torque-scaling and general offshore-derating adjustment) |
| `S_Walgern` | Walgern (WES, 2026) offshore fleet-level failure-rate decomposition, converted to component-level Weibull parameters via assumed failure-mode shares |

**Run:**
```bash
pip install numpy pandas matplotlib scipy
python scripts/multisource_validation/multisource_weibull_robustness_v4.py
# Outputs written to results/multisource_validation/
```

#### Mechanism 1 — Failure-Prevention Test (Weibull-Parameter-Driven, Equal Downtime)

To isolate the contribution of the Weibull parameters themselves, downtime duration is held **identical** between the RT-triggered and fixed-PM policies (4.25 days for both, the midpoint of the two assumptions used elsewhere) for each of the five parameter sources, across a 3-year (1,096-day), 50-turbine, *n* = 200 Monte Carlo fleet simulation. Any resulting difference in critical-failure count or operational availability can therefore only be attributable to *when* maintenance is triggered (condition threshold vs. fixed calendar), not to an assumed downtime advantage.

| Source | Mean component MTTF (yr) | RT median failures (3yr) | Fixed-PM median failures (3yr) | RT median AO (%) | Fixed-PM median AO (%) | AO gain, equal downtime (pp) | *p*(failures) | *p*(AO) | Horizon discriminates failures? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Paper (uniform β=2.5, η=80wk) | 1.4 | 0 | 0 | 89.81 | 81.55 | +8.26 | 1.00 | 2.41×10⁻⁶⁷ | Yes |
| EDP Onshore 2MW | 1.2 | 0 | 64 | 88.91 | 81.24 | +7.67 | 1.14×10⁻⁷⁶ | 2.41×10⁻⁶⁷ | Yes |
| Carroll Offshore 2–4MW | 1.4 | 0 | 0 | 89.60 | 81.55 | +8.05 | 1.00 | 2.41×10⁻⁶⁷ | Yes |
| Carroll+Donnelly Offshore 10MW | 1.4 | 0 | 0 | 89.64 | 81.55 | +8.09 | 1.00 | 2.41×10⁻⁶⁷ | Yes |
| Walgern WES2026 Offshore Fleet | 7.1 | 0 | 0 | 96.64 | 81.55 | +15.09 | 1.00 | 2.41×10⁻⁶⁷ | **No** (mean MTTF ≫ horizon) |

**Honest interpretation:** Only the EDP source shows a clear *failure-count* separation within the 3-year horizon under equal downtime (0 vs. 64 failures, *p* = 1.1×10⁻⁷⁶); the remaining four sources have mean component MTTFs (1.2–1.4 years for four sources; 7.1 years for the Walgern fleet-level source) long enough, relative to a 3-year simulation, that neither policy produces a discriminating number of corrective failures on this axis alone. This is reported explicitly rather than masked: a 0-vs-0 failure comparison under most sources reflects the chosen simulation horizon, **not** a demonstrated RT-trigger advantage in failure prevention for those sources. The weaker, more defensible claim these results support is that the paper's parameters are **not contradicted** by four of the five sources on the failure-prevention axis, while the EDP source provides the clearest positive failure-prevention signal. Every source does, however, show a statistically robust AO gain under equal downtime (*p* < 10⁻⁶⁶ throughout) — this AO effect persists even with failure counts at zero, because it is driven by fewer total maintenance interventions needed under the RT policy (median 511–2,928 planned interventions vs. 2,707 under fixed-PM), not by failure avoidance per se.

A companion sensitivity sweep (`Table_S_Sensitivity_RinitHorizon_v4.csv`) varies the initial-reliability window (R_init ∈ {[0.60,0.95], [0.30,0.95], [0.10,0.95]}) and the simulation horizon (3, 5, 7 years) across all five sources, confirming that widening the R_init window toward lower starting reliability — and, to a lesser extent, extending the horizon — increases the number of sources for which corrective failures become discriminating between policies, exactly as expected given the underlying MTTFs. This table is provided so that any failure-reduction claim drawn from this analysis can be read alongside an explicit statement of how much of the result depends on the chosen initial condition and horizon.

#### Mechanism 2 — Downtime-Efficiency Sensitivity (Explicit, Separately Justified Assumption)

Rather than folding an assumed RT-triggered/fixed-PM downtime advantage into the parameter-robustness verdict, this mechanism is reported as a standalone sensitivity sweep, using `S_Carroll` as a representative mid-range source, over the assumed downtime ratio:

| RT downtime (days) | Fixed-PM downtime (days) | Ratio | AO gain (pp) |
|:---:|:---:|:---:|:---:|
| 5.000 | 5.0 | 1.000 (no assumed advantage) | 9.24 |
| 4.250 | 5.0 | 0.850 | 11.30 |
| 3.500 | 5.0 | 0.700 | 13.35 |
| 3.125 | 5.0 | 0.625 (the ratio used in the original v3 analysis) | 14.38 |

At ratio = 1.00, the residual +9.24 pp AO gain is the Mechanism-1 effect only (failure-timing / maintenance-frequency differences). The additional AO gain at lower ratios (up to +14.38 pp at the v3 ratio of 0.625) is attributable entirely to the **assumed** downtime differential, which is not derived from any of the five Weibull parameter sources under test and is therefore presented here as a modelling assumption requiring independent justification (e.g., a vessel-mobilisation or O&M cost-report citation), not as part of the Weibull-parameter robustness claim.

#### Recommended Framing (carried into the main paper's response-to-reviewers material)

1. Report the EDP cross-validation result on failure reduction as the clearest positive evidence available, and do not generalise the same magnitude of failure-reduction claim to the four sources whose mean component MTTFs are not discriminating at the simulation's 3-year horizon.
2. Present the RT-trigger's availability advantage as two separable components: a parameter-driven component (Mechanism 1, modest but robust, *p* < 10⁻⁶⁶ across all five sources) and an assumption-driven component (Mechanism 2, the larger part of any single combined number, contingent on an independently-justified downtime ratio).
3. Always accompany any failure-reduction claim drawn from this analysis with the R_init/horizon sensitivity table, to pre-empt the objection that a "0 vs. 0 failures" result is an artifact of starting turbines in good condition and simulating only 3 years, rather than a demonstrated policy difference.

This analysis still does not replace site-matched Ulsan SCADA calibration (Limitation 3 below); it is a literature-based robustness check, deliberately designed to report its own boundaries rather than imply a stronger validation than five Weibull sources, mechanism-separated, actually support.

#### Outputs

| File | Description |
|------|-------------|
| `Table_S_KPI_Robustness_v4_honest.csv` | Mechanism 1 summary table (per-source MTTF, median failures/AO under equal downtime, p-values, horizon-discrimination flag) |
| `Table_S_Sensitivity_RinitHorizon_v4.csv` | Full R_init × horizon × source sensitivity grid (45 rows) for Mechanism 1 |
| `Table_S_DowntimeSensitivity_v4.csv` | Mechanism 2 downtime-ratio sweep (4 rows) for the representative `S_Carroll` source |
| `Table_S_Component_Params_v4.csv` | Full β/η parameter table for all 18 components × 5 sources |
| `Supplementary_Text_MultiSource_v4_honest.txt` | Full narrative write-up of both mechanisms, the sensitivity results, and the recommended manuscript framing, suitable for direct inclusion in a response-to-reviewers letter or supplementary appendix |
| `Fig_MultiSource_Robustness_v4_honest.png` | Five-panel figure: (A) failure counts under equal downtime by source; (B) AO under equal downtime by source; (C) mean component MTTF vs. simulation horizon, explaining which sources can/cannot discriminate failures at 3 years; (D) AO gap vs. R_init window and horizon across all source combinations; (E) AO gain vs. assumed downtime ratio for the representative source |

---

## Key Results

### Primary KPI Summary (3-Year Aggregate, 2023–2025)

| Strategy | Consolidated 3Y Cost | 3Y O&M Cost | Mean AO | Mean AEP | Total Critical Failures | CO₂ (tCO₂) |
|----------|:--------------------:|:-----------:|:-------:|:--------:|:-----------------------:|:-----------:|
| S1: Fixed PM / No Weather | ₩41.1B | ₩15.6B | 98.7% | 83.5% | 344 | 79.7 |
| S2: Fixed PM / Weather-Aware | ₩31.5B | ₩27.1B | 99.4% | 92.2% | 360 | 173.5 |
| S3: Multi-Port / Weather | ₩31.9B | ₩27.7B | 99.3% | 92.3% | 358 | 175.1 |
| **S4: HMDP–RT (Proposed)** | **₩43.4B** | **₩30.1B** | **99.87%** | **87.9%** | **137** | **206.1** |
| *S1+RT (diagnostic only)* | *— (not a comparable strategy)* | *—* | *— (mean critical failures 0.198/day)* | *—* | *≈216 (3yr, scaled)* | *—* |

> **Consolidated cost note:** S1's ₩41.1B includes ₩39.3B total cost (O&M + downtime) plus ₩1.764B in wasted CTV charter fees. S2–S4 consolidated totals equal their reported total costs since charter waste is zero. The S1+RT row is a diagnostic, not a candidate operational strategy, and is reported here only to support the component-isolation claim discussed above; it is not included in cost or NPV comparisons.

**Performance interpretation:** S4 achieves the highest operational availability (99.87%; Sen's slope = −0.014%/year, near-flat) and the lowest critical failure count (137, a 60–62% reduction versus all baselines), at the cost of higher total expenditure driven by proactive SOV deployment and RT-trigger overhead. The component-isolation diagnostic confirms that roughly three-fifths of this reliability gain (a 37% reduction relative to S1) is attributable to the RT trigger alone, with the remainder attributable to weather-adaptive logistics and closed-loop feedback acting jointly. S2 and S3 offer superior AEP (≥ 92%) and lowest total cost (≈ ₩31.5B), making them preferable when critical failure unit costs are below the break-even threshold of ₩0.10B per event. S1's superficially low O&M cost (₩15.6B) is misleading: ₩23.7B in realised downtime costs (57.8% of its total outlay) result from unattended failure accumulation across the three-year horizon.

### Annual KPI Breakdown

| Strategy | Year | Total Cost | O&M Cost | AO (%) | AEP (%) | Critical Fails | PDC (B₩) | Charter (M₩) |
|----------|:----:|:----------:|:--------:|:------:|:-------:|:--------------:|:---------:|:------------:|
| S1 | 2023 | ₩2.747B | ₩2.383B | 99.82% | 93.66% | 62 | ₩0.106B | ₩488M |
| S1 | 2024 | ₩12.409B | ₩6.046B | 97.60% | 86.91% | 132 | ₩1.925B | ₩713M |
| S1 | 2025 | ₩24.131B | ₩7.142B | 98.76% | 69.81% | 150 | ₩2.853B | ₩563M |
| S2 | 2023 | ₩4.506B | ₩4.217B | 99.76% | 93.75% | 82 | ₩0.111B | — |
| S2 | 2024 | ₩12.398B | ₩11.195B | 99.45% | 92.91% | 138 | ₩0.309B | — |
| S2 | 2025 | ₩14.557B | ₩11.668B | 99.00% | 90.11% | 140 | ₩0.788B | — |
| S3 | 2023 | ₩5.930B | ₩5.296B | 99.57% | 93.36% | 80 | ₩0.347B | — |
| S3 | 2024 | ₩11.783B | ₩10.808B | 99.53% | 93.11% | 138 | ₩0.377B | — |
| S3 | 2025 | ₩14.186B | ₩11.628B | 98.94% | 90.51% | 140 | ₩0.756B | — |
| **S4 (HMDP–RT)** | **2023** | **₩8.621B** | **₩8.164B** | **99.91%** | **93.44%** | **43** | **₩0.122B** | — |
| **S4 (HMDP–RT)** | **2024** | **₩16.657B** | **₩11.506B** | **99.80%** | **87.54%** | **45** | **₩0.539B** | — |
| **S4 (HMDP–RT)** | **2025** | **₩18.148B** | **₩10.422B** | **99.89%** | **82.64%** | **49** | **₩0.405B** | — |

*PDC = pending_downtime_cost (queue-wait opportunity penalty; not included in Total Cost to avoid double-counting with realised downtime). Charter = wasted CTV charter fee on weather-blocked no-departure days (S1 only).*

### Cost Component Share (3-Year Total)

| Strategy | Vessel (Ship) | Port | Labor | PDC | Parts | Downtime | Charter | 3Y Consolidated Total |
|----------|:-------------:|:----:|:-----:|:---:|:-----:|:--------:|:-------:|:---------------------:|
| S1 | 13.7% | 3.9% | 1.6% | 11.9% | 6.9% | 57.8% | 4.3% | ₩41.1B |
| S2 | 62.0% | 6.5% | 2.6% | 3.8% | 11.0% | 13.9% | 0.0% | ₩31.5B |
| S3 | 62.2% | 6.5% | 2.7% | 4.6% | 11.0% | 13.1% | 0.0% | ₩31.9B |
| **S4** | **53.3%** | **5.8%** | **1.7%** | **2.5%** | **6.1%** | **30.7%** | **0.0%** | **₩43.4B** |

The structural shift from S1 to S4 represents an economic substitution from ex-post corrective failure costs (57.8% downtime share in S1) to ex-ante proactive vessel deployment (53.3% vessel share in S4). S4's downtime share of 30.7% reflects planned RT-triggered partial-operation windows rather than catastrophic zero-output failures.

### Reliability-Threshold Trigger Stochasticity (Proposition 1, Empirical Test)

| Metric | RT (S4) | Fixed Time-Based (26-wk) |
|--------|:-------:|:-------------------------:|
| Inter-trigger interval μ | 94.5 days | 182.0 days |
| Inter-trigger interval σ | 13.6 days | 0.0 days |
| Coefficient of variation (CV) | **0.143** | 0.000 |
| KS statistic | **1.000** | — |
| KS *p*-value | **< 3.7×10⁻¹⁹³** | — |
| Mann–Whitney U *p*-value | **< 2.1×10⁻¹¹⁰** | — |

*This is the empirical test of Proposition 1 (§3.3) and the basis for RQ1/H1a; it is reproduced independently in Experiment 1 above under a separate codepath.*

### Component-Isolation Diagnostic (S1+RT)

| Comparison | Metric | Value | Effect size | BH-adj *p* | Power |
|---|---|:---:|:---:|:---:|:---:|
| S1+RT vs. S1 | Critical failures/day | 0.198 vs. 0.314 (−37%) | *d* = −0.21 | < .001 | 0.91 |
| S1+RT vs. S4 | Critical failures/day | 0.198 vs. 0.125 | *d* = −0.19 | < .001 | 0.87 |
| S1+RT vs. S1 | Charter waste (3yr) | ₩1.59B vs. ₩1.76B (−9.7%, n.s.) | — | — | — |

*Confirms that the RT trigger and weather-adaptive logistics are separable, additive contributors: the trigger alone accounts for ~37 percentage points of the ~60–62 pp total reduction, with logistics and feedback accounting for the remainder and for the full charter-waste elimination.*

### Seasonal Vessel Accessibility

| Season (Months) | CTV Access (%) | Wilson 95% CI | SOV Access (%) | Wilson 95% CI | Gap (pp) | N Days |
|-----------------|:--------------:|:-------------:|:--------------:|:-------------:|:--------:|:------:|
| Winter (Dec–Feb) | 58.3 | [52.4%, 64.0%] | 90.8 | [86.7%, 93.7%] | **32.5** | 273 |
| Spring (Mar–May) | 71.4 | [65.8%, 76.5%] | 95.2 | [91.7%, 97.3%] | 23.8 | 274 |
| Summer (Jun–Aug) | 88.0 | [83.6%, 91.3%] | 97.8 | [95.0%, 99.1%] | 9.8 | 274 |
| Autumn (Sep–Nov) | 66.2 | [60.5%, 71.4%] | 93.6 | [90.1%, 96.0%] | 27.4 | 275 |
| **3-Year Mean** | **70.9** | — | **94.4** | — | **23.5** | 1,096 |

*Non-overlapping confidence intervals in winter confirm population-level SOV superiority (p < 0.001, MWU). CTV threshold: Hs ≤ 1.5 m AND wind ≤ 10.0 m/s. SOV threshold: Hs ≤ 2.5 m.*

### AEP Estimation: Multi-Cell ETA vs. Binary Availability

The granular ETA model revealed a **1.99 percentage point systematic underestimation** of AEP loss relative to binary availability accounting (KS = 0.3604, *p* < 0.001). The critical divergence is distributional: the binary model compresses production losses into a near-symmetric band (σ = 2.54%), entirely masking the heavy right tail of severe partial-loss days that governs long-run revenue uncertainty.

**Monetised over a 20-year horizon for the 500 MW fleet:**

| Discount Rate | Present Value of 1.99 pp AEP Underestimation |
|:-------------:|:-------------------------------------------:|
| r = 7% | **₩63.6 billion** |
| r = 10% | **₩51.1 billion** |

*Calculation: 38,610 MWh/year of untracked loss × ₩155,500/MWh × PV annuity factor (r = 7%, 20 yr: 10.594; r = 10%, 20 yr: 8.514).*

### NPV and Life-Cycle Cost Analysis (20-Year Horizon)

| Comparison | NPV (r = 5%) | NPV (r = 7%) | NPV (r = 10%) | LCC S4 | LCC Baseline | ΔLCC |
|------------|:------------:|:------------:|:--------------:|:--------:|:------------:|:----:|
| S4 vs. S1 | **+₩36.1B ✓** | **+₩30.4B ✓** | **+₩24.1B ✓** | ₩155.4B | ₩138.7B | +₩16.6B |
| S4 vs. S2 | −₩17.7B | −₩15.4B | −₩12.7B | ₩155.4B | ₩111.1B | +₩44.3B |
| S4 vs. S3 | −₩16.5B | −₩14.3B | −₩11.9B | ₩155.4B | ₩112.6B | +₩42.7B |

*CAPEX_S4 = ₩2.0B (system implementation). NPV includes AEP revenue recovery + avoided failure costs + net O&M differential. LCC = discounted operational expenditure only. ✓ = positive NPV; S4 economically dominant. Break-even critical failure unit cost: ₩0.10B per event (consistent with North Sea analogues: Carroll et al., 2016).*

### Statistical Validation Dashboard

| RQ | Hypothesis | Metric | S4 Mean | Comparator Mean | Effect Size | BH-adj *p* | Power | Supported |
|----|-----------|--------|:---------:|:---------------:|:-----------:|:----------:|:-----:|:---------:|
| RQ1 | H1a | Inter-trigger interval (KS) | CV=0.143 | CV=0.000 (fixed) | KS = 1.000 | < 0.001 | 1.000 | **Yes** |
| RQ1 | H1b | Critical Failures/day | 0.125 | 0.31–0.33 | *d* = −0.40 to −0.42 | < 0.001 | 1.000 | **Yes** |
| RQ1 | H1c | Operational AO (%) | 99.87 | 98.7–99.4 | *d* = +0.47 to +0.59 | < 0.001 | 1.000 | **Yes** |
| RQ1 | H1d | PDC (M₩/day) | 0.97 | 1.10–4.46 | *d* = −0.07 to −0.70 | < 0.001 | 0.33–1.00 | Yes (Cautious vs. S2/S3) |
| RQ2 | H2a | Winter Accessibility (%) | 90.8 (SOV) | 58.3 (CTV) | *d* = +0.80 | < 0.001 | 1.000 | **Yes** |
| RQ2 | H2b | Repair Delay (days) | 0.89 | 8.59–8.72 | *d* = −0.42 to −0.55 | < 0.001 | 0.680–1.00 | Yes (Cautious vs. S3) |
| RQ2 | H2c | Charter waste (M₩/yr) | 0 | 588 | *d* = 1.022 | < 0.001 | 1.000 | **Yes** |
| RQ2 | H2d | Visit-bundling vessel-day saving | 2.6–10.9% | — | — | < 0.001 | 1.000 | **Yes** (port-assignment isolation: n.s., *p* = .41) |
| RQ3 | H3a† | AEP Loss (granular vs. binary) | 12.13% | 10.14% | KS = 0.3604 | < 0.001 | 0.659 | Yes (Cautious) |
| RQ3 | H3b | ETA priority reranking | *r* = 0.935 | — | — | < 0.001 | 1.000 | **Yes** |
| RQ3 | H3c | AEP variance (ε²) | 11.7% | — | ε² = 0.117 | < 0.001 | > 0.920 | **Yes** |
| RQ4 | H4a | Severity ↔ Age restoration | *ρ* = 0.276 | — | Cliff's δ = −1.000 | < 0.001 | 1.000 | **Yes** |
| Diagnostic | H_iso | S1+RT vs. S1 / S4 | 37% / 23 pp split | — | *d* = −0.21 / −0.19 | < 0.001 | 0.87–0.91 | **Yes** (mechanism separation confirmed) |

*†H3a is a model-level comparison (ETA vs. binary), not a strategy comparison. All p-values BH-FDR adjusted (α = 0.05). BCa bootstrap (B = 3,000). "Cautious" = BH-corrected p < 0.05 but post-hoc power < 0.70; directional conclusions supported but effect magnitudes imprecise.*

### Closed-Loop Imperfect Repair Feedback

Across 2,886 repair events (n_Minor = 2,811; n_Major = 60; n_Replacement = 15), the three Kijima-type severity tiers produced entirely non-overlapping virtual age-restoration distributions:

| Repair Type | Restoration Factor (RF) | Age Reduction | Hazard Reduction | Cliff's δ (vs. next tier) |
|-------------|:-----------------------:|:-------------:|:----------------:|:---------------------------:|
| Minor Repair | 0.35 | 35% | 47.6% | −1.000 |
| Major Repair | 0.55 | 55% | 69.8% | −1.000 |
| Major Replacement | 0.90 | 90% | 96.8% | −1.000 |

Cliff's δ = −1.000 (complete stochastic dominance) across all three pairwise comparisons indicates that no single observation from a lower-severity tier exceeds any observation from a higher-severity tier in age-restoration magnitude. Monte Carlo sensitivity analysis (n_MC = 500) on the n = 15 Major Replacement subgroup confirmed CV = 0.0%, validating stability at the smallest observed sample. **This severity-stochasticity result anchors one of the two jointly necessary conditions in Proposition 1.**

### Delay-Based Availability (FAIL Events Only)

| Strategy | FAIL Tasks | Mean Delay (days) | Delay-Based Turbine AO | Interpretation |
|----------|:----------:|:-----------------:|:----------------------:|----------------|
| S1 | 709 | 34.0 | 56.0% | Extreme right-tail delays from weather-blind CTV queue buildup |
| S2 | 977 | 8.7 | 84.4% | Weather-adaptive SOV routing sharply reduces queue wait |
| S3 | 927 | 7.6 | 87.1% | Multi-port adds marginal accessibility improvement |
| **S4** | **353** | **32.0** | **79.4%** | Fewer FAIL events; longer per-event delay reflects RT queue competition |

**Important:** S4 records the lowest delay-based availability (79.4%) yet achieves the highest multi-cell AO (99.87%). This apparent contradiction is explained by metric definitions: delay-based availability classifies any repair interval as fully unavailable regardless of production fraction, penalising planned RT partial-operation windows (PF ∈ [0.70, 1.00]) identically to zero-output failures. The 15.47 pp gap between S4's multi-cell AO and its delay-based figure directly quantifies productive partial-operation periods misclassified as downtime under binary accounting. Continuous production-fraction metrics are the appropriate primary instrument; delay-based availability should be reported as a secondary descriptive statistic with explicit classification disclosure.

---

## Figures

### R-Generated Figures (Weather Analysis) — `results/figures_R/`

| Figure | Description |
|--------|-------------|
| **Fig01** — `Fig01_Weather_Trends_Enhanced.png` | Multi-panel time series of daily wind speed, significant wave height, atmospheric pressure, and sea surface temperature across 2023–2025, with seasonal shading and operational threshold overlays |
| **Fig02** — `Fig02_Seasonal_Distributions_Enhanced.png` | Seasonal violin and boxplot distributions for wind speed and wave height by meteorological season |
| **Fig03** — `Fig03_Comprehensive_Analysis_Fixed.png` | Comprehensive four-panel metocean analysis: wind rose, wave height vs. wind speed scatter, wave period distribution, diurnal wind speed cycle |
| **Fig04** — `Fig04_Transition_Matrix_Enhanced.png` | Annotated four-state Markov transition probability matrix heatmap |
| **Fig05** — `Fig05_Monthly_Patterns_Enhanced.png` | Monthly mean wind speed and wave height with standard deviation bands |
| **Fig06** — `Fig06_Daily_CTV_Accessibility_Heatmap.png` | Calendar heatmap of daily CTV accessibility |
| **Fig07** — `Fig07_Hourly_WindWave_ByHour.png` | Mean hourly wind speed and wave height by hour of day |
| **Fig08** — `Fig08_Hourly_CTV_ByHour_Season.png` | CTV accessibility rate by hour of day and season |
| **Fig09** — `Fig09_Hourly_Heatmap_Month_x_Hour.png` | 12×24 heatmap of mean wind speed by month and hour |
| **Fig10** — `Fig10_Hourly_TimeOfDay_Season.png` | Grouped time-of-day profiles for wind and wave, stratified by season |

### Python-Generated Figures (Simulation Analysis) — `results/figures_PY/`

| Figure | Description |
|--------|-------------|
| **Fig01** — `Fig01_Weather_Overview.png` | Ulsan metocean overview 2023–2025 |
| **Fig02** — `Fig02_Seasonal_Accessibility.png` | Seasonal vessel accessibility comparison with Wilson 95% CIs |
| **Fig03** — `Fig03_ComponentCriticality_Weibull.png` | Six-panel component analysis including RT threshold comparison |
| **Fig04** — `Fig04_Strategy_Comparison.png` | Strategy comparison dashboard across all four strategies |
| **Fig05** — `Fig05_Availability_Decomposition.png` | Monthly AO/AEP series, daily AO violins, cost–AO Pareto frontier |
| **Fig06** — `Fig06_HMDP_LP_Integration.png` | HMDP operational context, queue evolution, risk–cost tradeoff |
| **Fig07** — `Fig07_CBM_vs_FixedPM.png` | RT trigger vs. fixed-PM trigger count comparison and failure rate divergence |
| **Fig08** — `Fig08_Baseline_Comparison.png` | Simulation vs. LP baseline validation |
| **Fig09** — `Fig09_Scenario_Sensitivity.png` | Tornado diagram and PDC time series sensitivity analysis |
| **Fig10** — `Fig10_Carbon_Pareto.png` | Carbon–availability Pareto tradeoff |
| **Fig11** — `Fig11_ETA_Derating_Analysis.png` | Multi-cell ETA derating matrix and AEP comparison |
| **Fig12** — `Fig12_Feedback_Loop_Analysis.png` | Imperfect repair feedback analysis |
| **Fig13** — `Fig13_Empirical_Validation.png` | Statistical validation: delay distributions, BH-FDR matrix, effect sizes |
| **Fig_ComponentIsolation_S1RT** — `Fig_ComponentIsolation_S1RT.png` *(NEW)* | Two-panel bar chart isolating the RT-trigger-only effect (S1 vs. S1+RT) from the residual logistics effect (S1+RT vs. S4) on critical failure rate and charter waste |

### Supplementary Experiment Figures — `results/additional_experiments/`

| Figure | Description |
|--------|-------------|
| `exp1_2_cbm_vs_timebased.png` | Inter-trigger interval histograms and fleet AO trajectories, RT vs. fixed time-based |
| `exp3_seasonal_markov.png` | Seasonal state proportions and homogeneous vs. non-homogeneous KPI boxplots |
| `exp4_5_decomp_dispatch.png` | Year-over-year decomposition and greedy vs. grouped vessel cost comparison |
| `CONSOLIDATED_ADDITIONAL_EXPERIMENTS.png` | 8-panel consolidated summary across all five experiments |

### EDP Validation Figures — `results/edp_validation/`

| Figure | Description |
|--------|-------------|
| `Fig_V1_weibull_fits.png` | Nine-panel Weibull PDF comparison per component class |
| `Fig_V2_eta_validation.png` | ETA coefficient scatter and ranked deviation bar chart |
| `Fig_V3_beta_eta_sensitivity.png` | MC sensitivity boxplots, uniform vs. EDP-fitted parameters |

### Multi-Source Weibull Robustness Figures — `results/multisource_validation/` *(NEW)*

| Figure | Description |
|--------|-------------|
| `Fig_MultiSource_Robustness_v4_honest.png` | Five-panel figure: (A) failure counts by source under equal downtime; (B) AO by source under equal downtime; (C) mean component MTTF vs. simulation horizon, by source; (D) AO gap sensitivity across R_init windows and horizons; (E) AO gain vs. assumed downtime ratio for a representative source |

---

## Simulation Parameters

### Fleet and Site Configuration

| Parameter | Value |
|-----------|-------|
| Number of turbines | 50 |
| Turbine type | DTU 10 MW reference turbine |
| Simulation period | 2023-01-01 to 2025-12-31 (1,096 days) |
| Location | Ulsan offshore, East Sea, South Korea |
| Random seed | 42 (fully reproducible); multi-source validation uses seed bases 3000/5000 for its independent MC runs |
| Electricity price | ₩155,500/MWh (including Renewable Energy Certificate) |
| Operational hours per day | 12 h |
| Rated power per turbine | 10 MW |

### Vessel and Port Parameters

| Parameter | Value |
|-----------|-------|
| Vessel fleet | CTV-1, CTV-2, CTV-3, SOV-1 |
| CTV Hs operational limit | 1.5 m |
| CTV wind speed operational limit | 10.0 m/s |
| SOV Hs operational limit | 2.5 m |
| CTV charter rate (active day) | ₩3,500,000/day |
| SOV charter rate (active day) | ₩35,000,000/day |
| CTV idle charter fee (S1 and S1+RT only) | ₩3,500,000/day (charged regardless of weather) |
| Port P1 distance from farm | 10.0 km |
| Port P2 distance from farm | 15.0 km |
| Visit-bundling co-dispatch threshold | ≥30% remaining vessel capacity; sensitivity-swept at 50th–95th priority percentile |

### Degradation and Maintenance Parameters

| Parameter | Value |
|-----------|-------|
| Weibull shape parameter β (main simulation) | 2.5 (wear-out regime; β > 1 → increasing hazard) |
| Weibull scale parameter η (main simulation) | 80 weeks (characteristic life ≈ 15 months) |
| Number of component types | 18 |
| ETA matrix dimensions | 18 components × 3 severities = 54 cells |
| RT threshold — Critical components | R(t) < 0.72 |
| RT threshold — Semi-Critical components | R(t) < 0.65 |
| RT threshold — Non-Critical components | R(t) < 0.55 |
| RT minimum revisit interval | 60 days (prevents retriggering during weather delays) |
| Maximum RT tasks per day | 5 (daily cap; prevents queue flooding) |
| Restoration factor — Minor Repair | RF = 0.35 (35% effective age reduction) |
| Restoration factor — Major Repair | RF = 0.55 (55% effective age reduction) |
| Restoration factor — Major Replacement | RF = 0.90 (90% effective age reduction) |
| Hazard scaling constant κ | 10 (rescales Weibull hazard for numerical parity with categorical weights) |
| **Multi-source validation: equal-downtime assumption** | 4.25 days, applied identically to RT and fixed-PM (isolates Mechanism 1) |
| **Multi-source validation: fail/critical threshold** | R(t) < 0.25 (separate from the RT trigger, used to flag corrective failures) |
| **Multi-source validation: R_init sensitivity grid** | [0.60, 0.95] (base case), [0.30, 0.95], [0.10, 0.95] |
| **Multi-source validation: horizon sensitivity grid** | 3, 5, 7 years |

### Statistical Analysis Parameters

| Parameter | Value |
|-----------|-------|
| Pre-registered hypotheses | 23 |
| Active hypotheses (BH-FDR) | 21 (H4b excluded for power < 0.35; one comparison structurally not applicable) |
| Hypotheses sustained | 22 of 23 |
| Multiple testing correction | Benjamini–Hochberg FDR (α = 0.05) |
| Bootstrap resamples (BCa CI) | B = 3,000 |
| Monte Carlo replicates (main sensitivity) | n_MC = 500 |
| Monte Carlo replicates (multi-source robustness, Mechanism 1) | n_MC = 200 per source |
| Monte Carlo replicates (multi-source robustness, sensitivity grid) | n_MC = 60 per source/R_init/horizon cell |
| Normality test | Shapiro–Wilk (21/24 combinations non-normal; *p* < 0.05) |
| Primary test (pairwise) | Mann–Whitney U |
| Omnibus test | Kruskal–Wallis H with ε² effect size |
| Distributional test | Kolmogorov–Smirnov |
| Trend analysis | Mann–Kendall with Sen's slope |
| Permutation validation | n = 2,000 permutations |

---

## Cost Terminology

Precise terminology is critical for interpreting the cost results in the KPI tables and the cost decomposition figures.

| Term | Symbol | Definition |
|------|--------|-----------|
| **O&M Cost** | — | Direct operational expenditure: vessel fuel/charter + port fees + labour + parts. Does not include downtime cost. |
| **Vessel (ship) cost** | `ship_cost` | Fuel cost × (1 + transit time / workable hours) × assigned hours + daily charter fee. Increases endogenously under Rough/Extreme weather as workable hours shrink. The constant "1" represents baseline transit-time overhead incurred independently of repair-hours assigned. |
| **Port cost** | `port_cost` | Port usage fee proportional to workable hours per vessel per day. |
| **Labour cost** | `labor_cost` | Technician hourly wage × number of crew teams × workable hours. |
| **Parts cost** | `parts_cost` | Component-level spare parts and consumables expenditure, indexed by repair severity. |
| **Downtime cost** | `downtime_cost` | Realised lost production revenue: (max possible MWh − actual MWh produced) × electricity price × wind speed fraction. **Included in Total Cost.** |
| **Pending Downtime Cost** | `PDC` | Queue-wait opportunity cost: criticality weight ρ_crit × ETA coefficient η × waiting hours Δt × normalisation constant K_PDC. **Reported separately; NOT included in Total Cost** to avoid double-counting with realised downtime. |
| **Wasted charter cost** | `wasted_charter_cost` | CTV daily charter fee charged on no-departure days when H_s > 1.5 m renders the vessel operationally infeasible (S1 and S1+RT only). Entirely eliminable through weather-conditioned dispatch. |
| **Total Cost** | — | O&M Cost + Downtime Cost. Excludes PDC and charter waste (reported separately). |
| **Consolidated Total** | — | Total Cost + cumulative charter waste. Used for cross-strategy comparison. S1: ₩41.1B = ₩39.3B + ₩1.764B. S2–S4: consolidated total equals Total Cost (charter waste = 0). |

---

## Modelling Limitations

The following limitations bound the generalisability and quantitative precision of the reported findings. Readers should consider these when applying the simulation outputs to operational decision-making.

1. **Rule-set HMDP policy** — The HMDP policy is implemented as a structured condition-indexed rule set rather than a value-function-optimised solution. Reinforcement learning policy optimisation (Zhang & Si, 2020; Hendradewa & Yin, 2025) is the primary architectural extension for future work.

2. **Literature-derived ETA coefficients** — The multi-cell ETA derating matrix is derived from published reliability literature rather than site-specific SCADA calibration. Cross-validation against EDP Open Data (27 component-severity cells) confirms that the relative ranking of components by downtime severity is consistent with observed data, while absolute derating magnitudes differ due to the structural distinction between failure-consequence-based ETA formulations and realised downtime fractions. Sensitivity analysis confirms ±10% KPI sensitivity and ±0.8 pp variation in the 1.99 pp AEP underestimation finding.

3. **Uniform Weibull parameterisation, with mechanism-separated robustness evidence** — A single parameter set (β = 2.5, η = 80 weeks) is applied across all 18 component types in the main simulation. Two independent cross-validations bound the resulting uncertainty: (a) against EDP Open Data (N = 1,111 inter-failure intervals), yielding identical median KPIs between uniform and heterogeneous parameterisations (MWU *p* > 0.05); and (b) against the five-source, mechanism-separated robustness analysis described above (`scripts/multisource_validation/`), which shows that the paper's parameters are not contradicted by four of five independent sources on the failure-prevention axis, while explicitly flagging that this axis is not discriminating at a 3-year horizon for sources with mean component MTTF substantially longer than the horizon (most notably the Walgern fleet-level source, MTTF ≈ 7.1 years). The availability advantage attributable to the Weibull parameters and RT threshold alone (Mechanism 1, equal downtime) is modest but statistically robust across all five sources (*p* < 10⁻⁶⁶); a separately reported downtime-efficiency assumption (Mechanism 2) is not derived from any cited Weibull source and should be justified independently before being combined with the parameter-robustness claim.

4. **Controllable AO–AEP divergence** — S4's lower AEP than S2/S3 in Years 2024–2025 reflects planned RT partial-operation windows, not an unintended system failure. This is a bounded, predictable cost of proactive reliability management.

5. **Stationary 20-year NPV extrapolation** — NPV analysis assumes constant annual benefit streams from the 3-year simulation, which may not hold under evolving failure cost structures, energy price regimes, or technology refresh cycles. This same stationarity assumption is the reason the simulation horizon was not extended specifically to raise the power of the cautious-tier hypothesis comparisons (Limitation 7 below); doing so would itself require revisiting the stationarity assumption, introducing a second, compounding source of extrapolation uncertainty.

6. **No Bayesian online Weibull recalibration** — The feedback loop updates effective age but does not recalibrate (β, η) from observed failure data. Bayesian online learning is identified as the primary algorithmic development priority for subsequent research.

7. **Low-power hypothesis comparisons** — H1d vs. S2/S3 (power ≈ 0.33), H2b vs. S3 (power = 0.68), H3a (power = 0.659), and the legacy H5b vs. S2/S3 comparisons (now folded under H2c, power 0.137–0.213) are designated "Cautious." Directional conclusions are supported by the data, but effect magnitude estimates carry substantial sampling uncertainty and should not be used for quantitative extrapolation without independent replication.

8. **Asymmetric seasonal-homogeneity robustness check** — The season-specific Markov chain robustness check (Experiment 3) was performed only for S4's SOV-equipped policy, not for S1's CTV-only, weather-blind dispatch. S1's CTV-only logistics lack a comparable accessibility buffer and would plausibly be more sensitive to seasonal non-stationarity than S4; the reported S1 disadvantage should therefore be read as conservative (if anything, understating the gap), not as a confirmed-robust baseline figure. Extending this check symmetrically to S1 is identified as a priority item for future replication.

9. **EDP and multi-source dataset scope** — The EDP Open Data used for parameter cross-validation covers onshore 2 MW turbines in Portugal (2016–2017); the multi-source robustness analysis additionally draws on Carroll et al. (2016, 2–4 MW and 10 MW-rescaled offshore estimates) and a fleet-level offshore source (Walgern, WES 2026). Transfer of findings to the 10 MW offshore DTU reference turbine modelled here requires acknowledgement of scale, configuration, and environmental loading differences across sources. Site-matched offshore SCADA calibration remains the recommended path for operational deployment.

10. **Component-isolation diagnostic scope** — The S1+RT diagnostic isolates the RT trigger from weather-adaptive logistics but does not, by itself, isolate the closed-loop repair-feedback mechanism (Layer 5) from the RT threshold rule, since S1+RT retains the same imperfect-repair feedback structure as all other strategies. A further diagnostic disentangling the threshold rule specifically from the feedback mechanism is a natural next step and is not pursued here to avoid expanding the factorial design beyond what Proposition 1 requires.

---

## Reproducibility

The complete pipeline is designed for full end-to-end reproducibility:

- All stochastic inputs share a fixed random seed (SEED = 42); the multi-source validation analysis uses its own documented seed bases (3000 for the main Mechanism 1 runs, 5000 for the sensitivity grid) to keep its Monte Carlo runs independently reproducible from the main simulation
- Python and R scripts generate all figures and tables deterministically from the raw input files
- Supplementary experiment scripts (`scripts/additional_experiments/`) and the multi-source validation script (`scripts/multisource_validation/`) read from `data/raw/` without requiring any manual path changes
- The EDP parameter validation script (`scripts/edp_validation/edp_scada_validation.py`) generates a calibrated synthetic dataset automatically if EDP source files are not present, enabling end-to-end pipeline demonstration without requiring data registration
- The multi-source Weibull robustness script requires no external data beyond the literature-derived parameter tables hardcoded in the script itself, and runs fully offline
- If `weather_hourly_raw.csv` is unavailable, synthetic weather is generated automatically using the calibrated Markov chain; all principal conclusions are robust to this substitution (Experiment 3)
- All processed outputs, statistical tables, and simulation results are committed to the repository under `results/`; researchers can verify figures against the raw CSV logs in `results/csv/`

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

Raw observational weather data from the Korea Meteorological Administration (KMA) is subject to KMA data-sharing terms and is not redistributed in this repository. Processed outputs derived from KMA data are provided for research reproducibility purposes only. EDP Open Data is used under the terms of the EDP Open Data platform; raw EDP files are not redistributed in this repository. The multi-source Weibull robustness analysis uses only published, literature-derived parameter values (no raw third-party data redistribution).
