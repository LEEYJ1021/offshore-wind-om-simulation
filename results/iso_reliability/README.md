# `results/iso_reliability/` — Output Index

```
iso_reliability/
├── outputs_v11/     raw files written by 03_iso_reliability_benchmark.py (not curated)
├── tables/          T01–T21: curated, manuscript-ready CSV tables
├── figures/         publication figures from 04_make_figures.py + FIGURES.md
└── manifest/        V11_manifest_curated.json
```

All differences are in **bn KRW** over the 3-year measurement window unless stated; negative values favour the first-named policy. Intervals are 95 % percentile-bootstrap intervals over 50 paired replicates (B = 20,000). Scenario labels: S0 homogeneous; S1 heterogeneous (σ = 0.35); S2 heterogeneous + faster wear (σ = 0.35, κ = 0.8); S3 strong heterogeneity (σ = 0.60). "crit" = critical failures over 3 years.

## Curated tables

| Table | Content | Raw source |
|-------|---------|-----------|
| `T01_simulation_parameters.csv` | Model and cost parameters (all assumptions except weather) | script constants |
| `T02_policy_set.csv` | Policies, trigger rules, information, tuning parameter, grid | script |
| `T03_protocol_validity.csv` | Equal-reliability targets; frequency-matching task-count ratios; fallback/grid-boundary flags | `Y1`, console log |
| `T04_iso_reliability_cost_vs_TBM.csv` | Cost vs. tuned TBM at q25/q50/q75 (CI at q50), % of TBM cost | `Y2` (`iso_vs_TBM`) |
| `T05_trigger_effect_decomposition.csv` | EAT−TBM, EAT−TBMm in failures and cost at cost-optimal parameters | `Y2` (`opt_vs_TBM`, `opt_vs_TBMm`) |
| `T06_equivalence_heuristic_check_S0.csv` | EAT vs. frequency-matched calendar under deterministic repair / unconstrained access | `Y12` |
| `T07_attribution_2x2_q50.csv` | State-tracking, learning and interaction effects (BAY) | `Y4` |
| `T08_oracle_recovery_q50.csv` | Share of oracle benefit recovered; LTBM share of BAY/BAYe gain | `Y3` |
| `T09_information_ladder_q50.csv` | All policies vs. TBM and vs. BAYe | `Y7` |
| `T10_structural_misspecification_q50.csv` | 20 settings × (EAT, LTBM, BAYe, ORA) vs. TBM, BAYe−LTBM, validity flag, retention | `Y5`, `Y5d` |
| `T11_cost_reweighting_400_draws.csv` | Share of favourable/significant draws and range of differences | `Y6c` |
| `T12_capacity_sweep_fixed_parameters.csv` | BAYe−TBM total cost and TBM queue by capacity and access mode | `Y9` |
| `T13_capacity_retuned_ctv_only.csv` | Re-tuned capacity analysis; fallback rows flagged *not interpreted* | `Y9b` |
| `T14_scale_experiment_q50.csv` | BAYe vs. TBM, N = 10–200, capacity fixed vs. scaled | `Y10` |
| `T15_tuning_budget.csv` | TBM3, EAT2, BAYe2 vs. one-parameter versions and each other | `Y11` |
| `T16_slope_correction_bias.csv` | Bias of Eq. (6) at displaced targets | `Y13` |
| `T17_cost_optimum_flatness.csv` | Grid points within 1 bn of the cost optimum and their failure span | `Y14` |
| `T18_runtime_scaling.csv` | Seconds per simulation and log-log slope | `Y15` |
| `T19_policy_search_and_learning_accuracy.csv` | PS training cost/weights; Spearman of posterior vs. true wear; uncovered targets | `Y0`, console log |
| `T20_prior_misspecification_supplementary.csv` | Fixed vs. empirical-Bayes prior sensitivity (earlier-stage run) | earlier-stage experiment |
| `T21_energy_output_q50.csv` | Mean annual energy production (% of rated) at q50 | `Y1` |

## Raw file mapping (`outputs_v11/`)

| File | Content |
|------|---------|
| `Y0_ps_weights.csv` | Cross-entropy policy-search weights per scenario |
| `Y1_core_means.csv` | Mean crit, adjusted cost, AEP, planned tasks per policy/operating point |
| `Y2_core_contrasts.csv` | All paired contrasts with CI and Holm p |
| `Y3_recovery.csv` | Oracle recovery and LTBM share |
| `Y4_attribution.csv` | 2 × 2 effects (BAY and BAYe learners, q25–q75) |
| `Y5_misspec.csv`, `Y5d_retention.csv` | Misspecification contrasts and BAYe benefit retention |
| `Y6_cost_oat.csv`, `Y6b_cost_joint.csv`, `Y6c_cost_summary.csv` | Cost re-pricing |
| `Y7_info_ladder.csv` | Information ladder |
| `Y9_capacity_sweep.csv`, `Y9b_capacity_retuned.csv` | Capacity analyses |
| `Y10_scale.csv` | Fleet-size experiment |
| `Y11_tuning_budget.csv` | Tuning-budget check |
| `Y12_propA.csv`, `Y13_propB.csv` | Equivalence heuristic; slope-correction bias |
| `Y14_opt_flatness.csv`, `Y15_runtime.csv` | Diagnostics and run time |
| `V11_manuscript_numbers.txt`, `V11_manifest.json` | Number sheet; versions, seeds, run time |

## Reading notes

* Contrasts at levels not covered by a baseline's frontier (PS at q25 in S1 and S2; SEN with independent noise 1.0 at q75 in S2 and S3) are not interpreted; all q50 contrasts are covered.
* Capacity cells marked `frame_not_applicable_not_interpreted` / `algorithm1_fallback_used = True` are artefacts of a collapsed reliability range and must not be quoted.
* `frame_valid = False` in T10 marks the two +25 % repair-induced-wear settings; they are reported but excluded from summary counts. The S3 +10 % setting is retained as valid but borderline.
* T20 comes from an earlier stage of the study (before the final protocol was fixed) and is supplementary evidence only.
