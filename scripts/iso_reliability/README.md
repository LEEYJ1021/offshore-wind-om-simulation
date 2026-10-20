# `scripts/iso_reliability/` — Iso-Reliability Benchmarking Protocol

| File | Purpose |
|------|---------|
| `03_iso_reliability_benchmark.py` | Single-file implementation of the protocol (v11): simulator (vectorised NumPy), Algorithm 1 (frontier interpolation + local-slope correction), Algorithm 2 (frequency matching), Algorithm 3 (2 × 2 attribution + paired bootstrap), baselines (IDX, IDXL, PS, SEN), structural misspecification, cost re-weighting, capacity/scale experiments, tuning budget, equivalence and slope-correction checks, run-time measurement |
| `04_make_figures.py` | Publication figures (v14): Fig. 1–5 and Fig. A1–A4, strictly black/white/gray, PDF + PNG (+ TIFF with `--tiff`) |

## Blocks of `03_iso_reliability_benchmark.py`

| Block | Content | Raw output |
|-------|---------|-----------|
| M1 | Core study S0–S3: TBM, EAT, LTBM, BAY, BAYe, ORA at q25/q50/q75 and cost optimum; TBMm/TBMmB calibration | `Y1`, `Y2`, `Y3`, `Y4` |
| M1b | Baselines on the **same target failure levels**: IDX, IDXL, PS (cross-entropy), SEN (5 variants); PS weights | `Y0`, `Y2`, `Y3` |
| M2 | Structural misspecification (10 settings × S1, S3) with applicability flag and retention | `Y5`, `Y5d` |
| M3 | Cost re-pricing (±50 % single component; 400 joint Latin-hypercube draws) | `Y6`, `Y6b`, `Y6c` |
| M4 | Information ladder (TBM … ORA) | `Y7` |
| M5 | Capacity sweep (fixed parameters) and re-tuned capacity analysis | `Y9`, `Y9b` |
| M6 | Fleet size N = 10…200, capacity fixed vs. scaled | `Y10` |
| M7 | Tuning budget (TBM3, EAT2, BAYe2) | `Y11` |
| M8 | Equivalence heuristic (S0) and slope-correction bias | `Y12`, `Y13` |
| M9 | Flat cost optimum diagnostic | `Y14` |
| M10 | Measured run time and N scaling | `Y15` |
| M11 | Manuscript number sheet, draft figures, manifest | `V11_manuscript_numbers.txt`, `V11_manifest.json`, `Fig2_iso_cost` … `Fig6_capacity_scale` |

## Usage

```bash
cd results/iso_reliability                      # script writes ./outputs_v11
V11_FAST=1 V11_HEADLESS=1 python ../../scripts/iso_reliability/03_iso_reliability_benchmark.py   # smoke test
NPROC=23 V11_HEADLESS=1 python ../../scripts/iso_reliability/03_iso_reliability_benchmark.py     # full run (~23 min, 23 workers)
python ../../scripts/iso_reliability/04_make_figures.py --indir outputs_v11 --outdir figures --dpi 600
```

| Variable | Meaning |
|----------|---------|
| `V11_FAST` | `1` = reduced design (3 test replicates, 1 training seed, smaller grids/bootstrap) for installation checks only |
| `V11_HEADLESS` | `1` = Agg backend, no plot windows |
| `NPROC` | worker processes (default: CPU count − 1) |
| `CALIB_TARGET` | annual critical failures per turbine; adds an externally calibrated scenario S4 |

## Notes

* **Weather path.** `KMA_PATHS` near the top of the script lists the raw KMA CSV locations; add `../../data/raw/weather_hourly_raw.csv` (repository-relative) instead of machine-specific absolute paths. Without the file the script uses a published fallback transition matrix and results will differ from the reported tables. The script prints the number of daily observations (reported run: 1,012).
* **Regression check.** If `outputs_v8/X1_raw.csv` is present the script compares BAYe q50 critical failures (S1) with the earlier code version (reported max difference 0.0000); otherwise the check is skipped.
* **Figure script inputs** (in `--indir`): `Y1, Y2, Y3, Y4, Y5, Y6b, Y7, Y9, Y10, Y11, Y13, Y14, Y15` (`*.csv`). Missing files are reported and the affected figure is skipped.
* **Platform.** Parallelism uses `multiprocessing` with the `fork` context (Linux/macOS); the script falls back to serial execution if the pool fails.
* **Assumptions.** All cost, repair, heterogeneity, misspecification and sensor parameters are modelling assumptions; only the weather process is estimated from data.
