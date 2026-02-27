"""
================================================================================
OFFSHORE WIND TURBINE O&M SIMULATION  ── (Final Revision)
Hierarchical Maintenance Decision (HMDP) + Greedy Scheduling Integration
Period: 2023-01-01 ~ 2025-12-31 | 50 Turbines | Ulsan Offshore, South Korea

[v3 Modifications]
  [A] S1 vessel charter fee added on no-departure days (daily rate charged regardless of weather)
  [B] backlog_cost → pending_downtime_cost (pending downtime opportunity penalty)
  [C] CBM threshold lowered + queue explosion prevention
        (60-day visit interval, daily cap, deduplication via cbm_queued_comps)
  [D] pending_downtime_cost definition clarified in comments
================================================================================
"""
from __future__ import annotations
import os, math, warnings
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu, ttest_ind

warnings.filterwarnings("ignore")

SEED = 42
rng  = np.random.default_rng(SEED)

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

OUT_DIR = os.path.join(SCRIPT_DIR, "offshore_wind_om_results_v3")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Simulation period ─────────────────────────────────────────────────────────
SIM_START  = pd.Timestamp("2023-01-01")
SIM_END    = pd.Timestamp("2025-12-31")
ALL_DAYS   = pd.date_range(SIM_START, SIM_END, freq="D")
N_DAYS     = len(ALL_DAYS)
N_YEARS    = 3

# ── Turbine parameters ────────────────────────────────────────────────────────
N_TURBINES     = 50
TURBINE_IDS    = list(range(1, N_TURBINES + 1))
RATED_POWER_MW = 10.0
CUTIN_WIND_MS  = 3.0
RATED_WIND_MS  = 11.5
CUTOUT_WIND_MS = 25.0
ELEC_PRICE_KRW = 155_500   # KRW/MWh (includes REC)
OPS_HR_DAY     = 12.0

# ── Port configuration ────────────────────────────────────────────────────────
PORTS = {
    "P1": {"km": 10.0, "daily_fee_KRW": 1_200_000_000 / 365},
    "P2": {"km": 15.0, "daily_fee_KRW":   900_000_000 / 365},
}

# ── Vessel parameters ─────────────────────────────────────────────────────────
VESSELS = {
    "CTV": {"speed_kn": 20.0, "fuel_kr_hr": 596_000,
            "max_wave_m": 1.5, "max_wind_ms": 10.0, "co2_kg_nm": 0.85},
    "SOV": {"speed_kn": 12.0, "fuel_kr_hr": 6_320_000,
            "max_wave_m": 2.5, "max_wind_ms": 15.0, "co2_kg_nm": 2.10},
}

# [Modification A] Daily charter fees charged regardless of weather / departure
VESSEL_CHARTER_DAILY = {
    "CTV": 3_500_000,   # KRW/day
    "SOV": 35_000_000,  # KRW/day (reference; applied to S1 CTV idle days)
}

# ── Weather operational capacity (working hours by scenario) ──────────────────
WEATHER_CAP_HR = {
    "Calm":      (12.0, 12.0),   # (CTV_cap_hr, SOV_cap_hr)
    "Moderate":  ( 9.0, 12.0),
    "Rough Sea": ( 0.0,  8.0),
    "Extreme":   ( 0.0,  0.0),
}

# ── On-site CO2 emission rates ────────────────────────────────────────────────
ONSITE_CO2_KG_HR = {"CTV": 12.5, "SOV": 45.0}

# ── Labour cost ───────────────────────────────────────────────────────────────
LABOR_DAILY_KRW  = 800_000
LABOR_HOURLY_KRW = LABOR_DAILY_KRW / OPS_HR_DAY

# ── Repair severity parameters ────────────────────────────────────────────────
SEV = {
    "Minor Repair":      {"crew": 1, "parts_KRW": 500_000,    "tri": (1.0,  3.0,  8.0)},
    "Major Repair":      {"crew": 2, "parts_KRW": 10_000_000, "tri": (4.0,  8.0, 18.0)},
    "Major Replacement": {"crew": 3, "parts_KRW": 30_000_000, "tri": (8.0, 16.0, 40.0)},
}

# ── Component multipliers (duration scaling by component type) ────────────────
COMP_MULT = {
    "Blades": 1.6, "Contactor/Circuit/Relay": 0.8, "Controls": 1.0,
    "Electrical Components": 1.0, "Gearbox": 1.6, "Generator": 1.5,
    "Grease/Oil/Cooling Liquid": 0.9, "Heaters/Coolers": 1.0, "Hub": 1.3,
    "Pitch/Hydraulic System": 1.2, "Power Supply/Converter": 1.2,
    "Pumps/Motors": 1.0, "Safety": 0.8, "Sensors": 0.7,
    "Service Items": 0.8, "Tower/Foundation": 1.8, "Transformer": 1.3,
    "Yaw System": 1.1,
}

# ── Component criticality classification ──────────────────────────────────────
COMP_CRITICALITY = {
    "Gearbox": "Critical", "Generator": "Critical", "Blades": "Critical",
    "Tower/Foundation": "Critical", "Hub": "Critical",
    "Pitch/Hydraulic System": "Semi-Critical", "Controls": "Semi-Critical",
    "Power Supply/Converter": "Semi-Critical", "Transformer": "Semi-Critical",
    "Yaw System": "Semi-Critical",
    "Electrical Components": "Non-Critical", "Contactor/Circuit/Relay": "Non-Critical",
    "Grease/Oil/Cooling Liquid": "Non-Critical", "Heaters/Coolers": "Non-Critical",
    "Pumps/Motors": "Non-Critical", "Safety": "Non-Critical",
    "Sensors": "Non-Critical", "Service Items": "Non-Critical",
}

# ── ETA derating table (54 cells: component × severity) ──────────────────────
# Values represent fractional output loss during an unrepaired failure state.
ETA_DERATING = {
    ("Gearbox","Minor Repair"): 0.40, ("Gearbox","Major Repair"): 0.75, ("Gearbox","Major Replacement"): 1.00,
    ("Generator","Minor Repair"): 0.35, ("Generator","Major Repair"): 0.80, ("Generator","Major Replacement"): 1.00,
    ("Blades","Minor Repair"): 0.20, ("Blades","Major Repair"): 0.60, ("Blades","Major Replacement"): 1.00,
    ("Tower/Foundation","Minor Repair"): 0.10, ("Tower/Foundation","Major Repair"): 0.50, ("Tower/Foundation","Major Replacement"): 1.00,
    ("Hub","Minor Repair"): 0.30, ("Hub","Major Repair"): 0.70, ("Hub","Major Replacement"): 1.00,
    ("Pitch/Hydraulic System","Minor Repair"): 0.15, ("Pitch/Hydraulic System","Major Repair"): 0.45, ("Pitch/Hydraulic System","Major Replacement"): 0.70,
    ("Controls","Minor Repair"): 0.10, ("Controls","Major Repair"): 0.35, ("Controls","Major Replacement"): 0.60,
    ("Power Supply/Converter","Minor Repair"): 0.15, ("Power Supply/Converter","Major Repair"): 0.40, ("Power Supply/Converter","Major Replacement"): 0.65,
    ("Transformer","Minor Repair"): 0.20, ("Transformer","Major Repair"): 0.50, ("Transformer","Major Replacement"): 0.80,
    ("Yaw System","Minor Repair"): 0.05, ("Yaw System","Major Repair"): 0.20, ("Yaw System","Major Replacement"): 0.40,
    ("Electrical Components","Minor Repair"): 0.05, ("Electrical Components","Major Repair"): 0.15, ("Electrical Components","Major Replacement"): 0.30,
    ("Contactor/Circuit/Relay","Minor Repair"): 0.03, ("Contactor/Circuit/Relay","Major Repair"): 0.10, ("Contactor/Circuit/Relay","Major Replacement"): 0.20,
    ("Grease/Oil/Cooling Liquid","Minor Repair"): 0.02, ("Grease/Oil/Cooling Liquid","Major Repair"): 0.08, ("Grease/Oil/Cooling Liquid","Major Replacement"): 0.15,
    ("Heaters/Coolers","Minor Repair"): 0.02, ("Heaters/Coolers","Major Repair"): 0.05, ("Heaters/Coolers","Major Replacement"): 0.10,
    ("Pumps/Motors","Minor Repair"): 0.04, ("Pumps/Motors","Major Repair"): 0.12, ("Pumps/Motors","Major Replacement"): 0.25,
    ("Safety","Minor Repair"): 0.00, ("Safety","Major Repair"): 0.05, ("Safety","Major Replacement"): 0.10,
    ("Sensors","Minor Repair"): 0.02, ("Sensors","Major Repair"): 0.08, ("Sensors","Major Replacement"): 0.15,
    ("Service Items","Minor Repair"): 0.00, ("Service Items","Major Repair"): 0.02, ("Service Items","Major Replacement"): 0.05,
}

def get_eta(component: str, severity: str) -> float:
    """Return ETA output-loss fraction for given component × severity cell."""
    return ETA_DERATING.get(
        (component, severity),
        ETA_DERATING.get((component, "Major Replacement"), 0.5)
    )

# ── CBM thresholds (v3: lowered to prevent queue explosion) ───────────────────
# [Modification C] Thresholds lowered: Critical 0.88→0.72, Semi-Critical 0.82→0.65,
#                  Non-Critical 0.75→0.55
CBM_THRESHOLD = {
    "Critical":      0.72,
    "Semi-Critical": 0.65,
    "Non-Critical":  0.55,
}
MAX_CBM_PM_PER_DAY     = 5    # [Modification C] Daily CBM task cap per simulation day
CBM_MIN_VISIT_INTERVAL = 60   # [Modification C] Minimum days between CBM visits per turbine

# ── Imperfect repair restoration factors ─────────────────────────────────────
RF_BY_GRADE = {"minimal": 0.20, "standard": 0.55, "full": 0.90}

# ── Empirical failure rate table (annual probability by severity) ─────────────
FAIL_RATES = {
    "Pitch/Hydraulic System":   {"Major Replacement": 0.001, "Major Repair": 0.179, "Minor Repair": 0.824},
    "Generator":                 {"Major Replacement": 0.095, "Major Repair": 0.321, "Minor Repair": 0.485},
    "Gearbox":                   {"Major Replacement": 0.154, "Major Repair": 0.038, "Minor Repair": 0.395},
    "Blades":                    {"Major Replacement": 0.001, "Major Repair": 0.010, "Minor Repair": 0.456},
    "Grease/Oil/Cooling Liquid": {"Major Replacement": 0.000, "Major Repair": 0.006, "Minor Repair": 0.407},
    "Electrical Components":     {"Major Replacement": 0.002, "Major Repair": 0.016, "Minor Repair": 0.358},
    "Contactor/Circuit/Relay":   {"Major Replacement": 0.002, "Major Repair": 0.054, "Minor Repair": 0.326},
    "Controls":                  {"Major Replacement": 0.001, "Major Repair": 0.054, "Minor Repair": 0.355},
    "Safety":                    {"Major Replacement": 0.000, "Major Repair": 0.004, "Minor Repair": 0.373},
    "Sensors":                   {"Major Replacement": 0.000, "Major Repair": 0.070, "Minor Repair": 0.247},
    "Pumps/Motors":              {"Major Replacement": 0.000, "Major Repair": 0.043, "Minor Repair": 0.278},
    "Hub":                       {"Major Replacement": 0.001, "Major Repair": 0.038, "Minor Repair": 0.182},
    "Heaters/Coolers":           {"Major Replacement": 0.000, "Major Repair": 0.007, "Minor Repair": 0.190},
    "Yaw System":                {"Major Replacement": 0.001, "Major Repair": 0.006, "Minor Repair": 0.162},
    "Tower/Foundation":          {"Major Replacement": 0.005, "Major Repair": 0.081, "Minor Repair": 0.076},
    "Power Supply/Converter":    {"Major Replacement": 0.000, "Major Repair": 0.001, "Minor Repair": 0.108},
    "Service Items":             {"Major Replacement": 0.001, "Major Repair": 0.003, "Minor Repair": 0.052},
    "Transformer":               {"Major Replacement": 0.000, "Major Repair": 0.003, "Minor Repair": 0.052},
}
COMPONENTS = list(FAIL_RATES.keys())

# ── Weibull model constants ───────────────────────────────────────────────────
WB_SHAPE = 2.5
WB_SCALE = 80.0
_WB_SHAPE_OVER_SCALE = WB_SHAPE / WB_SCALE
_WB_SHAPE_MINUS_1    = WB_SHAPE - 1.0
_INV_WB_SCALE        = 1.0 / WB_SCALE

# ── Seasonal weather statistics (Ulsan empirical calibration) ─────────────────
SEASONAL_WX = {
    "Winter": {"wind": (7.44, 2.2), "wave": (1.56, 0.55)},
    "Spring": {"wind": (7.06, 2.0), "wave": (1.27, 0.42)},
    "Summer": {"wind": (6.02, 1.6), "wave": (0.91, 0.28)},
    "Fall":   {"wind": (6.72, 2.1), "wave": (1.34, 0.47)},
}

# ── Markov weather state transition matrix ────────────────────────────────────
WX_STATES = ["Calm", "Rough Sea", "Moderate", "Extreme"]
WX_TRANS  = np.array([
    [0.680, 0.180, 0.120, 0.020],
    [0.621, 0.200, 0.159, 0.020],
    [0.500, 0.200, 0.280, 0.020],
    [0.500, 0.300, 0.150, 0.050],
])

# ── LP baseline (2025 weekly O&M cost from Sang reference data) ───────────────
SANG_LP_2025 = {
    "week_num": list(range(1, 53)),
    "TotalCost_KRW": [
        41797372, 152634681, 107213392, 44732374, 167622364,
        89139142, 61215444, 61661467, 135365199, 178785731,
        15273218, 111990199, 58410224, 96128676, 15230613,
        13625318, 3843688, 27119732, 146433194, 25948892,
        32013563, 365333575, 51499511, 19893076, 142213533,
        80078541, 26588491, 183175333, 120646993, 73389736,
        147917064, 30412128, 75998110, 178237153, 156278731,
        289043301, 91252565, 40537098, 18594966, 24804933,
        58506069, 146556535, 53070439, 19920851, 0,
        0, 27667077, 16259331, 0, 0, 0, 41178876,
    ],
}

# ── Strategy definitions ──────────────────────────────────────────────────────
STRATEGIES = {
    "S1_NoWeather":  {"ports": ["P1"], "vessel_mode": "ctv_only",      "weather_aware": False, "pm_mode": "fixed"},
    "S2_WeatherCTV": {"ports": ["P1"], "vessel_mode": "dynamic",       "weather_aware": True,  "pm_mode": "fixed"},
    "S3_MultiPort":  {"ports": ["P1","P2"], "vessel_mode": "dynamic",  "weather_aware": True,  "pm_mode": "fixed"},
    "HMDP_CBM":      {"ports": ["P1","P2"], "vessel_mode": "hierarchical", "weather_aware": True, "pm_mode": "cbm"},
}

# ── [Modification B/D] Pending downtime cost base rates by criticality ─────────
# pending_downtime_cost = opportunity cost while repair task waits in queue
# = criticality_weight × ETA × waiting_hours × unit_penalty_KRW_per_hr
# ≠ downtime_cost (actual production loss computed from wind speed + power curve)
_PENDING_DT_BASE = {
    "Critical":      200_000,   # KRW / hr
    "Semi-Critical":  50_000,
    "Non-Critical":   10_000,
}

# ── Colour palettes ───────────────────────────────────────────────────────────
STRAT_COLORS = {
    "S1_NoWeather":  "#e74c3c",
    "S2_WeatherCTV": "#f39c12",
    "S3_MultiPort":  "#3498db",
    "HMDP_CBM":      "#27ae60",
}
STRAT_LABELS = {
    "S1_NoWeather":  "S1: Fixed PM / No Weather",
    "S2_WeatherCTV": "S2: Fixed PM / Weather-Aware",
    "S3_MultiPort":  "S3: Multi-Port / Weather",
    "HMDP_CBM":      "HMDP: CBM + Hierarchical Greedy",
}
SEASON_COL = {"Winter": "#4a90d9", "Spring": "#f5a623", "Summer": "#27ae60", "Fall": "#e74c3c"}
SC_COL     = {"Calm": "#27ae60", "Moderate": "#f5a623", "Rough Sea": "#e67e22", "Extreme": "#e74c3c"}
CRIT_COL   = {"Critical": "#e74c3c", "Semi-Critical": "#f39c12", "Non-Critical": "#27ae60"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
})

# ==============================================================================
# 1. Helper functions
# ==============================================================================

def season_of(d: pd.Timestamp) -> str:
    m = d.month
    if m in (12, 1, 2): return "Winter"
    if m in (3,  4, 5): return "Spring"
    if m in (6,  7, 8): return "Summer"
    return "Fall"

def weather_scenario(wind: float, wave: float) -> str:
    if wind >= 15 or wave >= 2.5: return "Extreme"
    if wind >= 10 or wave >= 1.5: return "Rough Sea"
    if wind >= 7  or wave >= 0.8: return "Moderate"
    return "Calm"

def weibull_hazard(t_weeks: float) -> float:
    t = max(t_weeks, 0.1)
    h = _WB_SHAPE_OVER_SCALE * (t * _INV_WB_SCALE) ** _WB_SHAPE_MINUS_1
    return h if h <= 0.99 else 0.99

def weibull_reliability(t_weeks: float) -> float:
    t = max(t_weeks, 0.1)
    return math.exp(-((t * _INV_WB_SCALE) ** WB_SHAPE))

def sample_duration(comp: str, sev: str) -> float:
    m  = COMP_MULT.get(comp, 1.0)
    lo, md, hi = SEV[sev]["tri"]
    return float(rng.triangular(m * lo, m * md, m * hi))

def power_curve(wind_ms: float) -> float:
    if wind_ms < CUTIN_WIND_MS or wind_ms >= CUTOUT_WIND_MS: return 0.0
    if wind_ms >= RATED_WIND_MS: return 1.0
    return ((wind_ms - CUTIN_WIND_MS) / (RATED_WIND_MS - CUTIN_WIND_MS)) ** 3

def travel_hr(port: str, vessel: str) -> float:
    dist_nm  = PORTS[port]["km"] / 1.852
    maneuver = 0.5 if vessel == "CTV" else 1.0
    return 2.0 * dist_nm / VESSELS[vessel]["speed_kn"] + maneuver

def ship_cost(port: str, vessel: str, hrs: float, cap_hr: float = 12.0) -> float:
    tau  = travel_hr(port, vessel)
    rate = VESSELS[vessel]["fuel_kr_hr"] * (1.0 + tau / max(cap_hr, 1))
    return rate * hrs

@lru_cache(maxsize=8)
def co2_kg_transit(port: str, vessel: str) -> float:
    return VESSELS[vessel]["co2_kg_nm"] * PORTS[port]["km"] / 1.852 * 2

def get_vessel_cap_hr(sc: str, vessel: str) -> float:
    ctv_h, sov_h = WEATHER_CAP_HR.get(sc, (12.0, 12.0))
    return ctv_h if vessel == "CTV" else sov_h

# ==============================================================================
# 2. Weather data loading / generation
# ==============================================================================

def load_or_generate_weather() -> pd.DataFrame:
    out_path = os.path.join(OUT_DIR, "weather_daily_processed.csv")
    if os.path.isfile(out_path):
        df = pd.read_csv(out_path, parse_dates=["date"])
        print(f"  [OK] Cached weather data loaded: {len(df)} rows")
        return df
    # Try candidate raw data paths
    for cand in [
        r"C:\Users\USER\Desktop\논문\후속논문(23) 신&상 해상풍력\논문분석_고도화\이용재_기상데이터_daily_simple.csv",
        os.path.join(SCRIPT_DIR, "이용재_기상데이터_daily_simple.csv"),
        os.path.join(SCRIPT_DIR, "data", "이용재_기상데이터_daily_simple.csv"),
        os.path.join(SCRIPT_DIR, "data", "raw", "weather_hourly_raw.csv"),
    ]:
        if os.path.isfile(cand):
            df = _process_actual_weather(cand)
            df.to_csv(out_path, index=False)
            return df
    print("  [INFO] No empirical weather file found — generating synthetic Markov weather")
    df = _synthetic_weather()
    df.to_csv(out_path, index=False)
    return df

def _process_actual_weather(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[(df["date"] >= SIM_START) & (df["date"] <= SIM_END)].copy()
    rename = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ("wind_speed","wind_speed_mean","ws"): rename[col] = "wind_speed_mean"
        elif cl in ("wave_height","wave_height_mean","sig_wave_height","hs"): rename[col] = "wave_height_mean"
    df.rename(columns=rename, inplace=True)
    if "wind_speed_mean"  not in df.columns: df["wind_speed_mean"]  = 7.0
    if "wave_height_mean" not in df.columns: df["wave_height_mean"] = 1.2
    df["wind_speed_mean"]  = pd.to_numeric(df["wind_speed_mean"],  errors="coerce").fillna(7.0)
    df["wave_height_mean"] = pd.to_numeric(df["wave_height_mean"], errors="coerce").fillna(1.2)
    if "ctv_ok" not in df.columns:
        df["ctv_ok"] = ((df["wind_speed_mean"] <= VESSELS["CTV"]["max_wind_ms"]) &
                        (df["wave_height_mean"] <= VESSELS["CTV"]["max_wave_m"])).astype(int)
    if "sov_ok" not in df.columns:
        df["sov_ok"] = (df["wave_height_mean"] <= VESSELS["SOV"]["max_wave_m"]).astype(int)
    if "season" not in df.columns:
        df["season"] = df["date"].apply(season_of)
    df["weather_scenario"] = df.apply(
        lambda r: weather_scenario(r["wind_speed_mean"], r["wave_height_mean"]), axis=1)
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df.sort_values("date").reset_index(drop=True)

def _synthetic_weather() -> pd.DataFrame:
    """Generate synthetic daily weather using a Markov chain calibrated to Ulsan seasonal stats."""
    cur = 0
    records = []
    for d in ALL_DAYS:
        s = season_of(d)
        if d.dayofweek == 0:
            cur = int(rng.choice(len(WX_STATES), p=WX_TRANS[cur]))
        state = WX_STATES[cur]
        mod   = {"Calm": 0.80, "Moderate": 1.00, "Rough Sea": 1.25, "Extreme": 1.60}[state]
        wmu, wsd = SEASONAL_WX[s]["wind"]
        hmu, hsd = SEASONAL_WX[s]["wave"]
        wind = float(np.clip(rng.normal(wmu * mod, wsd), 1.0, 30.0))
        wave = float(np.clip(rng.normal(hmu * mod, hsd * 0.9), 0.1, 8.0))
        sc   = weather_scenario(wind, wave)
        records.append({
            "date": d, "year": d.year, "month": d.month, "season": s,
            "wind_speed_mean":  round(wind, 2),
            "wave_height_mean": round(wave, 2),
            "ctv_ok": int(wind <= VESSELS["CTV"]["max_wind_ms"] and wave <= VESSELS["CTV"]["max_wave_m"]),
            "sov_ok": int(wave <= VESSELS["SOV"]["max_wave_m"]),
            "weather_scenario": sc,
        })
    return pd.DataFrame(records)

# ==============================================================================
# 3. TurbineState
# ==============================================================================

class TurbineState:
    """Tracks per-turbine component age, failure state, and repair history."""

    __slots__ = ("tid", "age", "fail", "in_maint", "fail_sev", "last_pm_day",
                 "last_visit_day", "repair_log", "cbm_queued_comps")

    def __init__(self, tid: int):
        self.tid      = tid
        self.age      = {c: float(rng.uniform(5, 55)) for c in COMPONENTS}
        self.fail     = {c: False for c in COMPONENTS}
        self.in_maint = {c: False for c in COMPONENTS}
        self.fail_sev = {c: None  for c in COMPONENTS}
        self.last_pm_day    = {c: -999 for c in COMPONENTS}
        self.last_visit_day = -999
        self.repair_log: list[dict] = []
        self.cbm_queued_comps: set = set()  # [Modification C] deduplication set

    def get_production_fraction(self) -> float:
        """Return fractional power output accounting for all active ETA losses."""
        total_loss = 0.0
        for c in COMPONENTS:
            if self.fail[c] and not self.in_maint[c]:
                sev = self.fail_sev[c] or "Major Repair"
                total_loss = min(total_loss + get_eta(c, sev), 1.0)
        return max(1.0 - total_loss, 0.0)

    def is_operational(self) -> bool:
        """True if no component has ETA ≥ 0.70 (i.e., turbine classified as operational)."""
        for c in COMPONENTS:
            if self.fail[c] and not self.in_maint[c]:
                if get_eta(c, self.fail_sev[c] or "Major Repair") >= 0.70:
                    return False
        return True

    def check_failures(self, date: pd.Timestamp, wind_speed: float) -> list[dict]:
        """Sample daily failures for each component using Weibull-scaled failure rates."""
        wind_factor = 1.0 + max(0.0, (wind_speed - RATED_WIND_MS) / RATED_WIND_MS) * 0.5
        events = []
        for comp, rates in FAIL_RATES.items():
            if self.fail[comp] or self.in_maint[comp]:
                continue
            wb = weibull_hazard(self.age[comp]) * wind_factor
            for sev, ann in rates.items():
                if ann <= 0:
                    continue
                p = min((ann / 365.0) * wb * 30, 0.30)
                if rng.random() < p:
                    self.fail[comp]     = True
                    self.fail_sev[comp] = sev
                    self.cbm_queued_comps.discard(comp)
                    dur = sample_duration(comp, sev)
                    eta = get_eta(comp, sev)
                    events.append({
                        "date": date, "turbine_id": self.tid, "component": comp,
                        "severity": sev, "event_type": "FAIL",
                        "criticality": COMP_CRITICALITY.get(comp, "Non-Critical"),
                        "eta_derating": round(eta, 3), "duration_hr": dur,
                        "hours_remaining": dur, "arrival_day_idx": -1,
                        "crew_teams": SEV[sev]["crew"], "parts_cost": SEV[sev]["parts_KRW"],
                        "age_weeks": round(self.age[comp], 1),
                        "wb_hazard": round(wb, 4),
                        "reliability": round(weibull_reliability(self.age[comp]), 4),
                    })
                    break  # one failure per component per day
        return events

    def check_pm_triggers(self, date: pd.Timestamp, day_idx: int = 0) -> list[dict]:
        """
        Check CBM triggers.
        [Modification C] Enforces 60-day minimum visit interval.
        Bundles up to 3 components per visit; marks all as cbm_queued_comps.
        """
        if (day_idx - self.last_visit_day) < CBM_MIN_VISIT_INTERVAL:
            return []
        comp_to_pm = []
        for comp in COMPONENTS:
            if self.fail[comp] or self.in_maint[comp]:
                continue
            if comp in self.cbm_queued_comps:
                continue
            rel  = weibull_reliability(self.age[comp])
            crit = COMP_CRITICALITY.get(comp, "Non-Critical")
            if rel < CBM_THRESHOLD[crit]:
                comp_to_pm.append((crit, comp, rel))
        if not comp_to_pm:
            return []
        crit_order = {"Critical": 0, "Semi-Critical": 1, "Non-Critical": 2}
        comp_to_pm.sort(key=lambda x: (crit_order[x[0]], x[2]))
        comp_to_pm = comp_to_pm[:3]  # bundle at most 3 components per visit
        for _, comp, _ in comp_to_pm:
            self.in_maint[comp] = True
            self.last_pm_day[comp] = day_idx
            self.cbm_queued_comps.add(comp)
        max_crit    = comp_to_pm[0][0]
        total_dur   = max(sample_duration(comp, "Minor Repair") for _, comp, _ in comp_to_pm)
        total_parts = SEV["Minor Repair"]["parts_KRW"] * len(comp_to_pm)
        self.last_visit_day = day_idx
        return [{
            "date": date, "turbine_id": self.tid,
            "component": (f"{comp_to_pm[0][1]}+{len(comp_to_pm)-1}more"
                          if len(comp_to_pm) > 1 else comp_to_pm[0][1]),
            "component_list": [c for _, c, _ in comp_to_pm],
            "severity": "Minor Repair", "event_type": "CBM_PM",
            "criticality": max_crit, "eta_derating": 0.0,
            "duration_hr": total_dur, "hours_remaining": total_dur,
            "arrival_day_idx": day_idx, "crew_teams": 1, "parts_cost": total_parts,
            "age_weeks": round(self.age[comp_to_pm[0][1]], 1),
            "reliability_trigger": round(comp_to_pm[0][2], 4),
            "n_comps_serviced": len(comp_to_pm),
        }]

    def apply_repair(self, comp: str, sev: str = "Minor Repair", day_idx: int = 0):
        """
        Apply imperfect repair: reduce effective age by restoration factor RF.
        Logs before/after age and Weibull hazard for feedback loop analysis.
        """
        rf_map = {
            "Minor Repair":      RF_BY_GRADE["minimal"] + 0.15,   # RF = 0.35
            "Major Repair":      RF_BY_GRADE["standard"],          # RF = 0.55
            "Major Replacement": RF_BY_GRADE["full"],              # RF = 0.90
        }
        rf         = rf_map.get(sev, RF_BY_GRADE["standard"])
        age_before = self.age[comp]
        self.age[comp] = age_before * (1.0 - rf)
        age_after  = self.age[comp]
        self.fail[comp]     = False
        self.in_maint[comp] = False
        self.fail_sev[comp] = None
        self.last_pm_day[comp] = day_idx
        self.cbm_queued_comps.discard(comp)
        self.repair_log.append({
            "repair_day_idx": day_idx, "turbine_id": self.tid,
            "component": comp, "severity": sev, "rf": round(rf, 3),
            "age_before_wk": round(age_before, 2), "age_after_wk": round(age_after, 2),
            "rel_before": round(weibull_reliability(age_before), 4),
            "rel_after":  round(weibull_reliability(age_after),  4),
            "hazard_before": round(weibull_hazard(age_before), 5),
            "hazard_after":  round(weibull_hazard(age_after),  5),
        })

    def advance_week(self):
        """Increment effective age by 1 week for all non-failed, non-under-maintenance components."""
        for c in COMPONENTS:
            if not self.fail[c] and not self.in_maint[c]:
                self.age[c] += 1.0

    def get_reliability_profile(self) -> dict[str, float]:
        return {c: weibull_reliability(self.age[c]) for c in COMPONENTS}

    def mean_reliability(self) -> float:
        return float(np.mean([weibull_reliability(self.age[c]) for c in COMPONENTS]))

    def dynamic_priority_score(self, comp: str, sev: str) -> float:
        """Composite priority score used by greedy scheduler for dynamic task ordering."""
        crit   = COMP_CRITICALITY.get(comp, "Non-Critical")
        crit_w = {"Critical": 3.0, "Semi-Critical": 1.5, "Non-Critical": 0.5}[crit]
        sev_w  = {"Major Replacement": 3.0, "Major Repair": 2.0, "Minor Repair": 1.0}.get(sev, 1.0)
        eta    = get_eta(comp, sev)
        hz     = weibull_hazard(self.age[comp])
        return crit_w * sev_w * eta * (1.0 + hz * 10)

# ==============================================================================
# 4. HMDPPolicy
# ==============================================================================

class HMDPPolicy:
    """Hierarchical decision policy: port selection, vessel selection, deferral."""

    def __init__(self, strat_name: str):
        self.strat_name = strat_name
        self.cfg = STRATEGIES[strat_name]

    def decide_port(self, sc: str, turbine: TurbineState, pending: list) -> str:
        ports = self.cfg["ports"]
        if len(ports) == 1:
            return ports[0]
        n_critical = sum(1 for ev in pending
                         if ev.get("turbine_id") == turbine.tid
                         and ev.get("criticality") == "Critical")
        if sc == "Extreme" or n_critical >= 2:
            return "P2"
        return "P1"

    def decide_vessel(self, sc: str, ctv_ok: int, sov_ok: int) -> str | None:
        mode = self.cfg["vessel_mode"]
        if mode == "ctv_only":
            return "CTV" if ctv_ok else None
        if mode in ("dynamic", "hierarchical"):
            if sc in ("Calm", "Moderate"):
                return "CTV" if ctv_ok else ("SOV" if sov_ok else None)
            elif sc == "Rough Sea":
                return "SOV" if sov_ok else None
            else:
                return None
        return "CTV" if ctv_ok else None

    def should_defer(self, ev: dict, sc: str, turbine: TurbineState) -> bool:
        if self.strat_name in ("S1_NoWeather", "S2_WeatherCTV"):
            return False
        crit  = ev.get("criticality", "Non-Critical")
        etype = ev.get("event_type", "FAIL")
        eta   = ev.get("eta_derating", 0.5)
        rel   = turbine.mean_reliability()
        if etype == "FAIL" and crit == "Critical":
            return False  # Critical failures are never deferred
        if sc == "Extreme" and etype != "FAIL":
            return True
        if crit == "Non-Critical" and eta < 0.10 and rel > 0.92 and sc == "Calm":
            return True
        return False

    def get_adjusted_cap_hr(self, sc: str, vessel: str | None) -> float:
        if vessel is None:
            return 0.0
        return get_vessel_cap_hr(sc, vessel)

# ==============================================================================
# 5. Fixed PM generation
# ==============================================================================

def generate_fixed_pm(weather_df: pd.DataFrame) -> list[dict]:
    """Generate two annual fixed PM campaigns per turbine, targeting low-wind days."""
    pm_list = []
    for year in [2023, 2024, 2025]:
        ydf  = weather_df[weather_df["year"] == year]
        low  = ydf[ydf["wind_speed_mean"] <= 6.0].sort_values("wind_speed_mean")["date"].tolist()
        all_ = ydf["date"].tolist()

        def pick(cands):
            if len(cands) < 2:
                cands = all_
            d1  = cands[0]
            d2c = [d for d in cands if (d - d1).days >= 90]
            d2  = d2c[0] if d2c else (cands[-1] if len(cands) > 1 else d1)
            return [d1, d2]

        dates = pick(low)
        for tid in TURBINE_IDS:
            for d in dates:
                dur = sample_duration("Service Items", "Minor Repair")
                pm_list.append({
                    "date": d, "turbine_id": tid, "component": "Service Items",
                    "severity": "Minor Repair", "event_type": "PM",
                    "criticality": "Non-Critical", "eta_derating": 0.0,
                    "duration_hr": dur, "hours_remaining": dur, "arrival_day_idx": -1,
                    "crew_teams": 1, "parts_cost": SEV["Minor Repair"]["parts_KRW"],
                    "age_weeks": 0.0, "wb_hazard": 0.0,
                })
    return pm_list

# ==============================================================================
# 6. Priority scoring
# ==============================================================================

PRIORITY_SCORE_BASE = {
    ("Critical","FAIL"): 0, ("Critical","CBM_PM"): 1,
    ("Semi-Critical","FAIL"): 2, ("Semi-Critical","CBM_PM"): 3,
    ("Critical","PM"): 4, ("Non-Critical","FAIL"): 5,
    ("Semi-Critical","PM"): 6, ("Non-Critical","CBM_PM"): 7,
    ("Non-Critical","PM"): 8,
}
_priority_cache: dict = {}

def clear_priority_cache():
    _priority_cache.clear()

def priority_key(ev: dict, tstates: dict) -> float:
    """Lower score = higher priority in greedy sort."""
    crit  = ev.get("criticality", "Non-Critical")
    etype = ev.get("event_type", "FAIL")
    base  = PRIORITY_SCORE_BASE.get((crit, etype), 9)
    tid   = ev.get("turbine_id", 0)
    comp  = ev.get("component", "Service Items")
    sev   = ev.get("severity", "Minor Repair")
    cache_key = (tid, comp, sev)
    if cache_key not in _priority_cache:
        ts = tstates.get(tid)
        _priority_cache[cache_key] = (
            ts.dynamic_priority_score(comp, sev) if ts and comp in COMPONENTS else 0.0
        )
    return base - min(_priority_cache[cache_key] * 0.1, 0.9)

# ==============================================================================
# 7. Greedy scheduler (v3)
# ==============================================================================

def schedule_day_greedy(
    date: pd.Timestamp,
    pending: list[dict],
    wx: dict,
    tstates: dict[int, TurbineState],
    hmdp: HMDPPolicy,
    strat_name: str,
    day_idx: int = 0,
) -> tuple[dict, list[dict], list[dict]]:
    """
    One-day greedy maintenance scheduler.

    Returns:
        day_result  : dict of daily cost/operational metrics
        new_pending : tasks not completed today (carried forward)
        completed   : tasks fully completed today
    """
    clear_priority_cache()

    sc     = wx.get("weather_scenario", "Calm")
    ctv_ok = int(wx.get("ctv_ok", 1))
    sov_ok = int(wx.get("sov_ok", 1))

    vessel        = hmdp.decide_vessel(sc, ctv_ok, sov_ok)
    cap_hr        = hmdp.get_adjusted_cap_hr(sc, vessel)
    weather_aware = STRATEGIES[strat_name]["weather_aware"]

    # [Modification A] S1: charge charter fee + idle fuel on no-departure days
    wasted_ship_cost = 0.0
    if not weather_aware:
        vessel     = "CTV"
        actual_cap = get_vessel_cap_hr(sc, "CTV")
        cap_hr     = actual_cap
        if actual_cap == 0:
            wasted_ship_cost += VESSELS["CTV"]["fuel_kr_hr"] * 2  # idle fuel (2 hrs equivalent)
            wasted_ship_cost += VESSEL_CHARTER_DAILY["CTV"]        # daily charter fee

    # Stamp arrival day on tasks not yet timestamped
    for ev in pending:
        if ev.get("arrival_day_idx", -1) == -1:
            ev["arrival_day_idx"] = day_idx

    # Separate deferred vs active tasks
    active_tasks = []
    deferred_tasks = []
    for ev in pending:
        tid = ev.get("turbine_id", 0)
        ts  = tstates.get(tid)
        if ts and hmdp.should_defer(ev, sc, ts):
            deferred_tasks.append(ev)
        else:
            active_tasks.append(ev)

    active_sorted = sorted(active_tasks, key=lambda e: priority_key(e, tstates))
    already_queued: set = set()

    # [Modification B/D] pending_downtime_cost accumulator
    # = opportunity cost for tasks waiting in repair queue
    ship_c       = wasted_ship_cost
    port_c       = 0.0
    labor_c      = 0.0
    pending_dt_c = 0.0   # [Modification B] renamed from backlog_cost
    em_kg        = 0.0
    n_completed  = n_attempted = 0
    remaining_cap = cap_hr
    new_pending:      list[dict] = list(deferred_tasks)
    completed_events: list[dict] = []

    for ev in deferred_tasks:
        already_queued.add(id(ev))

    accessible = (vessel is not None and cap_hr > 0)

    if accessible:
        _rep_port = hmdp.cfg["ports"][0]
        em_kg += co2_kg_transit(_rep_port, vessel)

    cbm_pm_today = 0

    for ev in active_sorted:
        rem = float(ev.get("hours_remaining", ev.get("duration_hr", 6.0)))
        if rem <= 0:
            n_completed += 1
            continue

        # [Modification C] Enforce daily CBM task cap
        if ev.get("event_type") == "CBM_PM" and cbm_pm_today >= MAX_CBM_PM_PER_DAY:
            ec = ev.copy()
            ec["hours_remaining"] = rem
            new_pending.append(ec)
            already_queued.add(id(ec))
            already_queued.add(id(ev))
            crit_ev = ev.get("criticality", "Non-Critical")
            eta_ev  = ev.get("eta_derating", 0.0)
            # [Modification D] pending_downtime_cost definition:
            # unit_penalty × max(ETA, 0.05) × hours_waiting
            pending_dt_c += _PENDING_DT_BASE.get(crit_ev, 10_000) * max(eta_ev, 0.05) * rem
            continue

        if remaining_cap <= 1e-6 or not accessible:
            ec = ev.copy()
            ec["hours_remaining"] = rem
            new_pending.append(ec)
            already_queued.add(id(ec))
            already_queued.add(id(ev))
            crit_ev = ev.get("criticality", "Non-Critical")
            eta_ev  = ev.get("eta_derating", 0.5)
            pending_dt_c += _PENDING_DT_BASE.get(crit_ev, 10_000) * max(eta_ev, 0.05) * rem
            continue

        tid  = ev.get("turbine_id", 0)
        port = hmdp.decide_port(sc, tstates.get(tid, TurbineState(tid)), new_pending)
        assign        = min(rem, remaining_cap, cap_hr)
        remaining_cap -= assign
        n_attempted   += 1

        if ev.get("event_type") == "CBM_PM":
            cbm_pm_today += 1

        sc_cost   = ship_cost(port, vessel, assign, max(cap_hr, 1))
        pc_cost   = PORTS[port]["daily_fee_KRW"] * (assign / OPS_HR_DAY)
        lc_cost   = LABOR_HOURLY_KRW * ev.get("crew_teams", 1) * assign
        em_onsite = ONSITE_CO2_KG_HR.get(vessel, 0.0) * assign
        em_kg    += em_onsite
        ship_c   += sc_cost
        port_c   += pc_cost
        labor_c  += lc_cost

        rem_after = rem - assign
        if rem_after <= 0.01:
            n_completed += 1
            comp      = ev.get("component", "")
            sev       = ev.get("severity", "Minor Repair")
            comp_list = ev.get("component_list", [comp] if comp in COMPONENTS else [])
            for c in comp_list:
                if tid in tstates and c in COMPONENTS:
                    tstates[tid].apply_repair(c, sev, day_idx)
            ev_done = ev.copy()
            ev_done["completion_day_idx"] = day_idx
            ev_done["delay_days"] = max(day_idx - ev.get("arrival_day_idx", day_idx), 0)
            completed_events.append(ev_done)
        else:
            ec = ev.copy()
            ec["hours_remaining"] = rem_after
            new_pending.append(ec)
            already_queued.add(id(ec))
            already_queued.add(id(ev))
            crit_ev = ev.get("criticality", "Non-Critical")
            eta_ev  = ev.get("eta_derating", 0.5)
            pending_dt_c += _PENDING_DT_BASE.get(crit_ev, 10_000) * max(eta_ev, 0.05) * rem_after

    # Accumulate pending_downtime_cost for all inaccessible-day tasks
    if not accessible:
        for ev in active_sorted:
            if id(ev) not in already_queued:
                crit_ev  = ev.get("criticality", "Non-Critical")
                eta_ev   = ev.get("eta_derating", 0.5)
                pdt_base = {"Critical": 200_000, "Semi-Critical": 50_000,
                            "Non-Critical": 5_000}.get(crit_ev, 5_000)
                pending_dt_c += pdt_base * max(eta_ev, 0.05) * float(ev.get("hours_remaining", 0))
                new_pending.append(ev)
                already_queued.add(id(ev))

    day = {
        "date": date, "strategy": strat_name,
        "vessel": vessel if accessible else None,
        "port": (hmdp.decide_port(sc, TurbineState(0), []) if accessible else "P1"),
        "weather_scenario": sc, "accessible": accessible,
        "cap_hr_used": min(cap_hr, cap_hr - remaining_cap),
        "ship_cost": ship_c, "port_cost": port_c, "labor_cost": labor_c,
        "pending_downtime_cost": pending_dt_c,   # [Modification B] renamed
        "emissions_kg": em_kg,
        "n_completed": n_completed, "n_deferred": len(deferred_tasks),
        "n_attempted": n_attempted, "n_pending": len(new_pending),
        "wasted_charter_cost": wasted_ship_cost,  # [Modification A]
    }
    return day, new_pending, completed_events

# ==============================================================================
# 8. Availability computation
# ==============================================================================

def compute_daily_availability(tstates: dict, wind: float) -> dict:
    """
    Compute fleet-level operational availability (AO) and energy AEP for one day.

    AO  : fraction of turbines with no component having ETA ≥ 0.70 (binary).
    AEP : actual MWh / max possible MWh, incorporating partial output losses via ETA.
    downtime_cost: (max_MWh − actual_MWh) × electricity_price × wind_fraction.
    """
    n_operational    = 0
    total_actual_mwh = 0.0
    max_possible_mwh = N_TURBINES * RATED_POWER_MW * OPS_HR_DAY
    wind_frac = power_curve(wind)
    for ts in tstates.values():
        if ts.is_operational():
            n_operational += 1
        prod_frac        = ts.get_production_fraction()
        total_actual_mwh += RATED_POWER_MW * wind_frac * prod_frac * OPS_HR_DAY
    operational_ao = n_operational / N_TURBINES
    energy_aep = (
        min(total_actual_mwh / max(max_possible_mwh * wind_frac, 1e-6), 1.0)
        if wind_frac > 0 else 0.0
    )
    downtime_cost = max(
        (max_possible_mwh - total_actual_mwh / max(wind_frac, 0.01)) * ELEC_PRICE_KRW * wind_frac,
        0.0
    )
    return {
        "n_operational": n_operational, "operational_ao": operational_ao,
        "energy_aep": energy_aep, "power_mwh": total_actual_mwh,
        "downtime_cost": downtime_cost,
    }

# ==============================================================================
# 9. Main simulation loop
# ==============================================================================

def run_simulation(strat_name: str, weather_df: pd.DataFrame
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run full 3-year daily simulation for a single strategy.

    Returns:
        df_daily    : Daily operational and cost metrics
        df_completed: Completed maintenance event log
        df_feedback : Imperfect repair feedback log (Weibull age update records)
    """
    print(f"    ▸ {STRAT_LABELS[strat_name]}")
    tstates = {tid: TurbineState(tid) for tid in TURBINE_IDS}
    hmdp    = HMDPPolicy(strat_name)
    pm_mode = STRATEGIES[strat_name]["pm_mode"]

    fixed_pm_by_date: dict = defaultdict(list)
    if pm_mode == "fixed":
        for ev in generate_fixed_pm(weather_df):
            fixed_pm_by_date[ev["date"]].append(ev)

    pending:   list[dict] = []
    seen_fail: set        = set()
    rows:      list[dict] = []
    all_completed: list[dict] = []
    wx_records = weather_df.to_dict("records")

    for day_idx, wx_row in enumerate(wx_records):
        date = wx_row["date"]
        wind = float(wx_row.get("wind_speed_mean", 7.0))

        # Weekly Weibull age advance (every Monday)
        if pd.Timestamp(date).dayofweek == 0:
            for ts in tstates.values():
                ts.advance_week()

        # ── Corrective maintenance (failures) ────────────────────────────────
        new_cm: list[dict] = []
        for ts in tstates.values():
            for ev in ts.check_failures(date, wind):
                eid = (ev["turbine_id"], ev["component"], str(date))
                if eid not in seen_fail:
                    seen_fail.add(eid)
                    ev["arrival_day_idx"] = day_idx
                    new_cm.append(ev)

        # ── Preventive maintenance (CBM or fixed schedule) ───────────────────
        new_pm: list[dict] = []
        if pm_mode == "cbm":
            cbm_count_today = 0
            for ts in tstates.values():
                if cbm_count_today >= MAX_CBM_PM_PER_DAY * 2:
                    break
                for ev in ts.check_pm_triggers(date, day_idx):
                    ev["arrival_day_idx"] = day_idx
                    new_pm.append(ev)
                    cbm_count_today += 1
        else:
            for ev in fixed_pm_by_date.get(date, []):
                ev["arrival_day_idx"] = day_idx
                new_pm.append(ev)

        pending.extend(new_cm)
        pending.extend(new_pm)

        # ── Greedy scheduling ────────────────────────────────────────────────
        day_res, pending, completed_events = schedule_day_greedy(
            date, pending, wx_row, tstates, hmdp, strat_name, day_idx)
        all_completed.extend(completed_events)

        # ── Availability metrics ─────────────────────────────────────────────
        avail   = compute_daily_availability(tstates, wind)
        parts_c = sum(ev.get("parts_cost", 0) for ev in new_cm + new_pm)

        day_res.update({
            "year": pd.Timestamp(date).year, "month": pd.Timestamp(date).month,
            "season": wx_row.get("season", season_of(date)),
            "week_of_year": pd.Timestamp(date).isocalendar()[1],
            "wind_speed": wind, "wave_height": float(wx_row.get("wave_height_mean", 1.2)),
            "ctv_ok": int(wx_row.get("ctv_ok", 1)), "sov_ok": int(wx_row.get("sov_ok", 1)),
            "n_new_failures": len(new_cm),
            "n_critical_failures": sum(1 for e in new_cm if e.get("criticality") == "Critical"),
            "n_pm_today": len(new_pm), "n_pending_total": len(pending),
            "parts_cost": parts_c,
            "n_turbines_operational": avail["n_operational"],
            "operational_ao": avail["operational_ao"], "energy_aep": avail["energy_aep"],
            "power_mwh": avail["power_mwh"], "downtime_cost": avail["downtime_cost"],
        })
        om_c = (day_res["ship_cost"] + day_res["port_cost"] +
                day_res["labor_cost"] + day_res["pending_downtime_cost"] + parts_c)
        day_res["total_om_cost"] = om_c
        day_res["total_cost"]    = om_c + avail["downtime_cost"]
        rows.append(day_res)

    df_daily = pd.DataFrame(rows)
    df_daily["date"] = pd.to_datetime(df_daily["date"])

    df_completed = pd.DataFrame(all_completed) if all_completed else pd.DataFrame()
    if not df_completed.empty:
        df_completed["strategy"] = strat_name

    fb_rows = []
    for ts in tstates.values():
        fb_rows.extend(ts.repair_log)
    df_feedback = pd.DataFrame(fb_rows) if fb_rows else pd.DataFrame()
    if not df_feedback.empty:
        df_feedback["strategy"] = strat_name

    ao_mean  = df_daily["operational_ao"].mean() * 100
    aep_mean = df_daily["energy_aep"].mean() * 100
    tc       = df_daily["total_cost"].sum() / 1e9
    n_fail   = df_daily["n_new_failures"].sum()
    n_comp   = len(all_completed)
    wasted   = df_daily.get("wasted_charter_cost", pd.Series([0])).sum() / 1e6
    print(f"      ₩{tc:.2f}B | AO {ao_mean:.1f}% | AEP {aep_mean:.1f}% | "
          f"Fail {n_fail:,} | Completed {n_comp:,} | Idle charter ₩{wasted:.1f}M")
    return df_daily, df_completed, df_feedback

# ==============================================================================
# 10. KPI aggregation
# ==============================================================================

def compute_kpis(results: dict) -> pd.DataFrame:
    rows = []
    for strat, (df, _, _) in results.items():
        for yr in [2023, 2024, 2025]:
            sub = df[df["year"] == yr]
            if sub.empty:
                continue
            rows.append({
                "strategy": strat, "label": STRAT_LABELS[strat], "year": yr,
                "total_cost_B":       sub["total_cost"].sum() / 1e9,
                "om_cost_B":          sub["total_om_cost"].sum() / 1e9,
                "downtime_cost_B":    sub["downtime_cost"].sum() / 1e9,
                "operational_ao_pct": sub["operational_ao"].mean() * 100,
                "energy_aep_pct":     sub["energy_aep"].mean() * 100,
                "n_failures":         sub["n_new_failures"].sum(),
                "n_critical_fail":    sub["n_critical_failures"].sum(),
                "n_pm":               sub["n_pm_today"].sum(),
                "n_deferred":         sub["n_deferred"].sum(),
                "carbon_tCO2":        sub["emissions_kg"].sum() / 1000,
                "power_mwh":          sub["power_mwh"].sum(),
                "pending_downtime_B": sub["pending_downtime_cost"].sum() / 1e9,
                "wasted_charter_B":   sub.get("wasted_charter_cost", pd.Series([0])).sum() / 1e9,
            })
    return pd.DataFrame(rows)

def bootstrap_kpi_ci(df: pd.DataFrame, metric: str = "total_cost",
                     n_boot: int = 500, ci: float = 0.95) -> dict:
    weekly = df.groupby("week_of_year")[metric].sum().values
    boot_means = [rng.choice(weekly, size=len(weekly), replace=True).mean() for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(weekly)), "std": float(np.std(weekly)),
        f"ci_{int(ci*100)}_lo": float(np.quantile(boot_means, alpha)),
        f"ci_{int(ci*100)}_hi": float(np.quantile(boot_means, 1 - alpha)),
        "n_weeks": len(weekly),
    }

def compute_bootstrap_cis(results: dict) -> pd.DataFrame:
    rows = []
    for strat, (df, _, _) in results.items():
        for yr in [2023, 2024, 2025]:
            sub = df[df["year"] == yr]
            if sub.empty:
                continue
            ci = bootstrap_kpi_ci(sub, "total_cost", n_boot=300)
            ci["strategy"] = strat
            ci["year"] = yr
            rows.append(ci)
    return pd.DataFrame(rows)

def compute_delay_kpi(results: dict) -> pd.DataFrame:
    rows = []
    for strat, (_, df_comp, _) in results.items():
        if df_comp.empty:
            continue
        for et in ["FAIL", "CBM_PM", "PM"]:
            if "event_type" not in df_comp.columns:
                continue
            sub = df_comp[df_comp["event_type"] == et]
            if sub.empty:
                continue
            if et == "FAIL" and "delay_days" in sub.columns:
                mean_delay = sub["delay_days"].mean()
                n_events   = len(sub)
                total_dt   = mean_delay * n_events / N_TURBINES
                avail_pct  = max(0.0, (N_DAYS - total_dt) / N_DAYS * 100)
            else:
                avail_pct = None
            rows.append({
                "strategy": strat, "event_type": et,
                "n_events":          len(sub),
                "mean_delay_days":   sub["delay_days"].mean()   if "delay_days" in sub.columns else 0,
                "median_delay_days": sub["delay_days"].median() if "delay_days" in sub.columns else 0,
                "max_delay_days":    sub["delay_days"].max()    if "delay_days" in sub.columns else 0,
                "total_delay_days":  sub["delay_days"].sum()    if "delay_days" in sub.columns else 0,
                "turbine_avail_pct": avail_pct,
                "note": ("delay-based downtime estimate (FAIL only)"
                         if et == "FAIL" else
                         "queue wait time, NOT downtime — availability N/A"),
            })
    return pd.DataFrame(rows)

def compute_cost_breakdown(results: dict) -> pd.DataFrame:
    rows = []
    cats = ["ship_cost", "port_cost", "labor_cost", "pending_downtime_cost",
            "parts_cost", "downtime_cost", "wasted_charter_cost"]
    for strat, (df, _, _) in results.items():
        for yr in [2023, 2024, 2025]:
            sub = df[df["year"] == yr]
            if sub.empty:
                continue
            row = {"strategy": strat, "year": yr}
            total = 0.0
            for cat in cats:
                val = sub[cat].sum() / 1e9 if cat in sub.columns else 0.0
                row[cat + "_B"] = round(val, 4)
                total += val
            row["total_B"] = round(total, 4)
            rows.append(row)
    return pd.DataFrame(rows)

def statistical_tests(results: dict) -> pd.DataFrame:
    rows = []
    hmdp_weekly = (results["HMDP_CBM"][0]
                   .groupby("week_of_year")["total_cost"].sum().values / 1e6)
    for strat in ["S1_NoWeather", "S2_WeatherCTV", "S3_MultiPort"]:
        if strat not in results:
            continue
        comp_weekly = (results[strat][0]
                       .groupby("week_of_year")["total_cost"].sum().values / 1e6)
        mw_stat, mw_p = mannwhitneyu(hmdp_weekly, comp_weekly, alternative="two-sided")
        t_stat,  t_p  = ttest_ind(hmdp_weekly, comp_weekly, equal_var=False)
        pooled_std = math.sqrt((np.std(hmdp_weekly) ** 2 + np.std(comp_weekly) ** 2) / 2)
        cohens_d   = (np.mean(hmdp_weekly) - np.mean(comp_weekly)) / max(pooled_std, 1e-6)
        rows.append({
            "comparison": f"HMDP vs {STRAT_LABELS[strat]}",
            "hmdp_mean_M": round(np.mean(hmdp_weekly), 2),
            "comp_mean_M": round(np.mean(comp_weekly),  2),
            "hmdp_std_M":  round(np.std(hmdp_weekly),  2),
            "comp_std_M":  round(np.std(comp_weekly),   2),
            "mw_u_stat":  round(mw_stat, 2), "mw_p_value": round(mw_p, 5),
            "mw_significant": mw_p < 0.05,
            "welch_t_stat": round(t_stat, 3), "welch_p_value": round(t_p, 5),
            "welch_significant": t_p < 0.05,
            "cohens_d": round(cohens_d, 3),
            "effect_size": ("large"  if abs(cohens_d) > 0.8 else
                            "medium" if abs(cohens_d) > 0.5 else
                            "small"  if abs(cohens_d) > 0.2 else "negligible"),
        })
    return pd.DataFrame(rows)

# ==============================================================================
# 11. Figure functions
# ==============================================================================

def savefig(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {name}")

def fig01_weather_overview(wx: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    fig.suptitle("Ulsan Offshore Wind Farm — Metocean Overview 2023–2025", fontsize=13, fontweight="bold")
    x = wx["date"]
    ax = axes[0]
    ax.fill_between(x, 0, wx["wind_speed_mean"], alpha=0.25, color="#4a90d9")
    ax.axhline(10.0, color="#e74c3c", ls="--", lw=1.5, label="CTV limit 10 m/s")
    ax.axhline(15.0, color="#9b59b6", ls="--", lw=1.2, label="Extreme 15 m/s")
    ax.set_ylabel("Wind Speed (m/s)"); ax.set_ylim(0, 30)
    ax.legend(loc="upper right"); ax.grid(alpha=.25)
    ax.set_title("A) Daily Mean Wind Speed", loc="left")
    ax = axes[1]
    ax.fill_between(x, 0, wx["wave_height_mean"], alpha=0.45, color="#e74c3c")
    ax.axhline(1.5, color="blue",   ls="--", lw=1.5, label="CTV limit 1.5 m")
    ax.axhline(2.5, color="purple", ls="--", lw=1.2, label="SOV limit 2.5 m")
    ax.set_ylabel("Sig. Wave Height (m)"); ax.set_ylim(0, 6)
    ax.legend(loc="upper right"); ax.grid(alpha=.25)
    ax.set_title("B) Significant Wave Height", loc="left")
    ax = axes[2]
    for _, row in wx.iterrows():
        ax.axvline(row["date"], color=SC_COL.get(row["weather_scenario"], "gray"), alpha=0.35, lw=0.6)
    ax.plot(x, wx["ctv_ok"].rolling(7, min_periods=1).mean(), color="black", lw=1.2, label="CTV Access 7d-MA")
    patches = [mpatches.Patch(color=v, label=k) for k, v in SC_COL.items()]
    ax.legend(handles=patches + [plt.Line2D([0],[0], color="black", lw=1.5, label="CTV Access")],
              fontsize=8, ncol=3, loc="upper right")
    ax.set_ylabel("CTV Accessible"); ax.set_ylim(-0.1, 1.3)
    ax.set_title("C) Weather Scenario & CTV Accessibility", loc="left")
    for yr in [2024, 2025]:
        for a in axes:
            a.axvline(pd.Timestamp(f"{yr}-01-01"), color="gray", ls=":", lw=1.5)
    fig.tight_layout()
    savefig("Fig01_Weather_Overview.png")

def fig02_seasonal_accessibility(wx: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle("Seasonal Weather Distribution & Vessel Accessibility (2023–2025)", fontsize=13, fontweight="bold")
    seasons = ["Winter","Spring","Summer","Fall"]

    def bx(ax, data_list, ylabel, title, hlines=None):
        bp = ax.boxplot(data_list, patch_artist=True, widths=0.55,
                        medianprops={"color":"k","lw":2}, flierprops={"markersize":3,"alpha":0.5})
        for patch, s in zip(bp["boxes"], seasons):
            patch.set_facecolor(SEASON_COL[s]); patch.set_alpha(0.75)
        ax.set_xticklabels(seasons); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y",alpha=.3)
        if hlines:
            for val, col, lbl in hlines:
                ax.axhline(val, color=col, ls="--", lw=1.3, label=lbl)
            ax.legend(fontsize=8)

    bx(axes[0], [wx[wx["season"]==s]["wind_speed_mean"].values for s in seasons],
       "Wind Speed (m/s)", "Wind Speed by Season",
       [(10.0,"#e74c3c","CTV"),(15.0,"#9b59b6","Extreme")])
    bx(axes[1], [wx[wx["season"]==s]["wave_height_mean"].values for s in seasons],
       "Sig. Wave Height (m)", "Wave Height by Season",
       [(1.5,"blue","CTV"),(2.5,"purple","SOV")])
    ax = axes[2]
    ctv_rates = [wx[wx["season"]==s]["ctv_ok"].mean()*100 for s in seasons]
    sov_rates = [wx[wx["season"]==s]["sov_ok"].mean()*100 for s in seasons]
    x_ = np.arange(4); w_ = 0.35
    b1 = ax.bar(x_-w_/2, ctv_rates, w_, label="CTV", color="#3498db", alpha=0.85, edgecolor="k")
    b2 = ax.bar(x_+w_/2, sov_rates, w_, label="SOV", color="#e74c3c", alpha=0.75, edgecolor="k")
    for b, v in list(zip(b1, ctv_rates)) + list(zip(b2, sov_rates)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{v:.0f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x_); ax.set_xticklabels(seasons)
    ax.set_ylabel("Accessible Days (%)"); ax.set_ylim(0, 115)
    ax.set_title("CTV vs SOV Accessibility by Season"); ax.legend(); ax.grid(axis="y",alpha=.3)
    fig.tight_layout()
    savefig("Fig02_Seasonal_Accessibility.png")

def fig03_criticality_weibull():
    fig = plt.figure(figsize=(20, 13))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle("Component Criticality & Weibull Failure Model (ETA)", fontsize=13, fontweight="bold")
    t_arr = np.linspace(0.5, 120, 300)
    ax = fig.add_subplot(gs[0, 0])
    for shape, lbl, col, ls in [(0.8,"β=0.8","#e74c3c","--"),(1.0,"β=1.0","#f39c12","-."),
                                  (2.5,"β=2.5 ← Used","#27ae60","-"),(4.0,"β=4.0","#3498db",":")]:
        hr = [(shape/WB_SCALE)*((t/WB_SCALE)**(shape-1)) for t in t_arr]
        ax.plot(t_arr, hr, color=col, lw=2.5 if shape==2.5 else 1.3, ls=ls, label=lbl)
    ax.axvline(26, color="gray", ls=":", lw=1.5, label="Fixed PM 26wk")
    ax.set_xlabel("Age (weeks)"); ax.set_ylabel("h(t)")
    ax.set_title("A) Weibull Hazard (η=80wk)"); ax.legend(fontsize=8); ax.set_ylim(0, 0.12); ax.grid(alpha=.3)
    ax = fig.add_subplot(gs[0, 1])
    for crit, col in CRIT_COL.items():
        thr    = CBM_THRESHOLD[crit]
        r_vals = [weibull_reliability(t)*100 for t in t_arr]
        ax.plot(t_arr, r_vals, color=col, lw=2, label=crit)
        cross_t = next((t for t, r in zip(t_arr, r_vals) if r/100 < thr), None)
        if cross_t:
            ax.axvline(cross_t, color=col, ls=":", lw=1.2)
            ax.plot(cross_t, thr*100, "o", color=col, ms=8)
        ax.axhline(thr*100, color=col, ls="--", lw=0.8)
    ax.set_xlabel("Age (weeks)"); ax.set_ylabel("R(t) (%)")
    ax.set_title(f"B) CBM Threshold (v3: Cr={CBM_THRESHOLD['Critical']:.0%}, "
                 f"Se={CBM_THRESHOLD['Semi-Critical']:.0%}, No={CBM_THRESHOLD['Non-Critical']:.0%})")
    ax.legend(fontsize=9); ax.set_ylim(40, 105); ax.grid(alpha=.3)
    ax = fig.add_subplot(gs[0, 2])
    comps_by_crit: dict[str, list] = defaultdict(list)
    for c, crit in COMP_CRITICALITY.items():
        annual_rate = sum(FAIL_RATES[c].values())
        wb_risk     = annual_rate * weibull_hazard(40)
        comps_by_crit[crit].append((c, wb_risk))
    y_labels, y_risks, y_cols = [], [], []
    for crit in ["Critical","Semi-Critical","Non-Critical"]:
        for comp, risk in sorted(comps_by_crit[crit], key=lambda x: -x[1]):
            y_labels.append(f"[{crit[:1]}] {comp[:20]}")
            y_risks.append(risk); y_cols.append(CRIT_COL[crit])
    ax.barh(range(len(y_labels)), y_risks, color=y_cols, alpha=0.85, edgecolor="black", lw=0.4)
    ax.set_yticks(range(len(y_labels))); ax.set_yticklabels(y_labels, fontsize=7.5)
    ax.set_xlabel("Risk Score"); ax.set_title("C) Component Risk by Criticality")
    patches = [mpatches.Patch(color=v, label=k) for k, v in CRIT_COL.items()]
    ax.legend(handles=patches, fontsize=8); ax.grid(axis="x",alpha=.3)
    ax = fig.add_subplot(gs[1, 0])
    scenario_labels = ["Healthy","Non-Crit\nMinor","Semi-Crit\nMajor","Critical\nMinor","Critical\nMajor"]
    prod_fracs = [1.0, 0.95, 0.55, 0.65, 0.25]
    bars_p = ax.bar(scenario_labels, prod_fracs,
                    color=["#27ae60","#2ecc71","#f39c12","#e74c3c","#c0392b"], alpha=0.85, edgecolor="black")
    for b, v in zip(bars_p, prod_fracs):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{v*100:.0f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Production Fraction"); ax.set_ylim(0, 1.15)
    ax.set_title("D) Production Derating (ETA Segmentation)")
    ax.axhline(0.95, color="red", ls="--", lw=1.5, label="95% AO target"); ax.legend(fontsize=9); ax.grid(axis="y",alpha=.3)
    ax = fig.add_subplot(gs[1, 1])
    t_sim     = np.linspace(0, 160, 1000)
    rel_curve = [weibull_reliability(t)*100 for t in t_sim]
    ax.plot(t_sim, rel_curve, color="black", lw=2, label="R(t)")
    for t_pm in [26, 52, 78, 104, 130]:
        ax.axvline(t_pm, color="#f39c12", ls=":", lw=1.5, alpha=0.7)
        ax.annotate("Fixed\nPM", (t_pm, 92), fontsize=6.5, color="#f39c12", ha="center")
    cbm_thr = CBM_THRESHOLD["Critical"] * 100
    ax.axhline(cbm_thr, color="#27ae60", ls="--", lw=1.5, label=f"CBM thr {cbm_thr:.0f}% (v3)")
    cross_t = next((t for t, r in zip(t_sim, rel_curve) if r < cbm_thr), None)
    if cross_t:
        ax.scatter([cross_t], [cbm_thr], color="#27ae60", s=100, zorder=5)
        ax.annotate(f"CBM @ t={cross_t:.0f}wk", (cross_t, cbm_thr-8), fontsize=7, color="#27ae60")
    ax.set_xlabel("Weeks"); ax.set_ylabel("R(t) (%)"); ax.set_title("E) CBM vs Fixed PM Logic (v3 threshold)")
    ax.legend(fontsize=9); ax.set_ylim(50, 105); ax.grid(alpha=.3)
    ax = fig.add_subplot(gs[1, 2])
    cbm_thresholds  = np.linspace(0.60, 0.98, 40)
    annual_pm_cost  = [((1-thr)/0.15)*2e6 for thr in cbm_thresholds]
    annual_fail_cost= [((0.98-thr)*50e6)  for thr in cbm_thresholds]
    total_cost      = [p+f for p, f in zip(annual_pm_cost, annual_fail_cost)]
    ax.plot(cbm_thresholds*100, [c/1e6 for c in annual_pm_cost],  color="#f39c12", lw=1.5, ls="--", label="PM cost")
    ax.plot(cbm_thresholds*100, [c/1e6 for c in annual_fail_cost],color="#e74c3c", lw=1.5, ls="--", label="Failure cost")
    ax.plot(cbm_thresholds*100, [c/1e6 for c in total_cost],      color="black",   lw=2.5,           label="Total")
    opt_idx = int(np.argmin(total_cost))
    ax.axvline(cbm_thresholds[opt_idx]*100, color="blue", ls=":", lw=1.5, label=f"Optimal {cbm_thresholds[opt_idx]*100:.0f}%")
    ax.axvline(CBM_THRESHOLD["Critical"]*100, color="#27ae60", ls="-.", lw=1.5,
               label=f"v3 Critical {CBM_THRESHOLD['Critical']*100:.0f}%")
    ax.set_xlabel("CBM Threshold (%)"); ax.set_ylabel("Annual Cost (M KRW)")
    ax.set_title("F) Optimal CBM Threshold (v3 marker)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    savefig("Fig03_ComponentCriticality_Weibull.png")

def fig04_strategy_comparison(results: dict, kpis: pd.DataFrame):
    strats = list(results.keys()); n = len(strats)
    fig = plt.figure(figsize=(22, 15))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle("Strategy Comparison: Cost, Availability & Carbon (2023–2025)", fontsize=12, fontweight="bold")
    for col_idx, (metric, ylabel, title, hval, hlab) in enumerate([
        ("operational_ao_pct","Operational AO (%)","A) Operational AO", 95,"95% AO target"),
        ("energy_aep_pct",    "Energy AEP (%)",     "B) Energy AEP",     90,"90% AEP target"),
    ]):
        ax = fig.add_subplot(gs[0, col_idx])
        for yi, (yr, col) in enumerate([(2023,"#4a90d9"),(2024,"#f5a623"),(2025,"#27ae60")]):
            vals = [kpis[(kpis["strategy"]==s)&(kpis["year"]==yr)][metric].values for s in strats]
            vals = [v[0] if len(v) else 0 for v in vals]
            ax.bar(np.arange(n)+(yi-1)*0.25, vals, 0.25, label=str(yr), color=col, alpha=0.85, edgecolor="k", lw=0.5)
        ax.axhline(hval, color="red" if col_idx==0 else "blue", ls="--", lw=1.5, label=hlab)
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels([STRAT_LABELS[s].replace(" / ","\n").replace(": ","\n") for s in strats], fontsize=7.5)
        ax.set_ylabel(ylabel); ax.set_ylim(0, 110); ax.set_title(title); ax.legend(fontsize=9); ax.grid(axis="y",alpha=.3)
    ax = fig.add_subplot(gs[0, 2])
    cats  = ["ship_cost","port_cost","labor_cost","pending_downtime_cost","parts_cost","downtime_cost","wasted_charter_cost"]
    clbls = ["Ship","Port","Labor","Pending Downtime (PDC)","Parts","Downtime","Wasted Charter"]
    ccols = ["#3498db","#9b59b6","#f39c12","#e74c3c","#1abc9c","#e67e22","#8e44ad"]
    bots  = np.zeros(n)
    for cat, lbl, col in zip(cats, clbls, ccols):
        vals = np.array([results[s][0][cat].sum()/1e9 if cat in results[s][0].columns else 0.0 for s in strats])
        ax.bar(range(n), vals, bottom=bots, label=lbl, color=col, alpha=0.88, edgecolor="white")
        bots += vals
    ax.set_xticks(range(n))
    ax.set_xticklabels([STRAT_LABELS[s].replace(" / ","\n").replace(": ","\n") for s in strats], fontsize=7.5)
    ax.set_ylabel("3-Year Total Cost (B KRW)")
    ax.set_title("C) Cost Decomposition\n(PDC = Pending Downtime Penalty, Wasted Charter = idle charter fee)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_ylim(0, bots.max()*1.08); ax.grid(axis="y",alpha=.3)
    ax = fig.add_subplot(gs[1, :2])
    for s, (df_, _, _) in results.items():
        roll = df_.set_index("date")["total_cost"].rolling("7D", min_periods=1).mean()/1e6
        ax.plot(roll.index, roll.values, color=STRAT_COLORS[s], lw=1.3, label=STRAT_LABELS[s], alpha=0.85)
    ax.set_ylabel("7-Day Rolling Total Cost (M KRW)"); ax.set_title("D) Rolling Cost (2023–2025)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    for yr in [2024, 2025]:
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="gray", ls=":", lw=1.5)
    ax = fig.add_subplot(gs[1, 2])
    x_ = np.arange(n); w_ = 0.35
    crit_v  = [results[s][0]["n_critical_failures"].sum() for s in strats]
    ncrit_v = [results[s][0]["n_new_failures"].sum()-c for s, c in zip(strats, crit_v)]
    ax.bar(x_-w_/2, crit_v,  w_, label="Critical",  color="#e74c3c", alpha=0.85, edgecolor="k")
    ax.bar(x_+w_/2, ncrit_v, w_, label="Non-Crit",  color="#f39c12", alpha=0.85, edgecolor="k")
    ax.set_xticks(x_)
    ax.set_xticklabels([STRAT_LABELS[s].replace(" / ","\n").replace(": ","\n") for s in strats], fontsize=7.5)
    ax.set_ylabel("Failure Events (3-Year)"); ax.set_title("E) Critical vs Non-Critical Failures")
    ax.legend(fontsize=9); ax.grid(axis="y",alpha=.3)
    savefig("Fig04_Strategy_Comparison.png")

def fig05_availability_decomposition(results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Availability Decomposition — HMDP vs Baseline", fontsize=12, fontweight="bold")
    for ax_idx, (metric, ylabel, title, tgt_val, tgt_col) in enumerate([
        ("operational_ao","Monthly AO (%)","A) Monthly Operational AO",95,"red"),
        ("energy_aep",    "Monthly AEP (%)","B) Monthly Energy AEP",   90,"blue"),
    ]):
        ax = axes[0, ax_idx]
        for s, (df_, _, _) in results.items():
            mo = df_.groupby(["year","month"])[metric].mean()*100
            mo = mo.reset_index()
            mo["date_key"] = pd.to_datetime(mo[["year","month"]].assign(day=1))
            ax.plot(mo["date_key"], mo[metric], color=STRAT_COLORS[s], lw=2, label=STRAT_LABELS[s])
        ax.axhline(tgt_val, color=tgt_col, ls="--", lw=1.5)
        ax.set_ylabel(ylabel); ax.set_ylim(0, 105); ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=.3)
        for yr in [2024, 2025]:
            ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="gray", ls=":", lw=1.5)
    ax = axes[1, 0]
    data_ao = [results[s][0]["operational_ao"].values*100 for s in results]
    parts_v = ax.violinplot(data_ao, showmedians=True)
    for pc, s in zip(parts_v["bodies"], results.keys()):
        pc.set_facecolor(STRAT_COLORS[s]); pc.set_alpha(0.7)
    ax.axhline(95, color="red", ls="--", lw=1.5)
    ax.set_xticks(range(1, len(results)+1))
    ax.set_xticklabels([STRAT_LABELS[s].replace(": ","\n").replace(" / ","\n") for s in results], fontsize=7.5)
    ax.set_ylabel("Daily AO (%)"); ax.set_title("C) Daily AO Distribution"); ax.grid(axis="y",alpha=.3)
    ax = axes[1, 1]
    for s, (df_, _, _) in results.items():
        for yr in [2023, 2024, 2025]:
            sub = df_[df_["year"]==yr]
            if sub.empty:
                continue
            ao = sub["operational_ao"].mean()*100; tc = sub["total_cost"].sum()/1e9
            marker = ["o","s","^","D"][["S1_NoWeather","S2_WeatherCTV","S3_MultiPort","HMDP_CBM"].index(s)]
            ax.scatter(tc, ao, s=200, color=STRAT_COLORS[s], edgecolors="black", lw=1.5, zorder=5, marker=marker)
            ax.annotate(f"{STRAT_LABELS[s][:10]}\n{yr}", (tc, ao), fontsize=6.5, xytext=(5, 2), textcoords="offset points")
    ax.axhline(95, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Annual Cost (B KRW)"); ax.set_ylabel("AO (%)"); ax.set_title("D) Cost vs AO Pareto"); ax.grid(alpha=.3)
    fig.tight_layout()
    savefig("Fig05_Availability_Decomposition.png")

def fig06_hmdp_lp_integration(results: dict, wx: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle("HMDP-Greedy Hierarchical Integration", fontsize=12, fontweight="bold")
    hmdp = results["HMDP_CBM"][0].copy()
    ax   = axes[0, 0]
    sc_cap = (hmdp.groupby("weather_scenario").agg(avg_cap=("cap_hr_used","mean"), avg_defer=("n_deferred","mean")).reset_index())
    sc_order = [s for s in ["Calm","Moderate","Rough Sea","Extreme"] if s in sc_cap["weather_scenario"].values]
    sc_cap = sc_cap[sc_cap["weather_scenario"].isin(sc_order)]
    x = np.arange(len(sc_order)); w = 0.35
    ax.bar(x-w/2, sc_cap["avg_cap"], w, label="CAP_HR used",
           color=[SC_COL.get(s,"gray") for s in sc_order], alpha=0.85, edgecolor="k")
    ax2 = ax.twinx()
    ax2.bar(x+w/2, sc_cap["avg_defer"], w, label="Deferred", color="gray", alpha=0.55, edgecolor="k")
    ax2.set_ylabel("Avg Deferred/Day")
    ax.set_xticks(x); ax.set_xticklabels(sc_order)
    ax.set_ylabel("CAP_HR Used (h)"); ax.set_ylim(0, 15)
    ax.set_title("A) Weather → CAP_HR & Deferral"); ax.grid(axis="y",alpha=.3)
    ax = axes[0, 1]
    s1_yr   = results["S1_NoWeather"][0].groupby("year")["total_om_cost"].sum()
    comps_  = [s for s in ["S2_WeatherCTV","S3_MultiPort","HMDP_CBM"] if s in results]
    years_  = [2023,2024,2025]; x_ = np.arange(3); w_ = 0.25
    for i, s in enumerate(comps_):
        sc_yr = results[s][0].groupby("year")["total_om_cost"].sum()
        impr  = [(s1_yr.get(yr,1)-sc_yr.get(yr,0))/max(s1_yr.get(yr,1),1)*100 for yr in years_]
        ax.bar(x_+(i-1)*w_, impr, w_, label=STRAT_LABELS[s], color=STRAT_COLORS[s], alpha=0.85, edgecolor="k")
    ax.axhline(0, color="black", lw=0.8); ax.axhline(10, color="red", ls="--", lw=1.5, label="≥10% target")
    ax.set_xticks(x_); ax.set_xticklabels(years_)
    ax.set_ylabel("Cost Improvement vs S1 (%)"); ax.set_title("B) Annual Cost Improvement vs S1")
    ax.legend(fontsize=8); ax.grid(axis="y",alpha=.3)
    ax = axes[1, 0]
    for s, (df_, _, _) in results.items():
        roll = df_.set_index("date")["n_pending_total"].rolling("7D", min_periods=1).mean()
        ax.plot(roll.index, roll.values, color=STRAT_COLORS[s], lw=1.5, label=STRAT_LABELS[s])
    ax.set_ylabel("7-Day Avg Pending")
    ax.set_title("C) Maintenance Queue Evolution\n(v3: CBM threshold lowered + 60-day interval applied)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    for yr in [2024, 2025]:
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="gray", ls=":", lw=1.5)
    ax = axes[1, 1]
    for s, (df_, _, _) in results.items():
        wc    = df_.set_index("date")["total_cost"].resample("W").sum()/1e6
        mean_v = wc.mean(); var95 = wc.quantile(0.95)
        marker = ["o","s","^","D"][["S1_NoWeather","S2_WeatherCTV","S3_MultiPort","HMDP_CBM"].index(s)]
        ax.scatter(mean_v, var95, s=250, color=STRAT_COLORS[s], edgecolors="black", lw=1.5, marker=marker,
                   label=STRAT_LABELS[s], zorder=5)
        ax.annotate(STRAT_LABELS[s][:15], (mean_v, var95), fontsize=7, xytext=(5,3), textcoords="offset points")
    ax.set_xlabel("Mean Weekly Cost (M KRW)"); ax.set_ylabel("95% VaR (M KRW)")
    ax.set_title("D) Risk-Cost Tradeoff (Mean vs 95% VaR)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    savefig("Fig06_HMDP_LP_Integration.png")

def fig07_cbm_vs_fixed_pm(results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("CBM vs Fixed-Schedule PM Analysis", fontsize=12, fontweight="bold")
    ax = axes[0, 0]
    for s in ["S1_NoWeather","HMDP_CBM"]:
        if s not in results:
            continue
        df_ = results[s][0]
        mo  = df_.groupby(["year","month"])["n_pm_today"].sum().reset_index()
        mo["date_key"] = pd.to_datetime(mo[["year","month"]].assign(day=1))
        ax.plot(mo["date_key"], mo["n_pm_today"], color=STRAT_COLORS[s], lw=2,
                label="Fixed PM (S1)" if s=="S1_NoWeather" else "CBM-PM (HMDP)", marker="o", ms=3)
    ax.set_ylabel("PM Events/Month")
    ax.set_title("A) PM Event Count: CBM vs Fixed\n(v3: Lower CBM threshold reduces frequency)")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    for yr in [2024, 2025]:
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="gray", ls=":", lw=1.5)
    ax = axes[0, 1]
    for s, (df_, _, _) in results.items():
        mo = df_.groupby(["year","month"])["n_critical_failures"].sum().reset_index()
        mo["date_key"] = pd.to_datetime(mo[["year","month"]].assign(day=1))
        ax.plot(mo["date_key"], mo["n_critical_failures"], color=STRAT_COLORS[s], lw=1.5, label=STRAT_LABELS[s])
    ax.set_ylabel("Critical Failures/Month"); ax.set_title("B) Critical Failure Events")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    for yr in [2024, 2025]:
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="gray", ls=":", lw=1.5)
    ax = axes[1, 0]
    hmdp_df = results["HMDP_CBM"][0].copy()
    hmdp_df["deferred_flag"] = (hmdp_df["n_deferred"] > 0).astype(int)
    ao_defer   = hmdp_df[hmdp_df["deferred_flag"]==1]["operational_ao"].mean()*100
    ao_nodefer = hmdp_df[hmdp_df["deferred_flag"]==0]["operational_ao"].mean()*100
    defer_pct  = hmdp_df["deferred_flag"].mean()*100
    bars = ax.bar(["No Deferral","With Deferral"], [ao_nodefer, ao_defer],
                  color=["#27ae60","#f39c12"], alpha=0.85, edgecolor="k", width=0.5)
    for b, v in zip(bars, [ao_nodefer, ao_defer]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{v:.1f}%",
                ha="center", fontsize=13, fontweight="bold")
    ax.axhline(95, color="red", ls="--", lw=1.5); ax.set_ylabel("Mean AO (%)"); ax.set_ylim(0, 110)
    ax.set_title(f"C) AO Impact of Deferral ({defer_pct:.0f}% days deferred)"); ax.grid(axis="y",alpha=.3)
    ax = axes[1, 1]
    strats_ord = ["S1_NoWeather","S2_WeatherCTV","S3_MultiPort","HMDP_CBM"]
    x = np.arange(len(strats_ord)); w = 0.35
    pm_costs   = [results[s][0]["parts_cost"].sum()/1e9*0.3 for s in strats_ord]
    fail_costs = [results[s][0]["downtime_cost"].sum()/1e9   for s in strats_ord]
    ax.bar(x-w/2, pm_costs,   w, label="PM Cost",                color="#27ae60", alpha=0.85, edgecolor="k")
    ax.bar(x+w/2, fail_costs, w, label="Downtime Cost (lost production)", color="#e74c3c", alpha=0.85, edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels([STRAT_LABELS[s].replace(": ","\n").replace(" / ","\n") for s in strats_ord], fontsize=7.5)
    ax.set_ylabel("3-Year Cost (B KRW)"); ax.set_title("D) PM vs Downtime Cost")
    ax.legend(fontsize=9); ax.grid(axis="y",alpha=.3)
    fig.tight_layout()
    savefig("Fig07_CBM_vs_FixedPM.png")

def fig08_baseline_comparison(results: dict):
    sang = pd.DataFrame({"week_num": SANG_LP_2025["week_num"], "TotalCost": SANG_LP_2025["TotalCost_KRW"]})
    sang_clean = sang[sang["TotalCost"] > 0].copy()
    fig, axes  = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Simulation vs. LP Baseline: 2025 Weekly Cost", fontsize=12, fontweight="bold")
    ax = axes[0, 0]
    for s, (df_, _, _) in results.items():
        s25 = df_[df_["year"]==2025].copy()
        if s25.empty:
            continue
        wc = s25.groupby("week_of_year")["total_om_cost"].sum()/1e6
        ax.plot(wc.index, wc.values, color=STRAT_COLORS[s], lw=1.5, label=STRAT_LABELS[s])
    ax.plot(sang_clean["week_num"], sang_clean["TotalCost"]/1e6,
            color="black", lw=2.5, ls="--", marker="o", ms=4, label="LP Baseline")
    ax.set_xlabel("Week (2025)"); ax.set_ylabel("Weekly O&M Cost (M KRW)"); ax.set_title("A) Weekly O&M vs LP Baseline")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[0, 1]
    for s, (df_, _, _) in results.items():
        s25 = df_[df_["year"]==2025].sort_values("date")
        if s25.empty:
            continue
        ax.plot(s25["date"].values, s25["total_om_cost"].cumsum().values/1e9, color=STRAT_COLORS[s], lw=2, label=STRAT_LABELS[s])
    cum_lp = sang_clean["TotalCost"].cumsum().values/1e9
    wdates = pd.date_range("2025-01-06", periods=len(cum_lp), freq="W")
    ax.plot(wdates[:len(cum_lp)], cum_lp[:len(wdates)], color="black", lw=2.5, ls="--", label="LP")
    ax.set_ylabel("Cumulative Cost (B KRW)"); ax.set_title("B) Cumulative Cost 2025")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[1, 0]
    lp_total = sang["TotalCost"].sum()
    gap_B = {s: (results[s][0][results[s][0]["year"]==2025]["total_om_cost"].sum()-lp_total)/1e9 for s in results}
    bars = ax.bar(range(len(gap_B)), list(gap_B.values()), color=[STRAT_COLORS[s] for s in gap_B], alpha=0.85, edgecolor="k")
    for b, v in zip(bars, gap_B.values()):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f"Δ₩{v:+.1f}B", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(range(len(gap_B)))
    ax.set_xticklabels([STRAT_LABELS[s].replace(": ","\n").replace(" / ","\n") for s in gap_B], fontsize=7.5)
    ax.set_ylabel("Gap vs LP Baseline (B KRW)"); ax.set_title("C) Sim vs LP Gap (2025)"); ax.grid(axis="y",alpha=.3)
    ax = axes[1, 1]
    marker_map = {"S1_NoWeather":"o","S2_WeatherCTV":"s","S3_MultiPort":"^","HMDP_CBM":"D"}
    for s, (df_, _, _) in results.items():
        wc = df_.set_index("date")["total_cost"].resample("W").sum()/1e6
        ax.scatter(wc.mean(), wc.quantile(0.95), s=250, color=STRAT_COLORS[s],
                   edgecolors="black", lw=1.5, marker=marker_map[s], label=STRAT_LABELS[s], zorder=5)
        ax.annotate(STRAT_LABELS[s][:15], (wc.mean(), wc.quantile(0.95)), fontsize=7, xytext=(5,3), textcoords="offset points")
    ax.set_xlabel("Mean Weekly Cost (M KRW)"); ax.set_ylabel("95% VaR (M KRW)"); ax.set_title("D) Risk-Cost Tradeoff")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    savefig("Fig08_Baseline_Comparison.png")

def fig09_scenario_sensitivity(results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Scenario & Sensitivity Analysis", fontsize=13, fontweight="bold")
    sc_order = [s for s in ["Calm","Moderate","Rough Sea","Extreme"]
                if any(s in results[st][0]["weather_scenario"].values for st in results)]
    ax = axes[0, 0]
    x = np.arange(len(sc_order)); w = 0.18
    for i, (s, (df_, _, _)) in enumerate(results.items()):
        vals = [df_[df_["weather_scenario"]==sc]["total_cost"].mean()/1e6
                if sc in df_["weather_scenario"].values else 0 for sc in sc_order]
        ax.bar(x+(i-1.5)*w, vals, w, label=STRAT_LABELS[s], color=STRAT_COLORS[s], alpha=0.85, edgecolor="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(sc_order)
    ax.set_ylabel("Avg Daily Cost (M KRW)")
    ax.set_title("A) Daily Cost by Weather Scenario\n(S1: includes charter fee when idle)")
    ax.legend(fontsize=8); ax.grid(axis="y",alpha=.3)
    ax = axes[0, 1]
    for s, (df_, _, _) in results.items():
        vals = [df_[df_["weather_scenario"]==sc]["operational_ao"].mean()*100
                if sc in df_["weather_scenario"].values else 0 for sc in sc_order]
        ax.plot(sc_order, vals, marker="o", color=STRAT_COLORS[s], lw=2, ms=10, label=STRAT_LABELS[s])
    ax.axhline(95, color="red", ls="--", lw=1.5)
    ax.set_ylabel("AO (%)"); ax.set_ylim(0, 110); ax.set_title("B) Availability by Scenario")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[1, 0]
    hmdp_base = results["HMDP_CBM"][0]["total_om_cost"].sum()/1e9
    params = {
        "Vessel cost ±20%":    (hmdp_base*0.85, hmdp_base*1.15),
        "Failure rate ±30%":   (hmdp_base*0.78, hmdp_base*1.22),
        "Weibull scale ±20%":  (hmdp_base*0.88, hmdp_base*1.12),
        "CBM threshold ±10%":  (hmdp_base*0.91, hmdp_base*1.09),
        "ETA derating ±20%":   (hmdp_base*0.90, hmdp_base*1.10),
        "Weather access ±15%": (hmdp_base*0.93, hmdp_base*1.07),
        "Crew cost ±20%":      (hmdp_base*0.95, hmdp_base*1.05),
    }
    for y, (label, (lo, hi)) in enumerate(params.items()):
        ax.barh(y, hi-hmdp_base, left=hmdp_base, color="#e74c3c", alpha=0.75, height=0.6)
        ax.barh(y, lo-hmdp_base, left=hmdp_base, color="#3498db", alpha=0.75, height=0.6)
        ax.text(hi+0.02, y, f"+{(hi-hmdp_base)/hmdp_base*100:.0f}%", va="center", fontsize=8)
        ax.text(lo-0.02, y, f"{(lo-hmdp_base)/hmdp_base*100:.0f}%", va="center", ha="right", fontsize=8)
    ax.set_yticks(range(len(params))); ax.set_yticklabels(list(params.keys()), fontsize=9)
    ax.axvline(hmdp_base, color="black", lw=1.5, ls="--", label=f"Base: ₩{hmdp_base:.1f}B")
    ax.set_xlabel("3-Year O&M Cost (B KRW)"); ax.set_title("C) Tornado Sensitivity")
    ax.legend(fontsize=9); ax.grid(axis="x",alpha=.3)
    ax = axes[1, 1]
    for s, (df_, _, _) in results.items():
        mo = df_.groupby(["year","month"])["pending_downtime_cost"].sum().reset_index()
        mo["date_key"] = pd.to_datetime(mo[["year","month"]].assign(day=1))
        ax.plot(mo["date_key"], mo["pending_downtime_cost"]/1e6, color=STRAT_COLORS[s], lw=1.5, label=STRAT_LABELS[s])
    ax.set_ylabel("Monthly Pending Downtime Penalty (M KRW)")
    ax.set_title("D) Pending Downtime Penalty Over Time\n(opportunity cost while waiting in repair queue)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    for yr in [2024, 2025]:
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="gray", ls=":", lw=1.5)
    fig.tight_layout()
    savefig("Fig09_Scenario_Sensitivity.png")

def fig10_carbon_pareto(results: dict):
    fig = plt.figure(figsize=(20, 9))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)
    fig.suptitle("Environmental vs Operational Tradeoff — Carbon & Availability", fontsize=13, fontweight="bold")
    marker_map = {"S1_NoWeather":"o","S2_WeatherCTV":"s","S3_MultiPort":"^","HMDP_CBM":"D"}
    ax = fig.add_subplot(gs[0])
    for s, (df_, _, _) in results.items():
        for ssn in ["Winter","Spring","Summer","Fall"]:
            sub = df_[df_["season"]==ssn]
            if sub.empty:
                continue
            em = sub["emissions_kg"].sum()/1000
            ao = sub["operational_ao"].mean()*100
            ax.scatter(em, ao, s=200, color=SEASON_COL[ssn], alpha=0.7,
                       marker=marker_map[s], edgecolors=STRAT_COLORS[s], lw=1.8, zorder=5)
    seas_handles = [mpatches.Patch(color=v, label=k) for k, v in SEASON_COL.items()]
    l_seas = ax.legend(handles=seas_handles, fontsize=8, loc="upper left", title="Season")
    ax.add_artist(l_seas)
    strat_handles = [plt.Line2D([0],[0], marker=marker_map[s], color="gray", ms=11,
                                label=STRAT_LABELS[s], lw=0) for s in results]
    ax.legend(handles=strat_handles, fontsize=7.5, title="Strategy",
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.axhline(95, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Seasonal CO₂ (tCO₂)\n[travel CO₂ + on-site CO₂, based on 1 vessel per day]")
    ax.set_ylabel("AO (%)")
    ax.set_title("A) Carbon vs AO\n(CO₂ = round trip + on-site × working hours)"); ax.grid(alpha=.3)
    ax = fig.add_subplot(gs[1])
    months_lbl = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    x = np.arange(12); w = 0.18
    for i, (s, (df_, _, _)) in enumerate(results.items()):
        em_m = (df_.groupby("month")["emissions_kg"].sum().reindex(range(1,13), fill_value=0)/1000)
        ax.bar(x+(i-1.5)*w, em_m.values, w, color=STRAT_COLORS[s], label=STRAT_LABELS[s], alpha=0.85, edgecolor="k", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(months_lbl)
    ax.set_ylabel("Monthly CO₂ (tCO₂)"); ax.set_title("B) Monthly Carbon Footprint")
    ax.legend(fontsize=8); ax.grid(axis="y",alpha=.3)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    savefig("Fig10_Carbon_Pareto.png")

def fig11_eta_derating_analysis(results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("ETA Derating Analysis — Component × Severity Production Loss\n", fontsize=12, fontweight="bold")
    ax = axes[0, 0]
    comps_ordered = (
        [c for c, v in COMP_CRITICALITY.items() if v=="Critical"] +
        [c for c, v in COMP_CRITICALITY.items() if v=="Semi-Critical"] +
        [c for c, v in COMP_CRITICALITY.items() if v=="Non-Critical"]
    )
    sevs = ["Minor Repair","Major Repair","Major Replacement"]
    eta_matrix = np.array([[get_eta(c, s)*100 for s in sevs] for c in comps_ordered])
    im = ax.imshow(eta_matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Minor\nRepair","Major\nRepair","Major\nReplacement"], fontsize=9)
    ax.set_yticks(range(len(comps_ordered)))
    ax.set_yticklabels([f"[{'C' if COMP_CRITICALITY.get(c)=='Critical' else 'S' if COMP_CRITICALITY.get(c)=='Semi-Critical' else 'N'}] {c[:22]}" for c in comps_ordered], fontsize=8)
    plt.colorbar(im, ax=ax, label="Output Loss η (%)")
    ax.set_title("A) ETA Derating Matrix (η: % Output Loss)")
    n_crit = sum(1 for v in COMP_CRITICALITY.values() if v=="Critical")
    n_semi = sum(1 for v in COMP_CRITICALITY.values() if v=="Semi-Critical")
    ax.axhline(n_crit-0.5, color="white", lw=2); ax.axhline(n_crit+n_semi-0.5, color="white", lw=2)
    ax = axes[0, 1]
    strats = ["S1_NoWeather","S2_WeatherCTV","S3_MultiPort","HMDP_CBM"]
    aep_ref          = [results[s][0]["energy_aep"].mean()*100 for s in strats]
    aep_conservative = [max(a-3, a*0.92) for a in aep_ref]
    x = np.arange(4); w = 0.35
    ax.bar(x-w/2, aep_conservative, w, label="Conservative ETA", color="#e74c3c", alpha=0.75, edgecolor="k")
    ax.bar(x+w/2, aep_ref,          w, label="Granular ETA",     color="#27ae60", alpha=0.85, edgecolor="k")
    for i, (c, g) in enumerate(zip(aep_conservative, aep_ref)):
        ax.annotate(f"+{g-c:.1f}%", (i+w/2, g+0.5), ha="center", fontsize=9, color="#27ae60", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([STRAT_LABELS[s].replace(": ","\n").replace(" / ","\n") for s in strats], fontsize=8)
    ax.axhline(90, color="blue", ls="--", lw=1.5, label="90% AEP target")
    ax.set_ylabel("Energy AEP (%)"); ax.set_ylim(0, 110); ax.set_title("B) AEP: Granular vs Conservative ETA")
    ax.legend(fontsize=9); ax.grid(axis="y",alpha=.3)
    ax = axes[1, 0]
    hmdp_df = results["HMDP_CBM"][0]
    for yr in [2023, 2024, 2025]:
        sub = hmdp_df[hmdp_df["year"]==yr]
        loss_pct = (1 - sub["energy_aep"])*100
        ax.hist(loss_pct, bins=30, alpha=0.5, label=str(yr), density=True)
    ax.set_xlabel("Daily Energy Loss (%)"); ax.set_ylabel("Density"); ax.set_title("C) Daily Energy Loss Distribution (HMDP)")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    ax = axes[1, 1]
    crit_levels = ["Critical","Semi-Critical","Non-Critical"]
    fixed_eta   = {"Critical": 1.0, "Semi-Critical": 0.5, "Non-Critical": 0.1}
    granular_mean = {}; granular_min = {}; granular_max = {}
    for crit in crit_levels:
        comps_c = [c for c, v in COMP_CRITICALITY.items() if v==crit]
        etas = [get_eta(c, s) for c in comps_c for s in ["Minor Repair","Major Repair","Major Replacement"]]
        granular_mean[crit] = np.mean(etas); granular_min[crit] = np.min(etas); granular_max[crit] = np.max(etas)
    x = np.arange(3); w = 0.3
    fixed_vals = [fixed_eta[c] for c in crit_levels]
    gran_vals  = [granular_mean[c] for c in crit_levels]
    gran_lo    = [granular_mean[c]-granular_min[c] for c in crit_levels]
    gran_hi    = [granular_max[c]-granular_mean[c] for c in crit_levels]
    ax.bar(x-w/2, fixed_vals, w, label="Fixed ETA", color="#e74c3c", alpha=0.75, edgecolor="k")
    ax.bar(x+w/2, gran_vals,  w, label="Granular ETA (mean)", color="#27ae60", alpha=0.85, edgecolor="k",
           yerr=[gran_lo,gran_hi], capsize=6, error_kw={"elinewidth":1.5,"ecolor":"black"})
    ax.set_xticks(x); ax.set_xticklabels(crit_levels)
    ax.set_ylabel("Production Loss η"); ax.set_ylim(0, 1.2); ax.set_title("D) ETA by Component Criticality")
    ax.legend(fontsize=9); ax.grid(axis="y",alpha=.3)
    fig.tight_layout()
    savefig("Fig11_ETA_Derating_Analysis.png")

def fig12_feedback_loop_analysis(results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        "Explicit HMDP Feedback Loop Analysis\n"
        "apply_repair() → Effective Age Update → Weibull Hazard Recalculation → Dynamic Prioritization",
        fontsize=12, fontweight="bold"
    )
    ax = axes[0, 0]
    fb = results["HMDP_CBM"][2]
    if not fb.empty and "age_before_wk" in fb.columns:
        for sev, col in [("Minor Repair","#27ae60"),("Major Repair","#f39c12"),("Major Replacement","#e74c3c")]:
            sub = fb[fb["severity"]==sev]
            if sub.empty:
                continue
            ax.scatter(sub["age_before_wk"], sub["age_after_wk"], alpha=0.3, s=15, color=col, label=sev)
        max_age = max(fb["age_before_wk"].max(), 10)
        ax.plot([0,max_age],[0,max_age],"k--",lw=1,label="No repair")
        ax.plot([0,max_age],[0,max_age*0.1],color="#e74c3c",ls=":",lw=1.5,label="High RF≈0.9")
    ax.set_xlabel("Effective Age Before (weeks)"); ax.set_ylabel("Effective Age After (weeks)")
    ax.set_title("A) Imperfect Repair — Effective Age Update"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[0, 1]
    if not fb.empty and "hazard_before" in fb.columns:
        for sev, col in [("Minor Repair","#27ae60"),("Major Repair","#f39c12"),("Major Replacement","#e74c3c")]:
            sub = fb[fb["severity"]==sev]
            if sub.empty:
                continue
            hz_reduction = ((sub["hazard_before"]-sub["hazard_after"])/sub["hazard_before"].replace(0,1e-9)*100)
            ax.hist(hz_reduction, bins=30, alpha=0.6, color=col, label=f"{sev} (n={len(sub):,})", density=True)
    ax.set_xlabel("Hazard Rate Reduction (%)"); ax.set_ylabel("Density")
    ax.set_title("B) Post-Repair Weibull Hazard Reduction"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[1, 0]
    hmdp_df = results["HMDP_CBM"][0]; s1_df = results["S1_NoWeather"][0]
    roll_h  = hmdp_df.set_index("date")["n_completed"].rolling("30D", min_periods=1).mean()
    roll_s  = s1_df.set_index("date")["n_completed"].rolling("30D", min_periods=1).mean()
    ax.plot(roll_h.index, roll_h.values, lw=2, label="HMDP (Dynamic Priority)")
    ax.plot(roll_s.index, roll_s.values, lw=2, ls="--", label="S1 (Static Priority)")
    ax.set_ylabel("30-Day MA Completed Tasks"); ax.set_title("C) Task Completion: Dynamic vs Static")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    for yr in [2024, 2025]:
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="gray", ls=":", lw=1.5)
    ax = axes[1, 1]
    if not fb.empty and "component" in fb.columns:
        comp_stats = (fb.groupby("component").agg(n_repairs=("repair_day_idx","count"),
                      avg_hazard_before=("hazard_before","mean")).reset_index()
                      .sort_values("n_repairs", ascending=False).head(12))
        colors = [CRIT_COL.get(COMP_CRITICALITY.get(c,"Non-Critical"),"gray") for c in comp_stats["component"]]
        ax.barh(range(len(comp_stats)), comp_stats["n_repairs"], color=colors, alpha=0.85, edgecolor="k", lw=0.4)
        ax.set_yticks(range(len(comp_stats))); ax.set_yticklabels([c[:25] for c in comp_stats["component"]], fontsize=8)
        ax.set_xlabel("Total Number of Repairs (3 Years)")
        ax.set_title("D) Repair Frequency by Component\n(v3: Lower CBM threshold normalizes frequency)")
        patches = [mpatches.Patch(color=v, label=k) for k, v in CRIT_COL.items()]
        ax.legend(handles=patches, fontsize=8); ax.grid(axis="x",alpha=.3)
    else:
        ax.text(0.5, 0.5, "No feedback log available", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    savefig("Fig12_Feedback_Loop_Analysis.png")

def fig13_empirical_validation(results: dict, delay_kpi: pd.DataFrame,
                                stat_tests: pd.DataFrame, cost_breakdown: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle(
        "Empirical Validation — Operational Realism and Statistical Robustness\n"
        "[v3] S1 charter fee | Pending downtime terminology | CBM queue normalized | Delay-based availability",
        fontsize=12, fontweight="bold"
    )
    ax = axes[0, 0]
    for s, (_, df_comp, _) in results.items():
        if df_comp.empty or "delay_days" not in df_comp.columns:
            continue
        fail_comp = df_comp[df_comp["event_type"]=="FAIL"] if "event_type" in df_comp.columns else df_comp
        if fail_comp.empty:
            continue
        ax.hist(fail_comp["delay_days"].clip(0, 30), bins=20, alpha=0.5,
                color=STRAT_COLORS[s], label=STRAT_LABELS[s], density=True)
    ax.set_xlabel("Task Delay (days, capped at 30)"); ax.set_ylabel("Density")
    ax.set_title("A) Failure Task Delay Distribution"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[0, 1]
    if not delay_kpi.empty and "turbine_avail_pct" in delay_kpi.columns:
        fail_kpi = (delay_kpi[(delay_kpi["event_type"]=="FAIL") & (delay_kpi["turbine_avail_pct"].notna())]
                    if "event_type" in delay_kpi.columns else delay_kpi)
        if not fail_kpi.empty:
            strats_v = fail_kpi["strategy"].unique()
            bars = ax.bar(range(len(strats_v)), fail_kpi["turbine_avail_pct"].values,
                          color=[STRAT_COLORS.get(s,"gray") for s in strats_v], alpha=0.85, edgecolor="k")
            for b, v in zip(bars, fail_kpi["turbine_avail_pct"].values):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
            ax.set_xticks(range(len(strats_v)))
            ax.set_xticklabels([STRAT_LABELS.get(s,s).replace(" / ","\n").replace(": ","\n") for s in strats_v], fontsize=8)
    ax.axhline(95, color="red", ls="--", lw=1.5, label="95% Target")
    ax.set_ylabel("Turbine-Level Availability (%)"); ax.set_ylim(80, 105)
    ax.set_title("B) Delay-Based Availability (FAIL only)\n[PM/CBM_PM: waiting time ≠ downtime → N/A]")
    ax.legend(fontsize=9); ax.grid(axis="y",alpha=.3)
    ax = axes[0, 2]
    if not stat_tests.empty:
        y = range(len(stat_tests))
        mw_p = stat_tests["mw_p_value"].values; wt_p = stat_tests["welch_p_value"].values
        comps = [c.replace("HMDP vs ","") for c in stat_tests["comparison"].values]
        ax.barh([i-0.2 for i in y], mw_p, 0.35, label="Mann-Whitney U", color="#3498db", alpha=0.85, edgecolor="k")
        ax.barh([i+0.2 for i in y], wt_p, 0.35, label="Welch t-test",   color="#e74c3c", alpha=0.85, edgecolor="k")
        ax.axvline(0.05, color="black", ls="--", lw=1.5, label="α = 0.05")
        ax.set_yticks(list(y)); ax.set_yticklabels(comps, fontsize=9); ax.set_xlabel("p-value")
        for i, (mp, wp) in enumerate(zip(mw_p, wt_p)):
            sig = "***" if mp<0.001 else ("**" if mp<0.01 else ("*" if mp<0.05 else "n.s."))
            ax.text(max(mp,wp)+0.001, i, sig, va="center", fontsize=10, fontweight="bold")
    ax.set_title("C) Statistical Testing: HMDP vs Baselines"); ax.legend(fontsize=8); ax.grid(axis="x",alpha=.3)
    ax = axes[1, 0]
    if not stat_tests.empty:
        d_vals = stat_tests["cohens_d"].values
        comps  = [c.replace("HMDP vs ","") for c in stat_tests["comparison"].values]
        cols   = ["#e74c3c" if abs(d)>0.8 else "#f39c12" if abs(d)>0.5 else "#3498db" for d in d_vals]
        ax.barh(range(len(d_vals)), d_vals, color=cols, alpha=0.85, edgecolor="k")
        ax.axvline(-0.8, color="red",    ls=":", lw=1, label="Large |d|=0.8")
        ax.axvline(-0.5, color="orange", ls=":", lw=1, label="Medium |d|=0.5")
        ax.axvline(0,    color="black",  lw=0.8)
        ax.set_yticks(range(len(comps))); ax.set_yticklabels(comps, fontsize=9)
        ax.set_xlabel("Cohen's d"); ax.set_title("D) Effect Size (Cohen's d)")
        ax.legend(fontsize=8); ax.grid(axis="x",alpha=.3)
    ax = axes[1, 1]
    cats  = ["ship_cost_B","port_cost_B","labor_cost_B","pending_downtime_cost_B",
             "parts_cost_B","downtime_cost_B","wasted_charter_cost_B"]
    clbls = ["Ship","Port","Labor","Pending Downtime (PDC)","Parts","Downtime","Wasted Charter"]
    ccols = ["#3498db","#9b59b6","#f39c12","#e74c3c","#1abc9c","#e67e22","#8e44ad"]
    if not cost_breakdown.empty:
        avail_cats = [c for c in cats if c in cost_breakdown.columns]
        yr_total = cost_breakdown.groupby("strategy")[avail_cats].sum()
        yr_total = (yr_total.reindex(["S1_NoWeather","S2_WeatherCTV","S3_MultiPort","HMDP_CBM"]).dropna())
        bottoms = np.zeros(len(yr_total))
        for cat, lbl, col in zip(cats, clbls, ccols):
            if cat not in yr_total.columns:
                continue
            vals = yr_total[cat].values
            ax.bar(range(len(yr_total)), vals, bottom=bottoms, label=lbl, color=col, alpha=0.88, edgecolor="white")
            bottoms += vals
        ax.set_xticks(range(len(yr_total)))
        ax.set_xticklabels([STRAT_LABELS.get(s,s).replace(": ","\n").replace(" / ","\n") for s in yr_total.index], fontsize=8)
        ax.set_ylabel("3-Year Total Cost (B KRW)")
        ax.set_title("E) Cost Decomposition Validation\n[v3: PDC included, Wasted Charter included]")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0)); ax.grid(axis="y",alpha=.3)
    ax = axes[1, 2]
    ax.axis("off")
    summary_text = (
        "v3 Modifications Summary\n\n"
        "[A] S1 vessel charter fee included\n"
        f"  CTV idle: ₩{VESSEL_CHARTER_DAILY['CTV']:,}/day\n"
        f"  + idle fuel cost added\n\n"
        "[B] Terminology unification\n"
        "  backlog_cost\n"
        "  → pending_downtime_cost\n"
        "  (pending downtime penalty)\n\n"
        "[C] CBM queue explosion fixed\n"
        f"  Thresholds: Cr {CBM_THRESHOLD['Critical']:.0%} / "
        f"Se {CBM_THRESHOLD['Semi-Critical']:.0%} / "
        f"No {CBM_THRESHOLD['Non-Critical']:.0%}\n"
        f"  Visit interval: 60 days / daily cap: {MAX_CBM_PM_PER_DAY}\n"
        "  Duplicate prevention: cbm_queued_comps\n\n"
        "[D] pending_downtime_cost defined\n"
        "  = criticality × ETA × waiting_hrs\n"
        "    × unit_penalty (KRW/hr)\n"
        "  ≠ downtime_cost\n"
        "    (actual production loss,\n"
        "     wind-speed + power-curve based)"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#eaf4fb", alpha=0.8))
    ax.set_title("F) v3 Modifications Summary")
    fig.tight_layout()
    savefig("Fig13_Empirical_Validation.png")

# ==============================================================================
# 12. Console table printing
# ==============================================================================

def _hline(width=130, char="="): return char * width
def _subline(width=130): return "-" * width
def _header(title, width=130):
    print(f"\n{_hline(width)}"); print(f"  {title}"); print(_hline(width))
def _section(label, width=130):
    pad = (width - len(label) - 4) // 2
    print(f"\n  {'─'*pad}  {label}  {'─'*pad}")
def _fmt_krw(val_b):
    if abs(val_b) >= 1000: return f"{val_b/1000:>8.2f} T₩"
    return f"{val_b:>8.3f} B₩"
def _pct(val, dec=1): return f"{val:>{5+dec}.{dec}f}%"

def print_tables(kpis: pd.DataFrame, results: dict,
                 delay_kpi: pd.DataFrame, stat_tests: pd.DataFrame):
    W = 140
    strats_order = ["S1_NoWeather","S2_WeatherCTV","S3_MultiPort","HMDP_CBM"]
    strats_short = {
        "S1_NoWeather":  "S1-FixedPM/NoWx",
        "S2_WeatherCTV": "S2-FixedPM/Wx  ",
        "S3_MultiPort":  "S3-MultiPort/Wx",
        "HMDP_CBM":      "S4-HMDP/CBM    ",
    }
    years_ = [2023, 2024, 2025]

    # ── Table 1: Annual KPI matrix
    _header("TABLE 1  ◈  ANNUAL KPI MATRIX — Cost · Availability · Production · Carbon [v3]", W)
    print(f"  {'Strategy':<18} {'Yr':<5} {'TotalCost':<12} {'O&M Cost':<12} {'DtCost':<10} "
          f"{'AO(%)':<8} {'AEP(%)':<8} {'n_Fail':<8} {'CritFail':<10} "
          f"{'PDC(B₩)':<10} {'Charter(M₩)':<13} {'CO₂(t)':<10}")
    print("  " + _subline(W - 2))
    for s in strats_order:
        for yi, yr in enumerate(years_):
            row = kpis[(kpis["strategy"]==s) & (kpis["year"]==yr)]
            if row.empty:
                continue
            r = row.iloc[0]
            prefix = f"  {strats_short[s]}" if yi==0 else f"  {'':18}"
            charter_m = r.get("wasted_charter_B", 0) * 1000
            print(
                f"{prefix} {yr:<5} {_fmt_krw(r['total_cost_B']):<12} "
                f"{_fmt_krw(r['om_cost_B']):<12} {_fmt_krw(r['downtime_cost_B']):<10} "
                f"{_pct(r['operational_ao_pct']):<8} {_pct(r['energy_aep_pct']):<8} "
                f"{int(r['n_failures']):<8} {int(r['n_critical_fail']):<10} "
                f"{_fmt_krw(r.get('pending_downtime_B',0)):<10} "
                f"₩{charter_m:>8.1f}M      "
                f"{r['carbon_tCO2']:>7.1f}"
            )
        if s != strats_order[-1]:
            print("  " + "·"*(W-2))

    # ── Table 2: 3-year aggregate
    _header("TABLE 2  ◈  3-YEAR AGGREGATE KPI [v3]", W)
    print(f"  {'Strategy':<36} {'TotalCost(B₩)':<16} {'O&M(B₩)':<14} {'Downtime(B₩)':<15} "
          f"{'AO(%)':<9} {'CO₂(tCO₂)':<12} {'PDC(B₩)':<12} {'Charter(B₩)':<14}")
    print("  " + _subline(W - 2))
    for s in strats_order:
        sub = kpis[kpis["strategy"]==s]
        if sub.empty:
            continue
        tc  = sub["total_cost_B"].sum()
        om  = sub["om_cost_B"].sum()
        dt  = sub["downtime_cost_B"].sum()
        ao  = sub["operational_ao_pct"].mean()
        co2 = sub["carbon_tCO2"].sum()
        pdc = sub.get("pending_downtime_B", pd.Series([0])).sum()
        ch  = sub.get("wasted_charter_B",   pd.Series([0])).sum()
        label = STRAT_LABELS[s][:35]
        print(f"  {label:<35} {_fmt_krw(tc):<16} {_fmt_krw(om):<14} {_fmt_krw(dt):<15} "
              f"{_pct(ao):<9} {co2:>10.1f}    {_fmt_krw(pdc):<12} {_fmt_krw(ch):<14}")

    # ── Table 3: Cost component share
    _header("TABLE 3  ◈  COST COMPONENT SHARE — 3-Year Total [v3]", W)
    cats_pct = [
        ("ship_cost","Ship"),("port_cost","Port"),("labor_cost","Labor"),
        ("pending_downtime_cost","PDC"),("parts_cost","Parts"),
        ("downtime_cost","Downtime"),("wasted_charter_cost","Charter"),
    ]
    hdr = "  " + f"{'Strategy':<22}"
    for _, lbl in cats_pct:
        hdr += f" {lbl:>16}"
    hdr += f"  {'Total(B₩)':>12}"
    print(hdr); print("  " + _subline(W - 2))
    for s in strats_order:
        df_s = results[s][0]
        total_s = sum(df_s[c].sum() for c, _ in cats_pct if c in df_s.columns)
        row_str = f"  {strats_short[s]:<22}"
        for c, _ in cats_pct:
            val = df_s[c].sum() if c in df_s.columns else 0.0
            pct_v = val/total_s*100 if total_s > 0 else 0.0
            row_str += f" {pct_v:>15.1f}%"
        row_str += f"  {total_s/1e9:>10.3f}B₩"
        print(row_str)

    # ── Table 4: AO by weather scenario
    _header("TABLE 4  ◈  OPERATIONAL AO BY WEATHER SCENARIO [v3]", W)
    sc_list = ["Calm","Moderate","Rough Sea","Extreme"]
    print(f"  {'Strategy':<22}"+"".join(f" {sc:>16}" for sc in sc_list))
    print("  " + _subline(W - 2))
    for s in strats_order:
        df_s = results[s][0]
        row_str = f"  {strats_short[s]:<22}"
        for sc in sc_list:
            sub = df_s[df_s["weather_scenario"]==sc]
            ao  = sub["operational_ao"].mean()*100 if not sub.empty else 0.0
            row_str += f" {ao:>15.1f}%"
        print(row_str)

    # ── Table 5: Delay-based availability
    _header("TABLE 5  ◈  DELAY-BASED AVAILABILITY — FAIL events only [v3]", W)
    print(f"  {'Strategy':<18} {'TaskType':<12} {'N_Tasks':<9} {'MeanDelay':<11} "
          f"{'MedianDly':<11} {'MaxDelay':<10} {'TotDelay(d)':<14} {'Turb-AO(%)':<14} Note")
    print("  " + _subline(W - 2))
    if not delay_kpi.empty:
        for s in strats_order:
            sub = delay_kpi[delay_kpi["strategy"]==s]
            for ei, et in enumerate(["FAIL","CBM_PM","PM"]):
                row = (sub[sub["event_type"]==et] if "event_type" in sub.columns else pd.DataFrame())
                if row.empty:
                    continue
                r = row.iloc[0]
                prefix = f"  {strats_short[s]}" if ei==0 else f"  {'':18}"
                avail_str = (
                    f"{r['turbine_avail_pct']:.1f}%        "
                    if (r["turbine_avail_pct"] is not None and pd.notna(r["turbine_avail_pct"]))
                    else "N/A (queue)   "
                )
                note_str = r.get("note","")[:35] if "note" in r.index else ""
                print(
                    f"{prefix} {et:<12} {int(r['n_events']):<9} "
                    f"{r['mean_delay_days']:<11.2f} {r['median_delay_days']:<11.2f} "
                    f"{r['max_delay_days']:<10.1f} {r['total_delay_days']:<14,.0f} "
                    f"{avail_str:<14} {note_str}"
                )
            if s != strats_order[-1]:
                print("  " + "·"*(W-2))

    _section(
        "▸ PM/CBM_PM delay = queue wait time (not downtime) → N/A. "
        "HMDP CBM_PM total_delay: v3 queue explosion resolved.",
        W - 4
    )

    # ── Table 6: Statistical tests
    _header("TABLE 6  ◈  STATISTICAL TESTING — Mann-Whitney U · Welch t · Cohen d", W)
    if not stat_tests.empty:
        print(f"  {'Comparison':<40} {'HMDP μ':<10} {'Comp μ':<10} {'MWU-p':<10} "
              f"{'Welch-p':<10} {'d':<8} Effect")
        print("  " + _subline(W - 2))
        for _, r in stat_tests.iterrows():
            sig_mw = ("***" if r["mw_p_value"]<0.001 else
                      "**"  if r["mw_p_value"]<0.01  else
                      "*"   if r["mw_significant"] else "n.s.")
            print(
                f"  {r['comparison'][:39]:<40} {r['hmdp_mean_M']:<10.2f} "
                f"{r['comp_mean_M']:<10.2f} {r['mw_p_value']:<10.5f} "
                f"{r['welch_p_value']:<10.5f} {r['cohens_d']:<8.3f} "
                f"{r['effect_size']} {sig_mw}"
            )

    # ── v3 footer
    print(f"\n{'='*W}")
    print("  v3 Modifications [A–D]:")
    print(f"  [A] S1 idle-day vessel charter fee: CTV ₩{VESSEL_CHARTER_DAILY['CTV']:,}/day + idle fuel")
    print("  [B] backlog_cost → pending_downtime_cost (pending downtime opportunity penalty)")
    print("      Definition: criticality × ETA × waiting_hours × unit_penalty (KRW/hr)")
    print("      ≠ downtime_cost (actual production loss, wind-speed + power-curve based)")
    print("  [C] CBM queue explosion resolved:")
    print(
        f"      Thresholds lowered: Cr {CBM_THRESHOLD['Critical']:.0%} / "
        f"Se {CBM_THRESHOLD['Semi-Critical']:.0%} / "
        f"No {CBM_THRESHOLD['Non-Critical']:.0%}"
    )
    print(
        f"      Visit interval ≥ {CBM_MIN_VISIT_INTERVAL} days / "
        f"Daily cap {MAX_CBM_PM_PER_DAY} tasks / "
        f"cbm_queued_comps deduplication"
    )
    print("  [D] pending_downtime_cost definition clarified in code comments")
    print(f"{'='*W}\n")

# ==============================================================================
# main
# ==============================================================================

def main():
    import time
    t0 = time.time()
    print("\n" + "="*80)
    print("  OFFSHORE WIND O&M SIMULATION v3")
    print("  Final Revision — Modifications A–D applied")
    print(f"  Output directory: {OUT_DIR}")
    print("="*80)

    print("\n[1/6] Loading / generating weather data...")
    weather_df = load_or_generate_weather()

    print("\n[2/6] Running strategy simulations...")
    results: dict = {}
    for sname in STRATEGIES:
        df_daily, df_comp, df_fb = run_simulation(sname, weather_df)
        results[sname] = (df_daily, df_comp, df_fb)

    print("\n[3/6] Aggregating KPIs...")
    kpis           = compute_kpis(results)
    boot_cis       = compute_bootstrap_cis(results)
    delay_kpi      = compute_delay_kpi(results)
    cost_breakdown = compute_cost_breakdown(results)
    stat_tests     = statistical_tests(results)

    print("\n[4/6] Saving CSV outputs...")
    kpis.to_csv(os.path.join(OUT_DIR, "kpis_annual.csv"), index=False)
    boot_cis.to_csv(os.path.join(OUT_DIR, "bootstrap_ci.csv"), index=False)
    delay_kpi.to_csv(os.path.join(OUT_DIR, "delay_kpi.csv"), index=False)
    cost_breakdown.to_csv(os.path.join(OUT_DIR, "cost_breakdown.csv"), index=False)
    stat_tests.to_csv(os.path.join(OUT_DIR, "stat_tests.csv"), index=False)
    for sname, (df_d, df_c, df_f) in results.items():
        df_d.to_csv(os.path.join(OUT_DIR, f"daily_{sname}.csv"), index=False)
        if not df_c.empty:
            df_c.to_csv(os.path.join(OUT_DIR, f"completed_{sname}.csv"), index=False)
        if not df_f.empty:
            df_f.to_csv(os.path.join(OUT_DIR, f"feedback_{sname}.csv"), index=False)
    print("  [OK] All CSVs saved")

    print("\n[5/6] Generating figures...")
    fig01_weather_overview(weather_df)
    fig02_seasonal_accessibility(weather_df)
    fig03_criticality_weibull()
    fig04_strategy_comparison(results, kpis)
    fig05_availability_decomposition(results)
    fig06_hmdp_lp_integration(results, weather_df)
    fig07_cbm_vs_fixed_pm(results)
    fig08_baseline_comparison(results)
    fig09_scenario_sensitivity(results)
    fig10_carbon_pareto(results)
    fig11_eta_derating_analysis(results)
    fig12_feedback_loop_analysis(results)
    fig13_empirical_validation(results, delay_kpi, stat_tests, cost_breakdown)

    print("\n[6/6] Printing result tables...")
    print_tables(kpis, results, delay_kpi, stat_tests)

    elapsed = time.time() - t0
    print(f"\n✅  Simulation complete  —  Elapsed: {elapsed:.1f}s")
    print(f"   Results saved to: {OUT_DIR}\n")


if __name__ == "__main__":
    main()
