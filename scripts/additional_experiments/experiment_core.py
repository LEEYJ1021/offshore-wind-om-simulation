"""
=============================================================================
HMDP Offshore Wind O&M — Additional Experiments for RESS Revision
experiment_core.py  (revised: real KMA data integrated)
=============================================================================
Changes from original:
  * load_weather() now ingests weather_hourly_raw.csv (Korean column names)
    -> aggregates hourly -> daily (mean Hs, mean wind_speed)
    -> assigns season from month
    -> fills missing values via linear interpolation
  * load_weekly_baseline() reads weekly_cost_baseline.csv with comma-parsing
  * _synthetic_weather() retained as fallback when files absent
  * All other constants / Weibull helpers unchanged
=============================================================================
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

SEED = 42
rng  = np.random.default_rng(SEED)

# Simulation constants
N_TURBINES   = 50
N_DAYS       = 1096
BETA         = 2.5
ETA_WEEKS    = 80
ETA_DAYS     = ETA_WEEKS * 7

THETA = {"Critical": 0.72, "Semi-Critical": 0.65, "Non-Critical": 0.55}
RF    = {"Minor": 0.35, "Major": 0.55, "Replacement": 0.90}

CTV_HS_LIMIT   = 1.5
SOV_HS_LIMIT   = 2.5
CTV_WIND_LIMIT = 10.0

ETA_MATRIX = {
    "Gearbox":        {"Minor": 0.40, "Major": 0.75, "Replacement": 1.00},
    "Generator":      {"Minor": 0.35, "Major": 0.80, "Replacement": 1.00},
    "Blades":         {"Minor": 0.20, "Major": 0.60, "Replacement": 1.00},
    "Tower":          {"Minor": 0.10, "Major": 0.50, "Replacement": 1.00},
    "Hub":            {"Minor": 0.15, "Major": 0.55, "Replacement": 0.90},
    "PitchHydraulic": {"Minor": 0.15, "Major": 0.45, "Replacement": 0.70},
    "Yaw":            {"Minor": 0.05, "Major": 0.20, "Replacement": 0.40},
    "Safety":         {"Minor": 0.00, "Major": 0.05, "Replacement": 0.10},
}

CRITICALITY_MAP = {
    "Gearbox": "Critical", "Generator": "Critical",
    "Blades":  "Critical", "Tower":     "Critical",
    "Hub":     "Semi-Critical", "PitchHydraulic": "Semi-Critical",
    "Yaw":     "Non-Critical",  "Safety": "Non-Critical",
}

SEASON_MAP = {1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
              6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",
              11:"Autumn",12:"Winter"}

# Weibull helpers
def weibull_reliability(t_days, beta=BETA, eta=ETA_DAYS):
    return np.exp(-(np.asarray(t_days) / eta) ** beta)

def weibull_hazard(t_days, beta=BETA, eta=ETA_DAYS):
    t = np.maximum(np.asarray(t_days), 1e-6)
    return (beta / eta) * (t / eta) ** (beta - 1)

def cbm_trigger_day(t_start_days, theta, beta=BETA, eta=ETA_DAYS):
    t_trigger = eta * (-np.log(theta)) ** (1 / beta)
    return max(t_trigger - t_start_days, 0.0)


# Real KMA weather loader
def load_weather(path="/mnt/user-data/uploads/weather_hourly_raw.csv"):
    """
    Load KMA hourly CSV (Korean columns) -> aggregate to daily.
    Returns DataFrame: date, hs, wind_speed, season  (N_DAYS rows)
    """
    try:
        raw = pd.read_csv(path, parse_dates=["일시"])
    except Exception:
        print("[INFO] weather_hourly_raw.csv not found -- using synthetic weather")
        return _synthetic_weather()

    raw = raw.rename(columns={"일시": "datetime", "풍속(m/s)": "wind_speed", "유의파고(m)": "hs"})
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.set_index("datetime").sort_index()
    raw["hs"]         = raw["hs"].interpolate(method="time", limit=6)
    raw["wind_speed"] = raw["wind_speed"].interpolate(method="time", limit=6)
    raw = raw.reset_index()
    raw["date"] = raw["datetime"].dt.normalize()

    daily = (raw.groupby("date")
               .agg(hs=("hs","mean"), wind_speed=("wind_speed","mean"))
               .reset_index())
    daily["date"]   = pd.to_datetime(daily["date"])
    daily["season"] = daily["date"].dt.month.map(SEASON_MAP)

    start = pd.Timestamp("2023-01-01")
    end   = pd.Timestamp("2025-12-31")
    daily = daily[(daily["date"] >= start) & (daily["date"] <= end)].reset_index(drop=True)

    # Fill residual NaN with seasonal median
    for col in ["hs", "wind_speed"]:
        daily[col] = daily.groupby("season")[col].transform(lambda x: x.fillna(x.median()))
        daily[col] = daily[col].fillna(daily[col].median())

    if len(daily) < N_DAYS:
        extra = _synthetic_weather()
        extra = extra.iloc[len(daily):N_DAYS]
        daily = pd.concat([daily, extra[["date","hs","wind_speed","season"]]], ignore_index=True)

    print(f"[INFO] Real KMA weather loaded: {len(daily)} days | "
          f"Hs mu={daily['hs'].mean():.2f}m | wind mu={daily['wind_speed'].mean():.1f}m/s")
    print(f"[INFO] CTV access rate: {((daily['hs']<=1.5)&(daily['wind_speed']<=10)).mean()*100:.1f}%  "
          f"SOV access rate: {(daily['hs']<=2.5).mean()*100:.1f}%")
    return daily.head(N_DAYS)


def load_weekly_baseline(path="/mnt/user-data/uploads/weekly_cost_baseline.csv"):
    """
    Load LP scheduling weekly cost baseline.
    Returns DataFrame with numeric KRW columns.
    """
    try:
        df = pd.read_csv(path)
        for col in ["ShipCost","PortCost","LaborCost","DowntimeCost","PartsCost","TotalCost"]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",","").str.strip(), errors="coerce")
        df["TotalCost"] = df["TotalCost"].fillna(0)
        print(f"[INFO] Weekly baseline loaded: {len(df)} weeks | "
              f"Annual avg TotalCost={df['TotalCost'].mean()/1e6:.1f}M KRW/week")
        return df
    except Exception as e:
        print(f"[WARN] weekly_cost_baseline.csv error: {e}")
        return pd.DataFrame()


def _synthetic_weather(seed_offset=0):
    """Seasonal Markov chain synthetic weather (KMA-calibrated)."""
    local_rng = np.random.default_rng(SEED + seed_offset)
    P_season = {
        "Winter": np.array([[0.55,0.25,0.15,0.05],[0.20,0.40,0.30,0.10],
                             [0.10,0.25,0.45,0.20],[0.05,0.15,0.35,0.45]]),
        "Spring": np.array([[0.65,0.20,0.12,0.03],[0.25,0.45,0.25,0.05],
                             [0.15,0.30,0.45,0.10],[0.10,0.20,0.40,0.30]]),
        "Summer": np.array([[0.75,0.15,0.08,0.02],[0.30,0.50,0.18,0.02],
                             [0.20,0.35,0.40,0.05],[0.15,0.25,0.45,0.15]]),
        "Autumn": np.array([[0.60,0.22,0.13,0.05],[0.22,0.42,0.28,0.08],
                             [0.12,0.28,0.44,0.16],[0.08,0.18,0.38,0.36]]),
    }
    hs_p   = {"Calm":(0.7,0.2),"Moderate":(1.3,0.3),"Rough":(2.0,0.4),"Extreme":(3.2,0.6)}
    wind_p = {"Calm":(5.0,1.5),"Moderate":(8.0,1.5),"Rough":(11.5,2.0),"Extreme":(16.0,3.0)}
    states = ["Calm","Moderate","Rough","Extreme"]
    dates, state, rows = pd.date_range("2023-01-01", periods=N_DAYS), 0, []
    for d in dates:
        season = SEASON_MAP[d.month]
        state  = local_rng.choice(4, p=P_season[season][state])
        s      = states[state]
        rows.append({"date": d,
                     "hs":         max(0.1, local_rng.normal(*hs_p[s])),
                     "wind_speed": max(0.5, local_rng.normal(*wind_p[s])),
                     "season":     season})
    return pd.DataFrame(rows)


def load_components(path="component_params.csv"):
    try:
        return pd.read_csv(path)
    except Exception:
        comps = list(CRITICALITY_MAP.keys())
        return pd.DataFrame({
            "component":   comps,
            "beta":        [BETA]*len(comps),
            "eta_weeks":   [ETA_WEEKS]*len(comps),
            "criticality": [CRITICALITY_MAP[c] for c in comps],
        })


print("=" * 70)
print("  HMDP RESS Revision -- experiment_core.py  [real KMA data version]")
print("  weather_hourly_raw.csv -> hourly -> daily (Hs interpolated, 1096d)")
print("=" * 70)
