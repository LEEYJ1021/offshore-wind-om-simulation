# -*- coding: utf-8 -*-
"""
=====================================================================================
03_iso_reliability_benchmark.py  (protocol v11, single file)
Iso-reliability benchmarking of maintenance policies for capacity-constrained fleets
=====================================================================================
Goal: computational protocol with algorithms/complexity, strong comparison baselines,
extensive numerical experiments and reproducibility (Computers & Operations Research).
The v10 engine / common-random-number streams / grids / seeds are kept unchanged
(with a regression check) and the following blocks are added.

 [M1]  CORE      : as v10 (4 scenarios, N_BOOT=20000, paired EAT-TBMm CI, oracle recovery, 2x2 attribution)
 [M1b] Baselines : (a) IDX  cost-weighted hazard-index policy (nominal Weibull)
                   (b) IDXL index policy + learned wear rate
                   (c) PS   policy search (CEM): log-linear trigger on age / learned wear / tier / queue,
                            trained on training seeds
                   (d) SEN* degradation-sensor policies: iid noise / autocorrelated (AR1) / biased
                   -> all compared with CORE at the SAME target failure levels (q25/q50/q75), same sample (50 x 6 seeds)
 [M2]  Structural misspecification: 10 settings (incl. baseline) x S1,S3, same sample as CORE (50 reps / 6 seeds).
                   Validity criterion fixed beforehand; all settings reported + validity flag + benefit retention
 [M3]  Cost sensitivity : +-50% single / joint re-pricing of the 4 cost components, 400 draws
                   (equal-reliability operating points do not depend on costs)
 [M4]  Information ladder: TBM<EAT<IDX<IDXL<PS<LTBM<BAYe<SENSOR<ORA compared directly at the CORE targets
 [M5]  Capacity   : S1,S3 x (CTV-only / adaptive) capacity sweep (incl. 5), fixed parameters + per-capacity re-tuning,
                   queue metrics included
 [M6]  Scale      : fleet size N=10..200 (capacity fixed vs proportional), scale dependence of the benefit
 [M7]  Tuning budget: tier-specific calendar (TBM3), 2-parameter EAT2/BAYe2 -> does one-parameter tuning drive conclusions?
 [M8]  Theory checks: heuristic A (homogeneous + deterministic repair: effective-age trigger = per-component calendar),
                   check B (first-order consistency / bias of the slope correction)
 [M9]  Diagnostic : why BAYe's cost-optimal failure count is ~100 below BAY's (flat cost frontier region)
 [M10] Computational cost: measured time of one simulation, N scaling exponent, simulations per stage
 [M11] Manuscript number sheet, figures (5 main-text figures PDF+PNG), reproducibility manifest
 (option) environment variable CALIB_TARGET = annual critical failures per turbine -> adds an external calibration scenario S4

Usage:  cd results/iso_reliability && python ../../scripts/iso_reliability/03_iso_reliability_benchmark.py
        (writes ./outputs_v11)
        V11_FAST=1 first to check the whole flow, then the full run.
        Environment variables: NPROC, V11_HEADLESS=1 (no plot windows).
Run time: about 23 min with 23 worker processes (measured 1,357 s); nohup recommended.
NOTE: all cost / RF / heterogeneity / misspecification coefficients are ASSUMPTIONS. The learner's assumption
      (wear rate is component-specific) is deliberately broken in M2 and tested.
"""
import os, sys, time, json, platform, itertools, warnings
from dataclasses import dataclass, replace
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib
if os.environ.get("V11_HEADLESS"): matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")

# ============================== 0. Settings ==============================
FAST     = bool(int(os.environ.get("V11_FAST", "0")))
SMOKE    = bool(os.environ.get("EAT_SMOKE"))            # internal check (off by default)
HEADLESS = bool(os.environ.get("V11_HEADLESS"))
OUT      = os.path.join(os.getcwd(), "outputs_v11")
KMA_PATHS = ["../../data/raw/weather_hourly_raw.csv",    # repository-relative (run from results/iso_reliability)
             "data/raw/weather_hourly_raw.csv",          # run from repository root
             "/home/yjlee/Research/RESS_SnS_R1/data/raw/weather_hourly_raw.csv",
             "/mnt/user-data/uploads/weather_hourly_raw.csv"]
V8_RAW   = [os.path.join(os.getcwd(), "outputs_v8", "X1_raw.csv"), "/home/yjlee/Research/RESS_SnS_R1/outputs_v8/X1_raw.csv"]
NPROC    = int(os.environ.get("NPROC", max(1, (os.cpu_count() or 2) - 1)))
CALIB_TARGET = os.environ.get("CALIB_TARGET")
def F(full, fast): return fast if FAST else full

N_REP, N_TRAIN, N_CAL, CAL_ITERS = F(50, 3), F(6, 1), F(10, 2), F(6, 2)
N_BOOT = F(20000, 300)
N_REP_SC, N_TRAIN_SC = F(30, 3), F(4, 1)                # scale / capacity re-tuning blocks
N_COST_DRAWS = F(400, 30)
TBM_GRID_WK = F([6, 8, 10, 13, 16, 20, 24, 28, 32, 40, 52], [13, 26, 40])
SHIFT_GRID  = F([-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25], [-0.10, 0.0, 0.10])
TBM_GRID_B  = F([8, 12, 16, 20, 26, 32, 44], [13, 26, 40])
SHIFT_GRID_B = F([-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20], [-0.10, 0.0, 0.10])
IDX_GRID = F([0.02, 0.035, 0.06, 0.10, 0.16, 0.25, 0.40, 0.60, 0.90, 1.40], [0.1, 0.4, 1.0])
PS_GRID  = F([-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0], [-0.5, 0.0, 0.5])
PS_POP, PS_ITERS, PS_ELITE, PS_SEEDS = F(16, 3), F(6, 1), F(4, 1), F(3, 1)
CAP_SWEEP  = F([1, 2, 3, 4, 5, 6, 8, 12, 24], [2, 4, 12])
CAP_RETUNE = F([2, 4, 5, 6, 8, 12], [4])
SCALE_N = F([10, 25, 50, 100, 200], [4, 8])
os.makedirs(OUT, exist_ok=True)

N_T, N_BURN, N_MEAS = (50, 730, 1096) if not SMOKE else (8, 60, 150)
N_TOT = N_BURN + N_MEAS
BETA, ETA_D = 2.5, 80 * 7.0
MIN_GAP, KAPPA = 60, 10.0
R_LO, R_HI = 0.60, 0.95

COMP = ["Gearbox","Generator","Blades","Tower","Hub","PitchHyd","Controls","Converter","Transformer","Yaw","Other"]
C = len(COMP)
TIER = np.array([0,0,0,0,0, 1,1,1,1,1, 2]); CRIT = TIER == 0
W = np.array([3., 2., 1.])[TIER]
DER = np.array([[.40,.75,1.0],[.35,.80,1.0],[.20,.60,1.0],[.10,.50,1.0],[.30,.65,1.0],
                [.15,.45,.70],[.05,.20,.40],[.20,.55,.85],[.20,.60,.90],[.05,.20,.40],[.02,.05,.15]])
DFAIL = DER[:, 2]
RF_BY, DUR, COSTM = np.array([.35,.55,.90]), np.array([1,3,7]), np.array([1.,3.,8.])
CORR_C = np.array([120., 40., 8.])[TIER]; PREV_C = np.array([25., 10., 2.])[TIER]
REV_TD = 10 * 24 * 0.40 * 0.150
CTV_COST, SOV_COST, CHARTER = 5.0, 30.0, 5.0
CAP_CTV, CAP_SOV = 4, 8
SEV_P0 = (0.80, 0.15, 0.05)
DEF_THETA = (0.72, 0.65, 0.55)
# value of maintenance for the index policy: corrective repair (mean severity multiplier 1.65) + output loss during
# about 10 days of waiting, relative to the planned cost (multiplier 1.65)
LVAL = CORR_C * 1.65 + REV_TD * DFAIL * (1.6 + 10.0)
IDX_RATIO = LVAL / (PREV_C * 1.65)

# ---- weather ----
WX_P  = np.array([[.518,.290,.123,.069],[.357,.321,.239,.083],[.151,.312,.407,.131],[.133,.244,.343,.280]])
HS_MU, HS_SD = np.array([.7,1.3,2.0,3.2]), np.array([.2,.3,.4,.6])
WS_MU, WS_SD = np.array([5.,8.,11.5,16.]), np.array([1.5,1.5,2.,3.])
def try_load_kma(path):
    try:
        raw = pd.read_csv(path).rename(columns={"일시":"datetime","풍속(m/s)":"wind_speed","유의파고(m)":"hs"})
        raw["datetime"] = pd.to_datetime(raw["datetime"])
        d = (raw.set_index("datetime")[["hs","wind_speed"]].sort_index().interpolate(method="time", limit=6).resample("D").mean().dropna())
        st = np.digitize(d["hs"].values, [0.9, 1.5, 2.5]); P_ = np.zeros((4,4))
        for a, b in zip(st[:-1], st[1:]): P_[a, b] += 1
        for i in range(4): P_[i] = P_[i]/P_[i].sum() if P_[i].sum() > 0 else WX_P[i]
        f = lambda col, k, dflt: d[col].values[st == k].mean() if (st == k).any() else dflt
        g = lambda col, k, dflt: d[col].values[st == k].std() if (st == k).sum() > 2 else dflt
        return (P_, np.array([f("hs",k,HS_MU[k]) for k in range(4)]), np.array([g("hs",k,HS_SD[k]) for k in range(4)]),
                np.array([f("wind_speed",k,WS_MU[k]) for k in range(4)]), np.array([g("wind_speed",k,WS_SD[k]) for k in range(4)]), len(d))
    except Exception:
        return None
_k = None
for _p in KMA_PATHS:
    if os.path.exists(_p):
        _k = try_load_kma(_p)
        if _k is not None: break
WX_DAYS = None
if _k is not None: WX_P, HS_MU, HS_SD, WS_MU, WS_SD, WX_DAYS = _k; print(f"[weather] transition matrix re-estimated from KMA observations ({WX_DAYS} days) -- check against the day count stated in the manuscript")
else: print("[weather] KMA CSV not found -> using the built-in fallback transition matrix (results will NOT match the reported tables)")
_CUM = np.cumsum(WX_P, axis=1)
_WX = {}
def get_weather(seed):
    if seed in _WX: return _WX[seed]
    r = np.random.default_rng(seed*2 + 1); u = r.random(N_TOT); s = 0; st = np.empty(N_TOT, int)
    for d in range(N_TOT):
        s = min(int(np.searchsorted(_CUM[s], u[d])), 3); st[d] = s
    hs = np.maximum(.1, r.normal(HS_MU[st], HS_SD[st])); ws = np.maximum(.5, r.normal(WS_MU[st], WS_SD[st]))
    _WX[seed] = ((hs <= 1.5) & (ws <= 10.0), (hs <= 2.5) & (ws <= 15.0), st)
    return _WX[seed]

MG = np.exp(np.linspace(np.log(0.45), np.log(2.2), 13)); LAMG = MG ** (-BETA)

# ============================== 1. Simulator ==============================
@dataclass(frozen=True)
class Cfg:
    trigger: str = "EAT"          # EAT | TBM | LTBM
    rf_stoch: bool = True; access: str = "weather"; vessel: str = "adaptive"
    tbm_T: tuple = None; rf_scale: float = 1.0; sev_p: tuple = SEV_P0
    bayes: bool = False; oracle: bool = False
    sensor: bool = False; sen_sd: float = 0.3; sen_rho: float = 0.0; sen_bias: float = 0.0   # degradation sensor (iid / AR1 / bias)
    theta: tuple = DEF_THETA
    frail_sd: float = 0.0; eta_mult: float = 1.0; corr_mult: float = 1.0
    prior_sd: float = 0.35; prior_mu: float = 0.0
    eb: bool = False
    forget: float = 1.0; reset_rep: bool = False
    beta_true: float = 2.5                         # true Weibull shape (policies assume 2.5) -> structural misspecification
    rep_acc: float = 0.0                           # wear rate x(1+rep_acc*RF) at every repair
    renew: bool = False                            # redraw wear rate at replacement
    drift: float = 0.0                             # time drift of the wear rate
    het_dist: str = "lognormal"                    # lognormal | bimodal | student
    cap_ctv: int = CAP_CTV; cap_sov: int = CAP_SOV
    n_t: int = N_T                                 # number of turbines
    idx: str = None; idx_g: float = 0.2            # cost-weighted hazard-index policy: "nom" | "learn"
    ps_w: tuple = None; ps_shift: float = 0.0      # policy-search (CEM) log-linear trigger weights

def simulate(cfg, seed):
    nT = cfg.n_t
    ctv_ok, sov_ok, wst = get_weather(seed)
    if cfg.access == "uncon": acc = np.ones(N_TOT, bool)
    elif cfg.vessel == "ctv": acc = ctv_ok
    else: acc = ctv_ok | sov_ok
    rng = np.random.default_rng(10_000 + seed)                       # CRN (same streams as v8)
    R0 = rng.uniform(R_LO, R_HI, (nT, C)); phase = rng.random((nT, C)); frail = rng.normal(0, 1, (nT, C))
    Uf = rng.random((N_TOT, nT, C)); Us = rng.random((N_TOT, nT, C))
    r3 = np.random.default_rng(30_000 + seed)                        # separate stream for extensions (does not affect earlier results)
    ub = r3.random((nT, C)); zt = r3.standard_t(3, (nT, C)) / np.sqrt(3.0); znew = r3.normal(0, 1, (nT, C))
    sn = r3.normal(0, 1, (N_TOT // 30 + 2, nT, C)); sb = r3.normal(0, 1, (nT, C))   # sb appended at the end of the stream (earlier values unchanged)
    if cfg.sen_rho > 0:
        a_, b_ = cfg.sen_rho, np.sqrt(1 - cfg.sen_rho ** 2)
        for k_ in range(1, sn.shape[0]): sn[k_] = a_ * sn[k_ - 1] + b_ * sn[k_]
    if cfg.het_dist == "bimodal": z = (np.where(ub < 0.2, 2.0, -0.5) + 0.2 * frail) / np.sqrt(1.04)
    elif cfg.het_dist == "student": z = zt
    else: z = frail
    Bt = cfg.beta_true
    lam_true = (cfg.eta_mult * np.exp(cfg.frail_sd * z)) ** (-Bt)

    sp = np.array(cfg.sev_p); cum = np.cumsum(sp); RFv = np.minimum(RF_BY * cfg.rf_scale, 0.95)
    rf_fix = float((sp * RFv).sum()); dur_fix = max(1, int(round((sp * DUR).sum()))); der_fix = DER @ sp; cm_fix = float((sp * COSTM).sum())
    rho_bar = rf_fix
    THC = np.clip(np.array(cfg.theta)[TIER], 0.05, 0.97)
    LP0 = -0.5 * ((np.log(MG) - cfg.prior_mu) / cfg.prior_sd) ** 2; LP0 = LP0 - np.log(np.exp(LP0).sum())
    LPtab = np.tile(LP0, (C, 1)); LOGM = np.log(MG)
    if cfg.ps_w is not None:
        pw = np.array(cfg.ps_w, float); ps_bias = pw[1:4][TIER] + cfg.ps_shift

    t = ETA_D * (-np.log(R0)) ** (1/BETA); t_init = t.copy()
    failed = np.zeros((nT, C), bool); pending = np.zeros((nT, C), bool)
    t_trig = np.zeros((nT, C)); under = np.zeros((nT, C), int); under_der = np.zeros((nT, C))
    last_exec = -phase * 200.0
    TBM = cfg.trigger == "TBM"; LTBM = cfg.trigger == "LTBM"; learn = cfg.bayes
    if TBM: Tvec = np.array(cfg.tbm_T, float); last_trig = -phase * Tvec[None, :]
    if learn: ll = np.zeros((nT, C, len(MG)))
    if LTBM:
        lam0 = float((np.exp(LP0) * LAMG).sum())
        due = np.maximum(0.0, ETA_D * (-np.log(THC)[None, :] / lam0) ** (1/BETA) - t_init)
    lam_hat = None

    pf_day = np.zeros(N_MEAS); crit_day = np.zeros(N_MEAS)
    nfail = ncorr = nplan = 0; plan_cnt = np.zeros(C); c_prev = c_corr = c_ves = 0.0; vdays = 0; fcd = 0; qsum = 0

    for day in range(N_TOT):
        meas = day >= N_BURN; di = day - N_BURN
        mult = 1.0 + cfg.drift * (day / N_TOT)
        under = np.maximum(under - 1, 0)
        op = (~failed) & (under == 0)
        H0 = (t/ETA_D) ** BETA; H1 = ((t + 1)/ETA_D) ** BETA; dH = H1 - H0
        dHt = dH if Bt == BETA else ((t + 1)/ETA_D) ** Bt - (t/ETA_D) ** Bt
        fl = op & (Uf[day] < 1 - np.exp(-lam_true * mult * dHt))
        if learn:
            if cfg.forget < 1.0: ll *= cfg.forget
            ex = LAMG[None, None, :] * dH[..., None]
            ll += np.where(fl[..., None], np.log(1 - np.exp(-ex) + 1e-12), -ex) * op[..., None]
            lp = ll + LPtab[None, :, :]; lp = lp - lp.max(2, keepdims=True); w = np.exp(lp); w /= w.sum(2, keepdims=True)
            lam_hat = (w * LAMG[None, None, :]).sum(2)
            if cfg.eb and day >= 60 and day % 30 == 0:
                pm = (w * LOGM).sum(2); pv = (w * (LOGM - pm[..., None]) ** 2).sum(2)
                mu_c = np.clip(pm.mean(0), -0.7, 0.7); var_c = pv.mean(0) + ((pm - mu_c) ** 2).mean(0)
                sd_c = np.clip(np.sqrt(var_c), 0.15, 0.9)
                LPtab = -0.5 * ((LOGM[None, :] - mu_c[:, None]) / sd_c[:, None]) ** 2
                LPtab = LPtab - np.log(np.exp(LPtab).sum(1, keepdims=True))
        if fl.any():
            if meas: nfail += int(fl.sum()); crit_day[di] = int(fl[:, CRIT].sum())
            failed |= fl; pending |= fl
        t[op & ~fl] += 1
        cand = op & ~fl & ~pending
        if TBM:
            trig = cand & ((day - last_trig) >= Tvec[None, :])
        elif LTBM:
            trig = cand & (day >= due) & ((day - last_exec) >= MIN_GAP)
        else:
            Hb = (t/ETA_D) ** BETA
            Hbt = Hb if Bt == BETA else (t/ETA_D) ** Bt
            gap = (day - last_exec) >= MIN_GAP
            if cfg.idx:
                lam_u = lam_hat if cfg.idx == "learn" else 1.0
                hwk = 7 * (BETA/ETA_D) * lam_u * (np.maximum(t, 1e-6)/ETA_D) ** (BETA - 1)
                trig = cand & ((hwk * IDX_RATIO[None, :]) >= cfg.idx_g) & gap
            elif cfg.ps_w is not None:
                f1 = np.log(np.maximum(lam_hat * Hb, 1e-9)); qf = pending.sum() / (nT * C * 0.02)
                trig = cand & ((pw[0] * f1 + ps_bias[None, :] + pw[4] * qf) >= 0) & gap
            else:
                if cfg.oracle:   Rused = np.exp(-lam_true * mult * Hbt)
                elif cfg.sensor: Rused = np.exp(-lam_true * mult * Hbt * np.exp(cfg.sen_sd * sn[day // 30] + cfg.sen_bias * sb))
                elif learn:      Rused = np.exp(-lam_hat * Hb)
                else:            Rused = np.exp(-Hb)
                trig = cand & (Rused <= THC[None, :]) & gap
        if trig.any():
            pending |= trig; t_trig[trig] = day
            if TBM: last_trig[trig] = day
        npend = int(pending.sum())
        if meas: qsum += npend
        if npend > 0 and acc[day]:
            flat = np.flatnonzero(pending.ravel())
            cap = 10**9 if cfg.access == "uncon" else (cfg.cap_ctv if ctv_ok[day] else cfg.cap_sov)
            if flat.size > cap:
                h_wk = 7 * (BETA/ETA_D) * (t.ravel()[flat]/ETA_D) ** (BETA - 1)
                sc = failed.ravel()[flat] * 1000 + W[flat % C] * (1 + KAPPA * h_wk)
                sel = flat[np.argsort(-sc)[:cap]]
            else:
                sel = flat
            ti, ci = sel // C, sel % C
            wf = failed[ti, ci].copy(); pl = ~wf
            if cfg.rf_stoch:
                s = np.minimum(np.searchsorted(cum, Us[day][ti, ci]), 2); rf = RFv[s]; dur = DUR[s]; der = DER[ci, s]; cm = COSTM[s]
            else:
                s = np.full(sel.size, -1); rf = rf_fix; dur = dur_fix; der = der_fix[ci]; cm = cm_fix
            t[ti, ci] *= (1 - rf)                                         # Kijima Type II
            if cfg.rep_acc > 0: lam_true[ti, ci] *= (1.0 + cfg.rep_acc * rf)
            if cfg.rf_stoch and (cfg.renew or (learn and cfg.reset_rep)):
                m_ = (s == 2)
                if m_.any():
                    if cfg.renew: lam_true[ti[m_], ci[m_]] = (cfg.eta_mult * np.exp(cfg.frail_sd * znew[ti[m_], ci[m_]])) ** (-Bt)
                    if learn and cfg.reset_rep: ll[ti[m_], ci[m_], :] = 0.0
            under[ti, ci] = dur; under_der[ti, ci] = der; failed[ti, ci] = False; pending[ti, ci] = False
            if meas:
                c_prev += float((PREV_C[ci] * cm * pl).sum()); c_corr += float((CORR_C[ci] * cm * wf).sum()) * cfg.corr_mult
                ncorr += int(wf.sum()); nplan += int(pl.sum()); np.add.at(plan_cnt, ci[pl], 1)
                c_ves += CTV_COST if (cfg.access == "uncon" or cfg.vessel == "ctv" or ctv_ok[day]) else SOV_COST; vdays += 1
            last_exec[ti, ci] = day
            if TBM and wf.any(): last_trig[ti[wf], ci[wf]] = day
            if LTBM:
                due[ti, ci] = day + np.clip(rho_bar * ETA_D * (-np.log(THC[ci]) / lam_hat[ti, ci]) ** (1/BETA), MIN_GAP, 730)
        if meas and cfg.vessel == "ctv" and cfg.access != "uncon": c_ves += CHARTER
        if meas:
            d_tot = (failed * DFAIL[None, :]).sum(1) + ((under > 0) * under_der).sum(1)
            pf_day[di] = np.maximum(1 - d_tot, 0).mean(); fcd += int(failed.sum())

    loss = float((1 - pf_day).sum() * nT * REV_TD); tot = c_prev + c_corr + c_ves + loss
    out = dict(crit=crit_day.sum(), fails=nfail, aep=pf_day.mean()*100, planned=nplan, corrective=ncorr, vessel_days=vdays, fail_comp_days=fcd, queue_mean=qsum / N_MEAS,
               cost_prev=c_prev/1e3, cost_corr=c_corr/1e3, cost_vessel=c_ves/1e3, cost_loss=loss/1e3, cost_total=tot/1e3, plan_cnt=plan_cnt)
    if learn and cfg.frail_sd > 0:
        out["bayes_rho"] = float(stats.spearmanr(np.log(lam_hat).ravel(), np.log(lam_true).ravel())[0])
    return out

# ============================== 2. Parallel execution / statistics helpers ==============================
_POOL = None
def _sim_job(a): return simulate(*a)
def pmap_sim(jobs):
    global _POOL
    if NPROC > 1 and len(jobs) > 1:
        try:
            if _POOL is None: _POOL = mp.get_context("fork").Pool(NPROC)
            return _POOL.map(_sim_job, jobs, chunksize=1)
        except Exception as e:
            print("   [parallel execution failed -> serial fallback]", repr(e)[:80]); _POOL = None
    return [simulate(*j) for j in jobs]
def run_many(cfgs, seeds):
    res = pmap_sim([(c, s) for c in cfgs for s in seeds]); n = len(seeds)
    return [res[i*n:(i+1)*n] for i in range(len(cfgs))]
def runs(cfg, seeds): return run_many([cfg], seeds)[0]
def arr(rr, m): return np.array([r[m] for r in rr], float)

_BI = {}
def bidx(n, seed=1):
    k = (n, seed)
    if k not in _BI: _BI[k] = np.random.default_rng(seed).integers(0, n, (N_BOOT, n))
    return _BI[k]
def ci_diff(a, b):
    d = np.asarray(a, float) - np.asarray(b, float); n = len(d); bt = d[bidx(n)].mean(1)
    return d.mean(), np.percentile(bt, 2.5), np.percentile(bt, 97.5), max(2*min((bt > 0).mean(), (bt < 0).mean()), 1/N_BOOT)
def ratio_ci(num, den):
    n = len(num); I = bidx(n, 5); bt = num[I].mean(1) / den[I].mean(1)
    return num.mean()/den.mean(), np.percentile(bt, 2.5), np.percentile(bt, 97.5)
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o): run = max(run, (m - i) * p[j]); adj[j] = min(run, 1)
    return adj
def factorial_effects(Y, fnames):
    k = Y.ndim - 1; n = Y.shape[0]; g = np.indices((2,)*k) * 2 - 1; per, lab = [], []
    for r in range(1, k + 1):
        for sub in itertools.combinations(range(k), r):
            sg = np.prod([g[i] for i in sub], axis=0); per.append((Y * sg[None]).reshape(n, -1).sum(1) / 2**(k - 1)); lab.append("x".join(fnames[i] for i in sub))
    per = np.array(per); bt = per[:, bidx(n, 7)].mean(2)
    return pd.DataFrame({"term": lab, "effect": per.mean(1), "lo": np.percentile(bt, 2.5, axis=1), "hi": np.percentile(bt, 97.5, axis=1)})
def P(title, df, nd=2):
    print("\n" + title); print(df.round(nd).to_string(index=False) if isinstance(df, pd.DataFrame) else df)
SHEET = []                                             # manuscript number sheet (saved to file)
def S(line): SHEET.append(line); print(line)

# ---- policy construction ----
def tbm_cfg(base, T): return replace(base, trigger="TBM", tbm_T=tuple(T), bayes=False, oracle=False, eb=False, sensor=False, forget=1.0, reset_rep=False, idx=None, ps_w=None)
def pol(base, kind, shift=0.0, sen_sd=0.3, sen_rho=0.0, sen_bias=0.0, w=None):
    th = tuple(np.clip(np.array(DEF_THETA) + shift, 0.05, 0.97))
    k = dict(trigger="EAT", bayes=False, oracle=False, eb=False, sensor=False, forget=1.0, reset_rep=False, theta=th, idx=None, ps_w=None)
    if   kind == "EAT":  pass
    elif kind == "BAY":  k.update(bayes=True)
    elif kind == "BAYe": k.update(bayes=True, eb=True)
    elif kind == "ORA":  k.update(oracle=True)
    elif kind == "SEN":  k.update(sensor=True, sen_sd=sen_sd, sen_rho=sen_rho, sen_bias=sen_bias)
    elif kind == "LTBM": k.update(trigger="LTBM", bayes=True)
    elif kind == "IDX":  k.update(idx="nom", idx_g=shift)
    elif kind == "IDXL": k.update(idx="learn", idx_g=shift, bayes=True, eb=True)
    elif kind == "PS":   k.update(bayes=True, eb=True, ps_w=tuple(w), ps_shift=shift)
    else: raise ValueError(kind)
    return replace(base, **k)
def mk_tbm(b, g): return tbm_cfg(b, [g*7.0]*C)
def mk_pol(kind, **kw): return lambda b, g: pol(b, kind, g, **kw)
def specs_main(tg, sg, fams=("EAT", "LTBM", "BAY", "BAYe", "ORA")):
    sp = {"TBM": dict(make=mk_tbm, grid=tg, inc=True)}
    for f in fams: sp[f] = dict(make=mk_pol(f), grid=sg, inc=False)
    return sp

def calibrate_T(pcfg, n=N_CAL, iters=CAL_ITERS):
    """Algorithm 2: fixed-point calibration of a per-component calendar to the planned-task counts of pcfg."""
    seeds = [9000 + k for k in range(n)]; nT = pcfg.n_t
    cnt = sum(r["plan_cnt"] for r in runs(pcfg, seeds)); target = np.maximum(cnt / n, 1.0); T = np.clip(N_MEAS / np.maximum(target / nT, 0.5), 30, 730)
    for _ in range(iters):
        c2 = np.maximum(sum(r["plan_cnt"] for r in runs(tbm_cfg(pcfg, T), seeds)) / n, 1.0)
        if abs(c2.sum() / target.sum() - 1) < 0.015 and np.abs(c2 / target - 1).max() < 0.05: break
        T = np.clip(T * (c2 / target), 30, 730)
    return tuple(T)

# ---- iso-reliability engine (Algorithm 1: frontier interpolation + local-slope correction) ----
COMPS = ["cost_prev", "cost_corr", "cost_vessel", "cost_loss"]; W1 = np.ones(4)
ISO = ("q25", "q50", "q75")
def fr_cost(fr, w): return fr[COMPS].values @ np.asarray(w)
def mono(fr, inc):
    c = fr.sort_values("p").crit.values; return np.maximum.accumulate(c) if inc else np.minimum.accumulate(c)
def param_at(fr, target, inc):
    fr = fr.sort_values("p"); c = mono(fr, inc); p = fr.p.values.astype(float)
    if not inc: c, p = c[::-1], p[::-1]
    return float(np.interp(target, c + 1e-9 * np.arange(len(c)), p))
def local_slope(fr, target, w):
    c = fr.crit.values; k = np.argsort(np.abs(c - target))[:3]
    if np.ptp(c[k]) < 1e-6: return 0.0
    return float(np.polyfit(c[k], fr_cost(fr, w)[k], 1)[0])
def build_frontier(specs, names, base, train):
    cfgs, idx = [], []
    for nm in names:
        for g in specs[nm]["grid"]: cfgs.append(specs[nm]["make"](base, g)); idx.append((nm, g))
    out = run_many(cfgs, train); FR = {}
    for nm in names:
        rows = [dict(p=g, crit=arr(rr, "crit").mean(), aep=arr(rr, "aep").mean(), **{c: arr(rr, c).mean() for c in COMPS}) for (n_, g), rr in zip(idx, out) if n_ == nm]
        FR[nm] = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    return FR

def iso_study(base, specs, train, test, ops=ISO, with_opt=True):
    names = list(specs); FR = build_frontier(specs, names, base, train)
    lo = max(mono(FR[n], specs[n]["inc"]).min() for n in names); hi = min(mono(FR[n], specs[n]["inc"]).max() for n in names); fb = False
    if hi - lo < 5:
        ref = "TBM" if "TBM" in names else names[0]; c_ = mono(FR[ref], specs[ref]["inc"]); lo, hi = c_.min(), c_.max(); fb = True
    ops_q = {"q25": .25, "q50": .50, "q75": .75}; TARG = {op: lo + ops_q[op] * (hi - lo) for op in ops}
    PAR = {}
    for nm in names:
        for op in ops: PAR[(nm, op)] = param_at(FR[nm], TARG[op], specs[nm]["inc"])
        PAR[(nm, "opt")] = float(FR[nm].p.values[int(np.argmin(fr_cost(FR[nm], W1)))])
    tops = list(ops) + (["opt"] if with_opt else []); cfgs, idx = [], []
    for nm in names:
        for op in tops: cfgs.append(specs[nm]["make"](base, PAR[(nm, op)])); idx.append((nm, op))
    out = run_many(cfgs, test); RUN = {k: rr for k, rr in zip(idx, out)}
    return dict(FR=FR, TARG=TARG, PAR=PAR, RUN=RUN, names=names, specs=dict(specs), fallback=fb, test=test, ops=tops, base=base)

def add_policies(res, base, specs_new, train, test):
    """Add new policies to an existing study (res) at the SAME target failure levels (targets are not recomputed).
    cov=False if the frontier does not cover a target."""
    names = list(specs_new); FR = build_frontier(specs_new, names, base, train); cov = {}
    res["FR"].update(FR); res["specs"].update(specs_new)
    for nm in names:
        inc = specs_new[nm]["inc"]; c = mono(FR[nm], inc)
        for op in ISO:
            res["PAR"][(nm, op)] = param_at(FR[nm], res["TARG"][op], inc); cov[(nm, op)] = bool(c.min() <= res["TARG"][op] <= c.max())
        res["PAR"][(nm, "opt")] = float(FR[nm].p.values[int(np.argmin(fr_cost(FR[nm], W1)))])
    tops = list(ISO) + ["opt"]; cfgs, idx = [], []
    for nm in names:
        for op in tops: cfgs.append(specs_new[nm]["make"](base, res["PAR"][(nm, op)])); idx.append((nm, op))
    out = run_many(cfgs, test)
    for k, rr in zip(idx, out): res["RUN"][k] = rr
    res["names"] += names; res.setdefault("cov", {}).update(cov)

def cost_vec(res, nm, op, w=W1):
    rr = res["RUN"][(nm, op)]; tot = sum(w[i] * arr(rr, c) for i, c in enumerate(COMPS))
    if op in res["TARG"]:
        tot = tot - local_slope(res["FR"][nm], res["TARG"][op], w) * (arr(rr, "crit") - res["TARG"][op])
    return tot
def metric_vec(res, nm, op, m, w=W1): return cost_vec(res, nm, op, w) if m == "cost" else arr(res["RUN"][(nm, op)], m)
def contrast(res, a, oa, b, ob, m="cost", w=W1): return ci_diff(metric_vec(res, a, oa, m, w), metric_vec(res, b, ob, m, w))

t0 = time.time(); pd.set_option("display.width", 240); pd.set_option("display.max_columns", 60)
def ts(): return f"{time.time()-t0:.0f}s"
TEST = list(range(1, N_REP + 1)); TRAIN = [3000 + k for k in range(N_TRAIN)]
SCEN = {"S0 homogeneous": dict(frail_sd=0.0, eta_mult=1.0), "S1 hetero": dict(frail_sd=0.35, eta_mult=1.0),
        "S2 hetero+fast-wear": dict(frail_sd=0.35, eta_mult=0.8), "S3 strong-hetero": dict(frail_sd=0.60, eta_mult=1.0)}
SC3 = ["S1 hetero", "S2 hetero+fast-wear", "S3 strong-hetero"]
print(f"[settings] FAST={FAST} SMOKE={SMOKE} NPROC={NPROC} N_REP={N_REP} N_TRAIN={N_TRAIN} N_BOOT={N_BOOT}")

# (option) external calibration scenario: eta_mult matching annual critical failures per turbine (TBM, 26 weeks)
def calibrate_eta(target, frail_sd=0.35):
    lo, hi = 0.4, 2.5
    for _ in range(10):
        mid = float(np.sqrt(lo * hi)); b = Cfg(frail_sd=frail_sd, eta_mult=mid)
        rr = runs(tbm_cfg(b, [26*7.0]*C), [9100, 9101]); rate = arr(rr, "crit").mean() / (N_T * N_MEAS / 365.25)
        if rate > float(target): lo = mid
        else: hi = mid
    return float(np.sqrt(lo * hi))
if CALIB_TARGET:
    em = calibrate_eta(CALIB_TARGET); SCEN["S4 calibrated"] = dict(frail_sd=0.35, eta_mult=em); print(f"[calibration] target {CALIB_TARGET}/turbine-year -> eta_mult={em:.3f} (S4 added)")
BASE = {k: Cfg(**v) for k, v in SCEN.items()}
