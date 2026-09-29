"""Leak-free K-FGI experiment lab.

Timing convention (clean, one-day wait):
  - Row s holds information available after the close of trading day s
    (market data at close s, comments written on day s).
  - A weight decided from row s is traded at the close of day s+1 and
    earns the log return of day s+2.  held[d] = w[d-2].
  - Labels for row s use returns r[s+2 .. s+6]; they are fully realised at
    close s+6, so a model fitted at decision day t may only use rows s <= t-6.

Leakage controls:
  - purged walk-forward training (only fully realised labels)
  - scaler / 0-100 mapping / direction choice from past data only
  - EGARCH: fallback for fit failures and values < 1e-4, no full-sample
    quantile clip (hard cap 0.20 only)
  - development period ends 2023-06-30; 2023-07-01..2025-12-22 is a
    sealed holdout used once at the end.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr, skew, kurtosis, norm

warnings.filterwarnings("ignore")

# Paths are relative to this file: <BASE>/논문용/13_Leakfree_experiments
OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
RES = OUT / "results"

EVAL_START = pd.Timestamp("2016-01-06")
DEV_END = pd.Timestamp("2023-06-30")
HOLD_START = pd.Timestamp("2023-07-01")
EVAL_END = pd.Timestamp("2025-12-22")

FEE = 0.0003
GAP = 6  # label realised at s+6
TREND_MAP = {0: 0.0, 1: 0.35, 2: 0.70, 3: 1.0}
TGT = 0.15 / np.sqrt(252)

KF10 = [f"sub_index{i}" for i in range(2, 8)] + ["sent_composite_ma10", "egv", "vol_regime_high", "vol_ratio"]
DIR10 = np.array([1] * 7 + [-1] * 3)  # +1: greed direction
PRICE = ["rv5", "rv20", "rv60", "mom5", "mom20", "mom60", "ma_ratio_5_20", "ma_ratio_20_60",
         "ma_ratio_60_120", "rsi14", "trend_strength", "cum3"]
FALL = KF10 + PRICE


def _rsi(p: pd.Series, n: int = 14) -> pd.Series:
    dlt = p.diff()
    g = dlt.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-dlt.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))


def load(asset: str | None = None) -> pd.DataFrame:
    """asset=None: KOSPI 200 with the paper's EGARCH cache.
    asset='KOSPI'/'KOSDAQ'/...: that index's close and its own walk-forward EGARCH
    (data/egarch_<asset>.csv); market-wide sub-indices and sentiment are unchanged."""
    raw = pd.read_csv(BASE / "논문용/01_final_dataset/KFG_final_10y.csv", parse_dates=["date"])
    raw = raw.sort_values("date").reset_index(drop=True)
    if asset is None:
        egc = pd.read_csv(BASE / "kfgi_최종/paper_outputs/robustness/cache/robust_egarch_min252.csv")
    else:
        egc = pd.read_csv(OUT / "data" / f"egarch_{asset}.csv", parse_dates=["date"])
        assert (egc["date"].values == raw["date"].values).all()
        raw["kospi200_close"] = egc["close"].values
    assert len(egc) == len(raw)
    d = raw.copy()
    p = d["kospi200_close"]
    r = np.log(p / p.shift(1))
    d["r"] = r
    # EGARCH forecast for day t made with returns through t-1 (cache row t)
    fb = r.fillna(0).rolling(60, min_periods=20).std().shift(1)
    e = egc["egarch_vol"].copy()
    bad = (~np.isfinite(e)) | (e < 1e-4) | (e > 0.20)
    if asset is not None:
        # leak-free guard instead of the full-sample 99.5% clip: a forecast above
        # 3x the recent 60-day realised volatility is treated as a failed fit
        bad = bad | (e > 3 * fb)
    d["eg_fallback"] = bad
    e[bad] = fb[bad]
    d["egv"] = e.shift(-1)  # forecast made after close s (uses returns through s)

    sc = ["sent_norm_w", "sent_strength_w", "sent_std", "neg_z", "effective_n", "heat"]
    d[sc] = d[sc].ffill()
    d["sent_energy"] = d["sent_strength_w"] * d["sent_norm_w"]
    d["sent_composite"] = 0.4 * d["sent_norm_w"] + 0.3 * (-d["neg_z"]) + 0.3 * d["sent_energy"]
    d["sent_composite_ma10"] = d["sent_composite"].rolling(10).mean()

    d["vol_regime_high"] = (d["egv"] > d["egv"].rolling(60).quantile(0.7)).astype(float)
    d.loc[d["egv"].isna(), "vol_regime_high"] = np.nan
    d["vol_ratio"] = d["egv"] / (d["egv"].rolling(60).mean() + 1e-9)
    d["vol_shock"] = d["egv"].pct_change().clip(-5, 5)

    for n in (20, 60, 120):
        d[f"ma{n}"] = p.rolling(n).mean()
    d["trend_strength"] = sum((p > d[f"ma{n}"]).astype(float) for n in (20, 60, 120))
    d.loc[d["ma120"].isna(), "trend_strength"] = np.nan
    d["cum3"] = r.rolling(3).sum()
    d["rv5"] = r.rolling(5).std()
    d["rv20"] = r.rolling(20).std()
    d["rv60"] = r.rolling(60).std()
    d["mom5"] = p.pct_change(5)
    d["mom20"] = p.pct_change(20)
    d["mom60"] = p.pct_change(60)
    d["ma_ratio_5_20"] = p.rolling(5).mean() / d["ma20"] - 1
    d["ma_ratio_20_60"] = d["ma20"] / d["ma60"] - 1
    d["ma_ratio_60_120"] = d["ma60"] / d["ma120"] - 1
    d["rsi14"] = _rsi(p)

    fwd = pd.concat([r.shift(-k) for k in range(2, 7)], axis=1)
    ok = fwd.notna().all(axis=1)
    d["y_ret5"] = fwd.sum(axis=1).where(ok)
    d["y_vol5"] = np.log(np.sqrt((fwd ** 2).mean(axis=1)) + 1e-4).where(ok)
    d["y_tail"] = (d["y_ret5"] < -0.02).astype(float).where(ok)
    return d


# ---------------------------------------------------------------- models

def wf_signed(d: pd.DataFrame, feats, dirs, target, min_train=60, gap=GAP, alpha=1.0,
              logistic=False, center=True) -> np.ndarray:
    """Purged walk-forward sign-constrained ridge / logistic. Returns raw score per row
    (score = standardized features @ L1-normalized coefficients, as in the paper)."""
    X = d[feats].values.astype(float)
    y = d[target].values.astype(float)
    n = len(d)
    valid = ~np.isnan(X).any(axis=1)
    out = np.full(n, np.nan)
    k = len(feats)
    bounds = [(0, None) if s > 0 else ((None, 0) if s < 0 else (None, None)) for s in dirs]
    b = np.zeros(k)
    c0 = 0.0
    for t in range(n):
        if not valid[t]:
            continue
        hi = t - gap
        if hi < 0:
            continue
        idx = np.arange(0, hi + 1)
        m = valid[idx] & ~np.isnan(y[idx])
        idx = idx[m]
        if len(idx) < min_train:
            continue
        mu = X[idx].mean(0)
        sd = X[idx].std(0) + 1e-12
        Z = np.clip((X[idx] - mu) / sd, -3, 3)
        yy = y[idx]
        if logistic:
            s = 2 * yy - 1

            def f(v):
                bb, cc = v[:k], v[k]
                z = s * (Z @ bb + cc)
                return float(np.sum(np.logaddexp(0, -z)) + alpha * bb @ bb)

            def g(v):
                bb, cc = v[:k], v[k]
                z = s * (Z @ bb + cc)
                q = -s / (1 + np.exp(z))
                return np.concatenate([Z.T @ q + 2 * alpha * bb, [q.sum()]])

            res = minimize(f, np.concatenate([b, [c0]]), jac=g, method="L-BFGS-B",
                           bounds=bounds + [(None, None)], options={"maxiter": 300})
            b, c0 = res.x[:k], res.x[k]
        else:
            yc = yy - yy.mean() if center else yy

            def f(bb):
                e = yc - Z @ bb
                return float(e @ e + alpha * bb @ bb)

            def g(bb):
                e = yc - Z @ bb
                return -2 * Z.T @ e + 2 * alpha * bb

            res = minimize(f, b, jac=g, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 300, "ftol": 1e-10})
            b = res.x
        wts = b / (np.abs(b).sum() + 1e-12)
        zt = np.clip((X[t] - mu) / sd, -3, 3)
        out[t] = float(zt @ wts)
    return out


def wf_sklearn(d: pd.DataFrame, feats, target, make_model, kind="clf", min_train=250,
               refit=21, gap=GAP) -> np.ndarray:
    """Purged walk-forward for generic sklearn-like model, refit every `refit` rows.
    Returns risk score (probability for clf, prediction for reg)."""
    X = d[feats].values.astype(float)
    y = d[target].values.astype(float)
    n = len(d)
    valid = ~np.isnan(X).any(axis=1)
    out = np.full(n, np.nan)
    model = None
    last_fit = -10 ** 9
    for t in range(n):
        if not valid[t]:
            continue
        if t - last_fit >= refit or model is None:
            hi = t - gap
            if hi < 0:
                continue
            idx = np.arange(0, hi + 1)
            idx = idx[valid[idx] & ~np.isnan(y[idx])]
            if len(idx) < min_train:
                continue
            if kind == "clf" and len(np.unique(y[idx])) < 2:
                continue
            model = make_model()
            model.fit(X[idx], y[idx])
            last_fit = t
        xt = X[t:t + 1]
        out[t] = float(model.predict_proba(xt)[0, 1]) if kind == "clf" else float(model.predict(xt)[0])
    return out


# ---------------------------------------------------------------- mapping

def expanding_scale(score: np.ndarray, min_past=20, invert=False, full_sample=False) -> np.ndarray:
    """0-100 index via P1/P99 of *past* scores (leak-free). invert=True for risk scores."""
    s = np.asarray(score, float)
    out = np.full(len(s), np.nan)
    if full_sample:
        v = s[~np.isnan(s)]
        p1, p99 = np.percentile(v, 1), np.percentile(v, 99)
        out = 100 * (np.clip(s, p1, p99) - p1) / (p99 - p1 + 1e-12)
    else:
        past = []
        for t in range(len(s)):
            if np.isnan(s[t]):
                continue
            if len(past) >= min_past:
                p1, p99 = np.percentile(past, 1), np.percentile(past, 99)
                out[t] = 100 * (np.clip(s[t], p1, p99) - p1) / (p99 - p1 + 1e-12)
            past.append(s[t])
    if invert:
        out = 100 - out
    return out


def expanding_direction(score, y, gap=GAP, min_pairs=60):
    """momentum flag per row from Spearman(score, y) over past realised pairs."""
    s = np.asarray(score, float)
    y = np.asarray(y, float)
    out = np.ones(len(s), bool)
    for t in range(len(s)):
        hi = t - gap
        if hi < 0:
            continue
        m = ~np.isnan(s[:hi + 1]) & ~np.isnan(y[:hi + 1])
        if m.sum() >= min_pairs:
            rho = spearmanr(s[:hi + 1][m], y[:hi + 1][m]).correlation
            out[t] = bool(rho >= 0)
    return out


def rule_weights(d: pd.DataFrame, K, momentum=True, vol=None, fear=25, greed=65, kmin=0.5, kmax=1.6,
                 use_kfgi=True, use_vt=True, use_regime=True, use_safety=True, cap=1.0) -> pd.Series:
    K = pd.Series(np.asarray(K, float), index=d.index).fillna(50.0)  # no forecast yet -> neutral
    tw = d["trend_strength"].map(TREND_MAP)
    v = d["egv"] if vol is None else pd.Series(np.asarray(vol, float), index=d.index)
    vt = np.sqrt(TGT / (v + 1e-9)).clip(0.1, 2.0) if use_vt else 1.0
    mom = pd.Series(np.broadcast_to(momentum, len(d)), index=d.index)
    up = (kmin + K / 100 * (kmax - kmin)).clip(kmin, kmax)
    dn = (kmax - K / 100 * (kmax - kmin)).clip(kmin, kmax)
    mult = up.where(mom, dn) if use_kfgi else 1.0
    if use_regime:
        bull = (K > greed) & (d["vol_regime_high"] == 0) & (d["trend_strength"] >= 2)
        crisis = (K < fear) | (d["vol_shock"] > 2.0)
        # paper rule: bull is checked first, then crisis
        reg = pd.Series(1.0, index=d.index).mask(crisis, 0.7).mask(bull, 1.2)
    else:
        reg = 1.0
    w = (tw * vt * mult * reg).clip(0, cap)
    if use_safety:
        w = w.mask(d["vol_shock"] > 2.0, 0.0).mask(d["cum3"] < -0.07, 0.0)
    return w


def smooth(w: pd.Series, band=0.0, lam=1.0, safety=None) -> pd.Series:
    """No-trade band + partial adjustment toward target. safety mask forces 0 immediately."""
    wt = w.values.astype(float)
    sf = np.zeros(len(wt), bool) if safety is None else np.asarray(safety, bool)
    out = np.full(len(wt), np.nan)
    prev = 0.0
    for i, x in enumerate(wt):
        if np.isnan(x):
            out[i] = np.nan
            continue
        if sf[i]:
            prev = 0.0
        elif abs(x - prev) > band:
            prev = prev + lam * (x - prev)
        out[i] = prev
    return pd.Series(out, index=w.index)


def safety_mask(d):
    return ((d["vol_shock"] > 2.0) | (d["cum3"] < -0.07)).values


# ---------------------------------------------------------------- evaluation

def simulate(d: pd.DataFrame, w: pd.Series, lag=2, fee=FEE, start=EVAL_START, end=EVAL_END):
    held = w.fillna(0.0).shift(lag).fillna(0.0)
    turn = held.diff().abs().fillna(held.abs())
    ret = held * d["r"] - turn * fee
    m = (d["date"] >= start) & (d["date"] <= end)
    return ret[m], held[m], d["r"][m], d["date"][m]


def metrics(ret, held, bh):
    ret = pd.Series(np.asarray(ret, float))
    bh = pd.Series(np.asarray(bh, float))
    held = pd.Series(np.asarray(held, float))
    sh = ret.mean() / (ret.std() + 1e-12) * np.sqrt(252)
    cum = np.exp(ret.cumsum())
    mdd = (cum / cum.cummax() - 1).min()
    q = np.percentile(ret, 5)
    cvar = ret[ret <= q].mean()
    dn = bh < 0
    defense = (ret[dn] - bh[dn]).mean() * 1e4
    yrs = len(ret) / 252
    return dict(sharpe=sh, total=np.exp(ret.sum()) - 1, mdd=mdd, cvar=cvar, defense=defense,
                avgw=held.mean(), turnover=held.diff().abs().sum() / yrs, n=len(ret))


def evaluate(d, w, period="dev", lag=2, fee=FEE):
    if period == "dev":
        s, e = EVAL_START, DEV_END
    elif period == "hold":
        s, e = HOLD_START, EVAL_END
    else:
        s, e = EVAL_START, EVAL_END
    ret, held, bh, _ = simulate(d, w, lag=lag, fee=fee, start=s, end=e)
    return metrics(ret, held, bh)


def static_metrics(d, c, period="dev"):
    w = pd.Series(c, index=d.index)
    s, e = (EVAL_START, DEV_END) if period == "dev" else ((HOLD_START, EVAL_END) if period == "hold" else (EVAL_START, EVAL_END))
    m = (d["date"] >= s) & (d["date"] <= e)
    ret = c * d["r"][m]
    return metrics(ret, pd.Series(c, index=ret.index), d["r"][m])


def block_boot(a, b, stat, B=2000, block=20, seed=0):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    rng = np.random.default_rng(seed)
    outs = []
    for _ in range(B):
        st = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        ii = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        outs.append(stat(a[ii], b[ii]))
    return np.percentile(outs, [2.5, 97.5])


def deflated_sharpe(ret, n_trials, sr_trials_var):
    """Bailey & Lopez de Prado (2014). SR in per-period units."""
    r = np.asarray(ret, float)
    T = len(r)
    sr = r.mean() / r.std()
    g3 = skew(r)
    g4 = kurtosis(r, fisher=False)
    emc = 0.5772156649
    sr0 = np.sqrt(sr_trials_var) * ((1 - emc) * norm.ppf(1 - 1 / n_trials) + emc * norm.ppf(1 - 1 / (n_trials * np.e)))
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(norm.cdf(z)), float(sr0 * np.sqrt(252))
