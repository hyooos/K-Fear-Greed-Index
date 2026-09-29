"""Step 7 (plan 3): does sentiment improve daily risk forecasts? (more statistical power
than strategy Sharpe). Fixed, simple models; purged walk-forward; full period 2016-01-06
.. 2025-12-22 is out-of-sample.

Targets (for decision row s, horizon r[s+2..s+6]):
  (a) log 5-day realised volatility  -> OLS, squared-error loss
  (b) 5% quantile of the 5-day return (VaR) -> quantile regression, pinball loss
  (c) P(5-day return < -2%)          -> logistic, log loss
Baseline features: EGARCH (log), log |r|, log rv5, log rv20 (HAR + EGARCH).
Additions: +sentiment, +sub-indices, +sentiment+sub-indices.
Diebold-Mariano test on the loss difference with Newey-West (10 lags), one-sided
(H1: addition reduces loss).
Usage: python step7_forecast_tests.py KPI200 KOSPI KOSDAQ
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
from lab import *

assets = sys.argv[1:] or ["KPI200", "KOSPI", "KOSDAQ"]
SUB = [f"sub_index{i}" for i in range(2, 8)]
BASE_F = ["legv", "lrv1", "lrv5", "lrv20"]
ADD = {"base": [], "+sent": ["sent_composite_ma10"], "+sub": SUB, "+sent+sub": ["sent_composite_ma10"] + SUB}


def wf_predict(d, feats, target, kind, refit=5, min_train=250, tau=0.05):
    X = d[feats].values.astype(float)
    y = d[target].values.astype(float)
    n = len(d)
    ok = ~np.isnan(X).any(1)
    out = np.full(n, np.nan)
    coef = None
    mu = sd = None
    last = -10 ** 9
    for t in range(n):
        if not ok[t]:
            continue
        if coef is None or t - last >= refit:
            hi = t - GAP
            if hi < 0:
                continue
            idx = np.arange(0, hi + 1)
            idx = idx[ok[idx] & ~np.isnan(y[idx])]
            if len(idx) < min_train:
                continue
            mu, sd = X[idx].mean(0), X[idx].std(0) + 1e-12
            Z = sm.add_constant((X[idx] - mu) / sd, has_constant="add")
            if kind == "ols":
                coef = np.linalg.lstsq(Z, y[idx], rcond=None)[0]
            elif kind == "quant":
                coef = QuantReg(y[idx], Z).fit(q=tau, max_iter=2000).params
            else:
                m = LogisticRegression(C=1.0, max_iter=2000).fit(Z[:, 1:], y[idx])
                coef = np.concatenate([m.intercept_, m.coef_[0]])
            last = t
        z = np.concatenate([[1.0], (X[t] - mu) / sd])
        v = float(z @ coef)
        out[t] = 1 / (1 + np.exp(-v)) if kind == "logit" else v
    return out


def dm(loss_base, loss_new, lags=10):
    dlt = loss_base - loss_new
    dlt = dlt[~np.isnan(dlt)]
    T = len(dlt)
    m = dlt.mean()
    e = dlt - m
    v = e @ e / T
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / T
    t = m / np.sqrt(v / T)
    return t, 1 - norm.cdf(t)


rows = []
for a in assets:
    d = load(a)
    d["lrv1"] = np.log(d["r"].abs() + 1e-4)
    d["lrv5"] = np.log(d["rv5"] + 1e-4)
    d["lrv20"] = np.log(d["rv20"] + 1e-4)
    d["legv"] = np.log(d["egv"] + 1e-4)
    m = ((d["date"] >= EVAL_START) & (d["date"] <= EVAL_END)).values
    for tgt, kind in [("y_vol5", "ols"), ("y_ret5", "quant"), ("y_tail", "logit")]:
        preds = {k: wf_predict(d, BASE_F + v, tgt, kind) for k, v in ADD.items()}
        y = d[tgt].values
        valid = m & ~np.isnan(y) & np.all([~np.isnan(p) for p in preds.values()], axis=0)
        def loss(p):
            if kind == "ols":
                return (y - p) ** 2
            if kind == "quant":
                u = y - p
                return u * (0.05 - (u < 0))
            pp = np.clip(p, 1e-6, 1 - 1e-6)
            return -(y * np.log(pp) + (1 - y) * np.log(1 - pp))
        Lb = loss(preds["base"])[valid]
        for k in ADD:
            if k == "base":
                continue
            Lk = loss(preds[k])[valid]
            t, p = dm(Lb, Lk)
            extra = {}
            if kind == "quant":
                extra["hit_rate"] = float(np.mean(y[valid] < preds[k][valid]))
                extra["hit_rate_base"] = float(np.mean(y[valid] < preds["base"][valid]))
            rows.append(dict(asset=a, target=tgt, model=k, n=int(valid.sum()),
                             loss_reduction_pct=(1 - Lk.mean() / Lb.mean()) * 100, dm_t=t, p_one_sided=p, **extra))
        print(a, tgt, "done", flush=True)
res = pd.DataFrame(rows)
res.to_csv(RES / f"step7_forecast_{'_'.join(assets)}.csv", index=False)
pd.set_option("display.width", 220)
print(res.round(4).to_string(index=False))
