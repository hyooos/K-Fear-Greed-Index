"""Step 8: why does sentiment help the exposure rule? Tests of the 'slow and not overlapping
with price trend' hypothesis.

(1) Placebo: circularly shift the sentiment series by >= 1 year (keeps smoothness and
    distribution, destroys timing), rebuild the sentiment-only K-FGI, repeat 40 times.
(2) Smoothing: 10-day moving average (past only) of the K-FGI without sentiment and of the
    full K-FGI. If smoothing alone closes the gap, slowness is the mechanism.
(3) Cost-free comparison (0bp) of all / no sentiment / sentiment only.
(4) Overlap with price trend: Spearman correlation of each index with trend_strength,
    past 20-day return, and with the trend weight; daily index change size (smoothness).
Pre-specified rule (return target), full period, 3bp unless noted.
"""
import sys
import numpy as np
import pandas as pd
from lab import *

assets = sys.argv[1:] or ["KPI200", "KOSPI", "KOSDAQ"]
store = pd.read_pickle(RES / "step6_K_KPI200_KOSPI_KOSDAQ.pkl")
SUB = [f"sub_index{i}" for i in range(2, 8)]
EGA = ["egv", "vol_regime_high", "vol_ratio"]


def run_K(d, K, mom, fee=FEE):
    w = rule_weights(d, K, momentum=mom)
    return evaluate(d, w, "full", fee=fee)


def smooth_past(K, n=10):
    return pd.Series(K).rolling(n, min_periods=1).mean().values


rows, plac, ovl = [], [], []
rng = np.random.default_rng(7)
for a in assets:
    d = load(a)
    n = len(d)
    S = store[a]
    variants = {
        "all": (S["all"]["K_ret"], S["all"]["mom"]),
        "no_sent": (S["no_sent"]["K_ret"], S["no_sent"]["mom"]),
        "sent_only": (S["sent_only"]["K_ret"], S["sent_only"]["mom"]),
        "all_smoothed10": (smooth_past(S["all"]["K_ret"]), S["all"]["mom"]),
        "no_sent_smoothed10": (smooth_past(S["no_sent"]["K_ret"]), S["no_sent"]["mom"]),
    }
    m = ((d["date"] >= EVAL_START) & (d["date"] <= EVAL_END)).values
    past20 = d["kospi200_close"].pct_change(20)
    for name, (K, mom) in variants.items():
        r3 = run_K(d, K, mom)
        r0 = run_K(d, K, mom, fee=0.0)
        rows.append(dict(asset=a, variant=name, sharpe_3bp=r3["sharpe"], sharpe_0bp=r0["sharpe"],
                         mdd=r3["mdd"], turnover=r3["turnover"], avgw=r3["avgw"]))
        Ks = pd.Series(K)
        tw = d["trend_strength"].map(TREND_MAP)
        ovl.append(dict(asset=a, variant=name,
                        corr_trend=spearmanr(Ks[m], d["trend_strength"][m], nan_policy="omit").correlation,
                        corr_past20=spearmanr(Ks[m], past20[m], nan_policy="omit").correlation,
                        mean_abs_daily_change=Ks.diff().abs()[m].mean(),
                        lag1_autocorr=Ks[m].autocorr(1)))
    # placebo: circular shifts of the sentiment series by at least one year
    real = [r for r in rows if r["asset"] == a and r["variant"] == "sent_only"][0]
    base_col = d["sent_composite_ma10"].values.copy()
    shifts = rng.integers(252, n - 252, size=40)
    for k in shifts:
        d["sent_shift"] = np.roll(base_col, int(k))
        raw = wf_signed(d, ["sent_shift"], np.array([1]), "y_ret5", center=False)
        K = expanding_scale(raw)
        mom = expanding_direction(raw, d["y_ret5"].values)
        r3 = run_K(d, K, mom)
        r0 = run_K(d, K, mom, fee=0.0)
        plac.append(dict(asset=a, shift=int(k), sharpe_3bp=r3["sharpe"], sharpe_0bp=r0["sharpe"], turnover=r3["turnover"]))
    print(a, "done", flush=True)

res = pd.DataFrame(rows)
pl = pd.DataFrame(plac)
ov = pd.DataFrame(ovl)
res.to_csv(RES / "step8_variants.csv", index=False)
pl.to_csv(RES / "step8_placebo.csv", index=False)
ov.to_csv(RES / "step8_overlap.csv", index=False)
pd.set_option("display.width", 220)
print(res.round(3).to_string(index=False))
print()
summ = []
for a in assets:
    r = res[(res.asset == a) & (res.variant == "sent_only")].iloc[0]
    p = pl[pl.asset == a]
    summ.append(dict(asset=a, real_sharpe=r.sharpe_3bp, placebo_mean=p.sharpe_3bp.mean(),
                     placebo_p95=p.sharpe_3bp.quantile(0.95), p_value=(1 + (p.sharpe_3bp >= r.sharpe_3bp).sum()) / (1 + len(p)),
                     real_turnover=r.turnover, placebo_turnover=p.turnover.mean()))
print(pd.DataFrame(summ).round(3).to_string(index=False))
print()
print(ov.round(3).to_string(index=False))
