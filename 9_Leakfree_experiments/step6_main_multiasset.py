"""Step 6 (plans 1 and 2): pre-specified main analysis and cross-index validation.

Pre-specified rule = the paper's original design with only the leakage fixes and the
single-day wait (return target, 10 features, direction from past data, expanding scaling,
no turnover band). It is evaluated on the full walk-forward period 2016-01-06..2025-12-22,
which is entirely out-of-sample for the K-FGI weights. The same code is applied unchanged
to other indices (the design was never tuned on them).
Also reported: component comparison (all / without sentiment / sentiment only) for the
return target and the volatility target.
Usage: python step6_main_multiasset.py KPI200 KOSPI KOSDAQ [more...]
"""
import sys
import numpy as np
import pandas as pd
from lab import *

assets = sys.argv[1:] or ["KPI200", "KOSPI", "KOSDAQ"]
SUB = [f"sub_index{i}" for i in range(2, 8)]
SEN = ["sent_composite_ma10"]
EGA = ["egv", "vol_regime_high", "vol_ratio"]
SETS = {"all": SUB + SEN + EGA, "no_sent": SUB + EGA, "sent_only": SEN}
DIRS = {f: 1 for f in SUB + SEN} | {f: -1 for f in EGA}


def mdd(x):
    v = np.exp(np.cumsum(x))
    return (v / np.maximum.accumulate(v) - 1).min()


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(252)


def cvar(x):
    q = np.percentile(x, 5)
    return x[x <= q].mean()


rows, boots, store = [], [], {}
for a in assets:
    d = load(a)
    print(a, "EGARCH fallback", int(d["eg_fallback"].iloc[252:].sum()), flush=True)
    store[a] = {}
    for sname, feats in SETS.items():
        dv = np.array([DIRS[f] for f in feats])
        raw_r = wf_signed(d, feats, dv, "y_ret5", center=False)
        K_r = expanding_scale(raw_r)
        mom = expanding_direction(raw_r, d["y_ret5"].values)
        raw_v = wf_signed(d, feats, -dv, "y_vol5")
        K_v = expanding_scale(raw_v, invert=True)
        store[a][sname] = dict(K_ret=K_r, mom=mom, K_vol=K_v)
        for tgt, K, mm in [("ret", K_r, mom), ("vol", K_v, True)]:
            w = rule_weights(d, K, momentum=mm)
            for fee in [0.0, FEE, 0.0015]:
                m = evaluate(d, w, "full", fee=fee)
                rows.append(dict(asset=a, features=sname, target=tgt, fee_bp=round(fee * 1e4), **m))
            if sname == "all" or (sname == "sent_only" and tgt == "vol"):
                ret, held, bh, _ = simulate(d, w, start=EVAL_START, end=EVAL_END)
                ret, bh = ret.values, bh.values
                c = held.mean()
                st = c * bh
                dn = bh < 0
                lo_s, hi_s = block_boot(ret, st, lambda x, y: sharpe(x) - sharpe(y))
                lo_m, hi_m = block_boot(ret, bh, lambda x, y: (mdd(x) - mdd(y)) * 100)
                lo_c, hi_c = block_boot(ret, bh, lambda x, y: (cvar(x) - cvar(y)) * 100)
                lo_ms, hi_ms = block_boot(ret, st, lambda x, y: (mdd(x) - mdd(y)) * 100)
                boots.append(dict(asset=a, features=sname, target=tgt, avgw=c,
                                  dSharpe_static=sharpe(ret) - sharpe(st), ci=f"[{lo_s:.3f}, {hi_s:.3f}]",
                                  dMDD_static_pp=(mdd(ret) - mdd(st)) * 100, ci_mdd_static=f"[{lo_ms:.2f}, {hi_ms:.2f}]",
                                  dMDD_BH_pp=(mdd(ret) - mdd(bh)) * 100, ci_mdd=f"[{lo_m:.2f}, {hi_m:.2f}]",
                                  dCVaR_BH_pp=(cvar(ret) - cvar(bh)) * 100, ci_cvar=f"[{lo_c:.2f}, {hi_c:.2f}]"))
    for per in ["full"]:
        rows.append(dict(asset=a, features="BuyHold", target="-", fee_bp=0, **static_metrics(d, 1.0, per)))
        c = [r for r in rows if r["asset"] == a and r["features"] == "all" and r["target"] == "ret" and r["fee_bp"] == 3][0]["avgw"]
        rows.append(dict(asset=a, features="static_matched", target="-", fee_bp=0, **static_metrics(d, c, per)))

res = pd.DataFrame(rows)
bt = pd.DataFrame(boots)
tag = "_".join(assets)
res.to_csv(RES / f"step6_main_{tag}.csv", index=False)
bt.to_csv(RES / f"step6_boot_{tag}.csv", index=False)
pd.to_pickle(store, RES / f"step6_K_{tag}.pkl")
pd.set_option("display.width", 250)
cols = ["asset", "features", "target", "sharpe", "total", "mdd", "cvar", "defense", "avgw", "turnover"]
print(res[res.fee_bp.isin([3, 0]) & ~((res.fee_bp == 0) & res.features.isin(list(SETS)))][cols].round(4).to_string(index=False))
print(bt.round(3).to_string(index=False))
