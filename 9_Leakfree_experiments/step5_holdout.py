"""Step 5: pre-registered candidate selection on the development period, then a single
evaluation on the sealed holdout (2023-07-01 .. 2025-12-22).

Selection rule (fixed before looking at the holdout): for each candidate family, choose the
turnover band in {0, .05, .10, .20, .30} that maximises development Sharpe at 3bp.
All candidates are reported; none is dropped after seeing the holdout.
"""
import glob
import json
import numpy as np
import pandas as pd
from lab import *

d = load()
sm = safety_mask(d)
BANDS = [0.0, 0.05, 0.10, 0.20, 0.30]
b0 = pd.read_pickle(RES / "step1_B0.pkl")
s2 = pd.read_pickle(RES / "step2_scores.pkl")
s2b = pd.read_pickle(RES / "step2b_scores.pkl")
idx3 = pd.read_pickle(RES / "step3_idx.pkl")

fam = {}
fam["C1_current_rule_leakfree"] = lambda: rule_weights(d, b0["K_B0"].values, momentum=b0["mom_B0"].values)
fam["C2_KFGI10_vol_target"] = lambda: rule_weights(d, s2["A1_vol"]["K"], True)
fam["C3_sentiment_only_vol"] = lambda: rule_weights(d, s2b["sent_only"]["K_vol"], True)
fam["C4_ML_best_ENS_tail"] = lambda: rule_weights(d, idx3["ENS_tail"], True, use_regime=False, use_vt=False)
fam["C6_trend_vt_safety_noKFGI"] = lambda: rule_weights(d, np.full(len(d), 50.0), True, use_kfgi=False, use_regime=False)

# RL: ensemble (mean weight) of the 5 seeds for each mode/kappa; pick best on dev
rl = {}
for mode in ["free", "overlay"]:
    for k in ["0.0", "1.0"]:
        fs = sorted(glob.glob(str(RES / f"step4_rl_{mode}_k{k}_s*.pkl")))
        if len(fs) == 5:
            rl[f"{mode}_k{k}"] = pd.Series(np.nanmean(np.vstack([pd.read_pickle(f) for f in fs]), axis=0), index=d.index)
rl_dev = {k: evaluate(d, v, "dev")["sharpe"] for k, v in rl.items()}
best_rl = max(rl_dev, key=rl_dev.get)
fam["C5_RL_" + best_rl] = lambda: rl[best_rl]

rows, chosen = [], {}
for name, f in fam.items():
    w_t = f()
    grid = []
    for b in BANDS:
        w = smooth(w_t, band=b, safety=sm) if not name.startswith("C5") else smooth(w_t, band=b)
        grid.append((evaluate(d, w, "dev")["sharpe"], b, w))
    sh, b, w = max(grid, key=lambda x: x[0])
    chosen[name] = (b, w)
    for per in ["dev", "hold", "full"]:
        for fee in [0.0, FEE, 0.0015]:
            m = evaluate(d, w, per, fee=fee)
            rows.append(dict(candidate=name, band=b, period=per, fee_bp=round(fee * 1e4), **m))
# benchmarks
bm = {"BuyHold": pd.Series(1.0, index=d.index),
      "MA20_60": (d["ma20"] > d["ma60"]).astype(float),
      "RVtarget10": (0.10 / np.sqrt(252) / (d["rv60"] + 1e-9)).clip(0, 1),
      "EGtarget10": (0.10 / np.sqrt(252) / (d["egv"] + 1e-9)).clip(0, 1)}
for name, w in bm.items():
    for per in ["dev", "hold", "full"]:
        rows.append(dict(candidate=name, band=0, period=per, fee_bp=3, **evaluate(d, w, per)))
res = pd.DataFrame(rows)

# static benchmark matched to each candidate's own average exposure in each period
st = []
for name in fam:
    for per in ["dev", "hold", "full"]:
        c = res[(res.candidate == name) & (res.period == per) & (res.fee_bp == 3)]["avgw"].iloc[0]
        st.append(dict(candidate=name + "__static", band=0, period=per, fee_bp=0, **static_metrics(d, c, per)))
res = pd.concat([res, pd.DataFrame(st)], ignore_index=True)
res.to_csv(RES / "step5_holdout.csv", index=False)

# block bootstrap on holdout (3bp): candidate vs matched static and vs Buy & Hold
bt = []
for name, (b, w) in chosen.items():
    for per in ["hold", "full"]:
        s, e = (HOLD_START, EVAL_END) if per == "hold" else (EVAL_START, EVAL_END)
        ret, held, bh, _ = simulate(d, w, start=s, end=e)
        c = held.mean()
        stat_ret = c * bh
        sh = lambda a, b: a.mean() / a.std() * np.sqrt(252) - b.mean() / b.std() * np.sqrt(252)
        def mdd(x):
            v = np.exp(np.cumsum(x)); return (v / np.maximum.accumulate(v) - 1).min()
        md = lambda a, b: (mdd(a) - mdd(b)) * 100
        def dfn(a, b, bhv=bh.values):
            return 0.0
        lo, hi = block_boot(ret.values, stat_ret.values, sh)
        lo2, hi2 = block_boot(ret.values, bh.values, md)
        bt.append(dict(candidate=name, period=per, dSharpe_vs_static=sh(ret.values, stat_ret.values),
                       ci_lo=lo, ci_hi=hi, dMDD_vs_BH_pp=md(ret.values, bh.values), mdd_ci_lo=lo2, mdd_ci_hi=hi2))
bt = pd.DataFrame(bt)
bt.to_csv(RES / "step5_bootstrap.csv", index=False)

# deflated Sharpe on development period, all trials counted
trials = []
for f in ["step2_grid.csv", "step2b_components.csv", "step3_grid.csv"]:
    trials += list(pd.read_csv(RES / f)["sharpe"])
trials += list(pd.read_csv(RES / "step1_baseline.csv", index_col=0).filter(like="_dev", axis=0)["sharpe"])
trials += [evaluate(d, pd.Series(pd.read_pickle(f), index=d.index), "dev")["sharpe"]
           for f in glob.glob(str(RES / "step4_rl_*_s*.pkl"))]
trials = np.array(trials)
var_pp = np.var(trials / np.sqrt(252))
dsr = []
for name, (b, w) in chosen.items():
    ret, *_ = simulate(d, w, start=EVAL_START, end=DEV_END)
    p, sr0 = deflated_sharpe(ret.values, len(trials), var_pp)
    dsr.append(dict(candidate=name, dev_sharpe=evaluate(d, w, "dev")["sharpe"], dsr_prob=p, sr0_annual=sr0))
dsr = pd.DataFrame(dsr)
dsr.to_csv(RES / "step5_dsr.csv", index=False)

pd.set_option("display.width", 240)
print("RL dev sharpe by config:", {k: round(v, 3) for k, v in rl_dev.items()})
print("n trials:", len(trials))
cols = ["candidate", "band", "sharpe", "total", "mdd", "cvar", "defense", "avgw", "turnover"]
for per in ["dev", "hold", "full"]:
    print(f"\n=== {per} (3bp; static rows are cost-free) ===")
    t = res[(res.period == per) & (res.fee_bp.isin([3, 0])) & ~((res.fee_bp == 0) & ~res.candidate.str.endswith("__static"))]
    print(t[cols].round(4).to_string(index=False))
print("\n=== holdout Sharpe by fee (0/3/15bp) ===")
print(res[(res.period == "hold") & res.candidate.isin(fam.keys())].pivot(index="candidate", columns="fee_bp", values="sharpe").round(3))
print("\n=== bootstrap ===")
print(bt.round(3).to_string(index=False))
print("\n=== deflated Sharpe (dev) ===")
print(dsr.round(3).to_string(index=False))
