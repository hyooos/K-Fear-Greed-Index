"""Step 2: (A) change the K-FGI training target to downside risk, (B) turnover control.
All selection on the development period only."""
import itertools
import numpy as np
import pandas as pd
from lab import *

d = load()
b0 = pd.read_pickle(RES / "step1_B0.pkl")
sm = safety_mask(d)

scores = {}
# B0: original return target
scores["B0_ret"] = dict(K=b0["K_B0"].values, mom=b0["mom_B0"].values, vol=None)
# A1: future 5d volatility (log), sign constraint reversed (greed -> lower vol)
raw_v = wf_signed(d, KF10, -DIR10, "y_vol5")
scores["A1_vol"] = dict(K=expanding_scale(raw_v, invert=True), mom=True, vol=None)
# predicted vol level from the same features: refit unnormalized ridge for level
# (use rolling OLS on log vol with intercept -> sigma_hat); simple: map percentile to egv? keep separate:
# A2: tail-loss probability (5d < -2%), sign-constrained logistic
raw_t = wf_signed(d, KF10, -DIR10, "y_tail", logistic=True)
scores["A2_tail"] = dict(K=expanding_scale(raw_t, invert=True), mom=True, vol=None)
# A3: average of vol and tail indices
scores["A3_vol_tail"] = dict(K=(scores["A1_vol"]["K"] + scores["A2_tail"]["K"]) / 2, mom=True, vol=None)

pd.to_pickle({k: v for k, v in scores.items()}, RES / "step2_scores.pkl")

rows = []
bands = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
lams = [1.0, 0.5, 0.25]
structs = {
    "full": dict(),
    "no_regime": dict(use_regime=False),
    "kfgi_only_mult": dict(use_regime=False, use_vt=False),
}
for (name, sc), (sname, skw) in itertools.product(scores.items(), structs.items()):
    w_t = rule_weights(d, sc["K"], momentum=sc["mom"], **skw)
    for b, lam in itertools.product(bands, lams):
        w = smooth(w_t, band=b, lam=lam, safety=sm)
        m = evaluate(d, w, "dev")
        rows.append(dict(model=name, struct=sname, band=b, lam=lam, **m))
# reference: trend-only with turnover control (no K-FGI at all)
for b, lam in itertools.product(bands, lams):
    w_t = rule_weights(d, np.full(len(d), 50.0), True, use_kfgi=False, use_vt=False, use_regime=False, use_safety=False)
    w = smooth(w_t, band=b, lam=lam)
    rows.append(dict(model="trend_only", struct="-", band=b, lam=lam, **evaluate(d, w, "dev")))
    w_t = rule_weights(d, np.full(len(d), 50.0), True, use_kfgi=False, use_regime=False)
    w = smooth(w_t, band=b, lam=lam, safety=sm)
    rows.append(dict(model="trend_vt_safety", struct="-", band=b, lam=lam, **evaluate(d, w, "dev")))

res = pd.DataFrame(rows)
res.to_csv(RES / "step2_grid.csv", index=False)
pd.set_option("display.width", 220)
print("n trials:", len(res))
print("\n== no smoothing (band 0, lam 1)")
print(res[(res.band == 0) & (res.lam == 1)].sort_values("sharpe", ascending=False).round(4).to_string(index=False))
print("\n== best per model/struct over turnover grid")
best = res.loc[res.groupby(["model", "struct"]).sharpe.idxmax()].sort_values("sharpe", ascending=False)
print(best.round(4).to_string(index=False))
