"""Step 1: faithful replica of the paper pipeline (leaky), then the leak-free baseline.
Only the development period (2016-01-06 .. 2023-06-30) is inspected here, plus the full
period for the replica check (the paper already reported full-period numbers)."""
import json
import numpy as np
import pandas as pd
from lab import *

d = load()
print("EGARCH fallback rows (after 252 warm-up):", int(d["eg_fallback"].iloc[252:].sum()), "of", len(d))
rows = {}

# ---- replica of paper (leaky): training gap 1, full-sample scaling, momentum fixed, lag 3
raw_leaky = wf_signed(d, KF10, DIR10, "y_ret5", gap=1, center=False)
K_leaky = expanding_scale(raw_leaky, full_sample=True)
w = rule_weights(d, K_leaky, momentum=True)
rows["replica_leaky_lag3_full"] = evaluate(d, w, "full", lag=3)
rows["replica_leaky_lag3_dev"] = evaluate(d, w, "dev", lag=3)
rows["replica_leaky_lag2_dev"] = evaluate(d, w, "dev", lag=2)

# ---- leak-free baseline B0: purged (gap 6), expanding scaling, direction from past
raw = wf_signed(d, KF10, DIR10, "y_ret5", gap=GAP, center=False)
K0 = expanding_scale(raw)
mom = expanding_direction(raw, d["y_ret5"].values)
w0 = rule_weights(d, K0, momentum=mom)
rows["B0_leakfree_lag2_dev"] = evaluate(d, w0, "dev", lag=2)
rows["B0_leakfree_lag3_dev"] = evaluate(d, w0, "dev", lag=3)
# components (leak-free, lag 2)
rows["B0_trend_only_dev"] = evaluate(d, rule_weights(d, K0, mom, use_kfgi=False, use_vt=False, use_regime=False, use_safety=False), "dev")
rows["B0_trend_vt_dev"] = evaluate(d, rule_weights(d, K0, mom, use_kfgi=False, use_regime=False, use_safety=False), "dev")
rows["B0_trend_vt_kfgi_dev"] = evaluate(d, rule_weights(d, K0, mom, use_regime=False, use_safety=False), "dev")

# ---- simple benchmarks (lag 2)
ma = (d["ma20"] > d["ma60"]).astype(float)
rows["MA20_60_dev"] = evaluate(d, ma, "dev")
rv = (0.10 / np.sqrt(252) / (d["rv60"] + 1e-9)).clip(0, 1)
rows["RVtarget10_dev"] = evaluate(d, rv, "dev")
eg = (0.10 / np.sqrt(252) / (d["egv"] + 1e-9)).clip(0, 1)
rows["EGtarget10_dev"] = evaluate(d, eg, "dev")
rows["BuyHold_dev"] = static_metrics(d, 1.0, "dev")
for k in ["B0_leakfree_lag2_dev"]:
    rows[k.replace("_dev", "_static_dev")] = static_metrics(d, rows[k]["avgw"], "dev")

df = pd.DataFrame(rows).T
pd.set_option("display.width", 200)
print(df.round(4))
df.to_csv(RES / "step1_baseline.csv")
pd.DataFrame({"date": d["date"], "raw_B0": raw, "K_B0": K0, "mom_B0": mom, "w_B0": w0}).to_pickle(RES / "step1_B0.pkl")
