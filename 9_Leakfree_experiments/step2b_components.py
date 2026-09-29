"""Step 2b: K-FGI component subsets under the leak-free pipeline, for the
original return target (B0) and the volatility target (A1). Development period only."""
import numpy as np
import pandas as pd
from lab import *

d = load()
SUB = [f"sub_index{i}" for i in range(2, 8)]
SEN = ["sent_composite_ma10"]
EGA = ["egv", "vol_regime_high", "vol_ratio"]
sets = {"all": SUB + SEN + EGA, "sub+sent": SUB + SEN, "sub+egarch": SUB + EGA, "sent+egarch": SEN + EGA,
        "sub_only": SUB, "sent_only": SEN, "egarch_only": EGA}
dirs = {f: 1 for f in SUB + SEN} | {f: -1 for f in EGA}
rows = []
store = {}
for name, feats in sets.items():
    dv = np.array([dirs[f] for f in feats])
    raw_r = wf_signed(d, feats, dv, "y_ret5", center=False)
    K_r = expanding_scale(raw_r)
    mom = expanding_direction(raw_r, d["y_ret5"].values)
    raw_v = wf_signed(d, feats, -dv, "y_vol5")
    K_v = expanding_scale(raw_v, invert=True)
    store[name] = dict(K_ret=K_r, mom=mom, K_vol=K_v)
    for tgt, K, mm in [("ret", K_r, mom), ("vol", K_v, True)]:
        for sname, skw in {"full": {}, "kfgi_only_mult": dict(use_regime=False, use_vt=False)}.items():
            w = rule_weights(d, K, momentum=mm, **skw)
            rows.append(dict(set=name, target=tgt, struct=sname, **evaluate(d, w, "dev")))
res = pd.DataFrame(rows)
res.to_csv(RES / "step2b_components.csv", index=False)
pd.to_pickle(store, RES / "step2b_scores.pkl")
pd.set_option("display.width", 220)
print(res.sort_values(["target", "struct", "sharpe"], ascending=[True, True, False]).round(3).to_string(index=False))
