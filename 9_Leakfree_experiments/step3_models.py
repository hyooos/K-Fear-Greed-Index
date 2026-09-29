"""Step 3: richer risk models (interpretability not required). Purged walk-forward,
refit every 21 trading days. Development period only."""
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb  # must be imported before lightgbm on macOS (OpenMP clash)
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lab import *

d = load()
d["lrv1"] = np.log(d["r"].abs() + 1e-4)
d["lrv5"] = np.log(d["rv5"] + 1e-4)
d["lrv20"] = np.log(d["rv20"] + 1e-4)
d["legv"] = np.log(d["egv"] + 1e-4)
HAR = ["lrv1", "lrv5", "lrv20"]
HARX = HAR + ["legv", "sent_composite_ma10"] + [f"sub_index{i}" for i in range(2, 8)]
FA = FALL + ["lrv1", "legv"]
sm = safety_mask(d)

vol_models = {
    "HAR": (HAR, lambda: LinearRegression()),
    "HARX": (HARX, lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
    "Ridge_all": (FA, lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
    "LGBM_all": (FA, lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                                  min_child_samples=40, subsample=0.8, subsample_freq=1,
                                                  colsample_bytree=0.8, verbose=-1, random_state=0, n_jobs=1)),
    "RF_all": (FA, lambda: RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=20,
                                                  n_jobs=1, random_state=0)),
}
tail_models = {
    "Logit_all": (FA, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))),
    "LGBMc_all": (FA, lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                                    min_child_samples=40, subsample=0.8, subsample_freq=1,
                                                    colsample_bytree=0.8, verbose=-1, random_state=0, n_jobs=1)),
    "RFc_all": (FA, lambda: RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=20,
                                                    n_jobs=1, random_state=0)),
    "XGBc_all": (FA, lambda: xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3,
                                                  subsample=0.8, colsample_bytree=0.8, verbosity=0, n_jobs=1,
                                                  random_state=0)),
}

import sys
if len(sys.argv) > 1:  # fit one model in its own process
    name = sys.argv[1]
    if name in vol_models:
        feats, mk = vol_models[name]
        out = ("vol", wf_sklearn(d, feats, "y_vol5", mk, kind="reg"))
    else:
        feats, mk = tail_models[name]
        out = ("tail", wf_sklearn(d, feats, "y_tail", mk, kind="clf"))
    pd.to_pickle(out, RES / f"step3_pred_{name}.pkl")
    print("done", name, flush=True)
    sys.exit(0)
preds = {k: pd.read_pickle(RES / f"step3_pred_{k}.pkl") for k in list(vol_models) + list(tail_models)}
pd.to_pickle(preds, RES / "step3_preds.pkl")

# indices (higher = safer) from each model, leak-free scaling
idx = {k: expanding_scale(v, invert=True) for k, (_, v) in preds.items()}
idx["ENS_vol"] = np.nanmean(np.vstack([idx[k] for k in vol_models]), axis=0)
idx["ENS_tail"] = np.nanmean(np.vstack([idx[k] for k in tail_models]), axis=0)
idx["ENS_all"] = np.nanmean(np.vstack([idx[k] for k in preds]), axis=0)
pd.to_pickle(idx, RES / "step3_idx.pkl")

# forecast quality on dev (out-of-sample rank correlation with realised target)
dev = (d["date"] >= EVAL_START) & (d["date"] <= DEV_END)
q = []
for k, (kind, v) in preds.items():
    y = d["y_vol5"] if kind == "vol" else d["y_tail"]
    m = dev & ~np.isnan(v) & y.notna()
    q.append(dict(model=k, target=kind, spearman=spearmanr(v[m], y[m]).correlation))
q.append(dict(model="EGARCH_only", target="vol",
              spearman=spearmanr(d["legv"][dev & d["y_vol5"].notna()], d["y_vol5"][dev & d["y_vol5"].notna()]).correlation))
print(pd.DataFrame(q).round(3).to_string(index=False))

rows = []
bands = [0.0, 0.05, 0.10, 0.20, 0.30]
for name, K in idx.items():
    for sname, skw in {"full": {}, "kfgi_only_mult": dict(use_regime=False, use_vt=False)}.items():
        w_t = rule_weights(d, K, momentum=True, **skw)
        for b in bands:
            rows.append(dict(model=name, struct=sname, band=b, **evaluate(d, smooth(w_t, band=b, safety=sm), "dev")))
    # direct vol targeting with the model's own vol forecast (vol models only)
    if name in vol_models:
        sig = np.exp(pd.Series(preds[name][1], index=d.index)).fillna(d["egv"])
        for tv in (0.10, 0.15):
            w_t = (d["trend_strength"].map(TREND_MAP) * np.sqrt((tv / np.sqrt(252)) / (pd.Series(sig, index=d.index) + 1e-9)).clip(0.1, 2.0)).clip(0, 1)
            w_t = w_t.mask(pd.Series(sm, index=d.index), 0.0)
            for b in bands:
                rows.append(dict(model=name, struct=f"trend_x_voltarget{int(tv*100)}", band=b,
                                 **evaluate(d, smooth(w_t, band=b, safety=sm), "dev")))

res = pd.DataFrame(rows)
res.to_csv(RES / "step3_grid.csv", index=False)
pd.set_option("display.width", 220)
print("n trials:", len(res))
best = res.loc[res.groupby(["model", "struct"]).sharpe.idxmax()].sort_values("sharpe", ascending=False)
print(best.head(30).round(4).to_string(index=False))
