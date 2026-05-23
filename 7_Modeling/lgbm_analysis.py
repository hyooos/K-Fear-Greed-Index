import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from config import CFG

FEATURES_LIST = (
    [f"sub_index{i}_lag1" for i in range(2, 8)]
    + [
        "sent_norm_w", "sent_energy", "sent_std_inv", "neg_z_inv",
        "sent_norm_ma5", "neg_z_ma5", "sent_norm_diff",
        "sent_composite", "sent_composite_ma10", "sent_composite_diff",
        "egarch_vol_lag1", "egarch_vol_ma5_lag1",
        "vol_shock_lag1", "vol_regime_lag1", "neg_z_vol_ratio",
        "ma_ratio_5_20", "ma_ratio_20_60", "ma_ratio_60_120",
        "rsi14", "mom5", "mom20", "mom60",
        "above_ma20", "above_ma60", "above_ma120",
        "trend_strength", "vol_ratio",
        "K_FGI", "dayofweek", "month",
    ]
)

LGBM_PARAMS = dict(
    objective="regression", metric="rmse", learning_rate=0.02,
    num_leaves=31, feature_fraction=0.8, bagging_fraction=0.7,
    bagging_freq=5, lambda_l1=0.1, lambda_l2=0.1,
    min_child_samples=20, verbosity=-1, seed=CFG["seed"],
)


def get_features(df):
    return [f for f in FEATURES_LIST if f in df.columns]


def run_lgbm_analysis(df, features, n_splits=CFG["n_splits"]):
    """LGBM 분석 전용 — 포지션 결정에 사용하지 않음."""
    tscv  = TimeSeriesSplit(n_splits=n_splits)
    X, y  = df[features].values, df["target_5d"].values
    dates = df["date"].values
    rmse_list, preds_all, dates_all = [], [], []
    last_lgbm = None

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        lgbm = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(X[tr_idx], label=y[tr_idx]),
            num_boost_round=500,
        )
        pred = lgbm.predict(X[te_idx])
        rmse_list.append(np.sqrt(mean_squared_error(y[te_idx], pred)))
        preds_all.extend(pred)
        dates_all.extend(dates[te_idx])
        last_lgbm = lgbm

    print(f"\n[LightGBM (분석 전용)]  "
          f"RMSE: {np.mean(rmse_list):.6f} ± {np.std(rmse_list):.6f}")
    return pd.DataFrame({"date": dates_all, "pred_5d": preds_all}), last_lgbm


def plot_feature_importance(lgbm_model, features, save_path="feature_importance_v4.png"):
    imp = pd.DataFrame({
        "feature":    features,
        "importance": lgbm_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    sent_imp  = imp[imp["feature"].str.contains(
        "sent_|neg_z|composite", regex=True
    )]["importance"].sum()
    total_imp = imp["importance"].sum()
    print(f"\nTop-15 Feature Importance  "
          f"(감성 피처 비율: {sent_imp/total_imp*100:.1f}%)")
    print("-" * 50)
    print(imp.head(15).to_string(index=False))

    fig2, ax = plt.subplots(figsize=(9, 6))
    top15  = imp.head(15)
    colors = [
        "#DC2626" if any(k in f for k in ["sent_", "neg_z", "composite"])
        else "#F59E0B" if any(k in f for k in
             ["egarch", "vol_", "ma_ratio", "rsi", "mom", "above", "trend"])
        else "#2563EB"
        for f in top15["feature"]
    ]
    ax.barh(top15["feature"][::-1], top15["importance"][::-1], color=colors[::-1])
    ax.legend(handles=[
        Patch(color="#DC2626", label="감성 피처"),
        Patch(color="#F59E0B", label="추세/변동성 피처"),
        Patch(color="#2563EB", label="기타"),
    ], fontsize=9)
    ax.grid(alpha=0.2, axis="x")
    plt.tight_layout()
    plt.xlabel("Feature Importance (Gain)", fontsize=7)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"    저장: {save_path}")
    return imp