import numpy as np

CFG = dict(
    raw_path       = "KFG_final_2.csv",
    egarch_min_obs = 60,
    kfgi_min_obs   = 60,
    n_splits       = 5,
    fee            = 0.00015,
    target_vol_ann = 0.15,
    max_leverage   = 2.0,
    min_leverage   = 0.1,
    vol_shock_thr  = 2.0,
    tail_loss_thr  = -0.07,
    seed           = 42,
)

TARGET_DAILY_VOL = CFG["target_vol_ann"] / np.sqrt(252)

TREND_MAP = {0: 0.0, 1: 0.35, 2: 0.70, 3: 1.0}

KFGI_FEATS_BASE = (
    [f"sub_index{i}" for i in range(1, 8)]
    + [
        "sent_norm_w", "sent_energy", "sent_std_inv",
        "neg_z_inv", "sent_composite", "sent_composite_ma10",
        "egarch_vol", "vol_regime_high", "vol_ratio",
    ]
)

DIRECTION = {
    **{f"sub_index{i}": 1 for i in range(1, 8)},
    "sent_norm_w": 1, "sent_energy": 1, "sent_std_inv": 1,
    "neg_z_inv": 1, "sent_composite": 1, "sent_composite_ma10": 1,
    "egarch_vol": -1, "vol_regime_high": -1, "vol_ratio": -1,
}