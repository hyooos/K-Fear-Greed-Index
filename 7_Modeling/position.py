import numpy as np
import pandas as pd
from config import CFG, TARGET_DAILY_VOL, TREND_MAP


def classify_regime(row):
    if (row["K_FGI"] > 65
            and row["vol_regime_high"] == 0
            and row["trend_strength"] >= 2):
        return "bull"
    elif row["K_FGI"] < 25 or row["vol_shock"] > CFG["vol_shock_thr"]:
        return "crisis"
    return "normal"


def kfgi_scale_mult(kfgi_val, is_momentum=True):
    """K-FGI → 포지션 배율 변환 (범위: 0.5~1.6)."""
    if is_momentum:
        mult = 0.5 + (kfgi_val / 100.0) * 1.1
    else:
        mult = 1.6 - (kfgi_val / 100.0) * 1.1
    return float(np.clip(mult, 0.5, 1.6))


def compute_positions(df, kfgi_momentum=True, name="전략"):
    d = df.copy()
    d["tw"] = d["trend_strength"].apply(
        lambda x: TREND_MAP.get(int(x), 0.0)
    )
    raw_vt    = TARGET_DAILY_VOL / (d["egarch_vol"] + 1e-9)
    d["vt_pos"] = np.sqrt(raw_vt).clip(CFG["min_leverage"], CFG["max_leverage"])
    d["kfgi_mult"] = d["K_FGI"].apply(
        lambda k: kfgi_scale_mult(k, is_momentum=kfgi_momentum)
    )
    regime_mult = d["regime"].map({"bull": 1.2, "normal": 1.0, "crisis": 0.7})
    d["weight"] = (
        d["tw"] * d["vt_pos"] * d["kfgi_mult"] * regime_mult
    ).clip(0.0, CFG["max_leverage"] * 1.5)

    d.loc[d["vol_shock"] > CFG["vol_shock_thr"], "weight"] = 0.0
    d["cumret_3d"] = d["target_reg"].rolling(3).sum().shift(1).fillna(0)
    d.loc[d["cumret_3d"] < CFG["tail_loss_thr"], "weight"] = 0.0

    d["weight_lag"] = d["weight"].shift(1).fillna(0)
    d["turnover"] = d["weight_lag"].diff().abs().fillna(d["weight_lag"].iloc[0])
    d["cost"]     = d["turnover"] * CFG["fee"]
    d["strat_ret"] = d["weight_lag"] * d["target_reg"] - d["cost"]
    return d


def bench_pure_trend(df):
    """순수 추세추종 (K-FGI/EGARCH 없이)."""
    d = df.copy()
    d["weight"]     = d["trend_strength"].apply(
        lambda x: TREND_MAP.get(int(x), 0.0)
    )
    d["weight_lag"] = d["weight"].shift(1).fillna(0)
    d["turnover"]   = d["weight_lag"].diff().abs().fillna(d["weight_lag"].iloc[0])
    d["cost"]       = d["turnover"] * CFG["fee"]
    d["strat_ret"]  = d["weight_lag"] * d["target_reg"] - d["cost"]
    return d


def bench_trend_vol(df):
    """추세 + vol타게팅^0.5 (K-FGI 없이)."""
    d   = df.copy()
    tw  = d["trend_strength"].apply(lambda x: TREND_MAP.get(int(x), 0.0))
    vt  = np.sqrt(TARGET_DAILY_VOL / (d["egarch_vol"] + 1e-9)).clip(
        CFG["min_leverage"], CFG["max_leverage"]
    )
    d["weight"] = (tw * vt).clip(0, CFG["max_leverage"])
    d.loc[d["vol_shock"] > CFG["vol_shock_thr"], "weight"] = 0.0
    d["weight_lag"] = d["weight"].shift(1).fillna(0)
    d["turnover"]   = d["weight_lag"].diff().abs().fillna(d["weight_lag"].iloc[0])
    d["cost"]       = d["turnover"] * CFG["fee"]
    d["strat_ret"]  = d["weight_lag"] * d["target_reg"] - d["cost"]
    return d