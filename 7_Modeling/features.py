import numpy as np
import pandas as pd
from config import TARGET_DAILY_VOL


def compute_rsi(series, window=14):
    """RSI(Relative Strength Index) 계산. EWM 방식 사용."""
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def build_features(df):
    df = df.copy()
    sent_cols = [c for c in df.columns if any(k in c for k in
                 ["sent_", "neg_z", "effective", "heat", "pos_", "weight"])]
    df[sent_cols] = df[sent_cols].ffill()

    # 감성 파생 피처
    df["neg_z_inv"]      = -df["neg_z"]
    df["sent_std_inv"]   = -df["sent_std"]
    df["sent_energy"]    = df["sent_strength_w"] * df["sent_norm_w"]
    df["sent_norm_ma5"]  = df["sent_norm_w"].rolling(5).mean()
    df["sent_norm_diff"] = df["sent_norm_w"].diff()
    df["neg_z_ma5"]      = df["neg_z"].rolling(5).mean()
    df["sent_composite"]      = (df["sent_norm_w"]   * 0.4
                                 + df["neg_z_inv"]   * 0.3
                                 + df["sent_energy"] * 0.3)
    df["sent_composite_ma10"] = df["sent_composite"].rolling(10).mean()
    df["sent_composite_diff"] = df["sent_composite"].diff(5)

    # 추세/모멘텀 피처
    price = df["kospi_close"]
    df["ma20"]            = price.rolling(20).mean()
    df["ma60"]            = price.rolling(60).mean()
    df["ma120"]           = price.rolling(120).mean()
    df["ma_ratio_5_20"]   = price.rolling(5).mean() / (df["ma20"] + 1e-9) - 1
    df["ma_ratio_20_60"]  = df["ma20"] / (df["ma60"] + 1e-9) - 1
    df["ma_ratio_60_120"] = df["ma60"] / (df["ma120"] + 1e-9) - 1
    df["rsi14"]           = compute_rsi(price, 14)
    df["mom5"]            = price.pct_change(5)
    df["mom20"]           = price.pct_change(20)
    df["mom60"]           = price.pct_change(60)
    df["above_ma20"]      = (price > df["ma20"]).astype(float)
    df["above_ma60"]      = (price > df["ma60"]).astype(float)
    df["above_ma120"]     = (price > df["ma120"]).astype(float)
    df["trend_strength"]  = df["above_ma20"] + df["above_ma60"] + df["above_ma120"]
    df["vol_ratio"]       = df["egarch_vol"] / (
        df["egarch_vol"].rolling(60).mean() + 1e-9
    )

    # 서브인덱스 Lag 피처 (sub_index1 제외 — 모멘텀은 mom5/mom20/mom60으로 대체)
    for i in range(2, 8):
        df[f"sub_index{i}_lag1"] = df[f"sub_index{i}"].shift(1)

    # EGARCH Lag 피처
    df["egarch_vol_lag1"]     = df["egarch_vol"].shift(1)
    df["egarch_vol_ma5_lag1"] = df["vol_ma5"].shift(1)
    df["vol_shock_lag1"]      = df["vol_shock"].shift(1)
    df["vol_regime_lag1"]     = df["vol_regime_high"].shift(1)
    df["neg_z_vol_ratio"]     = (
        df["neg_z_inv"] / (df["egarch_vol_lag1"] + 1e-6)
    ).clip(-10, 10)

    df["dayofweek"]  = df["date"].dt.dayofweek
    df["month"]      = df["date"].dt.month
    df["target_reg"] = df["log_return_t+1"]
    df["target_5d"]  = df["log_return_t+1"].rolling(5).sum().shift(-4)

    before = len(df)

    # 전체 피처 1일 lag — 미래 정보 누출 원천 차단
    feature_cols = [col for col in df.columns
                    if col not in ["date", "target_reg", "target_5d"]]
    df[feature_cols] = df[feature_cols].shift(1)

    df = df.dropna().reset_index(drop=True)
    print(f"    dropna() 후: {before}→{len(df)}행  "
          f"(시작일={df['date'].min().date()})")
    return df