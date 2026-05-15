import numpy as np
import pandas as pd

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("[경고] arch 없음. rolling std 대체.")


def fit_egarch_walkforward(log_returns, min_obs):
    """
    walk-forward EGARCH(1,1) 변동성 추정.
    t 시점의 σ_t는 [0, t-1] 구간 데이터만 사용.
    min_obs(=60) 미만 구간은 NaN 처리.
    """
    n     = len(log_returns)
    sigma = np.full(n, np.nan)
    for t in range(min_obs, n):
        train = log_returns.iloc[:t].dropna()
        if not HAS_ARCH:
            sigma[t] = train.iloc[-20:].std() if len(train) >= 20 else np.nan
            continue
        try:
            am  = arch_model(train * 100, vol="EGARCH", p=1, q=1,
                             dist="skewt", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            fc  = res.forecast(horizon=1, reindex=False)
            sigma[t] = float(np.sqrt(fc.variance.values[-1, 0])) / 100
        except Exception:
            sigma[t] = np.nan
    return pd.Series(sigma, index=log_returns.index, name="egarch_vol")


def add_egarch_features(df_raw):
    """EGARCH 추정 후 파생 피처 추가."""
    from config import CFG
    log_ret = np.log(
        df_raw["kospi_close"] / df_raw["kospi_close"].shift(1)
    ).fillna(0)
    df_raw["egarch_vol"] = fit_egarch_walkforward(log_ret, CFG["egarch_min_obs"])

    df_raw["vol_shock"]       = df_raw["egarch_vol"].pct_change().clip(-5, 5)
    df_raw["vol_regime_high"] = (
        df_raw["egarch_vol"] > df_raw["egarch_vol"].rolling(60).quantile(0.7)
    ).astype(float)
    df_raw["vol_ma5"]   = df_raw["egarch_vol"].rolling(5).mean()
    df_raw["vol_trend"] = (df_raw["egarch_vol"] > df_raw["vol_ma5"]).astype(float)

    print(f"    유효 행 수={df_raw['egarch_vol'].notna().sum()}")
    return df_raw