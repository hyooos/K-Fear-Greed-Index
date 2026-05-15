import itertools
import numpy as np
import pandas as pd
from config import CFG, TARGET_DAILY_VOL, TREND_MAP


def run_sensitivity_analysis(df, kfgi_momentum=True):
    """포지션 파라미터 민감도 분석."""
    print("\n" + "="*62)
    print("  포지션 파라미터 민감도 분석")
    print("="*62)

    kfgi_ranges = [
        (0.3, 1.4),
        (0.5, 1.6),   # 채택값 ★
        (0.5, 2.0),
        (0.7, 1.4),
        (0.7, 1.6),
    ]
    regime_combos = [
        (1.1, 1.0, 0.8),
        (1.2, 1.0, 0.7),   # 채택값 ★
        (1.3, 1.0, 0.6),
        (1.2, 1.0, 0.5),
    ]

    sens_results = []

    for (kmin, kmax), (bull_r, norm_r, crisis_r) in itertools.product(
        kfgi_ranges, regime_combos
    ):
        d = df.copy()
        d["tw"] = d["trend_strength"].apply(lambda x: TREND_MAP.get(int(x), 0.0))
        raw_vt      = TARGET_DAILY_VOL / (d["egarch_vol"] + 1e-9)
        d["vt_pos"] = np.sqrt(raw_vt).clip(CFG["min_leverage"], CFG["max_leverage"])

        if kfgi_momentum:
            d["kfgi_mult"] = (kmin + (d["K_FGI"] / 100.0) * (kmax - kmin)).clip(kmin, kmax)
        else:
            d["kfgi_mult"] = (kmax - (d["K_FGI"] / 100.0) * (kmax - kmin)).clip(kmin, kmax)

        regime_mult = d["regime"].map({"bull": bull_r, "normal": norm_r, "crisis": crisis_r})
        d["weight"] = (d["tw"] * d["vt_pos"] * d["kfgi_mult"] * regime_mult).clip(
            0.0, CFG["max_leverage"] * 1.5
        )
        d.loc[d["vol_shock"] > CFG["vol_shock_thr"], "weight"] = 0.0
        d["cumret_3d"] = d["target_reg"].rolling(3).sum().shift(1).fillna(0)
        d.loc[d["cumret_3d"] < CFG["tail_loss_thr"], "weight"] = 0.0
        d["weight_lag"] = d["weight"].shift(1).fillna(0)
        d["turnover"]   = d["weight_lag"].diff().abs().fillna(d["weight_lag"].iloc[0])
        d["cost"]       = d["turnover"] * CFG["fee"]
        d["strat_ret"]  = d["weight_lag"] * d["target_reg"] - d["cost"]

        ann_ret = d["strat_ret"].mean() * 252
        ann_vol = d["strat_ret"].std() * np.sqrt(252) + 1e-9
        sharpe  = ann_ret / ann_vol
        cum     = np.exp(d["strat_ret"].cumsum())
        mdd     = (cum / cum.cummax() - 1).min()
        total   = np.exp(d["strat_ret"].sum()) - 1
        is_adopted = (kmin == 0.5 and kmax == 1.6 and bull_r == 1.2 and crisis_r == 0.7)

        sens_results.append({
            "kfgi_범위":      f"{kmin}~{kmax}",
            "regime(B/N/C)": f"{bull_r}/{norm_r}/{crisis_r}",
            "연수익(%)":      round(ann_ret * 100, 2),
            "Sharpe":        round(sharpe, 3),
            "MDD(%)":        round(mdd * 100, 2),
            "누적수익(%)":    round(total * 100, 2),
            "채택":          "★" if is_adopted else "",
        })

    df_sens = pd.DataFrame(sens_results).sort_values("Sharpe", ascending=False)
    print(df_sens.to_string(index=False))
    return df_sens