from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


TRADING_DAYS = 252
OUT = Path("paper_outputs")
DATA = OUT / "robustness" / "data" / "priority_ab_kfgi_timeseries_robust.csv"


def max_drawdown(r: pd.Series) -> float:
    wealth = np.exp(r.fillna(0).cumsum())
    return float((wealth / wealth.cummax() - 1).min())


def strategy_return(df: pd.DataFrame, weight: pd.Series, fee: float = 0.0015) -> pd.Series:
    weight_lag = weight.shift(1).fillna(0)
    turnover = weight_lag.diff().abs().fillna(weight_lag.iloc[0])
    return weight_lag * df["target_reg"] - turnover * fee


def metrics(df: pd.DataFrame, label: str, weight: pd.Series) -> dict:
    r = strategy_return(df, weight)
    b = df["target_reg"]
    down = b < 0
    dex = r[down] - b[down]
    ann_ret = float(r.mean() * TRADING_DAYS)
    ann_vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    mdd = max_drawdown(r)
    bh_mdd = max_drawdown(b)
    return {
        "variant": label,
        "total_return_pct": float((np.exp(r.sum()) - 1) * 100),
        "sharpe": ann_ret / (ann_vol + 1e-12),
        "mdd_pct": mdd * 100,
        "mdd_defense_pctp": (mdd - bh_mdd) * 100,
        "downside_excess_bp": float(dex.mean() * 10000),
        "downside_t": float(stats.ttest_1samp(dex.dropna(), 0).statistic),
        "downside_p": float(stats.ttest_1samp(dex.dropna(), 0).pvalue),
        "avg_exposure": float(weight.mean()),
        "zero_exposure_pct": float((weight <= 0.05).mean() * 100),
        "full_exposure_pct": float((weight >= 0.95).mean() * 100),
        "max_exposure": float(weight.max()),
    }


def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    target_daily_vol = 0.15 / math.sqrt(TRADING_DAYS)
    k_min, k_max = 0.5, 1.6
    fear, greed = 25, 65

    tw_map = {0: 0.0, 1: 0.35, 2: 0.70, 3: 1.0}
    tw = df["trend_strength"].fillna(0).astype(int).map(tw_map).fillna(0)
    vt = np.sqrt(target_daily_vol / (df["egarch_vol"] + 1e-9)).clip(0.1, 2.0)
    kfgi_mult = (k_min + (df["K_FGI"] / 100) * (k_max - k_min)).clip(k_min, k_max)

    bull = (df["K_FGI"] > greed) & (df["vol_regime_high"].eq(0)) & (df["trend_strength"] >= 2)
    crisis = (df["K_FGI"] < fear) | (df["vol_shock"] > 2.0)
    regime_mult = pd.Series(1.0, index=df.index)
    regime_mult[bull] = 1.2
    regime_mult[crisis] = 0.7

    raw_weight = (tw * vt * kfgi_mult * regime_mult).clip(0, 3.0)
    shock_cut = df["vol_shock"] > 2.0
    tail_cut = df["target_reg"].rolling(3).sum().shift(1).fillna(0) < -0.07
    circuit_cut = shock_cut | tail_cut

    final_weight = raw_weight.mask(circuit_cut, 0).clip(0, 1.0)
    no_circuit_weight = raw_weight.clip(0, 1.0)
    no_shock_weight = raw_weight.mask(tail_cut, 0).clip(0, 1.0)
    no_tail_weight = raw_weight.mask(shock_cut, 0).clip(0, 1.0)
    no_cap_weight = raw_weight.mask(circuit_cut, 0)

    perf = pd.DataFrame(
        [
            metrics(df, "final_cap1_with_circuit_breaker", final_weight),
            metrics(df, "cap1_without_circuit_breaker", no_circuit_weight),
            metrics(df, "cap1_without_vol_shock_cut", no_shock_weight),
            metrics(df, "cap1_without_3d_tail_cut", no_tail_weight),
            metrics(df, "with_circuit_no_1x_cap", no_cap_weight),
        ]
    )
    perf.to_csv(OUT / "tables" / "circuit_breaker_ablation.csv", index=False)

    corr_rows = []
    features = [
        "egarch_vol",
        "vol_ratio",
        "vol_regime_high",
        "sub_index2",
        "sub_index3",
        "sub_index4",
        "sub_index5",
        "sub_index6",
        "sub_index7",
        "sent_composite_ma10",
        "trend_strength",
    ]
    for f in features:
        tmp = df[[f, "vol_shock"]].replace([np.inf, -np.inf], np.nan).dropna()
        corr_rows.append(
            {
                "feature": f,
                "spearman_with_vol_shock": tmp[f].corr(tmp["vol_shock"], method="spearman"),
                "mean_on_shock_days": df.loc[shock_cut, f].mean(),
                "mean_on_non_shock_days": df.loc[~shock_cut, f].mean(),
            }
        )
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT / "tables" / "circuit_breaker_feature_overlap.csv", index=False)

    counts = pd.DataFrame(
        [
            {
                "rule": "vol_shock > 2.0",
                "n_days": int(shock_cut.sum()),
                "share_pct": float(shock_cut.mean() * 100),
                "market_avg_bp": float(df.loc[shock_cut, "target_reg"].mean() * 10000),
                "market_down_rate_pct": float((df.loc[shock_cut, "target_reg"] < 0).mean() * 100),
            },
            {
                "rule": "previous 3-day return < -7%",
                "n_days": int(tail_cut.sum()),
                "share_pct": float(tail_cut.mean() * 100),
                "market_avg_bp": float(df.loc[tail_cut, "target_reg"].mean() * 10000),
                "market_down_rate_pct": float((df.loc[tail_cut, "target_reg"] < 0).mean() * 100),
            },
            {
                "rule": "either circuit breaker",
                "n_days": int(circuit_cut.sum()),
                "share_pct": float(circuit_cut.mean() * 100),
                "market_avg_bp": float(df.loc[circuit_cut, "target_reg"].mean() * 10000),
                "market_down_rate_pct": float((df.loc[circuit_cut, "target_reg"] < 0).mean() * 100),
            },
        ]
    )
    counts.to_csv(OUT / "tables" / "circuit_breaker_rule_counts.csv", index=False)

    md = "# Circuit Breaker Overlay and Feature Overlap Check\n\n"
    md += "본 문서는 `vol_shock > 2.0` 및 최근 3일 누적수익률 -7% 조건이 최종 K-FGI 본체 피처와 역할이 겹치는지, 그리고 실제 성과에 어떤 영향을 주는지 확인한 결과이다.\n\n"
    md += "## 1. 적용 방식\n\n"
    md += "| 구성 | 적용 위치 | 설명 |\n| --- | --- | --- |\n"
    md += "| `sub_index2-7`, `sent_composite_ma10`, `egarch_vol`, `vol_regime_high`, `vol_ratio` | K-FGI 점수 | 지표 본체 |\n"
    md += "| `vol_shock > 2.0` | circuit breaker | EGARCH 변동성의 수준이 아니라 변동성의 급격한 변화율을 감지해 포지션을 0으로 축소 |\n"
    md += "| 최근 3일 누적수익률 < -7% | circuit breaker | 가격 급락이 이미 발생한 경우 추가 손실 확대를 막기 위한 tail-risk overlay |\n"
    md += "| cap 1.0x | 보수형 운용 제약 | K-FGI 신호가 강해도 레버리지 노출을 허용하지 않음 |\n\n"
    md += "## 2. 규칙 작동 빈도\n\n"
    md += counts.to_markdown(index=False, floatfmt=".3f")
    md += "\n\n"
    md += "## 3. Circuit breaker 제거 실험\n\n"
    md += perf.to_markdown(index=False, floatfmt=".3f")
    md += "\n\n"
    md += "## 4. 기존 피처와의 중복성\n\n"
    md += corr.to_markdown(index=False, floatfmt=".3f")
    md += "\n\n"
    md += "## 5. 결론\n\n"
    md += "- `vol_shock`는 `egarch_vol`의 수준이 아니라 변화율이므로, `egarch_vol`, `vol_ratio`, `vol_regime_high`와 완전히 같은 역할은 아니다.\n"
    md += "- 다만 `vol_shock` hard cut은 작동일이 15일, 전체 0.60%로 매우 적어 K-FGI 성과 전체를 좌우하는 주된 피처라고 보기는 어렵다.\n"
    md += "- 최근 3일 -7% tail cut도 11일, 전체 0.44%만 작동한다. 두 circuit breaker를 합쳐도 26일, 1.04% 수준이다.\n"
    md += "- 따라서 본문에서는 K-FGI 본체와 분리하여 실무적 위험관리 overlay로 설명하고, 강건성 표에서 제거 실험을 제시하는 것이 가장 안전하다.\n"
    md += "- cap 1.0x는 circuit breaker보다 훨씬 중요한 보수형 제약이다. 초기 구간의 과도한 레버리지 문제를 줄이는 핵심 장치이므로 본문에 명시해야 한다.\n"
    (OUT / "CIRCUIT_BREAKER_OVERLAY_ANALYSIS.md").write_text(md, encoding="utf-8")
    print(perf.to_string(index=False))
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
