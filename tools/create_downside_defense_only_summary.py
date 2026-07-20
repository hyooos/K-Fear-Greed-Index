from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT.parent / "논문용"
PAPER_OUT = PAPER_DIR / "10_Paper_outputs" if PAPER_DIR.exists() else ROOT / "paper_outputs"


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r.fillna(0)).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1).min())


def downside_deviation(r: pd.Series) -> float:
    downside = np.minimum(r.fillna(0), 0)
    return float(np.sqrt((downside**2).mean()) * np.sqrt(252))


def summarize_strategy(df: pd.DataFrame, strategy_col: str, label: str) -> dict:
    bh = df["buy_hold"].astype(float)
    st = df[strategy_col].astype(float)
    down_mask = bh < 0

    bh_down = bh.loc[down_mask]
    st_down = st.loc[down_mask]
    defense = st_down - bh_down

    bh_mdd = max_drawdown(bh)
    st_mdd = max_drawdown(st)
    bh_vol = bh.std(ddof=1) * np.sqrt(252)
    st_vol = st.std(ddof=1) * np.sqrt(252)
    bh_down_dev = downside_deviation(bh)
    st_down_dev = downside_deviation(st)

    return {
        "strategy": label,
        "n_days": len(df),
        "bh_down_days": int(down_mask.sum()),
        "avg_loss_on_bh_down_days_pct": st_down.mean() * 100,
        "bh_avg_loss_on_down_days_pct": bh_down.mean() * 100,
        "downside_defense_bp_per_down_day": defense.mean() * 10000,
        "downside_defense_t": defense.mean() / (defense.std(ddof=1) / np.sqrt(len(defense))),
        "hit_rate_on_bh_down_days_pct": (st_down > bh_down).mean() * 100,
        "mdd_pct": st_mdd * 100,
        "bh_mdd_pct": bh_mdd * 100,
        "mdd_improvement_pctp": (st_mdd - bh_mdd) * 100,
        "annual_vol_pct": st_vol * 100,
        "bh_annual_vol_pct": bh_vol * 100,
        "vol_reduction_pct": (1 - st_vol / bh_vol) * 100,
        "downside_deviation_pct": st_down_dev * 100,
        "bh_downside_deviation_pct": bh_down_dev * 100,
        "downside_deviation_reduction_pct": (1 - st_down_dev / bh_down_dev) * 100,
    }


def format_pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}%"


def format_bp(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}bp"


def main() -> None:
    returns_path = ROOT / "paper_outputs/robustness/data/priority_ab_strategy_returns_robust.csv"
    tests_path = ROOT / "paper_outputs/tables/extended_statistical_tests.csv"
    ablation_path = ROOT / "paper_outputs/tables/feature_drop_ablation.csv"

    df = pd.read_csv(returns_path)
    strategies = [
        ("trend_only", "Trend only"),
        ("trend_egarch", "Trend + EGARCH"),
        ("kfgi_without_sentiment", "K-FGI without sentiment"),
        ("kfgi_with_sentiment", "K-FGI with sentiment"),
    ]
    out = pd.DataFrame([summarize_strategy(df, col, label) for col, label in strategies])

    out_dir = ROOT / "paper_outputs/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "downside_defense_only_summary.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    tests = pd.read_csv(tests_path)
    defense_tests = tests[tests["category"].isin(["하방 방어", "위험 감소", "EGARCH 노출 조절"])].copy()
    defense_tests_path = out_dir / "downside_defense_only_stat_tests.csv"
    defense_tests.to_csv(defense_tests_path, index=False, encoding="utf-8-sig")

    ablation = pd.read_csv(ablation_path)
    ablation_labels = [
        "lean_sent_composite_ma10_only",
        "baseline_full",
        "drop_all_sentiment",
        "market_egarch_only",
        "drop_sent_composite_ma10",
        "drop_egarch_family",
    ]
    ablation_defense = ablation[ablation["label"].isin(ablation_labels)].copy()
    ablation_defense = ablation_defense[
        [
            "label",
            "n",
            "mdd",
            "downside_excess_bp",
            "downside_t",
            "downside_p",
            "n_kfgi_features",
            "kept_features",
        ]
    ]
    label_map = {
        "lean_sent_composite_ma10_only": "Lean K-FGI: sentiment composite 10D only",
        "baseline_full": "Full K-FGI: all sentiment features",
        "drop_all_sentiment": "No sentiment: market + EGARCH only",
        "market_egarch_only": "Market + EGARCH only",
        "drop_sent_composite_ma10": "Full K-FGI without sentiment composite 10D",
        "drop_egarch_family": "No EGARCH family",
    }
    ablation_defense.insert(1, "strategy_name", ablation_defense["label"].map(label_map))
    ablation_defense_path = out_dir / "downside_defense_feature_ablation_summary.csv"
    ablation_defense.to_csv(ablation_defense_path, index=False, encoding="utf-8-sig")

    paper_tables = PAPER_OUT / "tables"
    paper_tables.mkdir(parents=True, exist_ok=True)
    out.to_csv(paper_tables / out_path.name, index=False, encoding="utf-8-sig")
    defense_tests.to_csv(paper_tables / defense_tests_path.name, index=False, encoding="utf-8-sig")
    ablation_defense.to_csv(paper_tables / ablation_defense_path.name, index=False, encoding="utf-8-sig")

    main_row = out[out["strategy"] == "K-FGI with sentiment"].iloc[0]
    no_sent = out[out["strategy"] == "K-FGI without sentiment"].iloc[0]

    md = f"""# 하방방어 중심 결과 정리

본 논문의 핵심 주장은 전체 초과수익 창출이 아니라 **시장 하락일과 고변동성 구간에서 손실 노출을 줄이는 하방위험 방어 효과**이다. 따라서 본문 결과표는 누적수익률이나 Sharpe보다 아래 지표를 중심으로 제시하는 것이 적절하다.

## 메인 결과표

| 전략 | 시장 하락일 평균 손익 | B&H 하락일 평균 손실 | 하락일 방어폭 | 하락일 방어 적중률 | MDD | MDD 개선 | 연환산 변동성 감소 | 하방변동성 감소 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for _, row in out.iterrows():
        md += (
            f"| {row['strategy']} | "
            f"{format_pct(row['avg_loss_on_bh_down_days_pct'], 2)} | "
            f"{format_pct(row['bh_avg_loss_on_down_days_pct'], 2)} | "
            f"{format_bp(row['downside_defense_bp_per_down_day'], 1)} | "
            f"{format_pct(row['hit_rate_on_bh_down_days_pct'], 1)} | "
            f"{format_pct(row['mdd_pct'], 1)} | "
            f"{format_pct(row['mdd_improvement_pctp'], 1)}p | "
            f"{format_pct(row['vol_reduction_pct'], 1)} | "
            f"{format_pct(row['downside_deviation_reduction_pct'], 1)} |\n"
        )

    md += f"""
## 본문에 넣을 핵심 수치

- 표본 기간의 시장 하락일은 총 {int(main_row['bh_down_days'])}일이다.
- K-FGI with sentiment는 시장 하락일에 평균 {format_bp(main_row['downside_defense_bp_per_down_day'], 1)}의 손실 방어 효과를 보였다.
- K-FGI with sentiment의 하락일 방어 t 통계량은 {main_row['downside_defense_t']:.2f}이며, 기존 통계검정표 기준 p-value는 <0.001이다.
- Buy & Hold의 MDD는 {format_pct(main_row['bh_mdd_pct'], 1)}이고, K-FGI with sentiment의 MDD는 {format_pct(main_row['mdd_pct'], 1)}로 약 {format_pct(main_row['mdd_improvement_pctp'], 1)}p 개선되었다.
- K-FGI with sentiment는 Buy & Hold 대비 연환산 변동성을 {format_pct(main_row['vol_reduction_pct'], 1)}, 하방변동성을 {format_pct(main_row['downside_deviation_reduction_pct'], 1)} 낮췄다.
- 현재 full K-FGI 기준에서 감성 포함/제외의 하락일 방어폭 차이는 크지 않다. 따라서 감성의 역할은 단순히 전체 감성 피처를 많이 넣는 방식보다, 감성 정보를 압축한 `sent_composite_ma10` 중심의 ablation 결과로 제시하는 편이 적절하다.

## 감성 피처 축소형 하방방어 ablation

| 후보 | K-FGI 피처 수 | MDD | 하락일 방어폭 | t 통계량 | p-value | 해석 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
"""
    ablation_order = [
        "lean_sent_composite_ma10_only",
        "baseline_full",
        "drop_all_sentiment",
        "drop_sent_composite_ma10",
        "drop_egarch_family",
    ]
    interpretation = {
        "lean_sent_composite_ma10_only": "감성 정보를 10일 복합지표로 압축한 최종 후보",
        "baseline_full": "모든 감성 피처를 넣은 기준 모형",
        "drop_all_sentiment": "감성 제거 시 MDD가 크게 악화",
        "drop_sent_composite_ma10": "10일 감성 복합지표 제거 시 MDD 악화",
        "drop_egarch_family": "방어폭은 크지만 변동성 조절 논리 제거로 본 연구 구조와 맞지 않음",
    }
    for label in ablation_order:
        row = ablation_defense[ablation_defense["label"] == label].iloc[0]
        p = "<0.001" if row["downside_p"] < 0.001 else f"{row['downside_p']:.4f}"
        md += (
            f"| {row['strategy_name']} | "
            f"{int(row['n_kfgi_features'])} | "
            f"{format_pct(row['mdd'] * 100, 1)} | "
            f"{format_bp(row['downside_excess_bp'], 1)} | "
            f"{row['downside_t']:.2f} | "
            f"{p} | "
            f"{interpretation[label]} |\n"
        )

    md += f"""

이 표를 기준으로 하면, 논문 메인 K-FGI는 `sub_index2-7 + egarch_vol + vol_regime_high + vol_ratio + sent_composite_ma10`의 10개 피처 구성으로 제시하는 것이 가장 자연스럽다. 이 구성은 full K-FGI보다 피처 수가 적고, 감성 피처를 하나의 핵심 지표로 남기면서도 MDD와 하락일 방어폭이 유지된다.

## 논문 서술 방향

본 연구는 K-FGI를 수익률 예측 지표가 아니라 하방위험 관리 지표로 해석한다. 전체 기간의 평균 초과수익은 통계적으로 유의하지 않지만, 시장 하락일에 한정하면 K-FGI는 Buy & Hold 대비 유의한 손실 방어 효과를 보인다. 이는 K-FGI가 상승장을 공격적으로 예측하는 도구라기보다, 시장 약세와 고변동성 구간에서 노출을 낮춰 포트폴리오의 낙폭과 하방변동성을 줄이는 위험관리 지표로 기능함을 의미한다.

## 함께 제시할 통계검정

`downside_defense_only_stat_tests.csv`에는 하방방어, 위험 감소, EGARCH 노출 조절에 해당하는 검정만 따로 추렸다. 본문에는 하방방어 t-test, Newey-West HAC t-test, Wilcoxon signed-rank test, 분산비 F-test, Levene test 정도만 사용하면 충분하다.
"""

    md_path = ROOT / "paper_outputs/DOWNSIDE_DEFENSE_ONLY_RESULTS.md"
    md_path.write_text(md, encoding="utf-8")
    paper_doc_dir = PAPER_OUT
    paper_doc_dir.mkdir(parents=True, exist_ok=True)
    (paper_doc_dir / md_path.name).write_text(md, encoding="utf-8")

    print(out_path)
    print(defense_tests_path)
    print(md_path)


if __name__ == "__main__":
    main()
