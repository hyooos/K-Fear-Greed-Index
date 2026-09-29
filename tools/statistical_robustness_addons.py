from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "10_Paper_outputs"
TABLE_DIR = PAPER / "robustness" / "tables"
SUMMARY_PATH = PAPER / "robustness" / "STATISTICAL_ROBUSTNESS_ADDONS.md"


def p_adjust_bh(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float).to_numpy()
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted_ranked, 0, 1)
    return pd.Series(out, index=p_values.index)


def p_adjust_holm(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float).to_numpy()
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate((n - np.arange(n)) * ranked)
    out = np.empty(n)
    out[order] = np.clip(adjusted_ranked, 0, 1)
    return pd.Series(out, index=p_values.index)


def sig_label(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def one_sample_t_normal_p(values: pd.Series) -> tuple[float, float]:
    values = values.dropna()
    if len(values) < 2:
        return np.nan, np.nan
    se = values.std(ddof=1) / np.sqrt(len(values))
    if se == 0 or pd.isna(se):
        return np.nan, np.nan
    t_stat = values.mean() / se
    p_value = erfc(abs(t_stat) / sqrt(2))
    return float(t_stat), float(p_value)


def performance(ret: pd.Series) -> dict[str, float]:
    ret = ret.dropna()
    ann_ret = ret.mean() * 252
    ann_vol = ret.std(ddof=1) * np.sqrt(252)
    cum = np.exp(ret.cumsum())
    mdd = (cum / cum.cummax() - 1).min()
    return {
        "n": len(ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / (ann_vol + 1e-12),
        "mdd": mdd,
        "total_return": np.exp(ret.sum()) - 1,
    }


def subperiod_test(name: str, start: str, end: str, data: pd.DataFrame) -> dict[str, float | str]:
    mask = (data["date"] >= pd.Timestamp(start)) & (data["date"] <= pd.Timestamp(end))
    d = data.loc[mask].copy()
    bh = d["buy_hold_return"].dropna()
    kfgi = d["K-FGI final"].dropna()
    common = bh.index.intersection(kfgi.index)
    bh = bh.loc[common]
    kfgi = kfgi.loc[common]
    excess = kfgi - bh
    down = bh < 0
    excess_down = excess.loc[down]
    t_all, p_all = one_sample_t_normal_p(excess)
    t_down, p_down = one_sample_t_normal_p(excess_down)
    kperf = performance(kfgi)
    bperf = performance(bh)
    return {
        "period": name,
        "start": start,
        "end": end,
        "n": len(common),
        "down_days": int(down.sum()),
        "bh_ann_ret": bperf["ann_ret"],
        "kfgi_ann_ret": kperf["ann_ret"],
        "bh_sharpe": bperf["sharpe"],
        "kfgi_sharpe": kperf["sharpe"],
        "bh_mdd": bperf["mdd"],
        "kfgi_mdd": kperf["mdd"],
        "mdd_improvement_pctp": (kperf["mdd"] - bperf["mdd"]) * 100,
        "mean_excess_bp": excess.mean() * 10000,
        "excess_t": t_all,
        "excess_p": p_all,
        "downside_excess_bp": excess_down.mean() * 10000,
        "downside_t": t_down,
        "downside_p": p_down,
    }


def build_multiple_testing_table() -> pd.DataFrame:
    src = PAPER / "tables" / "extended_statistical_tests.csv"
    tests = pd.read_csv(src, encoding="utf-8-sig")
    tests = tests[pd.to_numeric(tests["p_value"], errors="coerce").notna()].copy()
    tests["p_value"] = tests["p_value"].astype(float)
    tests["p_holm_global"] = p_adjust_holm(tests["p_value"])
    tests["p_bh_global"] = p_adjust_bh(tests["p_value"])
    tests["sig_raw"] = tests["p_value"].map(sig_label)
    tests["sig_holm_global"] = tests["p_holm_global"].map(sig_label)
    tests["sig_bh_global"] = tests["p_bh_global"].map(sig_label)
    tests["p_value_formatted"] = tests["p_value"].map(format_p)
    tests["p_holm_global_formatted"] = tests["p_holm_global"].map(format_p)
    tests["p_bh_global_formatted"] = tests["p_bh_global"].map(format_p)

    family_frames = []
    for _, sub in tests.groupby("category", sort=False):
        sub = sub.copy()
        sub["p_holm_family"] = p_adjust_holm(sub["p_value"])
        sub["p_bh_family"] = p_adjust_bh(sub["p_value"])
        sub["sig_holm_family"] = sub["p_holm_family"].map(sig_label)
        sub["sig_bh_family"] = sub["p_bh_family"].map(sig_label)
        sub["p_holm_family_formatted"] = sub["p_holm_family"].map(format_p)
        sub["p_bh_family_formatted"] = sub["p_bh_family"].map(format_p)
        family_frames.append(sub)
    tests = pd.concat(family_frames, ignore_index=True)
    ordered = [
        "category",
        "hypothesis",
        "test",
        "stat",
        "p_value",
        "p_holm_global",
        "p_bh_global",
        "p_holm_family",
        "p_bh_family",
        "p_value_formatted",
        "p_holm_global_formatted",
        "p_bh_global_formatted",
        "p_holm_family_formatted",
        "p_bh_family_formatted",
        "sig_raw",
        "sig_holm_global",
        "sig_bh_global",
        "sig_holm_family",
        "sig_bh_family",
        "n",
        "effect",
        "unit",
        "interpretation",
    ]
    return tests[[c for c in ordered if c in tests.columns]]


def build_final_master_multiple_testing_table() -> pd.DataFrame:
    src = PAPER / "tables" / "final_master_statistical_tests.csv"
    tests = pd.read_csv(src, encoding="utf-8-sig")
    tests = tests[pd.to_numeric(tests["p_value"], errors="coerce").notna()].copy()
    tests["p_value"] = tests["p_value"].astype(float)
    tests["p_holm_global"] = p_adjust_holm(tests["p_value"])
    tests["p_bh_global"] = p_adjust_bh(tests["p_value"])
    tests["sig_raw"] = tests["p_value"].map(sig_label)
    tests["sig_holm_global"] = tests["p_holm_global"].map(sig_label)
    tests["sig_bh_global"] = tests["p_bh_global"].map(sig_label)
    tests["p_value_formatted"] = tests["p_value"].map(format_p)
    tests["p_holm_global_formatted"] = tests["p_holm_global"].map(format_p)
    tests["p_bh_global_formatted"] = tests["p_bh_global"].map(format_p)

    family_frames = []
    for _, sub in tests.groupby("category", sort=False):
        sub = sub.copy()
        sub["p_holm_family"] = p_adjust_holm(sub["p_value"])
        sub["p_bh_family"] = p_adjust_bh(sub["p_value"])
        sub["sig_holm_family"] = sub["p_holm_family"].map(sig_label)
        sub["sig_bh_family"] = sub["p_bh_family"].map(sig_label)
        sub["p_holm_family_formatted"] = sub["p_holm_family"].map(format_p)
        sub["p_bh_family_formatted"] = sub["p_bh_family"].map(format_p)
        family_frames.append(sub)
    tests = pd.concat(family_frames, ignore_index=True)
    ordered = [
        "category",
        "hypothesis",
        "test",
        "stat",
        "p_value",
        "p_holm_global",
        "p_bh_global",
        "p_holm_family",
        "p_bh_family",
        "p_value_formatted",
        "p_holm_global_formatted",
        "p_bh_global_formatted",
        "p_holm_family_formatted",
        "p_bh_family_formatted",
        "sig_raw",
        "sig_holm_global",
        "sig_bh_global",
        "sig_holm_family",
        "sig_bh_family",
        "effect",
        "unit",
    ]
    return tests[[c for c in ordered if c in tests.columns]]


def build_subperiod_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(PAPER / "data" / "q1_benchmark_timeseries.csv", parse_dates=["date"])
    data = data.sort_values("date").reset_index(drop=True)
    blocks = [
        ("2016_2019_pre_covid", "2016-01-06", "2019-12-31"),
        ("2020_covid", "2020-01-01", "2020-12-31"),
        ("2021_2022_post_covid_rate_hike", "2021-01-01", "2022-12-31"),
        ("2023_2025_recent_oos", "2023-01-01", "2025-12-31"),
        ("full_sample", "2016-01-06", "2025-12-22"),
    ]
    block_rows = [subperiod_test(*b, data=data) for b in blocks]
    years = []
    for year in sorted(data["date"].dt.year.unique()):
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        row = subperiod_test(str(year), start, end, data)
        if row["n"] >= 50:
            years.append(row)
    return pd.DataFrame(block_rows), pd.DataFrame(years)


def format_pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x * 100:.1f}%"


def format_p(x: float) -> str:
    if pd.isna(x):
        return ""
    if x == 0:
        return "<0.001"
    if x < 0.001:
        return "<0.001"
    return f"{x:.4f}"


def to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            if "p" in col.lower():
                text[col] = text[col].map(format_p)
            else:
                text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
        else:
            text[col] = text[col].fillna("").astype(str)
    headers = list(text.columns)
    rows = text.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_summary(
    multiple: pd.DataFrame,
    final_multiple: pd.DataFrame,
    blocks: pd.DataFrame,
    yearly: pd.DataFrame,
) -> None:
    global_keep = multiple[multiple["sig_holm_global"].ne("")]
    final_keep = final_multiple[final_multiple["sig_holm_global"].ne("")]
    final_downside = final_multiple[final_multiple["category"].eq("하방방어")]
    downside = multiple[multiple["category"].eq("하방 방어")]
    block_keep = blocks.assign(
        bh_mdd_fmt=blocks["bh_mdd"].map(format_pct),
        kfgi_mdd_fmt=blocks["kfgi_mdd"].map(format_pct),
        downside_p_fmt=blocks["downside_p"].map(format_p),
    )
    yearly_summary = {
        "years": len(yearly),
        "mdd_improved_years": int((yearly["mdd_improvement_pctp"] > 0).sum()),
        "downside_positive_years": int((yearly["downside_excess_bp"] > 0).sum()),
        "downside_sig_5pct_years": int((yearly["downside_p"] < 0.05).sum()),
    }
    lines = [
        "# 통계 강건성 보강 결과",
        "",
        "이 문서는 최종 본문 헤드라인으로 사용하는 lean K-FGI 결과와 숫자를 맞춘 강건성 보강표이다. 핵심 입력 시계열은 `10_Paper_outputs/data/q1_benchmark_timeseries.csv`이며, 전체 표본은 2,445거래일, B&H 하락일은 1,130일, 최종 K-FGI 하락일 방어폭은 43.1bp/day이다.",
        "",
        "## 1. 최종 lean K-FGI 기준 멀티플 테스팅 보정",
        "",
        f"- 보정 대상 최종 검정 수: {len(final_multiple)}개.",
        f"- 전체 검정 묶음 기준 Holm 5% 유의 검정: {(final_multiple['p_holm_global'] < 0.05).sum()}개.",
        f"- 전체 검정 묶음 기준 Benjamini-Hochberg/FDR 5% 유의 검정: {(final_multiple['p_bh_global'] < 0.05).sum()}개.",
        f"- 하방방어 family 내부 Holm 5% 유의 검정: {(final_downside['p_holm_family'] < 0.05).sum()} / {len(final_downside)}개.",
        "",
        "최종 모델 기준 보정표는 `10_Paper_outputs/robustness/tables/final_master_multiple_testing_adjusted_tests.csv`에 저장했다.",
        "",
        "### 최종 모델에서 global Holm 보정 후에도 유의한 검정",
        "",
        to_markdown_table(final_keep[["category", "test", "p_value", "p_holm_global", "p_bh_global", "effect", "unit"]]),
        "",
        "## 2. 서브기간 robustness",
        "",
        f"- 연도별 window 수: {yearly_summary['years']}개.",
        f"- K-FGI가 B&H보다 MDD를 개선한 연도: {yearly_summary['mdd_improved_years']} / {yearly_summary['years']}개.",
        f"- 하락일 초과방어가 양수인 연도: {yearly_summary['downside_positive_years']} / {yearly_summary['years']}개.",
        f"- 하락일 방어 p<0.05인 연도: {yearly_summary['downside_sig_5pct_years']} / {yearly_summary['years']}개.",
        "",
        "### 큰 서브기간별 결과",
        "",
        to_markdown_table(
            block_keep[
                [
                    "period",
                    "n",
                    "down_days",
                    "bh_mdd_fmt",
                    "kfgi_mdd_fmt",
                    "mdd_improvement_pctp",
                    "downside_excess_bp",
                    "downside_p_fmt",
                ]
            ]
        ),
        "",
        "## 3. 기존 2,505일 확장 검정표 처리",
        "",
        f"- 기존 `extended_statistical_tests.csv` 기준 보조 보정 대상 p-value: {len(multiple)}개.",
        f"- 기존 표 기준 Global Holm 5% 유의 검정: {(multiple['p_holm_global'] < 0.05).sum()}개.",
        f"- 기존 표 기준 하방방어 family 내부 Holm 5% 유의 검정: {(downside['p_holm_family'] < 0.05).sum()} / {len(downside)}개.",
        "",
        "이 기존 보조 보정표는 `10_Paper_outputs/robustness/tables/multiple_testing_adjusted_tests.csv`에 보관하지만, 본문이나 최종 부록의 대표 숫자로 사용하지 않는다. 이 표는 2,505일/1,161 하락일 기준 raw 또는 pre-lean 시계열에서 나온 35.34bp 결과이므로, 최종 lean K-FGI의 43.1bp 결과와 섞으면 안 된다.",
        "",
        "## 4. 논문 프레이밍 결론",
        "",
        "멀티플 테스팅 보정 이후에도 가장 방어 가능한 핵심 주장은 시장 하락일 방어, CVaR/손실 꼬리 개선, 변동성 감소, 고변동성 구간 노출 축소다. "
        "반대로 전체 초과수익, 단순 수익률 예측, K-FGI 극단 구간의 평균수익 차이는 보정 후 설득력이 약하거나 보조적이다. "
        "MDD는 full-sample에서 개선되지만 bootstrap 검정에서 5% 기준 유의하지 않으므로 기술통계상 보조 성과로만 제시한다. "
        "따라서 논문 전면에는 수익률 예측력이 아니라 `시장 하락일 손실 완화`, `CVaR 개선`, `고변동성/Fear 구간 노출 축소`, `연도별/비중첩 서브기간 반복성`을 배치하는 것이 안전하다.",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    multiple = build_multiple_testing_table()
    final_multiple = build_final_master_multiple_testing_table()
    blocks, yearly = build_subperiod_tables()
    multiple.to_csv(TABLE_DIR / "multiple_testing_adjusted_tests.csv", index=False, encoding="utf-8-sig")
    final_multiple.to_csv(TABLE_DIR / "final_master_multiple_testing_adjusted_tests.csv", index=False, encoding="utf-8-sig")
    blocks.to_csv(TABLE_DIR / "subperiod_block_robustness.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(TABLE_DIR / "subperiod_yearly_robustness.csv", index=False, encoding="utf-8-sig")
    write_summary(multiple, final_multiple, blocks, yearly)
    print(f"saved {TABLE_DIR / 'multiple_testing_adjusted_tests.csv'}")
    print(f"saved {TABLE_DIR / 'final_master_multiple_testing_adjusted_tests.csv'}")
    print(f"saved {TABLE_DIR / 'subperiod_block_robustness.csv'}")
    print(f"saved {TABLE_DIR / 'subperiod_yearly_robustness.csv'}")
    print(f"saved {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
