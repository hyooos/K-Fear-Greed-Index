from __future__ import annotations

import math
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy import stats


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def find_child(parent: Path, normalized_name: str) -> Path:
    for child in parent.iterdir():
        if nfc(child.name) == normalized_name:
            return child
    raise FileNotFoundError(f"{normalized_name} not found under {parent}")


ROOT = Path.cwd().resolve()
PROJECT_ROOT = ROOT.parent if nfc(ROOT.name) == "kfgi_최종" else ROOT
CODE_ROOT = find_child(PROJECT_ROOT, "kfgi_최종")
PAPER_ROOT = find_child(PROJECT_ROOT, "논문용")
REPO_OUT = CODE_ROOT / "paper_outputs"
PAPER_OUT = PAPER_ROOT / "10_Paper_outputs"
TRADING_DAYS = 252


def setup_plot() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    for path in [
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
    ]:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def perf_metrics(r: pd.Series, bench: pd.Series | None = None) -> dict[str, float]:
    r = r.dropna()
    if len(r) == 0:
        return {}
    cum = np.exp(r.cumsum())
    dd = cum / cum.cummax() - 1
    ann_ret = float(np.exp(r.mean() * TRADING_DAYS) - 1)
    ann_vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    out = {
        "n": len(r),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "mdd": float(dd.min()),
        "total_return": float(cum.iloc[-1] - 1),
        "positive_day_rate": float((r > 0).mean()),
    }
    if bench is not None:
        b = bench.loc[r.index].dropna()
        aligned = pd.concat([r, b], axis=1).dropna()
        if len(aligned) > 2:
            diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
            out["mean_excess_bp"] = float(diff.mean() * 10000)
            out["excess_t"] = float(stats.ttest_1samp(diff, 0).statistic)
            out["excess_p"] = float(stats.ttest_1samp(diff, 0).pvalue)
            out["win_rate_vs_bh"] = float((diff > 0).mean())
    return out


def max_drawdown(r: pd.Series) -> float:
    cum = np.exp(r.cumsum())
    return float((cum / cum.cummax() - 1).min())


def rolling_table(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    rows = []
    for i in range(window, len(df) + 1):
        w = df.iloc[i - window : i]
        main = w["main_kfgi_sentiment"]
        bh = w["buy_hold"]
        rows.append(
            {
                "date": w["date"].iloc[-1],
                "window_days": window,
                "kfgi_sharpe": perf_metrics(main)["sharpe"],
                "bh_sharpe": perf_metrics(bh)["sharpe"],
                "sharpe_diff": perf_metrics(main)["sharpe"] - perf_metrics(bh)["sharpe"],
                "kfgi_mdd": max_drawdown(main),
                "bh_mdd": max_drawdown(bh),
                "mdd_improvement": max_drawdown(main) - max_drawdown(bh),
                "kfgi_total_return": float(np.exp(main.sum()) - 1),
                "bh_total_return": float(np.exp(bh.sum()) - 1),
            }
        )
    return pd.DataFrame(rows)


def year_oos_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in df.groupby(df["date"].dt.year):
        if len(g) < 120:
            continue
        for label, col in [
            ("K-FGI", "main_kfgi_sentiment"),
            ("Buy&Hold", "buy_hold"),
            ("Trend only", "trend_only"),
            ("Trend+EGARCH", "trend_egarch_vol"),
        ]:
            metrics = perf_metrics(g.set_index("date")[col], g.set_index("date")["buy_hold"] if col != "buy_hold" else None)
            rows.append({"test_year": int(year), "strategy": label, **metrics})
    return pd.DataFrame(rows)


def conditional_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["year"] = d["date"].dt.year
    d["market_condition"] = np.where(d["buy_hold"] < 0, "B&H 하락일", "B&H 상승/보합일")
    d["kfgi_zone"] = pd.cut(d["K_FGI"], bins=[-np.inf, 25, 65, np.inf], labels=["Fear(<25)", "Neutral", "Greed(>65)"])
    d["vol_quartile"] = pd.qcut(d["egarch_vol"].replace([np.inf, -np.inf], np.nan).clip(lower=0, upper=0.20), 4, labels=["Q1 저변동", "Q2", "Q3", "Q4 고변동"])
    d["exposure_bucket"] = pd.cut(d["weight"], bins=[-np.inf, 0.5, 1.0, np.inf], labels=["방어(<0.5x)", "중립(0.5~1x)", "공격(>1x)"])

    rows = []
    for group_col in ["regime", "market_condition", "kfgi_zone", "vol_quartile", "exposure_bucket"]:
        for name, g in d.dropna(subset=[group_col]).groupby(group_col, observed=True):
            main = g.set_index("date")["main_kfgi_sentiment"]
            bh = g.set_index("date")["buy_hold"]
            rows.append({"dimension": group_col, "bucket": str(name), **perf_metrics(main, bh)})
    cond = pd.DataFrame(rows)

    tests = []
    for group_col in ["regime", "kfgi_zone", "vol_quartile", "exposure_bucket"]:
        groups = [g["main_kfgi_sentiment"].dropna().values for _, g in d.dropna(subset=[group_col]).groupby(group_col, observed=True)]
        if len(groups) >= 2 and all(len(x) > 2 for x in groups):
            f = stats.f_oneway(*groups)
            kw = stats.kruskal(*groups)
            tests.append({"dimension": group_col, "test": "one-way ANOVA", "stat": float(f.statistic), "p_value": float(f.pvalue)})
            tests.append({"dimension": group_col, "test": "Kruskal-Wallis", "stat": float(kw.statistic), "p_value": float(kw.pvalue)})
    # Paired downside defense in negative market condition.
    down = d[d["buy_hold"] < 0].copy()
    diff = down["main_kfgi_sentiment"] - down["buy_hold"]
    tests.append(
        {
            "dimension": "market_condition",
            "test": "B&H 하락일 paired t-test",
            "stat": float(stats.ttest_1samp(diff, 0).statistic),
            "p_value": float(stats.ttest_1samp(diff, 0).pvalue),
        }
    )
    return cond, pd.DataFrame(tests)


def save_fig(name: str) -> None:
    for out in [REPO_OUT / "figures", PAPER_OUT / "figures"]:
        out.mkdir(parents=True, exist_ok=True)
        plt.savefig(out / name, dpi=220, bbox_inches="tight")
    plt.close()


def draw_figures(rolling: pd.DataFrame, yearly: pd.DataFrame, cond: pd.DataFrame) -> None:
    setup_plot()
    fig, axes = plt.subplots(2, 1, figsize=(16, 7.6), sharex=True, gridspec_kw={"hspace": 0.18})
    axes[0].plot(rolling["date"], rolling["kfgi_sharpe"], color="#2166D5", lw=2, label="K-FGI 1Y rolling Sharpe")
    axes[0].plot(rolling["date"], rolling["bh_sharpe"], color="#AAB7C4", lw=1.6, label="B&H 1Y rolling Sharpe")
    axes[0].axhline(0, color="#666", lw=0.8)
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.92)
    axes[0].set_ylabel("Sharpe")
    axes[1].plot(rolling["date"], rolling["mdd_improvement"] * 100, color="#1A9A74", lw=2, label="MDD 개선폭")
    axes[1].axhline(0, color="#666", lw=0.8)
    axes[1].legend(loc="upper left", frameon=True, framealpha=0.92)
    axes[1].set_ylabel("MDD 개선폭 (%p)")
    axes[1].set_xlabel("Date")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(0.01, 0.01, "읽는 법: 위는 1년 rolling Sharpe, 아래는 같은 1년 구간에서 K-FGI가 Buy&Hold 대비 낙폭을 얼마나 줄였는지 보여준다.", fontsize=9)
    save_fig("17_rolling_OOS_성과안정성.png")

    y = yearly[yearly["strategy"].isin(["K-FGI", "Buy&Hold"])].copy()
    pivot = y.pivot(index="test_year", columns="strategy", values="sharpe")
    fig, ax = plt.subplots(figsize=(12, 5.4))
    x = np.arange(len(pivot.index))
    ax.bar(x - 0.18, pivot["Buy&Hold"], width=0.36, color="#C8D1DC", label="Buy&Hold")
    ax.bar(x + 0.18, pivot["K-FGI"], width=0.36, color="#2166D5", label="K-FGI")
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Sharpe")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)
    fig.text(0.01, 0.01, "읽는 법: 각 연도를 독립적인 OOS 검증 구간처럼 보고, K-FGI와 Buy&Hold의 연도별 Sharpe를 비교한다.", fontsize=9)
    save_fig("18_연도별_OOS_Sharpe.png")

    c = cond[(cond["dimension"] == "regime") & cond["bucket"].isin(["bull", "normal", "crisis"])].copy()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(c["bucket"], c["mean_excess_bp"], color=["#2D6CDF", "#9EC5E8", "#F4A261"][: len(c)])
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_ylabel("평균 초과수익 (bp/day)")
    for x, val in zip(c["bucket"], c["mean_excess_bp"]):
        ax.text(x, val + (0.4 if val >= 0 else -0.8), f"{val:.1f}", ha="center", fontsize=9)
    fig.text(0.01, 0.01, "읽는 법: 시장 국면별로 K-FGI가 Buy&Hold 대비 평균적으로 얼마나 초과/방어 성과를 냈는지 보여준다.", fontsize=9)
    save_fig("19_국면별_초과성과.png")


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    d = df[cols].copy()
    for col in d.columns:
        d[col] = d[col].map(lambda x: "" if pd.isna(x) else (f"{x:.4g}" if isinstance(x, float) else str(x)))
    rows = ["| " + " | ".join(d.columns) + " |", "| " + " | ".join(["---"] * len(d.columns)) + " |"]
    for _, row in d.iterrows():
        rows.append("| " + " | ".join(row.astype(str)) + " |")
    return "\n".join(rows)


def main() -> None:
    ret = pd.read_csv(REPO_OUT / "data" / "strategy_returns_10y.csv", parse_dates=["date"])
    kfgi = pd.read_csv(REPO_OUT / "data" / "kfgi_10y_timeseries.csv", parse_dates=["date"])
    df = ret.merge(kfgi[["date", "K_FGI", "regime", "egarch_vol", "weight", "vol_regime_high"]], on="date", how="inner")
    rolling = rolling_table(df, 252)
    yearly = year_oos_table(df)
    cond, tests = conditional_tables(df)

    for out in [REPO_OUT / "tables", PAPER_OUT / "tables"]:
        out.mkdir(parents=True, exist_ok=True)
        rolling.to_csv(out / "rolling_oos_1y_performance.csv", index=False, encoding="utf-8-sig")
        yearly.to_csv(out / "yearly_oos_detailed_performance.csv", index=False, encoding="utf-8-sig")
        cond.to_csv(out / "conditional_regime_performance.csv", index=False, encoding="utf-8-sig")
        tests.to_csv(out / "conditional_regime_tests.csv", index=False, encoding="utf-8-sig")

    draw_figures(rolling, yearly, cond)

    rsum = {
        "rolling_windows": len(rolling),
        "rolling_sharpe_win_rate": float((rolling["kfgi_sharpe"] > rolling["bh_sharpe"]).mean()),
        "rolling_mdd_improve_rate": float((rolling["mdd_improvement"] > 0).mean()),
        "rolling_avg_sharpe_diff": float(rolling["sharpe_diff"].mean()),
        "rolling_avg_mdd_improvement_pctp": float(rolling["mdd_improvement"].mean() * 100),
    }
    y_pivot = yearly.pivot(index="test_year", columns="strategy", values="sharpe")
    year_win = float((y_pivot["K-FGI"] > y_pivot["Buy&Hold"]).mean())
    cond_show = cond.sort_values(["dimension", "bucket"]).copy()
    tests_show = tests.copy()
    tests_show["p_label"] = tests_show["p_value"].map(lambda p: "<0.001" if p < 0.001 else f"{p:.4f}")

    md = f"""# Rolling OOS and Regime Evaluation

이 평가는 이미 walk-forward 방식으로 산출된 K-FGI 일별 전략 수익률을 사용해, 기간별 안정성과 조건부 성과를 추가 검증한 것이다.

## 핵심 요약

- 1년 rolling 평가창 수: {rsum['rolling_windows']:,}
- 1년 rolling Sharpe가 Buy&Hold보다 높았던 비율: {rsum['rolling_sharpe_win_rate']:.1%}
- 1년 rolling MDD가 Buy&Hold보다 개선된 비율: {rsum['rolling_mdd_improve_rate']:.1%}
- 평균 rolling Sharpe 차이: {rsum['rolling_avg_sharpe_diff']:.3f}
- 평균 rolling MDD 개선폭: {rsum['rolling_avg_mdd_improvement_pctp']:.2f}%p
- 연도별 OOS Sharpe 기준 K-FGI가 Buy&Hold를 이긴 연도 비율: {year_win:.1%}

## 논문에 쓸 수 있는 해석

K-FGI 전략은 전체 기간 초과수익의 유의성보다는, 1년 단위 rolling window와 시장 국면별 조건부 평가에서 위험 관리 성격이 더 뚜렷하게 나타난다. 특히 rolling MDD 개선 비율과 하락일 조건부 방어 효과를 함께 제시하면, 본 전략의 기여가 수익 극대화보다 하방 위험 완화에 있음을 더 명확히 설명할 수 있다.

주의: `B&H 상승/보합일`, `B&H 하락일`처럼 특정 조건만 뽑은 표본의 연율수익률은 경제적 직관과 다르게 과장될 수 있으므로 본문에서는 보조적으로만 사용한다. 해당 조건부 분석에서는 `mean_excess_bp`, `win_rate_vs_bh`, `paired t-test`를 중심으로 해석한다.

## Rolling OOS 요약

| 지표 | 값 |
| --- | --- |
| Rolling Sharpe 우위 비율 | {rsum['rolling_sharpe_win_rate']:.1%} |
| Rolling MDD 개선 비율 | {rsum['rolling_mdd_improve_rate']:.1%} |
| 평균 Rolling Sharpe 차이 | {rsum['rolling_avg_sharpe_diff']:.3f} |
| 평균 Rolling MDD 개선폭 | {rsum['rolling_avg_mdd_improvement_pctp']:.2f}%p |

## 조건부 성과표

{md_table(cond_show, ['dimension','bucket','n','ann_ret','ann_vol','sharpe','mdd','mean_excess_bp','excess_p','win_rate_vs_bh'])}

## 조건부 검정표

{md_table(tests_show, ['dimension','test','stat','p_label'])}

## 생성 파일

- `tables/rolling_oos_1y_performance.csv`
- `tables/yearly_oos_detailed_performance.csv`
- `tables/conditional_regime_performance.csv`
- `tables/conditional_regime_tests.csv`
- `figures/17_rolling_OOS_성과안정성.png`
- `figures/18_연도별_OOS_Sharpe.png`
- `figures/19_국면별_초과성과.png`
"""
    for out in [REPO_OUT, CODE_ROOT / "docs", PAPER_OUT]:
        (out / "ROLLING_OOS_REGIME_EVALUATION.md").write_text(md, encoding="utf-8")

    print("rolling_summary", rsum)
    print("yearly_sharpe_win_rate", year_win)
    print("saved", REPO_OUT / "ROLLING_OOS_REGIME_EVALUATION.md")


if __name__ == "__main__":
    main()
