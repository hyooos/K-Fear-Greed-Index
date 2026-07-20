from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_outputs"
TABLES = OUT / "tables"
FIGS = OUT / "final_figures"
DATA = OUT / "data"


def setup_plot() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 220


def load_panel() -> pd.DataFrame:
    robust = pd.read_csv(OUT / "robustness/data/priority_ab_kfgi_timeseries_robust.csv", parse_dates=["date"])
    q1 = pd.read_csv(DATA / "q1_benchmark_timeseries.csv", parse_dates=["date"])
    df = robust.merge(
        q1[["date", "buy_hold_return", "K-FGI final", "K-FGI final_weight"]],
        on="date",
        how="inner",
    )
    df["final_weight"] = df["K-FGI final_weight"].fillna(df["weight"].clip(0, 1))
    df["kfgi_zone"] = np.select([df["K_FGI"] < 25, df["K_FGI"] > 65], ["Fear", "Greed"], default="Neutral")
    df["sent_pct"] = df["sent_composite_ma10"].rank(pct=True) * 100
    df["egarch_pct"] = df["egarch_vol"].rank(pct=True) * 100
    return df.sort_values("date").reset_index(drop=True)


def make_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_defs = pd.DataFrame(
        [
            ["Sortino ratio", "Annualized return divided by downside deviation", "수익률 대비 하방변동성 보상", "보조 지표. B&H가 더 높을 수 있으므로 과장 금지"],
            ["Calmar ratio", "Annualized return divided by absolute MDD", "최대낙폭 대비 연수익률", "보조 지표. MDD 방어와 함께 해석"],
            ["VaR 5%", "5th percentile of daily returns", "나쁜 5% 경계 손실", "손실 꼬리 시작점"],
            ["CVaR 5% / Expected Shortfall", "Average return below VaR 5%", "최악 5% 평균 손실", "downside risk 핵심 지표로 본문에 사용"],
            ["Downside deviation", "Annualized root mean square of negative returns", "음수 수익률만 본 변동성", "Sortino 계산과 하방위험 보조"],
            ["Market-down-day defense", "Average K-FGI return minus B&H return on B&H down days", "시장 하락일 손실 방어폭", "본 논문의 핵심 성과지표"],
        ],
        columns=["metric", "definition", "economic_meaning", "paper_usage"],
    )
    metric_defs.to_csv(TABLES / "downside_metric_definitions.csv", index=False)

    interp = pd.DataFrame(
        [
            [
                "Fear zone",
                "K-FGI < 25",
                "평균 노출 0.226x, 하락일 방어 109.8bp",
                "시장 하위지표와 감성/변동성 신호가 동시에 위험 상태를 가리키면 노출을 급격히 축소한다.",
                "K-FGI가 공포 국면에서 손실을 방어하는 가장 직접적인 근거.",
            ],
            [
                "High volatility Q4",
                "EGARCH volatility top quartile",
                "평균 노출 0.453x, 하락일 방어 68.2bp",
                "조건부 변동성이 높을수록 risk-scaling block이 노출을 낮춘다.",
                "기존 volatility timing과 연결되지만, K-FGI는 감성/시장상태까지 결합한다.",
            ],
            [
                "COVID shock",
                "2020 market stress",
                "2020 하락일 방어 78.7bp",
                "가격 급락, 옵션/변동성 공포, 댓글 감성 악화가 동시에 나타난 전형적 stress regime.",
                "K-FGI가 위기 구간에서 작동했다는 event-study 근거.",
            ],
            [
                "2022 rate-hike bear market",
                "2022 tightening/bear regime",
                "2022 하락일 방어 82.8bp",
                "금리 상승과 위험회피가 지속되며 안전자산 선호와 변동성 신호가 방어적으로 작동했다.",
                "단기 이벤트가 아니라 지속 약세장에서도 방어 가능함을 보여줌.",
            ],
            [
                "Korean retail sentiment",
                "Naver comment sentiment",
                "sent_composite_ma10 RF 23.4%, ElasticNet 17.3%",
                "한국 시장은 개인투자자 비중과 온라인 금융 커뮤니케이션이 커서 댓글 감성이 행동 상태 변수로 의미를 갖는다.",
                "한국 시장 특화 novelty의 핵심.",
            ],
            [
                "Low total return",
                "Average exposure 0.555x",
                "B&H보다 수익률 낮음",
                "평균적으로 시장 노출을 절반 수준으로 낮췄기 때문에 강세장에서 기회비용이 발생한다.",
                "약점이 아니라 downside-risk 목적함수의 trade-off로 설명.",
            ],
        ],
        columns=["topic", "condition", "key_result", "economic_interpretation", "paper_usage"],
    )
    interp.to_csv(TABLES / "economic_interpretation_map.csv", index=False)

    dates = ["2020-03-13", "2020-03-19", "2020-03-23", "2022-09-30", "2025-04-07"]
    rows = []
    for d in dates:
        row = df[df["date"].eq(pd.Timestamp(d))]
        if row.empty:
            continue
        r = row.iloc[0]
        rows.append(
            {
                "date": r["date"].date().isoformat(),
                "K_FGI": r["K_FGI"],
                "zone": r["kfgi_zone"],
                "exposure": r["final_weight"],
                "market_return_pct": r["buy_hold_return"] * 100,
                "strategy_return_pct": r["K-FGI final"] * 100,
                "sub_index2_strength": r["sub_index2"],
                "sub_index3_breadth": r["sub_index3"],
                "sub_index4_options": r["sub_index4"],
                "sub_index5_volatility": r["sub_index5"],
                "sub_index6_safe_demand": r["sub_index6"],
                "sub_index7_credit_risk": r["sub_index7"],
                "sent_composite_ma10": r["sent_composite_ma10"],
                "sent_percentile": r["sent_pct"],
                "egarch_vol_pct": r["egarch_vol"] * 100,
                "egarch_percentile": r["egarch_pct"],
                "vol_ratio": r["vol_ratio"],
            }
        )
    case = pd.DataFrame(rows)
    case.to_csv(TABLES / "case_study_exposure_explainability.csv", index=False)
    return metric_defs, interp, case


def make_case_figure(df: pd.DataFrame) -> None:
    setup_plot()
    FIGS.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp("2020-02-20")
    end = pd.Timestamp("2020-04-10")
    w = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    event = pd.Timestamp("2020-03-19")
    erow = df[df["date"].eq(event)].iloc[0]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.05, 1.15], hspace=0.22)
    ax1 = fig.add_subplot(gs[0])
    colors = np.where(w["buy_hold_return"] >= 0, "#94a3b8", "#ef4444")
    ax1.bar(w["date"], w["buy_hold_return"] * 100, color=colors, width=0.8)
    ax1.axvline(event, color="#111827", linewidth=1.0, linestyle="--")
    ax1.set_ylabel("Market return (%)")
    ax1.grid(axis="y", alpha=0.28)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(w["date"], w["K_FGI"], color="#7c3aed", linewidth=2.0, label="K-FGI")
    ax2.axhline(25, color="#3b82f6", linestyle=":", linewidth=1.2, label="Fear threshold")
    ax2b = ax2.twinx()
    ax2b.step(w["date"], w["final_weight"], color="#2563eb", linewidth=1.8, where="post", label="Exposure")
    ax2.axvline(event, color="#111827", linewidth=1.0, linestyle="--")
    ax2.set_ylabel("K-FGI")
    ax2b.set_ylabel("Exposure (x)")
    ax2.set_ylim(-2, 102)
    ax2b.set_ylim(-0.05, 1.05)
    ax2.grid(axis="y", alpha=0.28)
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left", ncol=3, fontsize=9)

    ax3 = fig.add_subplot(gs[2])
    components = pd.Series(
        {
            "Market strength\nsub2": erow["sub_index2"],
            "Breadth\nsub3": erow["sub_index3"],
            "Options\nsub4": erow["sub_index4"],
            "Volatility\nsub5": erow["sub_index5"],
            "Safe demand\nsub6": erow["sub_index6"],
            "Credit risk\nsub7": erow["sub_index7"],
            "Sentiment pct": erow["sent_pct"],
            "EGARCH pct": erow["egarch_pct"],
        }
    )
    bar_colors = ["#ef4444" if v < 25 else "#f59e0b" if v < 50 else "#2563eb" for v in components.values]
    bars = ax3.bar(components.index, components.values, color=bar_colors)
    ax3.axhline(25, color="#3b82f6", linestyle=":", linewidth=1)
    ax3.set_ylim(0, 105)
    ax3.set_ylabel("Score / percentile")
    ax3.grid(axis="y", alpha=0.28)
    for b, v in zip(bars, components.values):
        ax3.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}", ha="center", fontsize=8)

    for ax in [ax1, ax2]:
        ax.tick_params(axis="x", labelbottom=False)
    ax3.tick_params(axis="x", rotation=0)
    fig.text(
        0.01,
        0.01,
        f"Case date {event.date()}: K-FGI={erow['K_FGI']:.1f}, exposure={erow['final_weight']:.2f}x, EGARCH vol={erow['egarch_vol']*100:.2f}%, vol ratio={erow['vol_ratio']:.2f}.",
        fontsize=9,
        color="#475569",
    )
    fig.savefig(FIGS / "17_case_study_20200319_exposure_explainability.png", bbox_inches="tight")
    plt.close(fig)


def write_md(metric_defs: pd.DataFrame, interp: pd.DataFrame, case: pd.DataFrame) -> None:
    md = []
    md.append("# Economic Interpretation and Explainability Layer\n")
    md.append("본 문서는 downside risk 지표, 경제적 해석, 특정 날짜 exposure explainability를 reviewer 대응용으로 정리한 것이다.\n")
    md.append("## 1. Downside Risk Metrics\n")
    md.append(metric_defs.to_markdown(index=False))
    md.append("\n\n핵심은 Expected Shortfall(CVaR 5%)이다. K-FGI의 CVaR 5%는 Buy & Hold보다 작으므로, 제목의 downside risk management와 가장 잘 맞는 보조 지표로 사용할 수 있다.\n")
    md.append("\n## 2. Economic Interpretation Map\n")
    md.append(interp.to_markdown(index=False))
    md.append("\n\nReviewer에게는 숫자 자체보다 왜 그런 숫자가 나왔는지 설명해야 한다. 특히 Fear zone의 109.8bp 방어는 평균 노출이 0.226x까지 낮아진 exposure-control mechanism과 연결해 설명한다.\n")
    md.append("\n## 3. Case Study: 2020-03-19\n")
    tmp = case.copy()
    for c in tmp.columns:
        if c not in ["date", "zone"]:
            tmp[c] = tmp[c].astype(float).round(3)
    md.append(tmp.to_markdown(index=False))
    md.append("\n\n2020-03-19 전후에는 K-FGI가 0에 가까운 crisis/fear 상태로 하락했고, 최종 exposure가 0.00x까지 축소되었다. 이는 모델이 단순히 사후적으로 좋은 구간을 고른 것이 아니라, 시장 breadth 붕괴, 변동성 상승, 감성 악화가 동시에 나타난 시점에 노출을 줄였음을 보여주는 사례로 사용할 수 있다.\n")
    md.append("\n## 4. 사용할 그림\n")
    md.append("- `paper_outputs/final_figures/17_case_study_20200319_exposure_explainability.png`\n")
    md.append("\n## 5. 논문 삽입 위치\n")
    md.append("| 항목 | 위치 | 목적 |\n| --- | --- | --- |\n| Downside metric definitions | Experimental Design | Sortino, Calmar, CVaR, Expected Shortfall 정의 |\n| Economic interpretation map | Discussion | 왜 Fear/COVID/한국 시장에서 의미가 있는지 설명 |\n| 2020-03-19 case study | Explainability 또는 Event Study | 특정 날짜 exposure 결정 근거 시각화 |\n| Exposure table | Empirical Results | 평균 노출과 현금보유 의심에 대응 |\n")
    (OUT / "ECONOMIC_INTERPRETATION_EXPLAINABILITY.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    df = load_panel()
    metric_defs, interp, case = make_tables(df)
    make_case_figure(df)
    write_md(metric_defs, interp, case)
    print("Wrote economic interpretation and explainability outputs.")


if __name__ == "__main__":
    main()
