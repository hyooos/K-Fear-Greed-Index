from __future__ import annotations

import json
import shutil
import textwrap
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


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
REPO_FIG = CODE_ROOT / "paper_outputs" / "figures"
REPO_TABLE = CODE_ROOT / "paper_outputs" / "tables"
REPO_DATA = CODE_ROOT / "paper_outputs" / "data"
PAPER_OUTPUT = PAPER_ROOT / "10_Paper_outputs"
PAPER_FIG = PAPER_OUTPUT / "figures"
PAPER_TABLE = PAPER_OUTPUT / "tables"
PAPER_DATA = PAPER_OUTPUT / "data"
FINAL_DATA = PAPER_ROOT / "01_final_dataset" / "KFG_final_10y.csv"
SENT_DATA = PAPER_ROOT / "05_Sentiment_analysis" / "data" / "recollect_sentiment_final_model_toxicity" / "daily_sentiment.csv"


FIGURES: dict[str, dict[str, str]] = {
    "01_filtering_retention_by_year.png": {
        "title": "최종 감성 분석 표본이 연도별로 충분히 확보되었는가",
        "message": "연도별 최종 댓글 수와 독성 가중치 적용 후 유효표본을 함께 보여 주어, 감성 피처가 특정 연도에만 과도하게 의존하지 않는지 확인한다.",
        "read": "막대는 최종 감성 분석에 사용된 댓글 수, 선은 독성 가중치 반영 후 유효표본 수다. 두 값의 차이가 크면 독성/노이즈 조정 효과가 큰 연도다.",
    },
    "02_toxicity_weight_curve.png": {
        "title": "독성 점수를 제거가 아니라 가중치로 반영한 이유",
        "message": "본 연구는 독성 점수가 높을수록 댓글 영향력을 완만하게 낮추는 gamma=2 가중치를 사용한다.",
        "read": "x축은 독성 점수, y축은 감성 계산에 반영되는 댓글 가중치다. 0.95 이상은 hard drop 후보 구간이다.",
    },
    "03_sentiment_normalization_before_after.png": {
        "title": "감성 점수 정규화가 필요한 이유",
        "message": "원 점수는 0 부근에 지나치게 몰려 있어 시장 심리 변화를 충분히 드러내지 못한다. 정규화 후에는 음/양 방향 변동성이 살아난다.",
        "read": "왼쪽은 원 감성 점수, 오른쪽은 정규화 점수다. 오른쪽 분포가 더 넓으면 모델이 감성 변화를 피처로 사용할 여지가 커진다.",
    },
    "04_sentiment_activity_and_trend.png": {
        "title": "감성 데이터의 양과 방향성이 시간에 따라 유지되는가",
        "message": "댓글 수와 정규화 감성의 월별 흐름을 함께 보아 감성 피처의 시계열 안정성을 확인한다.",
        "read": "막대는 월별 댓글 수, 선은 월별 평균 감성이다. 댓글 수가 충분하고 감성 방향이 시간에 따라 움직이면 피처로 사용할 근거가 생긴다.",
    },
    "05_kfgi_timeseries_thresholds.png": {
        "title": "K-FGI가 공포/탐욕 구간을 어떻게 식별하는가",
        "message": "K-FGI는 0~100 스케일에서 낮을수록 fear, 높을수록 greed로 해석한다. 25/65 기준선은 극단 구간을 보는 참조선이다.",
        "read": "보라색 선은 20일 이동평균 K-FGI다. 파란 음영은 fear, 붉은 음영은 greed 구간이다.",
    },
    "06_subindex_yearly_heatmap.png": {
        "title": "개별 subindex가 연도별로 어떤 방향을 보였는가",
        "message": "K-FGI를 구성하는 하위 지표들이 특정 한두 지표에만 의존하지 않는지 확인한다.",
        "read": "색이 붉을수록 greed/risk-on 쪽, 푸를수록 fear/risk-off 쪽에 가깝다. 행별로 연도 변화 패턴을 비교한다.",
    },
    "07_strategy_cumulative_return.png": {
        "title": "K-FGI 전략은 Buy&Hold 대비 어떤 성과 경로를 보였는가",
        "message": "K-FGI 전략은 Buy&Hold보다 총수익률은 약간 낮지만, 장기적으로 비슷한 성과를 유지하면서 하락 구간의 손실을 줄이는 전략이다.",
        "read": "선은 1에서 시작한 누적 수익률이다. 성과 수준뿐 아니라 하락장에서 얼마나 완만하게 떨어지는지도 함께 본다.",
    },
    "08_strategy_drawdown.png": {
        "title": "K-FGI 전략의 핵심 장점은 낙폭 방어인가",
        "message": "Buy&Hold의 최대 낙폭은 -41.2%, K-FGI 전략은 -22.9%로 하방 위험이 크게 줄었다.",
        "read": "0에 가까울수록 손실이 작다. 회색 영역보다 파란 선이 위에 있으면 K-FGI가 손실을 덜 낸 구간이다.",
    },
    "09_egarch_volatility_exposure.png": {
        "title": "EGARCH 변동성이 시장 노출 조절에 어떻게 쓰였는가",
        "message": "조건부 변동성이 높아지는 구간에서 시장 노출을 낮추어 손실 확대를 방어하는 구조다.",
        "read": "주황색은 변동성, 파란색은 시장 노출도다. 변동성 피크 전후로 노출이 낮아지는지 확인한다.",
    },
    "10_feature_importance_grouped.png": {
        "title": "모델이 실제로 어떤 피처를 중요하게 보았는가",
        "message": "추세/변동성 피처가 가장 강하지만, 감성 피처도 상위권에 포함되어 보조 정보로 작동한다.",
        "read": "막대가 길수록 LightGBM gain 기준 중요도가 높다. 색은 피처 그룹을 의미한다.",
    },
    "11_weight_sensitivity_sharpe_heatmap.png": {
        "title": "왜 특정 가중치 조합을 선택했는가",
        "message": "가중치 선택은 한 번의 임의 선택이 아니라 여러 후보 조합의 Sharpe 민감도를 비교해 결정했다.",
        "read": "진한 색일수록 Sharpe가 높다. 별표는 실제 채택 조합이다.",
    },
    "12_weight_candidate_ranking.png": {
        "title": "채택 가중치는 후보군 안에서 어느 정도 성능인가",
        "message": "가장 공격적인 최고 Sharpe 조합만 택하지 않고, 성과와 안정성을 함께 고려한 조합을 채택했다.",
        "read": "막대는 후보 조합별 Sharpe다. 파란색은 채택 조합이다.",
    },
    "13_sentiment_ablation_performance.png": {
        "title": "감성 피처를 넣으면 무엇이 달라지는가",
        "message": "감성 포함 K-FGI는 감성 제외 버전보다 총수익률이 높다. 단 Sharpe는 감성 제외 버전이 더 높아 해석에 균형이 필요하다.",
        "read": "왼쪽은 총수익률, 오른쪽은 MDD다. 감성 피처의 효과는 수익률 개선과 변동성 증가를 함께 본다.",
    },
    "14_statistical_test_pvalues.png": {
        "title": "통계적으로 강하게 주장할 수 있는 부분은 무엇인가",
        "message": "전체 초과수익보다 하락 구간 방어 성능이 통계적으로 강하게 나타난다.",
        "read": "막대가 붉은 기준선 위면 p<0.05다. downside defense만 명확히 유의하다.",
    },
    "15_yearly_and_oos_performance.png": {
        "title": "성과가 특정 구간에만 몰려 있지는 않은가",
        "message": "연도별 편차는 존재하지만 2025년 OOS 구간에서 높은 성과를 보였다. 과적합 가능성은 연도별 성과와 함께 해석한다.",
        "read": "왼쪽은 연도별 총수익률, 오른쪽은 train/test 성과 비교다.",
    },
    "16_position_size.png": {
        "title": "K-FGI 전략이 실제로 시장 노출을 얼마나 조절했는가",
        "message": "포지션 크기는 단순 매수/매도가 아니라 K-FGI, 추세, 변동성 조건을 반영해 0배에서 2배 사이에서 조절된다.",
        "read": "위 패널은 일별 노출도와 20일 평균, 아래 패널은 연도별 평균 노출도다. 낮은 노출 구간은 위험 축소 국면으로 해석한다.",
    },
}

KOREAN_FILENAMES: dict[str, str] = {
    "01_filtering_retention_by_year.png": "01_댓글_필터링_통과율.png",
    "02_toxicity_weight_curve.png": "02_독성점수_가중치_함수.png",
    "03_sentiment_normalization_before_after.png": "03_감성점수_정규화_전후.png",
    "04_sentiment_activity_and_trend.png": "04_댓글수와_감성추세.png",
    "05_kfgi_timeseries_thresholds.png": "05_KFGI_공포탐욕_시계열.png",
    "06_subindex_yearly_heatmap.png": "06_하위지표_연도별_히트맵.png",
    "07_strategy_cumulative_return.png": "07_전략별_누적수익률.png",
    "08_strategy_drawdown.png": "08_KFGI_낙폭방어.png",
    "09_egarch_volatility_exposure.png": "09_EGARCH_변동성과_시장노출.png",
    "10_feature_importance_grouped.png": "10_피처중요도.png",
    "11_weight_sensitivity_sharpe_heatmap.png": "11_가중치_민감도_Sharpe.png",
    "12_weight_candidate_ranking.png": "12_가중치_후보_성능순위.png",
    "13_sentiment_ablation_performance.png": "13_감성피처_포함제외_성능비교.png",
    "14_statistical_test_pvalues.png": "14_통계검정_p값_요약.png",
    "15_yearly_and_oos_performance.png": "15_연도별_OOS_성과.png",
    "16_position_size.png": "16_포지션크기_시장노출.png",
}


def setup() -> None:
    REPO_FIG.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
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
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["axes.titlepad"] = 10


def remove_plot_titles() -> None:
    fig = plt.gcf()
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        ax.set_title("")


def save(name: str) -> None:
    remove_plot_titles()
    out_name = KOREAN_FILENAMES.get(name, name)
    for out_dir in [REPO_FIG, PAPER_FIG]:
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / out_name, dpi=220, bbox_inches="tight")
    plt.close()


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def load() -> dict[str, pd.DataFrame]:
    return {
        "kfgi": pd.read_csv(REPO_DATA / "kfgi_10y_timeseries.csv", parse_dates=["date"]),
        "ret": pd.read_csv(REPO_DATA / "strategy_returns_10y.csv", parse_dates=["date"]),
        "perf": pd.read_csv(REPO_TABLE / "performance_summary.csv"),
        "imp": pd.read_csv(REPO_TABLE / "feature_importance.csv"),
        "sens": pd.read_csv(REPO_TABLE / "sensitivity_analysis.csv"),
        "yearly": pd.read_csv(REPO_TABLE / "yearly_performance.csv"),
        "stat": pd.read_csv(REPO_TABLE / "statistical_tests.csv"),
        "oos": pd.read_csv(REPO_TABLE / "oos_2025_performance.csv"),
        "final": pd.read_csv(FINAL_DATA, parse_dates=["date"]),
        "sent": pd.read_csv(SENT_DATA, parse_dates=["date"]),
    }


def add_note(fig, text: str, y: float = 0.01) -> None:
    fig.text(
        0.01,
        y,
        textwrap.fill(text, 120),
        fontsize=9,
        color="#3A3A3A",
        ha="left",
        va="bottom",
    )


def format_year_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
    ax.grid(True, axis="y", alpha=0.32)
    ax.grid(True, axis="x", which="major", alpha=0.16)
    ax.margins(x=0.01)


def legend_above(ax, ncol: int = 3):
    return ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        frameon=False,
        borderaxespad=0.0,
    )


def filtering(sent: pd.DataFrame | None = None) -> None:
    rows = []
    base = CODE_ROOT / "3_Filtering_final" / "recollect_final_filtered_model_toxicity"
    for year in range(2015, 2026):
        p = base / str(year) / f"stock_classification_summary_{year}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        total = data.get("total_comments") or data.get("total") or data.get("input_rows")
        kept = data.get("final_stock_comments") or data.get("stock_comments") or data.get("kept_rows")
        if total and kept:
            rows.append({"year": year, "total": total, "kept": kept, "retention": kept / total})
    df = pd.DataFrame(rows)
    if df.empty:
        if sent is None:
            return
        d = sent[sent["date"].dt.year.between(2015, 2025)].copy()
        df = (
            d.groupby(d["date"].dt.year)
            .agg(total=("comment_count", "sum"), kept=("effective_n", "sum"))
            .reset_index()
            .rename(columns={"date": "year"})
        )
        df["retention"] = df["kept"] / df["total"].replace(0, np.nan)
    fig, ax1 = plt.subplots(figsize=(11, 5.4))
    ax1.bar(df["year"], df["total"] / 1000, color="#D9E8F5", label="수집 댓글 수")
    ax1.bar(df["year"], df["kept"] / 1000, color="#2D6CDF", alpha=0.78, label="가중치 반영 유효표본")
    ax1.set_ylabel("댓글 수 (천 개)")
    ax1.set_title("연도별 최종 감성 표본 규모")
    ax2 = ax1.twinx()
    ax2.plot(df["year"], df["retention"] * 100, color="#E67E22", marker="o", lw=2.2, label="유효표본/댓글 비율")
    ax2.set_ylabel("유효표본 비율 (%)")
    ax2.set_ylim(0, min(110, max(35, df["retention"].max() * 115)))
    for x, y in zip(df["year"], df["retention"] * 100):
        ax2.text(x, y + 0.8, f"{y:.1f}%", ha="center", fontsize=8, color="#9A4F00")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncol=3)
    add_note(fig, "읽는 법: 연한 막대는 최종 댓글 수, 진한 막대와 주황색 선은 독성 가중치 적용 후 실제 감성 평균에 반영되는 유효표본 규모를 뜻한다.")
    save("01_filtering_retention_by_year.png")


def toxicity() -> None:
    x = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(x, 1 - x, "--", color="#BFC5CE", lw=2, label="선형 감소")
    ax.plot(x, (1 - x) ** 2, color="#2D6CDF", lw=3, label="본 연구: gamma=2")
    ax.plot(x, (1 - x) ** 3, ":", color="#6C757D", lw=2.5, label="강한 패널티 gamma=3")
    ax.axvline(0.95, color="#E74C3C", ls="--", lw=1.7)
    ax.text(0.955, 0.88, "Hard drop 후보\nτ=0.95", color="#C0392B", fontsize=9)
    for tx in [0.2, 0.5, 0.8]:
        ty = (1 - tx) ** 2
        ax.scatter(tx, ty, color="#2D6CDF", s=45, zorder=4)
        ax.text(tx + 0.015, ty + 0.045, f"독성 {tx:.1f}\n가중치 {ty:.2f}", fontsize=8, color="#1F4E9A")
    ax.set_title("독성 점수에 따른 댓글 반영 가중치")
    ax.set_xlabel("독성 점수: 0=낮음, 1=높음")
    ax.set_ylabel("감성 계산 반영 가중치")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    add_note(fig, "핵심: 댓글을 단순 삭제하지 않고 독성이 높을수록 영향력을 줄여, 표본 손실과 노이즈 통제를 동시에 노린다.")
    save("02_toxicity_weight_curve.png")


def sentiment_norm(sent: pd.DataFrame) -> None:
    sample = sent[["sent_raw_mean_w", "sent_norm_w"]].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    axes[0].hist(sample["sent_raw_mean_w"], bins=70, color="#9EC5E8", edgecolor="white")
    axes[0].axvline(sample["sent_raw_mean_w"].median(), color="#1F4E9A", ls="--", label="median")
    axes[0].set_title("정규화 전: 0 부근 집중")
    axes[0].set_xlabel("원 감성 점수")
    axes[0].set_ylabel("일수")
    axes[1].hist(sample["sent_norm_w"], bins=70, color="#F4A261", edgecolor="white")
    axes[1].axvline(0, color="#555", ls="--", label="중립")
    axes[1].set_title("정규화 후: 음/양 방향 확대")
    axes[1].set_xlabel("정규화 감성 점수 (-1~1)")
    for ax in axes:
        ax.legend()
    fig.suptitle("감성 점수 정규화 전/후 분포 비교", fontweight="bold", fontsize=16)
    add_note(fig, "핵심: 원 감성 점수는 변동 폭이 작아 모델이 사용하기 어렵다. 정규화는 감성의 방향성과 강도를 더 분명하게 만든다.")
    save("03_sentiment_normalization_before_after.png")


def sentiment_activity(sent: pd.DataFrame) -> None:
    d = sent[sent["date"].dt.year.between(2015, 2025)].copy()
    monthly = d.set_index("date").resample("ME").agg(comment_count=("comment_count", "sum"), sent=("sent_norm_w", "mean")).reset_index()
    fig, ax1 = plt.subplots(figsize=(12, 5.2))
    ax1.bar(monthly["date"], monthly["comment_count"] / 1000, width=25, color="#D9E8F5", label="월별 댓글 수")
    ax1.set_ylabel("댓글 수 (천 개)")
    ax2 = ax1.twinx()
    ax2.plot(monthly["date"], monthly["sent"], color="#7C3AED", lw=2, label="월평균 정규화 감성")
    ax2.axhline(0, color="#666", ls="--", lw=1)
    ax2.set_ylim(-1, 1)
    ax2.set_ylabel("정규화 감성")
    ax1.set_title("월별 댓글 수와 감성 방향성")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    add_note(fig, "읽는 법: 댓글 수가 지나치게 적은 구간은 감성 피처 신뢰도가 낮을 수 있다. 보라색 선은 감성이 어느 방향으로 움직였는지 보여 준다.")
    save("04_sentiment_activity_and_trend.png")


def kfgi_series(kfgi: pd.DataFrame) -> None:
    d = kfgi.copy()
    d["K_FGI_20d"] = d["K_FGI"].rolling(20).mean()
    d["year"] = d["date"].dt.year
    d["zone"] = np.select(
        [d["K_FGI_20d"] < 25, d["K_FGI_20d"] > 65],
        ["Fear", "Greed"],
        default="Neutral",
    )
    zone_share = d.dropna(subset=["K_FGI_20d"]).pivot_table(
        index="year", columns="zone", values="K_FGI_20d", aggfunc="count", fill_value=0
    )
    zone_share = zone_share.div(zone_share.sum(axis=1), axis=0) * 100
    for col in ["Fear", "Neutral", "Greed"]:
        if col not in zone_share:
            zone_share[col] = 0
    zone_share = zone_share[["Fear", "Neutral", "Greed"]]

    fig, (ax, ax_bar) = plt.subplots(
        2, 1, figsize=(16, 8.2), sharex=False, gridspec_kw={"height_ratios": [3.2, 1.25], "hspace": 0.28}
    )
    ax.fill_between(d["date"], 0, 25, color="#DCEBFA", alpha=0.7, label="Fear zone (<25)")
    ax.fill_between(d["date"], 65, 100, color="#FADBD8", alpha=0.7, label="Greed zone (>65)")
    ax.plot(d["date"], d["K_FGI"], color="#BFA7FF", lw=0.55, alpha=0.28, label="K-FGI daily")
    ax.plot(d["date"], d["K_FGI_20d"], color="#5B21B6", lw=2.4, label="K-FGI 20D MA")
    ax.axhline(25, color="#2D6CDF", ls="--", lw=1)
    ax.axhline(65, color="#E74C3C", ls="--", lw=1)
    ax.set_ylim(0, 100)
    ax.set_title("10개년 K-FGI 시계열")
    ax.set_ylabel("K-FGI (0=Fear, 100=Greed)")
    format_year_axis(ax)
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92)

    bottom = np.zeros(len(zone_share))
    colors = {"Fear": "#7FB3E6", "Neutral": "#D8DEE7", "Greed": "#F28B82"}
    for col in ["Fear", "Neutral", "Greed"]:
        ax_bar.bar(zone_share.index, zone_share[col], bottom=bottom, color=colors[col], label=col, width=0.72)
        bottom += zone_share[col].values
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel("연도별 비중 (%)")
    ax_bar.set_xlabel("Year")
    ax_bar.set_title("연도별 Fear/Neutral/Greed 체류 비중", fontsize=12)
    ax_bar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False)
    ax_bar.grid(True, axis="y", alpha=0.28)
    ax_bar.set_xticks(zone_share.index)
    fig.subplots_adjust(top=0.91, bottom=0.13)
    add_note(fig, "핵심: 이 그림은 예측 성능 자체가 아니라, K-FGI가 시간에 따라 공포/탐욕 구간을 어떻게 오가는지 보여 주는 지표 설명용 그림이다.")
    save("05_kfgi_timeseries_thresholds.png")


def subindex_heatmap(final: pd.DataFrame) -> None:
    labels = {
        "sub_index2": "시장 강도",
        "sub_index3": "시장 폭",
        "sub_index4": "옵션 심리",
        "sub_index5": "변동성",
        "sub_index6": "안전자산 수요",
        "sub_index7": "채권 스프레드",
    }
    cols = list(labels)
    yearly = final.groupby(final["date"].dt.year)[cols].mean()
    fig, ax = plt.subplots(figsize=(11, 5.4))
    im = ax.imshow(yearly.T.values, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(yearly.index)))
    ax.set_xticklabels(yearly.index, rotation=45)
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels([labels[c] for c in cols])
    for i in range(len(cols)):
        for j in range(len(yearly.index)):
            ax.text(j, i, f"{yearly.iloc[j, i]:.0f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("평균 점수 (0=Fear, 100=Greed)")
    ax.set_title("연도별 K-FGI 하위지표 평균")
    add_note(fig, "읽는 법: 행별로 보면 각 subindex의 연도별 변화가 보인다. sub_index1은 최종 모델링에서 제외했기 때문에 여기서도 제외했다.")
    save("06_subindex_yearly_heatmap.png")


def performance(ret: pd.DataFrame, perf_df: pd.DataFrame) -> None:
    labels = {
        "buy_hold": "KOSPI200 B&H",
        "trend_only": "Trend only",
        "trend_egarch_vol": "Trend+EGARCH",
        "main_kfgi_sentiment": "K-FGI",
    }
    colors = {"buy_hold": "#AAB7C4", "trend_only": "#13A386", "trend_egarch_vol": "#F39C12", "main_kfgi_sentiment": "#2D6CDF"}
    d = ret.copy()
    fig, ax = plt.subplots(figsize=(16, 6.2))
    end_points = []
    for col, label in labels.items():
        y = np.exp(d[col].cumsum())
        lw = 2.8 if col == "main_kfgi_sentiment" else 1.7
        alpha = 1.0 if col == "main_kfgi_sentiment" else 0.78
        ax.plot(d["date"], y, label=label, color=colors[col], lw=lw, alpha=alpha)
        end_points.append((col, label, y.iloc[-1]))
    for idx, (col, label, value) in enumerate(sorted(end_points, key=lambda x: x[2], reverse=True)):
        ax.annotate(
            f"{label}: {(value - 1) * 100:.0f}%",
            xy=(d["date"].iloc[-1], value),
            xytext=(18, (1.5 - idx) * 15),
            textcoords="offset points",
            color=colors[col],
            fontsize=9,
            va="center",
            arrowprops={"arrowstyle": "-", "color": colors[col], "lw": 0.8, "alpha": 0.75},
        )
    ax.set_title("전략별 누적 수익률 비교")
    ax.set_ylabel("누적 수익률 (Base=1.0)")
    format_year_axis(ax)
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92)
    fig.subplots_adjust(top=0.90, bottom=0.13, right=0.86)
    add_note(fig, "핵심: K-FGI는 Buy&Hold의 최종 수익률을 완전히 넘기기보다, 하락 구간 손실을 줄이면서 장기 성과를 유지하는 전략이다. 오른쪽 끝 라벨은 10개년 총수익률이다.")
    save("07_strategy_cumulative_return.png")

    fig, ax = plt.subplots(figsize=(16, 6.0))
    def drawdown(s: pd.Series) -> pd.Series:
        cum = np.exp(s.cumsum())
        return (cum / cum.cummax() - 1) * 100
    bh = drawdown(d["buy_hold"])
    main = drawdown(d["main_kfgi_sentiment"])
    ax.fill_between(d["date"], bh, 0, color="#DDE5EF", alpha=0.88, label="KOSPI200 B&H drawdown")
    ax.plot(d["date"], main, color="#2166D5", lw=2.35, label="K-FGI drawdown")
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_title("낙폭 비교: K-FGI의 하방 방어 효과")
    ax.set_ylabel("고점 대비 낙폭 (%)")
    ax.set_ylim(min(bh.min(), main.min()) * 1.12, 2)
    format_year_axis(ax)
    ax.legend(loc="lower right", ncol=2, frameon=True, framealpha=0.92)
    main_row = perf_df[perf_df["label"] == "main_kfgi_sentiment"].iloc[0]
    bench_row = perf_df[perf_df["label"] == "buy_hold_kospi200"].iloc[0]
    bh_min_date = d.loc[bh.idxmin(), "date"]
    main_min_date = d.loc[main.idxmin(), "date"]
    ax.scatter([bh_min_date, main_min_date], [bh.min(), main.min()], color=["#7B8794", "#2166D5"], s=38, zorder=5)
    ax.annotate(f"B&H MDD {bench_row['mdd']*100:.1f}%", xy=(bh_min_date, bh.min()), xytext=(12, -22), textcoords="offset points", fontsize=9, color="#5E6A75")
    ax.annotate(f"K-FGI MDD {main_row['mdd']*100:.1f}%", xy=(main_min_date, main.min()), xytext=(12, 18), textcoords="offset points", fontsize=9, color="#2166D5")
    ax.text(
        0.015,
        0.09,
        f"Max DD 개선: {bench_row['mdd']*100:.1f}% -> {main_row['mdd']*100:.1f}%",
        transform=ax.transAxes,
        fontsize=12,
        color="#1F4E9A",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "#D6DEE8", "boxstyle": "round,pad=0.35", "alpha": 0.92},
    )
    fig.subplots_adjust(top=0.90, bottom=0.13)
    save("08_strategy_drawdown.png")


def egarch(kfgi: pd.DataFrame) -> None:
    d = kfgi.copy()
    raw_vol = d["egarch_vol"].replace([np.inf, -np.inf], np.nan)
    sane_cap = raw_vol[raw_vol.between(0, 0.20)].quantile(0.99)
    if not np.isfinite(sane_cap) or sane_cap <= 0:
        sane_cap = 0.05
    d["egarch_vol_clean"] = raw_vol.clip(lower=0, upper=sane_cap)
    d["vol_ma"] = (d["egarch_vol_clean"] * 100).rolling(60).mean()
    d["weight_ma"] = d["weight"].rolling(20).mean()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 7.4), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.45], "hspace": 0.12}
    )
    ax1.fill_between(d["date"], 0, d["egarch_vol_clean"] * 100, color="#F6C77A", alpha=0.22, label="EGARCH daily vol (display-clipped)")
    ax1.plot(d["date"], d["vol_ma"], color="#D97706", lw=2.2, label="EGARCH vol 60D MA")
    vol_thr = d["vol_ma"].quantile(0.85)
    ax1.axhline(vol_thr, color="#B45309", ls="--", lw=1.1, alpha=0.75, label="상위 15% 변동성")
    ax1.set_ylabel("조건부 변동성 (%)")
    ax1.set_title("EGARCH 조건부 변동성과 시장 노출 조절")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.plot(d["date"], d["weight"], color="#93B7FF", lw=0.75, alpha=0.45, label="일별 시장 노출도")
    ax2.plot(d["date"], d["weight_ma"], color="#2166D5", lw=2.15, label="시장 노출도 20D MA")
    ax2.axhline(1.0, color="#666", ls=":", lw=1.0, label="기준 노출 1.0x")
    ax2.set_ylabel("시장 노출도 (x)")
    ax2.set_xlabel("Date")
    ax2.set_ylim(-0.05, max(2.1, d["weight"].quantile(0.99) * 1.08))
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    format_year_axis(ax2)
    ax1.margins(x=0.01)
    fig.subplots_adjust(top=0.88, bottom=0.16)
    add_note(fig, "읽는 법: 위 패널은 변동성 수준, 아래 패널은 실제 시장 노출도다. 같은 축에 겹치지 않도록 분리했으며, 표시용 변동성은 초기 EGARCH 추정 폭발치가 축을 망치지 않도록 상위 1% 기준으로 winsorizing했다.")
    save("09_egarch_volatility_exposure.png")


def position_size(kfgi: pd.DataFrame) -> None:
    d = kfgi.copy()
    d["weight_ma20"] = d["weight"].rolling(20).mean()
    yearly = d.groupby(d["date"].dt.year)["weight"].mean()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 7.4), gridspec_kw={"height_ratios": [2.5, 1.2], "hspace": 0.28}
    )
    ax1.plot(d["date"], d["weight"], color="#93B7FF", lw=0.8, alpha=0.5, label="일별 포지션 크기")
    ax1.plot(d["date"], d["weight_ma20"], color="#2166D5", lw=2.2, label="20일 평균")
    ax1.axhline(1.0, color="#666", ls=":", lw=1.0, label="기준 노출 1.0x")
    ax1.fill_between(d["date"], 0, d["weight"], where=d["weight"] < 0.5, color="#FADBD8", alpha=0.25, label="위험 축소 구간(<0.5x)")
    ax1.set_title("K-FGI 전략 포지션 크기")
    ax1.set_ylabel("시장 노출도 (x)")
    ax1.set_ylim(-0.05, max(2.1, d["weight"].quantile(0.99) * 1.08))
    format_year_axis(ax1)
    ax1.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92)

    ax2.bar(yearly.index, yearly.values, color="#D7E4F5", edgecolor="#7EA6D9", width=0.7)
    ax2.axhline(1.0, color="#666", ls=":", lw=1.0)
    ax2.set_ylabel("연평균 노출도")
    ax2.set_xlabel("Year")
    ax2.set_xticks(yearly.index)
    ax2.grid(True, axis="y", alpha=0.28)
    for year, val in yearly.items():
        ax2.text(year, val + 0.035, f"{val:.2f}x", ha="center", fontsize=8, color="#24578C")
    fig.subplots_adjust(top=0.90, bottom=0.12)
    add_note(fig, "핵심: 포지션 크기 그래프는 전략이 언제 공격적으로, 언제 방어적으로 움직였는지를 직접 보여 준다. 긴 10개년 구간에서는 일별선보다 20일 평균과 연평균을 함께 보는 것이 더 읽기 쉽다.")
    save("16_position_size.png")


def feature_importance(imp: pd.DataFrame) -> None:
    name_map = {
        "sent_composite_ma10": "감성 종합 10D",
        "sent_norm_ma5": "감성 정규화 5D",
        "neg_z_ma5": "부정감성 z 5D",
        "sub_index5_lag1": "변동성지수 lag1",
        "sub_index7_lag1": "채권스프레드 lag1",
        "sub_index6_lag1": "안전자산수요 lag1",
        "sub_index3_lag1": "시장폭 lag1",
    }
    top = imp.head(12).copy()
    top["display"] = top["feature"].map(name_map).fillna(top["feature"])
    def group(f: str) -> str:
        if any(k in f for k in ["sent", "neg_z", "composite"]):
            return "감성"
        if any(k in f for k in ["egarch", "vol", "ma_ratio", "mom", "rsi"]):
            return "추세/변동성"
        return "시장/기타"
    top["group"] = top["feature"].map(group)
    palette = {"감성": "#E53935", "추세/변동성": "#F39C12", "시장/기타": "#2D6CDF"}
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ordered = top.iloc[::-1]
    ax.barh(ordered["display"], ordered["importance"], color=[palette[g] for g in ordered["group"]])
    ax.set_title("LightGBM 피처 중요도: 모델이 가장 많이 사용한 정보")
    ax.set_xlabel("Gain importance")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    ax.legend(handles, palette.keys(), loc="lower right")
    add_note(fig, "핵심: 감성 피처가 1순위는 아니지만 상위권에 포함되어 시장/변동성 피처를 보완한다.")
    save("10_feature_importance_grouped.png")


def weights(sens: pd.DataFrame) -> None:
    d = sens.copy()
    pivot = d.pivot_table(index="kfgi_range", columns="regime_mult", values="sharpe", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    adopted = d[d["adopted"] == True]
    for i, idx in enumerate(pivot.index):
        for j, col in enumerate(pivot.columns):
            val = pivot.loc[idx, col]
            mark = "★\n" if not adopted.empty and adopted.iloc[0]["kfgi_range"] == idx and adopted.iloc[0]["regime_mult"] == col else ""
            ax.text(j, i, f"{mark}{val:.3f}", ha="center", va="center", fontsize=8, color="#111")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Sharpe")
    ax.set_title("가중치 조합별 Sharpe 민감도")
    ax.set_xlabel("시장 국면별 노출 배수 (bull/normal/crisis)")
    ax.set_ylabel("K-FGI 점수별 노출 범위")
    add_note(fig, "읽는 법: 진한 칸일수록 Sharpe가 높다. 별표는 실제 채택한 조합이며, 최고점만이 아니라 안정성과 해석 가능성을 함께 고려한다.")
    save("11_weight_sensitivity_sharpe_heatmap.png")

    top = d.sort_values("sharpe", ascending=False).head(8).copy()
    top["display"] = top["kfgi_range"] + " / " + top["regime_mult"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    colors = np.where(top["adopted"], "#2D6CDF", "#C7D0DB")
    ax.barh(top["display"].iloc[::-1], top["sharpe"].iloc[::-1], color=colors[::-1])
    ax.set_title("가중치 후보별 성능 순위")
    ax.set_xlabel("Sharpe")
    for y, val in enumerate(top["sharpe"].iloc[::-1]):
        ax.text(val + 0.004, y, f"{val:.3f}", va="center", fontsize=9)
    add_note(fig, "핵심: 채택 조합이 최고 Sharpe만 좇는 과도한 공격형 조합인지, 후보군 안에서 합리적인 수준인지 확인하는 그림이다.")
    save("12_weight_candidate_ranking.png")


def ablation(perf: pd.DataFrame, stat: pd.DataFrame) -> None:
    label_map = {"buy_hold_kospi200": "Buy&Hold", "main_kfgi_no_sentiment": "K-FGI\n감성 제외", "main_kfgi_sentiment": "K-FGI\n감성 포함"}
    rows = perf[perf["label"].isin(label_map)].copy()
    rows["display"] = rows["label"].map(label_map)
    rows = rows.set_index("display").loc[["Buy&Hold", "K-FGI\n감성 제외", "K-FGI\n감성 포함"]].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    metrics = [("total_return", "총수익률 (%)", 100), ("sharpe", "Sharpe", 1), ("mdd", "MDD (%)", 100)]
    for ax, (col, title, mult) in zip(axes, metrics):
        vals = rows[col] * mult
        ax.bar(rows["display"], vals, color=["#AAB7C4", "#8EA6C8", "#2D6CDF"])
        ax.set_title(title)
        ax.axhline(0, color="#555", lw=0.8)
        for x, y in zip(rows["display"], vals):
            ax.text(x, y + (2 if col != "sharpe" else 0.03), f"{y:.1f}" if col != "sharpe" else f"{y:.2f}", ha="center", fontsize=9)
    fig.suptitle("감성 피처 ablation: 포함/제외 성능 차이", fontweight="bold", fontsize=16)
    add_note(fig, "해석: 감성 포함 버전은 총수익률이 개선되지만, Sharpe는 감성 제외 버전이 높다. 따라서 감성 피처는 '무조건 우월'이 아니라 수익률 개선 보조 요인으로 해석한다.")
    save("13_sentiment_ablation_performance.png")

    names = {
        "overall_excess_ttest": "전체 초과수익",
        "downside_excess_ttest": "하락구간 방어",
        "kfgi_extreme_return_ttest": "극단 K-FGI 구간",
        "sentiment_bootstrap_delta_rho": "감성 기여 bootstrap",
    }
    d = stat.copy()
    d["display"] = d["test"].map(names).fillna(d["test"])
    d["score"] = -np.log10(d["p_value"].clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(10, 5.1))
    colors = np.where(d["p_value"] < 0.05, "#2D6CDF", "#C7D0DB")
    ax.bar(d["display"], d["score"], color=colors)
    ax.axhline(-np.log10(0.05), color="#E74C3C", ls="--", label="p=0.05")
    ax.set_title("통계 검정 p-value 요약")
    ax.set_ylabel("-log10(p-value), 높을수록 유의")
    ax.tick_params(axis="x", rotation=15)
    for x, p in zip(d["display"], d["p_value"]):
        ax.text(x, -0.2, f"p={p:.3g}", ha="center", va="top", fontsize=8)
    ax.legend()
    add_note(fig, "핵심: 전체 초과수익은 유의하지 않지만, 하락구간 방어는 매우 유의하다. 논문 주장은 하방 위험 관리에 두는 것이 안전하다.")
    save("14_statistical_test_pvalues.png")


def yearly_oos(yearly: pd.DataFrame, oos: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    y = yearly.copy()
    colors = np.where(y["total_return"] >= 0, "#2D6CDF", "#E74C3C")
    axes[0].bar(y["year"], y["total_return"] * 100, color=colors)
    axes[0].axhline(0, color="#555", lw=0.8)
    axes[0].set_title("연도별 K-FGI 전략 수익률")
    axes[0].set_ylabel("연도별 수익률 (%)")
    axes[0].set_xticks(y["year"])
    axes[0].tick_params(axis="x", rotation=45)
    o = oos.copy()
    o["display"] = o["label"].map({"train_pre_2025": "2015-2024\nTrain", "test_2025": "2025\nOOS"})
    axes[1].bar(o["display"], o["sharpe"], color=["#AAB7C4", "#2D6CDF"])
    axes[1].set_title("Train vs 2025 OOS Sharpe")
    axes[1].set_ylabel("Sharpe")
    for x, val in zip(o["display"], o["sharpe"]):
        axes[1].text(x, val + 0.08, f"{val:.2f}", ha="center", fontsize=10)
    add_note(fig, "읽는 법: 왼쪽은 성과가 어느 연도에 몰렸는지, 오른쪽은 2025년 외표본 구간에서 전략이 유지되는지 확인한다.")
    save("15_yearly_and_oos_performance.png")


def write_guides() -> None:
    rows = ["# Figure Guide\n", "논문 삽입용 그림은 이미지 내부 제목을 제거했고, 파일명은 한글 설명형으로 정리했다.\n"]
    for i, (name, meta) in enumerate(FIGURES.items(), start=1):
        display_name = KOREAN_FILENAMES.get(name, name)
        rows.append(f"## Figure {i}. `{display_name}`")
        rows.append(f"**그림 역할:** {meta['title']}")
        rows.append(f"**논문 인사이트:** {meta['message']}")
        rows.append(f"**해석 포인트:** {meta['read']}")
        rows.append(f"**본문 연결:** 이 그림은 `{display_name}` 파일을 사용하고, 논문 본문/캡션에서 제목을 따로 부여한다.")
        rows.append("")
    text = "\n".join(rows)
    for out_dir in [CODE_ROOT / "paper_outputs", PAPER_OUTPUT]:
        (out_dir / "FIGURE_GUIDE.md").write_text(text, encoding="utf-8")

    index = "# Paper Figure Index\n\n" + "\n".join(
        f"{i}. `{KOREAN_FILENAMES.get(name, name)}` - {meta['title']}"
        for i, (name, meta) in enumerate(FIGURES.items(), start=1)
    ) + "\n"
    for out_dir in [CODE_ROOT / "paper_outputs", PAPER_OUTPUT]:
        (out_dir / "FIGURE_INDEX.md").write_text(index, encoding="utf-8")


def sync_to_repo_readme() -> None:
    src = CODE_ROOT / "paper_outputs" / "FIGURE_GUIDE.md"
    dst = CODE_ROOT / "docs" / "FIGURE_GUIDE.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    setup()
    data = load()
    filtering(data["sent"])
    toxicity()
    sentiment_norm(data["sent"])
    sentiment_activity(data["sent"])
    kfgi_series(data["kfgi"])
    subindex_heatmap(data["final"])
    performance(data["ret"], data["perf"])
    egarch(data["kfgi"])
    position_size(data["kfgi"])
    feature_importance(data["imp"])
    weights(data["sens"])
    ablation(data["perf"], data["stat"])
    yearly_oos(data["yearly"], data["oos"])
    write_guides()
    sync_to_repo_readme()
    print(f"redrawn={REPO_FIG}")
    print(f"guide={CODE_ROOT / 'paper_outputs' / 'FIGURE_GUIDE.md'}")


if __name__ == "__main__":
    main()
