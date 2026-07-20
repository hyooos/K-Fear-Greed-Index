from __future__ import annotations

import json
import shutil
import textwrap
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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


def resolve_paths() -> tuple[Path, Path, Path]:
    cwd = Path.cwd().resolve()
    project_root = cwd.parent if nfc(cwd.name) == "kfgi_최종" else cwd
    if nfc(project_root.name) == "kfgi_최종":
        project_root = project_root.parent
    paper_dir = find_child(project_root, "논문용")
    try:
        package_dir = find_child(paper_dir, "10y_kfgi_paper_package")
    except FileNotFoundError:
        package_dir = paper_dir
    source_code_dir = find_child(project_root, "kfgi_최종")
    return project_root, package_dir, source_code_dir


PROJECT_ROOT, PACKAGE_DIR, SOURCE_CODE_DIR = resolve_paths()
EXP_DIR = PACKAGE_DIR / "06_10y_experiments"
FIG_DIR = PACKAGE_DIR / "figure 모음"
PAPER_FIG_DIR = FIG_DIR / "paper_figures"
TABLE_DIR = EXP_DIR / "tables"
DASH_DIR = PACKAGE_DIR / "8_Dashboard"


def setup() -> None:
    for path in [
        PACKAGE_DIR / "1_KFGI_subindex",
        PACKAGE_DIR / "2_Naver_crawling",
        PACKAGE_DIR / "3_Filtering_final",
        PACKAGE_DIR / "4_Sentiment_analysis",
        PACKAGE_DIR / "5_Merge_to_final_csv",
        PACKAGE_DIR / "6_KFGI_weight",
        PACKAGE_DIR / "7_Modeling",
        DASH_DIR,
        PACKAGE_DIR / "검증",
        PACKAGE_DIR / "data",
        FIG_DIR,
        PAPER_FIG_DIR,
        PACKAGE_DIR / "tools",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    for font_path in [
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
        Path("/Library/Fonts/AppleGothic.ttf"),
    ]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


def copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                continue
            shutil.copytree(item, target, ignore=shutil.ignore_patterns(".DS_Store", "__pycache__", "*.pyc"))
        elif item.name != ".DS_Store" and not target.exists():
            shutil.copy2(item, target)


def organize_package() -> None:
    mapping = {
        "02_subindices": "1_KFGI_subindex",
        "03_sentiment": "4_Sentiment_analysis",
        "01_final_dataset": "5_Merge_to_final_csv",
    }
    for src_name, dst_name in mapping.items():
        src = PACKAGE_DIR / src_name
        dst = PACKAGE_DIR / dst_name
        copy_tree_contents(src, dst)

    scripts = PACKAGE_DIR / "04_scripts"
    copy_tree_contents(scripts / "crawling", PACKAGE_DIR / "2_Naver_crawling")
    copy_tree_contents(scripts / "filtering", PACKAGE_DIR / "3_Filtering_final")
    copy_tree_contents(scripts / "merge", PACKAGE_DIR / "5_Merge_to_final_csv")
    copy_tree_contents(scripts / "modeling", PACKAGE_DIR / "7_Modeling")
    for name in ["sentiment.py", "sent_feature.py", "sent_viz.py"]:
        src = scripts / name
        if src.exists():
            shutil.copy2(src, PACKAGE_DIR / "4_Sentiment_analysis" / name)

    for src in [
        EXP_DIR / "kfgi_10y_timeseries.csv",
        EXP_DIR / "strategy_returns_10y.csv",
        TABLE_DIR / "sensitivity_analysis.csv",
        TABLE_DIR / "performance_summary.csv",
    ]:
        if src.exists():
            shutil.copy2(src, PACKAGE_DIR / "6_KFGI_weight" / src.name)
    copy_tree_contents(TABLE_DIR, PACKAGE_DIR / "검증")

    for src in [
        PACKAGE_DIR / "01_final_dataset" / "KFG_final_10y.csv",
        PACKAGE_DIR / "01_final_dataset" / "KFG_final_10y_raw.csv",
        PACKAGE_DIR / "01_final_dataset" / "KFG_final_10y_missing_report.csv",
        EXP_DIR / "kfgi_10y_timeseries.csv",
        EXP_DIR / "strategy_returns_10y.csv",
    ]:
        if src.exists():
            shutil.copy2(src, PACKAGE_DIR / "data" / src.name)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def load_data() -> dict[str, pd.DataFrame]:
    final_df = pd.read_csv(PACKAGE_DIR / "01_final_dataset" / "KFG_final_10y.csv", parse_dates=["date"])
    kfgi = pd.read_csv(EXP_DIR / "kfgi_10y_timeseries.csv", parse_dates=["date"])
    returns = pd.read_csv(EXP_DIR / "strategy_returns_10y.csv", parse_dates=["date"])
    perf = pd.read_csv(TABLE_DIR / "performance_summary.csv")
    imp = pd.read_csv(TABLE_DIR / "feature_importance.csv")
    yearly = pd.read_csv(TABLE_DIR / "yearly_performance.csv")
    sensitivity = pd.read_csv(TABLE_DIR / "sensitivity_analysis.csv")
    stat = pd.read_csv(TABLE_DIR / "statistical_tests.csv")
    oos = pd.read_csv(TABLE_DIR / "oos_2025_performance.csv")
    sentiment = pd.read_csv(
        PACKAGE_DIR / "03_sentiment" / "recollect_sentiment_final_model_toxicity" / "daily_sentiment.csv",
        parse_dates=["date"],
    )
    return {
        "final": final_df,
        "kfgi": kfgi,
        "returns": returns,
        "perf": perf,
        "imp": imp,
        "yearly": yearly,
        "sensitivity": sensitivity,
        "stat": stat,
        "oos": oos,
        "sentiment": sentiment,
    }


def plot_filtering_retention() -> None:
    summary_dir = SOURCE_CODE_DIR / "3_Filtering_final" / "recollect_final_filtered_model_toxicity"
    rows = []
    for year in range(2014, 2026):
        p = summary_dir / str(year) / f"stock_classification_summary_{year}.json"
        if not p.exists():
            continue
        with p.open() as f:
            data = json.load(f)
        total = data.get("total_comments") or data.get("total") or data.get("input_rows")
        kept = data.get("final_stock_comments") or data.get("stock_comments") or data.get("kept_rows")
        if total and kept:
            rows.append({"year": year, "kept": kept, "removed": max(total - kept, 0), "retention": kept / total})
    if not rows:
        return
    df = pd.DataFrame(rows)
    recent = df[df["year"] >= 2015].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(recent["year"], recent["kept"], color="#DCEAF7", label="최종 사용")
    axes[0].bar(recent["year"], recent["removed"], bottom=recent["kept"], color="#E85050", label="필터 제거")
    axes[0].set_title("연도별 댓글 필터링 결과")
    axes[0].set_ylabel("댓글 수")
    axes[0].legend()
    axes[1].bar(recent["year"], recent["retention"] * 100, color="#2A6FBA")
    axes[1].set_title("최종 통과율")
    axes[1].set_ylabel("통과율 (%)")
    for x, y in zip(recent["year"], recent["retention"] * 100):
        axes[1].text(x, y + 0.5, f"{y:.1f}%", ha="center", fontsize=8)
    savefig(PAPER_FIG_DIR / "01_filtering_retention_by_year.png")


def plot_toxicity_weight_curve() -> None:
    x = np.linspace(0, 1, 501)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, 1 - x, "--", color="#B8B8B8", lw=2, label="γ=1")
    ax.plot(x, (1 - x) ** 2, color="#2E86DE", lw=2.5, label="γ=2 (본 연구)")
    ax.plot(x, (1 - x) ** 3, ":", color="#0D4F8B", lw=2.5, label="γ=3")
    ax.axvline(0.95, color="#E74C3C", ls="--", label="Hard Drop τ=0.95")
    ax.axvspan(0, 0.3, color="#FADBD8", alpha=0.23)
    for tx in [0.2, 0.5]:
        ax.scatter([tx], [(1 - tx) ** 2], color="#2E86DE", zorder=3)
        ax.text(tx + 0.02, (1 - tx) ** 2 + 0.04, f"({tx:.1f}, {(1-tx)**2:.2f})", color="#2E86DE")
    ax.set_xlabel("독성 점수")
    ax.set_ylabel("가중치")
    ax.set_title("독성 점수 기반 댓글 가중치 선택 근거")
    ax.set_ylim(0, 1.05)
    ax.legend()
    savefig(PAPER_FIG_DIR / "02_toxicity_weight_curve.png")


def plot_sentiment_normalization(sentiment: pd.DataFrame) -> None:
    sample = sentiment[["sent_raw_mean_w", "sent_norm_w"]].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].hist(sample["sent_raw_mean_w"], bins=80, color="#6FA8DC", alpha=0.85)
    axes[0].set_title("정규화 전 감성점수")
    axes[0].set_xlabel("raw weighted sentiment")
    axes[1].hist(sample["sent_norm_w"], bins=80, color="#E88963", alpha=0.85)
    axes[1].axvline(0, color="#555", ls=":")
    axes[1].set_title("정규화 후 감성점수")
    axes[1].set_xlabel("normalized sentiment (-1~1)")
    fig.suptitle("감성점수 정규화 전/후 분포 비교", fontweight="bold")
    savefig(PAPER_FIG_DIR / "03_sentiment_normalization_before_after.png")


def plot_sentiment_activity(sentiment: pd.DataFrame) -> None:
    d = sentiment.copy()
    d["year"] = d["date"].dt.year
    annual = d.groupby("year").agg(
        comment_count=("comment_count", "sum"),
        effective_n=("effective_n", "mean"),
        sent_std=("sent_std", "mean"),
        heat=("heat", "mean"),
    ).reset_index()
    annual = annual[annual["year"].between(2015, 2025)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].bar(annual["year"], annual["comment_count"], color="#6FA8DC")
    axes[0].set_title("연도별 감성 댓글 수")
    axes[0].set_ylabel("댓글 수")
    axes[1].plot(d["date"], d["sent_norm_w"].rolling(20).mean(), color="#6A3D9A", lw=1.4, label="sent_norm_w 20D MA")
    axes[1].fill_between(d["date"], -1, 1, color="#EFEAF7", alpha=0.25)
    axes[1].axhline(0, color="#777", lw=0.8, ls=":")
    axes[1].set_title("정규화 감성점수 추이")
    axes[1].set_ylim(-1, 1)
    axes[1].legend()
    savefig(PAPER_FIG_DIR / "04_sentiment_activity_and_trend.png")


def plot_kfgi_series(kfgi: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    d = kfgi.copy()
    d["K_FGI_5d"] = d["K_FGI"].rolling(5).mean()
    ax.plot(d["date"], d["K_FGI_5d"], color="#7C3AED", lw=1.8, label="K-FGI 5D MA")
    ax.axhline(65, color="#E74C3C", ls=":", label="Greed threshold 65")
    ax.axhline(25, color="#4A90E2", ls=":", label="Fear threshold 25")
    ax.fill_between(d["date"], 0, 100, where=d["K_FGI"] > 65, color="#FADBD8", alpha=0.25)
    ax.fill_between(d["date"], 0, 100, where=d["K_FGI"] < 25, color="#D6EAF8", alpha=0.30)
    ax.set_ylim(0, 100)
    ax.set_ylabel("K-FGI")
    ax.set_title("10개년 K-FGI 시계열")
    ax.legend(ncol=3, fontsize=8)
    savefig(PAPER_FIG_DIR / "05_kfgi_timeseries_thresholds.png")


def plot_subindex_heatmap(final_df: pd.DataFrame) -> None:
    cols = [f"sub_index{i}" for i in range(1, 8)]
    d = final_df[["date"] + cols].copy()
    yearly = d.groupby(d["date"].dt.year)[cols].mean()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.imshow(yearly.T.values, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(yearly.index)))
    ax.set_xticklabels(yearly.index, rotation=45)
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(cols)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("평균 점수")
    ax.set_title("연도별 Subindex 평균 히트맵")
    ax.set_xlabel("연도")
    ax.set_ylabel("Subindex")
    savefig(PAPER_FIG_DIR / "06_subindex_yearly_heatmap.png")


def plot_strategy_performance(returns: pd.DataFrame) -> None:
    d = returns.copy()
    strategy_cols = ["buy_hold", "trend_only", "trend_egarch_vol", "main_kfgi_sentiment"]
    labels = {
        "buy_hold": "KOSPI200 B&H",
        "trend_only": "Trend",
        "trend_egarch_vol": "Trend + EGARCH",
        "main_kfgi_sentiment": "K-FGI Strategy",
    }
    colors = ["#C8D3E1", "#13A386", "#F2A541", "#2D6CDF"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for col, color in zip(strategy_cols, colors):
        ax.plot(d["date"], np.exp(d[col].cumsum()), label=labels[col], color=color, lw=2 if col == "main_kfgi_sentiment" else 1.4)
    ax.set_title("전략별 누적 수익률 비교")
    ax.set_ylabel("누적 수익률 (Base=1.0)")
    ax.legend()
    savefig(PAPER_FIG_DIR / "07_strategy_cumulative_return.png")

    fig, ax = plt.subplots(figsize=(12, 4.3))
    def dd(s):
        c = np.exp(s.cumsum())
        return (c / c.cummax() - 1) * 100
    ax.fill_between(d["date"], dd(d["buy_hold"]), color="#DDE5EF", alpha=0.75, label="KOSPI200 DD")
    ax.plot(d["date"], dd(d["main_kfgi_sentiment"]), color="#2D6CDF", lw=1.7, label="K-FGI Strategy DD")
    ax.set_title("KOSPI200 대비 K-FGI 전략 낙폭 비교")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    savefig(PAPER_FIG_DIR / "08_strategy_drawdown.png")


def plot_egarch_exposure(kfgi: pd.DataFrame) -> None:
    d = kfgi.copy()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.fill_between(d["date"], d["egarch_vol"] * 100, color="#F7D794", alpha=0.35, label="EGARCH volatility")
    ax.plot(d["date"], (d["egarch_vol"] * 100).rolling(100).mean(), color="#E67E22", lw=1.8, label="EGARCH vol 100D MA")
    ax2 = ax.twinx()
    ax2.plot(d["date"], d["weight"], color="#82A9FF", lw=1.0, alpha=0.8, label="Market exposure")
    ax.set_title("EGARCH 조건부 변동성과 시장 노출도")
    ax.set_ylabel("Daily volatility (%)")
    ax2.set_ylabel("Exposure (x)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    savefig(PAPER_FIG_DIR / "09_egarch_volatility_exposure.png")


def plot_feature_importance(imp: pd.DataFrame) -> None:
    top = imp.head(15).copy()
    def group(feature: str) -> str:
        if any(k in feature for k in ["sent_", "neg_z", "composite"]):
            return "감성 피처"
        if any(k in feature for k in ["egarch", "vol_", "ma_ratio", "rsi", "mom", "above", "trend"]):
            return "추세/변동성 피처"
        return "기타"
    top["group"] = top["feature"].map(group)
    palette = {"감성 피처": "#E53935", "추세/변동성 피처": "#F39C12", "기타": "#2D6CDF"}
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=[palette[g] for g in top["group"][::-1]])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    ax.legend(handles, palette.keys(), loc="lower right")
    ax.set_title("LightGBM Gain 기반 피처 중요도")
    ax.set_xlabel("Feature importance (gain)")
    savefig(PAPER_FIG_DIR / "10_feature_importance_grouped.png")


def plot_weight_sensitivity(sensitivity: pd.DataFrame) -> None:
    d = sensitivity.copy()
    pivot = d.pivot_table(index="kfgi_range", columns="regime_mult", values="sharpe", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Sharpe")
    ax.set_title("가중치 조합별 Sharpe 민감도")
    ax.set_xlabel("Regime multiplier (bull/normal/crisis)")
    ax.set_ylabel("K-FGI multiplier range")
    savefig(PAPER_FIG_DIR / "11_weight_sensitivity_sharpe_heatmap.png")

    adopted = d[d["adopted"] == True]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    top = d.sort_values("sharpe", ascending=False).head(10).copy()
    top["label"] = top["kfgi_range"] + " | " + top["regime_mult"]
    colors = ["#2D6CDF" if x else "#BDC3C7" for x in top["adopted"]]
    ax.barh(top["label"][::-1], top["sharpe"][::-1], color=colors[::-1])
    if not adopted.empty:
        ax.axvline(adopted.iloc[0]["sharpe"], color="#2D6CDF", ls="--", lw=1, label="채택 조합")
    ax.set_title("가중치 선택 전 후보 성능 비교")
    ax.set_xlabel("Sharpe")
    ax.legend()
    savefig(PAPER_FIG_DIR / "12_weight_candidate_ranking.png")


def plot_ablation_and_tests(perf: pd.DataFrame, stat: pd.DataFrame) -> None:
    rows = perf[perf["label"].isin(["main_kfgi_sentiment", "main_kfgi_no_sentiment", "buy_hold_kospi200"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(rows["label"], rows["total_return"] * 100, color=["#DDE5EF", "#2D6CDF", "#7C3AED"])
    axes[0].set_title("감성 포함/제외 누적수익률 비교")
    axes[0].set_ylabel("누적수익률 (%)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(rows["label"], rows["mdd"] * 100, color=["#DDE5EF", "#2D6CDF", "#7C3AED"])
    axes[1].set_title("감성 포함/제외 MDD 비교")
    axes[1].set_ylabel("MDD (%)")
    axes[1].tick_params(axis="x", rotation=25)
    savefig(PAPER_FIG_DIR / "13_sentiment_ablation_performance.png")

    d = stat.copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(d["test"], -np.log10(d["p_value"].clip(lower=1e-300)), color="#6FA8DC")
    ax.axhline(-np.log10(0.05), color="#E74C3C", ls="--", label="p=0.05")
    ax.set_title("통계 검정 유의성 요약")
    ax.set_ylabel("-log10(p-value)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    savefig(PAPER_FIG_DIR / "14_statistical_test_pvalues.png")


def plot_yearly_and_oos(yearly: pd.DataFrame, oos: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].bar(yearly["year"], yearly["total_return"] * 100, color=np.where(yearly["total_return"] >= 0, "#2D6CDF", "#E74C3C"))
    axes[0].set_title("연도별 전략 수익률")
    axes[0].set_ylabel("누적수익률 (%)")
    axes[0].axhline(0, color="#555", lw=0.8)
    axes[1].bar(oos["label"], oos["sharpe"], color=["#AAB7B8", "#2D6CDF"])
    axes[1].set_title("Train/Test Sharpe 비교")
    axes[1].tick_params(axis="x", rotation=15)
    savefig(PAPER_FIG_DIR / "15_yearly_and_oos_performance.png")


def build_dashboard() -> None:
    app = r'''
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="K-FGI 10Y Dashboard", layout="wide")

BASE = Path(__file__).resolve().parents[1]
EXP = BASE / "06_10y_experiments"
TABLES = EXP / "tables"


@st.cache_data
def load_data():
    kfgi = pd.read_csv(EXP / "kfgi_10y_timeseries.csv", parse_dates=["date"])
    ret = pd.read_csv(EXP / "strategy_returns_10y.csv", parse_dates=["date"])
    perf = pd.read_csv(TABLES / "performance_summary.csv")
    imp = pd.read_csv(TABLES / "feature_importance.csv")
    sens = pd.read_csv(TABLES / "sensitivity_analysis.csv")
    yearly = pd.read_csv(TABLES / "yearly_performance.csv")
    stat = pd.read_csv(TABLES / "statistical_tests.csv")
    return kfgi, ret, perf, imp, sens, yearly, stat


kfgi, ret, perf, imp, sens, yearly, stat = load_data()

st.title("K-FGI x EGARCH 10-Year Dashboard")
st.caption("sub_index1은 원자료에는 보존하되, 모멘텀 중복 방지를 위해 K-FGI 및 모델 피처에서는 제외.")

main = perf[perf["label"] == "main_kfgi_sentiment"].iloc[0]
bench = perf[perf["label"] == "buy_hold_kospi200"].iloc[0]
cols = st.columns(5)
cols[0].metric("Experiment period", f"{kfgi['date'].min().date()} ~ {kfgi['date'].max().date()}")
cols[1].metric("Main total return", f"{main['total_return']*100:.1f}%")
cols[2].metric("Main Sharpe", f"{main['sharpe']:.3f}")
cols[3].metric("Main MDD", f"{main['mdd']*100:.1f}%", delta=f"{(main['mdd']-bench['mdd'])*100:.1f}%p vs B&H")
cols[4].metric("Buy&Hold total", f"{bench['total_return']*100:.1f}%")

tabs = st.tabs(["Performance", "K-FGI", "Weights", "Feature Importance", "Validation", "Data"])

with tabs[0]:
    st.subheader("전략 성과")
    fig = go.Figure()
    labels = {
        "buy_hold": "KOSPI200 B&H",
        "trend_only": "Trend",
        "trend_egarch_vol": "Trend + EGARCH",
        "main_kfgi_sentiment": "K-FGI Strategy",
    }
    for col, name in labels.items():
        fig.add_trace(go.Scatter(x=ret["date"], y=np.exp(ret[col].cumsum()), mode="lines", name=name))
    fig.update_layout(height=460, yaxis_title="Cumulative return (base=1.0)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(perf, use_container_width=True)

with tabs[1]:
    st.subheader("K-FGI 시계열")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kfgi["date"], y=kfgi["K_FGI"].rolling(5).mean(), mode="lines", name="K-FGI 5D MA"))
    fig.add_hline(y=65, line_dash="dot", line_color="red", annotation_text="Greed 65")
    fig.add_hline(y=25, line_dash="dot", line_color="blue", annotation_text="Fear 25")
    fig.update_layout(height=440, yaxis_range=[0, 100], yaxis_title="K-FGI")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(kfgi[["date", "K_FGI", "regime", "egarch_vol", "weight", "strat_ret"]], use_container_width=True)

with tabs[2]:
    st.subheader("가중치 민감도")
    heat = sens.pivot_table(index="kfgi_range", columns="regime_mult", values="sharpe", aggfunc="mean")
    fig = px.imshow(heat, text_auto=".3f", aspect="auto", color_continuous_scale="YlGnBu")
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sens.sort_values("sharpe", ascending=False), use_container_width=True)

with tabs[3]:
    st.subheader("LightGBM Feature Importance")
    top = imp.head(20).copy()
    fig = px.bar(top.iloc[::-1], x="importance", y="feature", orientation="h", height=560)
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("검증 결과")
    left, right = st.columns(2)
    with left:
        st.write("연도별 성과")
        st.plotly_chart(px.bar(yearly, x="year", y="total_return", color="total_return", height=360), use_container_width=True)
    with right:
        st.write("통계 검정")
        stat_plot = stat.copy()
        stat_plot["minus_log10_p"] = -np.log10(stat_plot["p_value"].clip(lower=1e-300))
        st.plotly_chart(px.bar(stat_plot, x="test", y="minus_log10_p", height=360), use_container_width=True)
    st.dataframe(stat, use_container_width=True)

with tabs[5]:
    st.subheader("파일 위치")
    st.code(str(BASE))
    st.write("주요 입력 및 산출물은 패키지 내부 `data`, `06_10y_experiments`, `figure 모음`에 저장되어 있습니다.")
'''
    (DASH_DIR / "streamlit_app.py").write_text(app.strip() + "\n")
    (DASH_DIR / "requirements.txt").write_text("streamlit\nplotly\npandas\nnumpy\n")
    (DASH_DIR / "README.md").write_text(
        "Run with:\n\n"
        "```bash\n"
        "cd /Users/hyowon/Desktop/uni/3-1/기계학습/논문용\n"
        "conda run -n kfgi streamlit run 8_Dashboard/streamlit_app.py\n"
        "```\n"
    )


def write_figure_index() -> None:
    rows = []
    for idx, path in enumerate(sorted(PAPER_FIG_DIR.glob("*.png")), start=1):
        rows.append(f"{idx}. `{path.name}`")
    (PAPER_FIG_DIR / "FIGURE_INDEX.md").write_text(
        "# Paper Figure Index\n\n" + "\n".join(rows) + "\n"
    )


def main() -> None:
    setup()
    organize_package()
    data = load_data()
    plot_filtering_retention()
    plot_toxicity_weight_curve()
    plot_sentiment_normalization(data["sentiment"])
    plot_sentiment_activity(data["sentiment"])
    plot_kfgi_series(data["kfgi"])
    plot_subindex_heatmap(data["final"])
    plot_strategy_performance(data["returns"])
    plot_egarch_exposure(data["kfgi"])
    plot_feature_importance(data["imp"])
    plot_weight_sensitivity(data["sensitivity"])
    plot_ablation_and_tests(data["perf"], data["stat"])
    plot_yearly_and_oos(data["yearly"], data["oos"])
    build_dashboard()
    write_figure_index()

    manifest = PACKAGE_DIR / "00_docs" / "FILE_MANIFEST.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(str(p) for p in sorted(PACKAGE_DIR.rglob("*")) if p.is_file()) + "\n")
    print(f"figures={PAPER_FIG_DIR}")
    print(f"dashboard={DASH_DIR / 'streamlit_app.py'}")


if __name__ == "__main__":
    main()
