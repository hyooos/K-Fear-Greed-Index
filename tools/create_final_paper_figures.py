from __future__ import annotations

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
CODE_ROOT = ROOT if (ROOT / "paper_outputs").exists() else find_child(PROJECT_ROOT, "kfgi_최종")
try:
    PAPER_ROOT = find_child(PROJECT_ROOT, "논문용")
except FileNotFoundError:
    PAPER_ROOT = CODE_ROOT

REPO_OUT = CODE_ROOT / "paper_outputs"
REPO_FIG = REPO_OUT / "final_figures"
REPO_TABLE = REPO_OUT / "tables"
REPO_DATA = REPO_OUT / "data"
PAPER_OUT = PAPER_ROOT / "10_Paper_outputs" if PAPER_ROOT != CODE_ROOT else REPO_OUT
PAPER_FIG = PAPER_OUT / "final_figures"
PAPER_TABLE = PAPER_OUT / "tables" if PAPER_OUT.exists() else REPO_TABLE


BLUE = "#2F6FDB"
LIGHT_BLUE = "#D9E6F7"
RED = "#D94B4B"
ORANGE = "#E68632"
GREEN = "#2E9D64"
PURPLE = "#6E44C9"
GRAY = "#8A96A3"
DARK = "#30343B"


def setup() -> None:
    for path in [REPO_FIG, PAPER_FIG]:
        path.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    for path in [
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
        Path("/Library/Fonts/AppleGothic.ttf"),
    ]:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.titlepad": 12,
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    title = Path(filename).stem
    if fig.axes:
        fig.axes[0].set_title(title)
    fig.subplots_adjust(top=0.88, bottom=0.18)
    for out in [REPO_FIG, PAPER_FIG]:
        fig.savefig(out / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_data() -> dict[str, pd.DataFrame]:
    ts = pd.read_csv(REPO_DATA / "conservative_kfgi_position_caps_timeseries.csv", parse_dates=["date"])
    final_ts = ts[ts["variant"].eq("lean_sent_composite_ma10__cap_1_0_no_leverage")].copy()
    if final_ts.empty:
        raise ValueError("final conservative K-FGI time series not found")
    return {
        "ts": final_ts,
        "conditional": pd.read_csv(REPO_TABLE / "final_master_conditional_downside_defense.csv"),
        "stats": pd.read_csv(REPO_TABLE / "final_master_statistical_tests.csv"),
        "sens": pd.read_csv(REPO_TABLE / "final_conservative_kfgi_sensitivity.csv"),
        "rf_imp": pd.read_csv(REPO_TABLE / "final_master_rf_defense_permutation_importance.csv"),
        "en_imp": pd.read_csv(REPO_TABLE / "final_master_elasticnet_defense_feature_importance.csv"),
        "pca": pd.read_csv(REPO_TABLE / "final_master_pca_explained_variance.csv"),
        "smoothed": pd.read_csv(REPO_TABLE / "smoothed_feature_experiments.csv"),
        "features": pd.read_csv(REPO_TABLE / "final_master_feature_role_map.csv"),
    }


def drawdown(r: pd.Series) -> pd.Series:
    wealth = np.exp(r.fillna(0).cumsum())
    return wealth / wealth.cummax() - 1


def format_date_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
    ax.grid(axis="x", which="major", alpha=0.16)
    ax.grid(axis="y", alpha=0.28)
    ax.margins(x=0.01)


def legend_outside(ax, ncol: int = 3) -> None:
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=ncol, frameon=False)


def apply_grid(ax, x: bool = True, y: bool = True) -> None:
    ax.grid(axis="x", alpha=0.18 if x else 0)
    ax.grid(axis="y", alpha=0.24 if y else 0)


def label_barh(ax, values: pd.Series | np.ndarray, fmt: str = "{:.1f}") -> None:
    xmax = max(float(np.nanmax(values)), 1e-9)
    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_width() + xmax * 0.015,
            patch.get_y() + patch.get_height() / 2,
            fmt.format(float(value)),
            va="center",
            ha="left",
            fontsize=8.5,
            color=DARK,
        )


def label_bar(ax, values: pd.Series | np.ndarray, fmt: str = "{:.1f}") -> None:
    ymax = max(float(np.nanmax(values)), 1e-9)
    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + ymax * 0.015,
            fmt.format(float(value)),
            va="bottom",
            ha="center",
            fontsize=8.0,
            color=DARK,
        )


def fig_kfgi_timeseries(ts: pd.DataFrame) -> None:
    d = ts.copy()
    d["K_FGI_20D"] = d["K_FGI"].rolling(20, min_periods=5).mean()
    fig, ax = plt.subplots(figsize=(14, 5.4))
    ax.plot(d["date"], d["K_FGI_20D"], color=PURPLE, lw=2.0, label="K-FGI 20D MA")
    ax.axhline(25, color=BLUE, ls="--", lw=1.3, label="Fear 25")
    ax.axhline(65, color=RED, ls="--", lw=1.3, label="Greed 65")
    ax.fill_between(d["date"], 0, 25, color=LIGHT_BLUE, alpha=0.42)
    ax.fill_between(d["date"], 65, 100, color="#F7D7D7", alpha=0.35)
    ax.set_ylim(0, 100)
    ax.set_ylabel("K-FGI")
    format_date_axis(ax)
    legend_outside(ax)
    save(fig, "01_KFGI_시계열.png")


def fig_drawdown(ts: pd.DataFrame) -> None:
    d = ts.copy()
    d["bh_dd"] = drawdown(d["buy_hold_return"]) * 100
    d["kfgi_dd"] = drawdown(d["strategy_return"]) * 100
    fig, ax = plt.subplots(figsize=(14, 5.4))
    ax.fill_between(d["date"], d["bh_dd"], 0, color=LIGHT_BLUE, alpha=0.70, label="KOSPI200 B&H DD")
    ax.plot(d["date"], d["kfgi_dd"], color=BLUE, lw=2.2, label="K-FGI DD")
    ax.axhline(0, color=DARK, lw=0.8)
    ax.set_ylabel("Drawdown (%)")
    ax.set_ylim(min(d["bh_dd"].min(), d["kfgi_dd"].min()) - 5, 3)
    format_date_axis(ax)
    legend_outside(ax, ncol=2)
    save(fig, "02_최종_KFGI_낙폭방어.png")


def fig_exposure_volatility(ts: pd.DataFrame) -> None:
    d = ts.copy()
    d["weight_20D"] = d["weight"].rolling(20, min_periods=5).mean()
    fig, ax1 = plt.subplots(figsize=(14, 5.4))
    ax1.plot(d["date"], d["weight_20D"], color=BLUE, lw=2.0, label="Exposure 20D MA")
    ax1.set_ylabel("Exposure (x)")
    ax1.set_ylim(-0.02, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(d["date"], d["K_FGI"].rolling(20, min_periods=5).mean(), color=PURPLE, lw=1.4, alpha=0.82, label="K-FGI 20D MA")
    ax2.set_ylabel("K-FGI")
    ax2.set_ylim(0, 100)
    format_date_axis(ax1)
    ax2.grid(False)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    save(fig, "03_KFGI와_시장노출.png")


def fig_conditional_defense(cond: pd.DataFrame) -> None:
    keep = [
        "전체 시장 하락일",
        "고변동성 & 시장 하락일",
        "저변동성 & 시장 하락일",
        "Fear 구간 & 시장 하락일",
        "Neutral 구간 & 시장 하락일",
        "Greed 구간 & 시장 하락일",
    ]
    d = cond[cond["condition"].isin(keep)].copy()
    d["condition"] = pd.Categorical(d["condition"], keep, ordered=True)
    d = d.sort_values("condition")
    fig, ax = plt.subplots(figsize=(12, 5.8))
    colors = [BLUE if "전체" in c else RED if "고변동성" in c or "Fear" in c else GRAY for c in d["condition"].astype(str)]
    ax.barh(d["condition"].astype(str), d["downside_excess_bp"], color=colors, alpha=0.9)
    ax.set_xlabel("Market-down-day defense (bp/day)")
    ax.invert_yaxis()
    label_barh(ax, d["downside_excess_bp"], "{:.1f}bp")
    ax.set_xlim(0, d["downside_excess_bp"].max() * 1.18)
    apply_grid(ax, x=True, y=False)
    save(fig, "04_국면별_하방방어.png")


def fig_yearly_defense(cond: pd.DataFrame) -> None:
    d = cond[cond["condition"].str.contains("시장 하락일") & cond["condition"].str.contains("20|2016|2022|2025", regex=True)].copy()
    d["year"] = d["condition"].str.extract(r"(\d{4})").astype(int)
    d = d.sort_values("year")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(d["year"].astype(str), d["downside_excess_bp"], color=BLUE, alpha=0.88)
    ax.set_ylabel("Defense (bp/down day)")
    ax.set_xlabel("Year")
    label_bar(ax, d["downside_excess_bp"], "{:.1f}")
    ax.set_ylim(0, d["downside_excess_bp"].max() * 1.18)
    apply_grid(ax, x=True, y=True)
    save(fig, "05_연도별_하방방어.png")


def fig_feature_importance(rf: pd.DataFrame) -> None:
    d = rf.sort_values("rf_permutation_auc_drop", ascending=True).tail(10)
    d["importance_plot"] = d["rf_permutation_auc_drop"].clip(lower=0)
    colors = [RED if f == "sent_composite_ma10" else ORANGE if "egarch" in f or "vol" in f else BLUE for f in d["feature"]]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.barh(d["feature"], d["importance_plot"], color=colors, alpha=0.92)
    ax.set_xlabel("Permutation importance: AUC drop")
    label_barh(ax, d["importance_plot"], "{:.3f}")
    ax.set_xlim(0, max(d["importance_plot"].max() * 1.18, 0.01))
    apply_grid(ax, x=True, y=False)
    save(fig, "06_하방방어_피처중요도.png")


def fig_elasticnet(en: pd.DataFrame) -> None:
    d = en.copy()
    d["importance_abs"] = d["elasticnet_coef"].abs()
    d = d.sort_values("importance_abs", ascending=True).tail(10)
    colors = [RED if f == "sent_composite_ma10" else ORANGE if "egarch" in f or "vol" in f else BLUE for f in d["feature"]]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.barh(d["feature"], d["importance_abs"], color=colors, alpha=0.9)
    ax.set_xlabel("Absolute standardized coefficient")
    label_barh(ax, d["importance_abs"], "{:.3f}")
    ax.set_xlim(0, max(d["importance_abs"].max() * 1.18, 0.1))
    apply_grid(ax, x=True, y=False)
    save(fig, "07_ElasticNet_하방방어_계수.png")


def fig_sensitivity_cap(sens: pd.DataFrame) -> None:
    d = sens[sens["sensitivity_type"].eq("position_cap")].copy().sort_values("position_cap")
    fig, ax1 = plt.subplots(figsize=(10.5, 5.4))
    ax1.plot(d["position_cap"], d["mdd"] * 100, marker="o", color=BLUE, lw=2.2, label="MDD")
    ax1.set_xlabel("Position cap (x)")
    ax1.set_ylabel("MDD (%)")
    ax2 = ax1.twinx()
    ax2.plot(d["position_cap"], d["downside_excess_bp"], marker="s", color=RED, lw=2.0, label="Downside defense")
    ax2.set_ylabel("Defense (bp/down day)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    apply_grid(ax1, x=True, y=True)
    ax2.grid(False)
    save(fig, "08_포지션상한_민감도.png")


def fig_sensitivity_summary(sens: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    axes = axes.ravel()

    cap = sens[sens["sensitivity_type"].eq("position_cap")].copy().sort_values("position_cap")
    axes[0].plot(cap["position_cap"], cap["mdd_improvement_pctp"], marker="o", color=BLUE, lw=2.0)
    axes[0].set_xlabel("Position cap (x)")
    axes[0].set_ylabel("MDD improvement (%p)")

    fee = sens[sens["sensitivity_type"].eq("transaction_cost")].copy().sort_values("fee_bp")
    axes[1].plot(fee["fee_bp"], fee["mdd_improvement_pctp"], marker="o", color=RED, lw=2.0)
    axes[1].set_xlabel("Transaction cost (bp)")
    axes[1].set_ylabel("MDD improvement (%p)")

    th = sens[sens["sensitivity_type"].eq("fear_greed_threshold")].copy()
    th = th.sort_values(["fear", "greed"])
    axes[2].bar(th["setting"], th["downside_excess_bp"], color=BLUE, alpha=0.88)
    axes[2].set_xlabel("Fear/Greed threshold")
    axes[2].set_ylabel("Defense (bp/down day)")
    axes[2].tick_params(axis="x", rotation=25)

    mult = sens[sens["sensitivity_type"].eq("kfgi_multiplier")].copy().sort_values(["k_min", "k_max"])
    axes[3].bar(mult["setting"], mult["downside_excess_bp"], color=GREEN, alpha=0.88)
    axes[3].set_xlabel("K-FGI multiplier range")
    axes[3].set_ylabel("Defense (bp/down day)")
    axes[3].tick_params(axis="x", rotation=25)

    for ax in axes:
        apply_grid(ax, x=True, y=True)
    label_bar(axes[2], th["downside_excess_bp"], "{:.1f}")
    label_bar(axes[3], mult["downside_excess_bp"], "{:.1f}")
    fig.tight_layout()
    save(fig, "09_파라미터_민감도_요약.png")


def fig_sensitivity_heatmap(sens: pd.DataFrame) -> None:
    d = sens[sens["sensitivity_type"].isin(["fear_greed_threshold", "kfgi_multiplier", "transaction_cost"])].copy()
    order_map = {
        "fear_greed_threshold": ["fear20_greed80", "fear25_greed65", "fear30_greed70", "fear20_greed65", "fear30_greed65"],
        "kfgi_multiplier": ["k0.4_1.4", "k0.5_1.4", "k0.5_1.6", "k0.6_1.6", "k0.5_1.8"],
        "transaction_cost": ["fee_0bp", "fee_5bp", "fee_10bp", "fee_15bp", "fee_20bp"],
    }
    label_map = {
        "fear20_greed80": "20/80",
        "fear25_greed65": "25/65",
        "fear30_greed70": "30/70",
        "fear20_greed65": "20/65",
        "fear30_greed65": "30/65",
        "k0.4_1.4": "0.4-1.4",
        "k0.5_1.4": "0.5-1.4",
        "k0.5_1.6": "0.5-1.6",
        "k0.6_1.6": "0.6-1.6",
        "k0.5_1.8": "0.5-1.8",
        "fee_0bp": "0bp",
        "fee_5bp": "5bp",
        "fee_10bp": "10bp",
        "fee_15bp": "15bp",
        "fee_20bp": "20bp",
    }
    mat = []
    cell_labels = []
    for typ in ["fear_greed_threshold", "kfgi_multiplier", "transaction_cost"]:
        sub = d[d["sensitivity_type"].eq(typ)].set_index("setting").reindex(order_map[typ]).reset_index()
        mat.append(sub["downside_excess_bp"].to_numpy())
        cell_labels.append([label_map.get(s, str(s)) for s in sub["setting"]])
    max_len = max(len(x) for x in mat)
    arr = np.full((len(mat), max_len), np.nan)
    ylabels = ["Fear/Greed", "Multiplier", "Fee"]
    for i, row in enumerate(mat):
        arr[i, : len(row)] = row
    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    im = ax.imshow(arr, aspect="auto", cmap="Blues")
    ax.grid(False)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(max_len))
    ax.set_xticklabels([f"후보 {i + 1}" for i in range(max_len)])
    ax.set_xticks(np.arange(-0.5, max_len, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Defense (bp/down day)")
    threshold = np.nanpercentile(arr, 82)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isnan(arr[i, j]):
                continue
            color = "white" if arr[i, j] >= threshold else DARK
            ax.text(
                j,
                i,
                f"{cell_labels[i][j]}\n{arr[i, j]:.1f}bp",
                ha="center",
                va="center",
                fontsize=8.0,
                color=color,
                fontweight="bold" if arr[i, j] >= threshold else "normal",
            )
    save(fig, "12_민감도_히트맵_부록용.png")


def fig_sentiment_smoothing(smoothed: pd.DataFrame) -> None:
    keep = [
        "smoothed_sentiment_only",
        "sent_composite_ma10_only",
        "baseline_full_mixed_raw_smooth",
        "raw_sentiment_only",
        "market_egarch_only",
    ]
    d = smoothed[smoothed["label"].isin(keep)].copy()
    order = keep
    d["label"] = pd.Categorical(d["label"], order, ordered=True)
    d = d.sort_values("label")
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    colors = [RED if "sent_composite" in str(x) else GREEN if "smoothed" in str(x) else GRAY for x in d["label"]]
    ax.barh(d["label"].astype(str), d["mdd_improvement_pctp"], color=colors, alpha=0.9)
    ax.set_xlabel("MDD improvement vs B&H (%p)")
    ax.invert_yaxis()
    label_barh(ax, d["mdd_improvement_pctp"], "{:.1f}%p")
    ax.set_xlim(0, d["mdd_improvement_pctp"].max() * 1.18)
    apply_grid(ax, x=True, y=False)
    save(fig, "10_감성평활화_효과.png")


def fig_pca(pca: pd.DataFrame) -> None:
    d = pca.copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(d["component"], d["explained_variance_ratio"] * 100, color=LIGHT_BLUE, label="Explained variance")
    ax.plot(d["component"], d["cumulative_explained_variance"] * 100, color=BLUE, marker="o", lw=2.0, label="Cumulative")
    ax.set_ylabel("Explained variance (%)")
    ax.set_ylim(0, max(85, d["cumulative_explained_variance"].max() * 110))
    label_bar(ax, d["explained_variance_ratio"] * 100, "{:.1f}%")
    apply_grid(ax, x=True, y=True)
    legend_outside(ax, ncol=2)
    save(fig, "11_PCA_요인구조_부록용.png")


def write_guide() -> None:
    md = """# 최종 시각화 사용 가이드

## PCA 사용 여부

PCA는 최종 K-FGI 산출 방식으로 사용하지 않는다. 최종 모델은 `sub_index2-7 + sent_composite_ma10 + EGARCH/변동성 계열`의 원 피처를 그대로 사용한다. PCA는 **부록용 진단**이다. 즉, 피처들이 하나의 중복 축에만 몰려 있는지 확인하는 용도이며, 본문에서 K-FGI를 설명하거나 가중치를 정당화하는 주된 근거로 쓰지 않는다. 그래서 설명 가능성 저하는 피할 수 있다.

## 히트맵 사용 방식

히트맵은 메인 성과 그림이 아니다. `Fear/Greed`, `multiplier`, `거래비용` 후보를 바꿔도 하락일 방어폭이 유지되는지 한 장으로 압축한 **부록용 robustness 요약 그림**이다. 각 셀에는 `설정값 + 하락일 방어폭(bp)`을 함께 넣었다. 본문에서는 `08_포지션상한_민감도.png`와 민감도 표를 우선 사용하고, 히트맵은 “여러 파라미터 조합에서도 결론이 크게 바뀌지 않았다”는 보조 증거로만 둔다.

## 피처 중요도 해석

`06_하방방어_피처중요도.png`는 Random Forest permutation importance 기준이다. 값을 크게 해석하면 “해당 피처를 섞었을 때 OOS AUC가 얼마나 떨어지는가”이다. `sub_index6`, `sub_index2`, `sent_composite_ma10`가 상위 3개로 나타나므로, 안전자산 선호·시장 강도·평활화 감성이 하락일 방어 성공 분류에서 가장 큰 정보량을 제공한다.

`07_ElasticNet_하방방어_계수.png`는 계수의 방향이 아니라 절대 크기 기준이다. 방향성은 표에서 해석하고, 그림에서는 어떤 피처가 강하게 선택되었는지만 보여준다. `sent_composite_ma10`는 ElasticNet에서도 3위이므로, 감성 피처는 단독 신호라기보다 변동성·시장 내부강도와 함께 쓰일 때 의미 있는 보완축으로 해석한다.

## 최종 그림 배치

| 파일 | 권장 위치 | 그래프 형태 | 사용 목적 |
| --- | --- | --- | --- |
| `01_KFGI_시계열.png` | 4장 결과 초입 | 선그래프 | K-FGI 공포/탐욕 구간을 시간축으로 제시 |
| `02_최종_KFGI_낙폭방어.png` | 4장 메인 결과 | 면+선그래프 | B&H 대비 MDD 방어를 시계열로 제시 |
| `03_KFGI와_시장노출.png` | 방법론 또는 결과 | 이중축 선그래프 | K-FGI와 노출 조절 관계 설명 |
| `04_국면별_하방방어.png` | 조건부 성과 | 가로 막대 | 어떤 시장 국면에서 방어가 강한지 제시 |
| `05_연도별_하방방어.png` | 연도별/이벤트 해석 | 막대 | 2016, 2020, 2022, 2025 구간 해석 |
| `06_하방방어_피처중요도.png` | 피처 검증 | 가로 막대 | RF permutation 기준 중요 피처 제시 |
| `07_ElasticNet_하방방어_계수.png` | 피처 검증/부록 | 가로 막대 | 표준화 계수 크기 기준 피처 중요도 |
| `08_포지션상한_민감도.png` | robustness | 꺾은선 | 1.0배 상한 선택 근거 |
| `09_파라미터_민감도_요약.png` | robustness | 2x2 선/막대 | position cap, 거래비용, threshold, multiplier 안정성 |
| `10_감성평활화_효과.png` | 감성 피처 설계 | 가로 막대 | raw 감성보다 평활화 감성이 적절한 이유 |
| `11_PCA_요인구조_부록용.png` | 부록 | 막대+선 | 피처 구조 점검, 메인 주장에는 사용하지 않음 |
| `12_민감도_히트맵_부록용.png` | 부록 | 히트맵 | 여러 민감도 결과를 한 장에 압축한 보조 그림 |

## 작성 원칙

- 그래프 안에 해석 문장을 넣지 않는다.
- 막대그래프와 히트맵에는 수치 라벨을 넣어 독자가 표 없이도 크기를 읽을 수 있게 한다.
- 제목은 파일명과 동일하게 둔다.
- 자세한 해석은 본문 또는 figure caption에서 한다.
- 메인 결과는 선그래프/막대그래프 위주로 쓰고, 히트맵과 PCA는 부록에 둔다.
"""
    for root in [REPO_OUT, PAPER_OUT]:
        (root / "FINAL_FIGURE_USAGE_GUIDE.md").write_text(md, encoding="utf-8")


def main() -> None:
    setup()
    data = load_data()
    fig_kfgi_timeseries(data["ts"])
    fig_drawdown(data["ts"])
    fig_exposure_volatility(data["ts"])
    fig_conditional_defense(data["conditional"])
    fig_yearly_defense(data["conditional"])
    fig_feature_importance(data["rf_imp"])
    fig_elasticnet(data["en_imp"])
    fig_sensitivity_cap(data["sens"])
    fig_sensitivity_summary(data["sens"])
    fig_sensitivity_heatmap(data["sens"])
    fig_sentiment_smoothing(data["smoothed"])
    fig_pca(data["pca"])
    write_guide()
    print(REPO_FIG)
    print(PAPER_FIG)


if __name__ == "__main__":
    main()
