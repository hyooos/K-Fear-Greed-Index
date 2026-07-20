from __future__ import annotations

import math
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
    plt.rcParams["font.family"] = [
        "AppleGothic",
        "Malgun Gothic",
        "NanumGothic",
        "DejaVu Sans",
    ]
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 220


def max_drawdown(r: pd.Series) -> float:
    wealth = np.exp(r.fillna(0).cumsum())
    dd = wealth / wealth.cummax() - 1
    return float(dd.min())


def cvar(r: pd.Series, alpha: float = 0.05) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    var = r.quantile(alpha)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) else float(var)


def perf_metrics(r: pd.Series, bh: pd.Series | None = None, weight: pd.Series | None = None) -> dict:
    r = r.astype(float).fillna(0)
    ann_ret = np.exp(r.mean() * 252) - 1
    ann_vol = r.std(ddof=1) * math.sqrt(252)
    downside = r[r < 0]
    downside_dev = math.sqrt((np.minimum(r, 0) ** 2).mean()) * math.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    sortino = ann_ret / downside_dev if downside_dev > 0 else np.nan
    mdd = max_drawdown(r)
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan
    out = {
        "n": len(r),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "downside_dev": downside_dev,
        "sharpe": sharpe,
        "sortino": sortino,
        "mdd": mdd,
        "calmar": calmar,
        "var_5": float(r.quantile(0.05)),
        "cvar_5": cvar(r, 0.05),
        "total_return": float(np.exp(r.sum()) - 1),
        "positive_day_rate": float((r > 0).mean()),
    }
    if bh is not None:
        bh = bh.astype(float).fillna(0)
        down = bh < 0
        if down.any():
            excess = r[down] - bh[down]
            out.update(
                {
                    "downside_excess_bp": float(excess.mean() * 10000),
                    "downside_hit_rate": float((excess > 0).mean()),
                    "market_down_days": int(down.sum()),
                }
            )
    if weight is not None:
        w = weight.astype(float).fillna(0)
        out.update(
            {
                "avg_exposure": float(w.mean()),
                "median_exposure": float(w.median()),
                "cash_day_rate": float((w <= 0.05).mean()),
                "full_exposure_day_rate": float((w >= 0.95).mean()),
                "turnover": float(w.diff().abs().fillna(0).mean()),
            }
        )
    return out


def block_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    metric_fn,
    n_boot: int = 3000,
    block: int = 20,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(a)
    base = metric_fn(a) - metric_fn(b)
    vals = np.empty(n_boot)
    starts = np.arange(n)
    n_blocks = math.ceil(n / block)
    for i in range(n_boot):
        idx_parts = []
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        for s in chosen:
            idx_parts.append((np.arange(s, s + block) % n))
        idx = np.concatenate(idx_parts)[:n]
        vals[i] = metric_fn(a[idx]) - metric_fn(b[idx])
    if base >= 0:
        p = float((vals <= 0).mean())
    else:
        p = float((vals >= 0).mean())
    return {
        "observed_diff": float(base),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
        "one_sided_p_against_zero": p,
    }


def ann_ret_np(x: np.ndarray) -> float:
    s = pd.Series(x).fillna(0)
    return float(np.exp(s.mean() * 252) - 1)


def sharpe_np(x: np.ndarray) -> float:
    s = pd.Series(x).fillna(0)
    vol = s.std(ddof=1) * math.sqrt(252)
    return ann_ret_np(x) / vol if vol > 0 else np.nan


def sortino_np(x: np.ndarray) -> float:
    s = pd.Series(x).fillna(0)
    dd = math.sqrt((np.minimum(s, 0) ** 2).mean()) * math.sqrt(252)
    return ann_ret_np(x) / dd if dd > 0 else np.nan


def mdd_np(x: np.ndarray) -> float:
    return max_drawdown(pd.Series(x))


def cvar_np(x: np.ndarray) -> float:
    return cvar(pd.Series(x), 0.05)


def make_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q1 = pd.read_csv(DATA / "q1_benchmark_timeseries.csv", parse_dates=["date"])
    bh = q1["buy_hold_return"]

    strategy_cols = [
        "Buy & Hold",
        "K-FGI final",
        "MA 20/60 timing",
        "Realized-vol targeting",
        "EGARCH-only vol targeting",
        "VKOSPI/sub_index5 only",
        "Market-only FGI",
        "Sentiment-only index",
    ]
    rows = []
    for s in strategy_cols:
        rows.append(
            {
                "strategy": s,
                **perf_metrics(q1[s], bh, q1.get(f"{s}_weight")),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics["mdd_defense_rate_pct"] = (
        (abs(metrics.loc[metrics["strategy"] == "Buy & Hold", "mdd"].iloc[0]) - abs(metrics["mdd"]))
        / abs(metrics.loc[metrics["strategy"] == "Buy & Hold", "mdd"].iloc[0])
        * 100
    )
    metrics.to_csv(TABLES / "downside_risk_metrics_extended.csv", index=False)

    # Sentiment ablation: combine computable benchmark metrics and existing lean/drop experiments.
    ab = pd.read_csv(TABLES / "downside_defense_feature_ablation_summary.csv")
    keep = [
        "lean_sent_composite_ma10_only",
        "baseline_full",
        "drop_all_sentiment",
        "drop_sent_composite_ma10",
        "drop_egarch_family",
    ]
    ab = ab[ab["label"].isin(keep)].copy()
    ab["source"] = "feature-drop ablation"
    bench_ab = metrics[metrics["strategy"].isin(["K-FGI final", "Market-only FGI", "Sentiment-only index"])].copy()
    bench_ab = bench_ab.rename(columns={"strategy": "label"})
    bench_ab["strategy_name"] = bench_ab["label"]
    bench_ab["source"] = "q1 benchmark"
    ab_cols = [
        "label",
        "strategy_name",
        "source",
        "n",
        "total_return",
        "sharpe",
        "sortino",
        "calmar",
        "mdd",
        "mdd_defense_rate_pct",
        "downside_excess_bp",
        "downside_hit_rate",
    ]
    for c in ab_cols:
        if c not in ab:
            ab[c] = np.nan
    sent_ab = pd.concat([bench_ab[ab_cols], ab[ab_cols]], ignore_index=True)
    sent_ab.to_csv(TABLES / "sentiment_ablation_downside_extended.csv", index=False)

    robust = pd.read_csv(OUT / "robustness/data/priority_ab_kfgi_timeseries_robust.csv", parse_dates=["date"])
    robust = robust[robust["date"].isin(q1["date"])].copy()
    robust = robust.merge(
        q1[["date", "K-FGI final_weight", "buy_hold_return"]],
        on="date",
        how="left",
        suffixes=("", "_q1"),
    )
    robust["final_weight"] = robust["K-FGI final_weight"].fillna(robust["weight"].clip(0, 1))
    bh_col = "buy_hold_return_q1" if "buy_hold_return_q1" in robust.columns else "buy_hold_return"
    robust["market_down"] = robust[bh_col].fillna(robust["log_return"]) < 0
    robust["vol_bucket"] = pd.qcut(robust["egarch_vol"], 4, labels=["Q1 low vol", "Q2", "Q3", "Q4 high vol"], duplicates="drop")
    robust["kfgi_zone"] = np.select(
        [robust["K_FGI"] < 25, robust["K_FGI"] > 65],
        ["Fear", "Greed"],
        default="Neutral",
    )
    exp_rows = []
    for name, mask in {
        "All days": robust.index == robust.index,
        "Market down days": robust["market_down"],
        "Market up/flat days": ~robust["market_down"],
        "High volatility Q4": robust["vol_bucket"].astype(str).eq("Q4 high vol"),
        "Low volatility Q1": robust["vol_bucket"].astype(str).eq("Q1 low vol"),
        "Fear zone": robust["kfgi_zone"].eq("Fear"),
        "Neutral zone": robust["kfgi_zone"].eq("Neutral"),
        "Greed zone": robust["kfgi_zone"].eq("Greed"),
    }.items():
        d = robust.loc[mask].copy()
        exp_rows.append(
            {
                "condition": name,
                "n": len(d),
                "avg_exposure": d["final_weight"].mean(),
                "median_exposure": d["final_weight"].median(),
                "cash_day_rate": (d["final_weight"] <= 0.05).mean(),
                "full_exposure_day_rate": (d["final_weight"] >= 0.95).mean(),
                "avg_kfgi": d["K_FGI"].mean(),
                "avg_egarch_vol": d["egarch_vol"].mean(),
            }
        )
    exposure = pd.DataFrame(exp_rows)
    exposure.to_csv(TABLES / "exposure_by_regime_extended.csv", index=False)

    boot_rows = []
    pairs = [
        ("K-FGI final", "Buy & Hold"),
        ("K-FGI final", "EGARCH-only vol targeting"),
        ("K-FGI final", "Realized-vol targeting"),
        ("K-FGI final", "Market-only FGI"),
        ("K-FGI final", "Sentiment-only index"),
    ]
    metric_fns = {
        "annual_return_diff": ann_ret_np,
        "sharpe_diff": sharpe_np,
        "sortino_diff": sortino_np,
        "mdd_diff": mdd_np,
        "cvar5_diff": cvar_np,
    }
    for a_name, b_name in pairs:
        a = q1[a_name].to_numpy(float)
        b = q1[b_name].to_numpy(float)
        for metric, fn in metric_fns.items():
            res = block_bootstrap_diff(a, b, fn)
            boot_rows.append(
                {
                    "strategy_a": a_name,
                    "strategy_b": b_name,
                    "metric": metric,
                    **res,
                }
            )
        down = bh.to_numpy(float) < 0
        res = block_bootstrap_diff(a[down] - b[down], np.zeros(down.sum()), lambda x: float(np.mean(x) * 10000))
        boot_rows.append(
            {
                "strategy_a": a_name,
                "strategy_b": b_name,
                "metric": "market_down_day_excess_bp",
                **res,
            }
        )
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(TABLES / "bootstrap_strategy_difference_tests.csv", index=False)
    return metrics, sent_ab, exposure, boot


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def make_figures(metrics: pd.DataFrame, sent_ab: pd.DataFrame, exposure: pd.DataFrame, boot: pd.DataFrame) -> None:
    setup_plot()
    FIGS.mkdir(parents=True, exist_ok=True)

    # 13: Downside metric comparison
    plot_df = metrics[metrics["strategy"].isin(["Buy & Hold", "K-FGI final", "EGARCH-only vol targeting", "Realized-vol targeting", "Market-only FGI", "Sentiment-only index"])].copy()
    labels = ["B&H", "K-FGI", "EGARCH", "Realized vol", "Market FGI", "Sentiment"]
    plot_df["short"] = labels
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    specs = [
        ("mdd", "MDD (%)", lambda s: s * 100, "lower"),
        ("sortino", "Sortino", lambda s: s, "higher"),
        ("cvar_5", "CVaR 5% (%)", lambda s: s * 100, "lower"),
    ]
    for ax, (col, ylabel, trans, _) in zip(axes, specs):
        vals = trans(plot_df[col])
        colors = ["#8d99ae" if s != "K-FGI final" else "#2563eb" for s in plot_df["strategy"]]
        bars = ax.bar(plot_df["short"], vals, color=colors)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.28)
        ax.tick_params(axis="x", rotation=25)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "13_downside_risk_metrics_extended.png", bbox_inches="tight")
    plt.close(fig)

    # 14: Sentiment ablation
    ab = sent_ab[sent_ab["label"].isin(["K-FGI final", "Market-only FGI", "Sentiment-only index", "lean_sent_composite_ma10_only", "drop_all_sentiment", "drop_sent_composite_ma10"])].copy()
    name_map = {
        "K-FGI final": "K-FGI final",
        "Market-only FGI": "Market-only",
        "Sentiment-only index": "Sentiment-only",
        "lean_sent_composite_ma10_only": "Lean sent MA10",
        "drop_all_sentiment": "No sentiment",
        "drop_sent_composite_ma10": "Drop sent MA10",
    }
    ab["short"] = ab["label"].map(name_map)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    colors = ["#2563eb" if x in ["K-FGI final", "Lean sent MA10"] else "#94a3b8" for x in ab["short"]]
    axes[0].barh(ab["short"], ab["mdd"] * 100, color=colors)
    axes[0].set_xlabel("MDD (%)")
    axes[0].grid(axis="x", alpha=0.28)
    axes[0].axvline(0, color="#111827", linewidth=0.8)
    axes[1].barh(ab["short"], ab["downside_excess_bp"], color=colors)
    axes[1].set_xlabel("Downside defense (bp/down day)")
    axes[1].grid(axis="x", alpha=0.28)
    fig.tight_layout()
    fig.savefig(FIGS / "14_sentiment_ablation_downside_extended.png", bbox_inches="tight")
    plt.close(fig)

    # 15: Exposure by regime
    exp = exposure.copy()
    fig, ax = plt.subplots(figsize=(12, 5.2))
    bars = ax.bar(exp["condition"], exp["avg_exposure"], color="#2563eb")
    ax.set_ylabel("Average exposure (x)")
    ax.set_ylim(0, max(1.0, exp["avg_exposure"].max() * 1.2))
    ax.grid(axis="y", alpha=0.28)
    ax.tick_params(axis="x", rotation=30)
    for b, v in zip(bars, exp["avg_exposure"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}x", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "15_regime_exposure_analysis.png", bbox_inches="tight")
    plt.close(fig)

    # 16: Bootstrap tests
    boot_plot = boot[
        (boot["strategy_b"].isin(["Buy & Hold", "EGARCH-only vol targeting", "Realized-vol targeting", "Market-only FGI"]))
        & (boot["metric"].isin(["sortino_diff", "mdd_diff", "market_down_day_excess_bp"]))
    ].copy()
    metric_labels = {
        "sortino_diff": "Sortino diff",
        "mdd_diff": "MDD diff",
        "market_down_day_excess_bp": "Down-day excess bp",
    }
    boot_plot["label"] = boot_plot["strategy_b"].replace(
        {
            "Buy & Hold": "vs B&H",
            "EGARCH-only vol targeting": "vs EGARCH",
            "Realized-vol targeting": "vs Realized vol",
            "Market-only FGI": "vs Market FGI",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, metric in zip(axes, ["sortino_diff", "mdd_diff", "market_down_day_excess_bp"]):
        d = boot_plot[boot_plot["metric"] == metric]
        y = np.arange(len(d))
        ax.errorbar(
            d["observed_diff"],
            y,
            xerr=[d["observed_diff"] - d["ci_low"], d["ci_high"] - d["observed_diff"]],
            fmt="o",
            color="#2563eb",
            ecolor="#94a3b8",
            capsize=3,
        )
        ax.axvline(0, color="#111827", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(d["label"])
        ax.set_xlabel(metric_labels[metric])
        ax.grid(axis="x", alpha=0.28)
    fig.tight_layout()
    fig.savefig(FIGS / "16_bootstrap_strategy_difference_tests.png", bbox_inches="tight")
    plt.close(fig)


def write_md(metrics: pd.DataFrame, sent_ab: pd.DataFrame, exposure: pd.DataFrame, boot: pd.DataFrame) -> None:
    m = metrics.copy()
    show_cols = [
        "strategy",
        "total_return",
        "sharpe",
        "sortino",
        "calmar",
        "mdd",
        "var_5",
        "cvar_5",
        "downside_excess_bp",
        "avg_exposure",
    ]
    md = []
    md.append("# Downside Risk Extension Results\n")
    md.append("본 문서는 sentiment ablation, downside risk metrics, exposure analysis, bootstrap 검정을 추가로 정리한 결과이다.\n")
    md.append("## 1. Extended Downside Metrics\n")
    tmp = m[show_cols].copy()
    for c in ["total_return", "mdd", "var_5", "cvar_5"]:
        tmp[c] = (tmp[c] * 100).round(2)
    for c in ["sharpe", "sortino", "calmar", "downside_excess_bp", "avg_exposure"]:
        tmp[c] = tmp[c].round(3)
    md.append(tmp.to_markdown(index=False))
    md.append("\n\n해석: K-FGI final은 Buy & Hold보다 누적수익률과 Sharpe는 낮지만, MDD와 CVaR 측면에서 손실 꼬리를 줄이는 데 강점이 있다. 따라서 Sortino, Calmar, CVaR은 본 논문의 downside risk management 프레임과 잘 맞는 보조 성과지표다.\n")

    md.append("\n## 2. Sentiment Ablation\n")
    ab = sent_ab[["label", "strategy_name", "source", "mdd", "downside_excess_bp", "downside_hit_rate"]].copy()
    ab["mdd"] = (ab["mdd"] * 100).round(2)
    ab["downside_excess_bp"] = ab["downside_excess_bp"].round(2)
    ab["downside_hit_rate"] = (ab["downside_hit_rate"] * 100).round(1)
    md.append(ab.to_markdown(index=False))
    md.append("\n\n해석: 감성 피처 전체를 늘리는 것보다 `sent_composite_ma10` 중심의 lean 구조가 가장 설명 가능하다. 감성 제거 계열은 MDD가 악화되는 경향이 있으므로, 감성은 독립 alpha라기보다 K-FGI의 하방위험 방어를 보완하는 평활화 심리 정보축으로 쓰는 것이 적절하다.\n")

    md.append("\n## 3. Exposure Analysis\n")
    ex = exposure.copy()
    ex["avg_exposure"] = ex["avg_exposure"].round(3)
    ex["median_exposure"] = ex["median_exposure"].round(3)
    ex["cash_day_rate"] = (ex["cash_day_rate"] * 100).round(1)
    ex["full_exposure_day_rate"] = (ex["full_exposure_day_rate"] * 100).round(1)
    ex["avg_kfgi"] = ex["avg_kfgi"].round(2)
    ex["avg_egarch_vol"] = (ex["avg_egarch_vol"] * 100).round(2)
    md.append(ex.to_markdown(index=False))
    md.append("\n\n해석: K-FGI의 낮은 누적수익률은 상당 부분 평균 노출이 약 0.55x로 낮기 때문이다. 특히 Fear zone과 고변동성 구간에서 노출이 크게 낮아진다. 이는 기회비용을 만들지만, 동시에 MDD와 하락일 손실을 줄이는 직접 메커니즘이다.\n")

    md.append("\n## 4. Bootstrap Strategy Difference Tests\n")
    bt = boot.copy()
    for c in ["observed_diff", "ci_low", "ci_high", "one_sided_p_against_zero"]:
        bt[c] = bt[c].round(4)
    md.append(bt.to_markdown(index=False))
    md.append("\n\n해석: bootstrap 결과는 전략 간 평균/위험조정 성과 차이를 보수적으로 점검하기 위한 것이다. 논문에서는 전체 수익률 차이보다 `market_down_day_excess_bp`, `mdd_diff`, `cvar5_diff`를 우선적으로 해석해야 한다.\n")

    md.append("\n## 5. 생성된 그림\n")
    md.append("- `paper_outputs/final_figures/13_downside_risk_metrics_extended.png`\n")
    md.append("- `paper_outputs/final_figures/14_sentiment_ablation_downside_extended.png`\n")
    md.append("- `paper_outputs/final_figures/15_regime_exposure_analysis.png`\n")
    md.append("- `paper_outputs/final_figures/16_bootstrap_strategy_difference_tests.png`\n")
    (OUT / "DOWNSIDE_RISK_EXTENSION_RESULTS.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    metrics, sent_ab, exposure, boot = make_metrics()
    make_figures(metrics, sent_ab, exposure, boot)
    write_md(metrics, sent_ab, exposure, boot)
    print("Wrote downside risk extension outputs.")


if __name__ == "__main__":
    main()
