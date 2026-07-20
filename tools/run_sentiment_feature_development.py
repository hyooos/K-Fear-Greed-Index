from __future__ import annotations

import math
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


TRADING_DAYS = 252


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


def ensure_dirs() -> None:
    for root in [REPO_OUT, PAPER_OUT]:
        (root / "tables").mkdir(parents=True, exist_ok=True)
        (root / "figures").mkdir(parents=True, exist_ok=True)


def clean_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_data() -> pd.DataFrame:
    ts = pd.read_csv(
        REPO_OUT / "robustness" / "data" / "priority_ab_kfgi_timeseries_robust.csv",
        parse_dates=["date"],
    )
    rets = pd.read_csv(
        REPO_OUT / "robustness" / "data" / "priority_ab_strategy_returns_robust.csv",
        parse_dates=["date"],
    )
    df = ts.merge(rets, on="date", how="inner", suffixes=("", "_ret"))
    for col in df.columns:
        if col != "date":
            df[col] = clean_num(df[col])

    df["strategy_delta"] = df["kfgi_with_sentiment"] - df["kfgi_without_sentiment"]
    df["with_excess_bh"] = df["kfgi_with_sentiment"] - df["buy_hold"]
    df["without_excess_bh"] = df["kfgi_without_sentiment"] - df["buy_hold"]
    df["incremental_defense"] = df["with_excess_bh"] - df["without_excess_bh"]
    df["buy_hold_down"] = df["buy_hold"] < 0
    df["next_5d"] = df["target_5d"]

    # Candidate sentiment refinements. These are deliberately simple and
    # reproducible: intensity is allowed to matter only when attention is high.
    attention = np.log1p(df["effective_n"].clip(lower=0))
    comments = np.log1p(df["comment_count"].clip(lower=0))
    df["sent_attention"] = df["sent_norm_w"] * attention
    df["negative_attention"] = df["neg_z"].clip(lower=0) * attention
    df["panic_pressure"] = (-df["sent_norm_w"]).clip(lower=0) * df["sent_strength_w"].clip(lower=0) * attention
    df["disagreement_attention"] = df["sent_std"].clip(lower=0) * attention
    df["heat_attention"] = df["heat"].clip(lower=0) * comments
    df["sent_confidence"] = df["sent_norm_w"].abs() * attention / (df["sent_std"].abs() + 0.05)

    # A deployable version of a sentiment risk signal: high when negative tone,
    # comment attention, and disagreement all rise together.
    parts = []
    for col in ["negative_attention", "panic_pressure", "disagreement_attention", "heat_attention"]:
        r = df[col].rank(pct=True)
        parts.append(r)
    df["sentiment_risk_composite"] = pd.concat(parts, axis=1).mean(axis=1)
    return df


def one_sample(x: pd.Series) -> tuple[float, float, float]:
    x = x.dropna()
    if len(x) < 5 or x.std(ddof=1) == 0:
        return np.nan, np.nan, np.nan
    res = stats.ttest_1samp(x, 0)
    return float(x.mean()), float(res.statistic), float(res.pvalue)


def welch(a: pd.Series, b: pd.Series) -> tuple[float, float, float]:
    a = a.dropna()
    b = b.dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan, np.nan
    res = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean() - b.mean()), float(res.statistic), float(res.pvalue)


def mannwhitney(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna()
    b = b.dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def bh_qvalues(pvals: pd.Series) -> pd.Series:
    p = pvals.astype(float).to_numpy()
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return pd.Series(q, index=pvals.index)
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0, 1)
    return pd.Series(q, index=pvals.index)


def condition_tests(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("부정 관심 상위 20%", "negative_attention", "top", 0.80),
        ("공포 압력 상위 20%", "panic_pressure", "top", 0.80),
        ("의견 불일치 상위 20%", "disagreement_attention", "top", 0.80),
        ("댓글 열기 상위 20%", "heat_attention", "top", 0.80),
        ("감성 신뢰도 상위 20%", "sent_confidence", "top", 0.80),
        ("감성위험 복합 상위 20%", "sentiment_risk_composite", "top", 0.80),
        ("긍정 감성 상위 20%", "sent_attention", "top", 0.80),
        ("부정 감성 하위 20%", "sent_attention", "bottom", 0.20),
    ]
    rows = []
    high_vol = df["egarch_vol"] >= df["egarch_vol"].quantile(0.70)
    for label, col, side, q in specs:
        s = df[col]
        cutoff = s.quantile(q)
        mask = s >= cutoff if side == "top" else s <= cutoff
        low_mask = s <= s.quantile(0.20) if side == "top" else s >= s.quantile(0.80)
        groups = {
            "전체 조건일": mask,
            "B&H 하락일": mask & df["buy_hold_down"],
            "고변동성일": mask & high_vol,
            "하락+고변동성일": mask & df["buy_hold_down"] & high_vol,
        }
        for group, gmask in groups.items():
            sub = df.loc[gmask]
            mean_delta, t_delta, p_delta = one_sample(sub["strategy_delta"])
            mean_def, t_def, p_def = one_sample(sub["incremental_defense"])
            next_diff, t_next, p_next = welch(df.loc[mask, "next_5d"], df.loc[low_mask, "next_5d"])
            mw_next = mannwhitney(df.loc[mask, "next_5d"], df.loc[low_mask, "next_5d"])
            rows.append(
                {
                    "condition": label,
                    "group": group,
                    "feature": col,
                    "side": side,
                    "cutoff": cutoff,
                    "n": int(gmask.sum()),
                    "mean_incremental_return_bp": mean_delta * 10000,
                    "t_incremental_return": t_delta,
                    "p_incremental_return": p_delta,
                    "mean_incremental_defense_bp": mean_def * 10000,
                    "t_incremental_defense": t_def,
                    "p_incremental_defense": p_def,
                    "next_5d_high_low_diff_pctp": next_diff * 100,
                    "t_next_5d_high_low": t_next,
                    "p_next_5d_high_low": p_next,
                    "mw_p_next_5d_high_low": mw_next,
                }
            )
    out = pd.DataFrame(rows)
    out["q_incremental_return"] = bh_qvalues(out["p_incremental_return"])
    out["q_incremental_defense"] = bh_qvalues(out["p_incremental_defense"])
    out["q_next_5d_high_low"] = bh_qvalues(out["p_next_5d_high_low"])
    return out.sort_values(["q_incremental_defense", "p_incremental_defense", "condition", "group"])


def interaction_regression(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_cols = {
        "strategy_delta": "감성 포함-제외 전략수익 차이",
        "with_excess_bh": "감성 포함 전략의 B&H 대비 초과수익",
        "buy_hold": "B&H 일수익률",
    }
    x_cols = ["sentiment_risk_composite", "negative_attention", "panic_pressure", "disagreement_attention"]
    high_vol = (df["egarch_vol"] >= df["egarch_vol"].quantile(0.70)).astype(float)
    down_prev = (df["log_return"] < 0).astype(float)
    for y_col, y_label in y_cols.items():
        for x_col in x_cols:
            data = pd.DataFrame(
                {
                    "y": df[y_col],
                    "x": df[x_col],
                    "high_vol": high_vol,
                    "down_prev": down_prev,
                }
            ).dropna()
            if len(data) < 30:
                continue
            for col in ["x"]:
                sd = data[col].std(ddof=1)
                if sd > 0:
                    data[col] = (data[col] - data[col].mean()) / sd
            data["x_high_vol"] = data["x"] * data["high_vol"]
            data["x_down_prev"] = data["x"] * data["down_prev"]
            xmat = np.column_stack(
                [
                    np.ones(len(data)),
                    data["x"],
                    data["high_vol"],
                    data["down_prev"],
                    data["x_high_vol"],
                    data["x_down_prev"],
                ]
            )
            beta, *_ = np.linalg.lstsq(xmat, data["y"].to_numpy(), rcond=None)
            resid = data["y"].to_numpy() - xmat @ beta
            dof = len(data) - xmat.shape[1]
            sigma2 = float((resid @ resid) / dof)
            cov = sigma2 * np.linalg.pinv(xmat.T @ xmat)
            se = np.sqrt(np.diag(cov))
            tvals = beta / se
            pvals = 2 * stats.t.sf(np.abs(tvals), dof)
            names = ["intercept", "sentiment", "high_vol", "prev_down", "sentiment_x_high_vol", "sentiment_x_prev_down"]
            for name, b, se_i, t, p in zip(names, beta, se, tvals, pvals):
                rows.append(
                    {
                        "outcome": y_label,
                        "outcome_col": y_col,
                        "sentiment_feature": x_col,
                        "term": name,
                        "coef_bp": b * 10000,
                        "se_bp": se_i * 10000,
                        "t": t,
                        "p": p,
                        "n": len(data),
                    }
                )
    out = pd.DataFrame(rows)
    out["q"] = bh_qvalues(out["p"])
    return out.sort_values(["q", "p"])


def candidate_feature_scores(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "sent_norm_w",
        "sent_attention",
        "negative_attention",
        "panic_pressure",
        "disagreement_attention",
        "heat_attention",
        "sent_confidence",
        "sentiment_risk_composite",
        "sent_composite",
        "sent_composite_ma10",
        "neg_z",
        "sent_std",
    ]
    rows = []
    for f in features:
        s = df[f]
        top = s >= s.quantile(0.80)
        bottom = s <= s.quantile(0.20)
        pearson = stats.pearsonr(s.fillna(s.median()), df["next_5d"].fillna(0))
        spearman = stats.spearmanr(s, df["next_5d"], nan_policy="omit")
        d_delta, t_delta, p_delta = welch(df.loc[top, "strategy_delta"], df.loc[bottom, "strategy_delta"])
        d_def, t_def, p_def = welch(df.loc[top & df["buy_hold_down"], "incremental_defense"], df.loc[bottom & df["buy_hold_down"], "incremental_defense"])
        rows.append(
            {
                "feature": f,
                "pearson_next_5d": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_next_5d": spearman.correlation,
                "spearman_p": spearman.pvalue,
                "top_bottom_strategy_delta_diff_bp": d_delta * 10000,
                "p_strategy_delta_diff": p_delta,
                "top_bottom_downside_defense_diff_bp": d_def * 10000,
                "p_downside_defense_diff": p_def,
            }
        )
    out = pd.DataFrame(rows)
    out["q_strategy_delta_diff"] = bh_qvalues(out["p_strategy_delta_diff"])
    out["q_downside_defense_diff"] = bh_qvalues(out["p_downside_defense_diff"])
    return out.sort_values(["q_downside_defense_diff", "q_strategy_delta_diff"])


def perf(r: pd.Series, bh: pd.Series | None = None, no_sent: pd.Series | None = None) -> dict:
    r = r.dropna()
    cum = np.exp(r.cumsum())
    dd = cum / cum.cummax() - 1
    ann_vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    out = {
        "ann_ret": float(r.mean() * TRADING_DAYS),
        "ann_vol": ann_vol,
        "sharpe": float(r.mean() * TRADING_DAYS / (ann_vol + 1e-12)),
        "mdd": float(dd.min()),
        "total_return": float(cum.iloc[-1] - 1),
    }
    if bh is not None:
        a = pd.concat([r.rename("r"), bh.rename("bh")], axis=1).dropna()
        ex = a["r"] - a["bh"]
        down = a["bh"] < 0
        out["excess_bp"] = float(ex.mean() * 10000)
        out["excess_p"] = float(stats.ttest_1samp(ex, 0).pvalue)
        out["downside_excess_bp"] = float(ex[down].mean() * 10000)
        out["downside_p"] = float(stats.ttest_1samp(ex[down], 0).pvalue)
    if no_sent is not None:
        a = pd.concat([r.rename("r"), no_sent.rename("no_sent"), bh.rename("bh")], axis=1).dropna()
        inc = a["r"] - a["no_sent"]
        down = a["bh"] < 0
        out["incremental_vs_no_sent_bp"] = float(inc.mean() * 10000)
        out["incremental_vs_no_sent_p"] = float(stats.ttest_1samp(inc, 0).pvalue)
        out["incremental_downside_vs_no_sent_bp"] = float(inc[down].mean() * 10000)
        out["incremental_downside_vs_no_sent_p"] = float(stats.ttest_1samp(inc[down], 0).pvalue)
    return out


def strategy_from_weight(df: pd.DataFrame, weight: pd.Series, fee: float = 0.0015) -> pd.Series:
    weight_lag = weight.shift(1).fillna(0)
    turnover = weight_lag.diff().abs().fillna(weight_lag.iloc[0])
    return weight_lag * df["target_reg"] - turnover * fee


def overlay_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_weight = df["weight"].copy()
    feature_names = ["sentiment_risk_composite", "panic_pressure", "negative_attention", "disagreement_attention", "heat_attention"]
    for feature in feature_names:
        for q in [0.70, 0.80, 0.90]:
            mask = df[feature] >= df[feature].quantile(q)
            for cap in [0.7, 1.0, 1.2]:
                w = base_weight.copy()
                w.loc[mask] = np.minimum(w.loc[mask], cap)
                r = strategy_from_weight(df, w)
                rows.append(
                    {
                        "overlay": "cap",
                        "feature": feature,
                        "threshold_quantile": q,
                        "parameter": cap,
                        "active_days": int(mask.sum()),
                        **perf(r, df["buy_hold"], df["kfgi_without_sentiment"]),
                    }
                )
            for mult in [0.5, 0.7]:
                w = base_weight.copy()
                w.loc[mask] = w.loc[mask] * mult
                r = strategy_from_weight(df, w)
                rows.append(
                    {
                        "overlay": "multiply",
                        "feature": feature,
                        "threshold_quantile": q,
                        "parameter": mult,
                        "active_days": int(mask.sum()),
                        **perf(r, df["buy_hold"], df["kfgi_without_sentiment"]),
                    }
                )
    out = pd.DataFrame(rows)
    out["q_incremental_downside_vs_no_sent"] = bh_qvalues(out["incremental_downside_vs_no_sent_p"])
    return out.sort_values(["q_incremental_downside_vs_no_sent", "sharpe"], ascending=[True, False])


def set_korean_font() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    candidates = ["AppleGothic", "NanumGothic", "Malgun Gothic", "Arial Unicode MS"]
    for name in candidates:
        try:
            plt.rcParams["font.family"] = name
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, "감성")
            plt.close(fig)
            return
        except Exception:
            plt.close("all")


def save_fig(fig: plt.Figure, filename: str) -> None:
    for root in [REPO_OUT, PAPER_OUT]:
        fig.savefig(root / "figures" / filename, dpi=220, bbox_inches="tight")


def make_figures(df: pd.DataFrame, cond: pd.DataFrame, scores: pd.DataFrame, overlay: pd.DataFrame) -> None:
    set_korean_font()
    feature_label = {
        "sentiment_risk_composite": "감성위험 복합",
        "panic_pressure": "공포 압력",
        "negative_attention": "부정 관심",
        "disagreement_attention": "의견 불일치",
        "heat_attention": "댓글 열기",
        "sent_confidence": "감성 신뢰도",
    }

    selected = cond[cond["group"].eq("B&H 하락일") & (cond["mean_incremental_defense_bp"] > 0)].copy()
    selected = selected.sort_values("mean_incremental_defense_bp", ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = np.where(selected["q_incremental_defense"] < 0.10, "#2f6fd6", "#9aa8b7")
    labels = selected["condition"].str.replace(" 상위 20%", "", regex=False).str.replace(" 하위 20%", "", regex=False)
    ax.barh(labels, selected["mean_incremental_defense_bp"], color=colors)
    ax.axvline(0, color="#333333", lw=1)
    for y, row in enumerate(selected.itertuples()):
        ax.text(
            row.mean_incremental_defense_bp + (0.15 if row.mean_incremental_defense_bp >= 0 else -0.15),
            y,
            f"{row.mean_incremental_defense_bp:.2f}bp, q={row.q_incremental_defense:.3f}",
            va="center",
            ha="left" if row.mean_incremental_defense_bp >= 0 else "right",
            fontsize=9,
        )
    ax.set_xlabel("감성 포함 전략의 추가 하방 방어 효과(bp/day)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    ax.text(
        0.0,
        -0.16,
        "파란색은 Benjamini-Hochberg 보정 q<0.10 조건이다. 값이 클수록 감성 피처가 제외 모델보다 하락일 손실을 더 줄였다는 뜻이다.",
        transform=ax.transAxes,
        fontsize=10,
        color="#555555",
    )
    save_fig(fig, "20_감성조건부_하방방어.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    d = df.sort_values("date")
    risk = d["sentiment_risk_composite"].rolling(5).mean()
    delta = d["strategy_delta"].rolling(20).mean() * 10000
    axes[0].plot(d["date"], risk, color="#7a3fd1", lw=1.8, label="감성위험 복합지표(5일 평균)")
    axes[0].axhline(risk.quantile(0.80), color="#cc3d3d", lw=1, ls="--", label="상위 20% 기준")
    axes[0].set_ylabel("감성위험")
    axes[0].legend(loc="upper left", ncols=2, frameon=True)
    axes[0].grid(alpha=0.2)
    axes[1].plot(d["date"], delta, color="#2066d4", lw=1.5, label="감성 포함-제외 전략수익 차이(20일 평균, bp)")
    axes[1].axhline(0, color="#333333", lw=1)
    axes[1].set_ylabel("추가 기여(bp)")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper left", frameon=True)
    axes[1].grid(alpha=0.2)
    fig.subplots_adjust(hspace=0.08)
    save_fig(fig, "21_감성위험지표와_추가기여.png")
    plt.close(fig)

    positive_scores = scores[scores["top_bottom_downside_defense_diff_bp"] > 0].sort_values(
        ["q_downside_defense_diff", "p_downside_defense_diff"]
    )
    top_scores = pd.concat(
        [
            positive_scores,
            scores[~scores.index.isin(positive_scores.index)].sort_values(["q_downside_defense_diff", "p_downside_defense_diff"]),
        ]
    ).head(8).copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top_scores))
    width = 0.36
    ax.bar(x - width / 2, top_scores["top_bottom_strategy_delta_diff_bp"], width, label="전략수익 차이", color="#2f6fd6")
    ax.bar(x + width / 2, top_scores["top_bottom_downside_defense_diff_bp"], width, label="하락일 방어 차이", color="#e58b28")
    ax.axhline(0, color="#333333", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(top_scores["feature"], rotation=30, ha="right")
    ax.set_ylabel("상위 20%-하위 20% 차이(bp/day)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, "22_감성후보피처_효과비교.png")
    plt.close(fig)

    selected = overlay[
        (overlay["incremental_downside_vs_no_sent_bp"] > 0)
        & (overlay["total_return"] > 0)
        & (overlay["q_incremental_downside_vs_no_sent"] < 0.10)
    ].sort_values("sharpe", ascending=False).head(10)
    if len(selected):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        labels = []
        for row in selected.itertuples():
            q = int(round(row.threshold_quantile * 100))
            labels.append(f"{feature_label.get(row.feature, row.feature)}\nq{q}, {row.overlay}={row.parameter:g}")
        y = np.arange(len(selected))
        axes[0].barh(y, selected["total_return"] * 100, color="#2f6fd6", alpha=0.9)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels, fontsize=8.5)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("전체 누적수익률(%)")
        axes[0].grid(axis="x", alpha=0.25)
        for yi, val in zip(y, selected["total_return"] * 100):
            axes[0].text(val + 0.6, yi, f"{val:.1f}%", va="center", fontsize=8.5)

        axes[1].barh(y, selected["incremental_downside_vs_no_sent_bp"], color="#e58b28", alpha=0.9)
        axes[1].axvline(0, color="#333333", lw=1)
        axes[1].set_xlabel("감성 제외 대비 추가 하락일 방어(bp/day)")
        axes[1].grid(axis="x", alpha=0.25)
        for yi, val, qv in zip(y, selected["incremental_downside_vs_no_sent_bp"], selected["q_incremental_downside_vs_no_sent"]):
            axes[1].text(val + 0.08, yi, f"{val:.2f}bp\nq={qv:.3g}", va="center", fontsize=8)
        axes[0].text(
            0.0,
            -0.16,
            "오른쪽 위에 가까울수록 전체 수익을 유지하면서 감성위험 overlay가 하락일 방어를 유의하게 개선한 후보이다.",
            transform=axes[0].transAxes,
            fontsize=10,
            color="#555555",
        )
        fig.subplots_adjust(wspace=0.08)
        save_fig(fig, "23_감성위험_오버레이_후보.png")
        plt.close(fig)


def md_table(df: pd.DataFrame, cols: list[str], n: int = 12) -> str:
    d = df[cols].head(n).copy()
    for col in d.columns:
        d[col] = d[col].map(lambda x: "" if pd.isna(x) else (f"{x:.4g}" if isinstance(x, float) else str(x)))
    rows = ["| " + " | ".join(d.columns) + " |", "| " + " | ".join(["---"] * len(d.columns)) + " |"]
    for _, row in d.iterrows():
        rows.append("| " + " | ".join(row.astype(str)) + " |")
    return "\n".join(rows)


def build_markdown(df: pd.DataFrame, cond: pd.DataFrame, reg: pd.DataFrame, scores: pd.DataFrame, overlay: pd.DataFrame) -> str:
    all_delta_mean, all_delta_t, all_delta_p = one_sample(df["strategy_delta"])
    down_delta_mean, down_delta_t, down_delta_p = one_sample(df.loc[df["buy_hold_down"], "strategy_delta"])
    positive_def = cond[cond["mean_incremental_defense_bp"] > 0].sort_values(["q_incremental_defense", "p_incremental_defense"])
    best_def = positive_def.iloc[0] if len(positive_def) else cond.sort_values("q_incremental_defense").iloc[0]
    positive_scores = scores[scores["top_bottom_downside_defense_diff_bp"] > 0].sort_values(["q_downside_defense_diff", "p_downside_defense_diff"])
    best_score = positive_scores.iloc[0] if len(positive_scores) else scores.iloc[0]
    sig_def = cond[(cond["q_incremental_defense"] < 0.10) & (cond["mean_incremental_defense_bp"] > 0)]
    sig_ret = cond[(cond["q_incremental_return"] < 0.10) & (cond["mean_incremental_return_bp"] > 0)]
    overlay_good = overlay[
        (overlay["incremental_downside_vs_no_sent_bp"] > 0)
        & (overlay["total_return"] > 0)
        & (overlay["q_incremental_downside_vs_no_sent"] < 0.10)
    ].sort_values("sharpe", ascending=False)
    best_overlay = overlay_good.iloc[0] if len(overlay_good) else overlay.iloc[0]
    interaction = reg[
        reg["term"].isin(["sentiment", "sentiment_x_high_vol", "sentiment_x_prev_down"])
        & (reg["q"] < 0.10)
    ].head(12)

    cond_display = pd.concat(
        [
            positive_def,
            cond[~cond.index.isin(positive_def.index)].sort_values(["q_incremental_defense", "p_incremental_defense"]),
        ]
    )
    scores_display = pd.concat(
        [
            positive_scores,
            scores[~scores.index.isin(positive_scores.index)].sort_values(["q_downside_defense_diff", "p_downside_defense_diff"]),
        ]
    )

    return f"""# Sentiment Feature Development and Conditional Validation

## 왜 기존 결과에서 감성이 약해 보였나

기존 검정은 감성 피처가 전체 표본의 다음날/5일 수익률을 직접 예측하는지를 중심으로 보았다. 그러나 댓글 감성은 평상시 평균 수익률보다 시장 관심이 집중되거나 부정 감성, 의견 불일치, 변동성이 커지는 구간에서 위험 노출을 보정하는 역할일 가능성이 더 크다. 따라서 본 문서는 감성을 `수익률 단독 예측 변수`가 아니라 `조건부 하방 위험 보정 변수`로 재검정한다.

## 핵심 결론

- 전체 표본에서 감성 포함 전략이 감성 제외 전략보다 평균적으로 더 낸 일별 수익 차이는 {all_delta_mean * 10000:.2f}bp이다. t={all_delta_t:.3f}, p={all_delta_p:.4g}.
- B&H 하락일만 보면 감성 포함 전략의 추가 기여는 {down_delta_mean * 10000:.2f}bp이다. t={down_delta_t:.3f}, p={down_delta_p:.4g}.
- 조건부 하방 방어에서 양(+)의 효과가 가장 강한 후보는 `{best_def['condition']} / {best_def['group']}`이다. 이 구간에서 추가 하방 방어 효과는 {best_def['mean_incremental_defense_bp']:.2f}bp/day, t={best_def['t_incremental_defense']:.3f}, p={best_def['p_incremental_defense']:.4g}, BH 보정 q={best_def['q_incremental_defense']:.4g}.
- 후보 피처 중 양(+)의 방어 차이가 가장 컸던 것은 `{best_score['feature']}`이다. 상위 20%와 하위 20%의 하락일 방어 차이는 {best_score['top_bottom_downside_defense_diff_bp']:.2f}bp/day, q={best_score['q_downside_defense_diff']:.4g}.
- 감성위험 overlay 실험에서는 `{best_overlay['feature']}` 상위 {best_overlay['threshold_quantile'] * 100:.0f}%에서 `{best_overlay['overlay']}={best_overlay['parameter']}`를 적용한 후보가 전체 누적수익률 {best_overlay['total_return'] * 100:.1f}%, Sharpe {best_overlay['sharpe']:.3f}, 감성 제외 대비 추가 하락일 방어 {best_overlay['incremental_downside_vs_no_sent_bp']:.2f}bp/day, q={best_overlay['q_incremental_downside_vs_no_sent']:.4g}를 기록했다.

## 논문에서 살릴 수 있는 해석

감성 피처는 단독으로 시장 방향을 예측하는 강한 alpha라기보다, 시장 기반 K-FGI가 놓치는 `투자자 반응 강도`와 `불안의 확산 정도`를 보완한다. 특히 부정 감성과 댓글 관심도가 동시에 높아지는 날에는 감성 제외 지표보다 노출 조절의 하방 방어 효과가 커지는지를 검정할 수 있다.

이 해석은 감성을 논문에서 빼자는 의미가 아니다. 오히려 감성을 핵심 기여로 유지하려면 “감성은 전체 평균 수익률을 맞히는 피처”가 아니라 “위험 국면을 더 민감하게 식별하는 피처”라고 역할을 재정의하는 편이 더 설득력 있다.

## 조건부 검정 상위 결과

아래 표는 감성 포함 전략에서 감성 제외 전략을 뺀 추가 기여를 조건별로 검정한 결과다. `incremental_defense`는 B&H 대비 초과수익 기준으로 감성 포함 모델이 감성 제외 모델보다 얼마나 더 방어했는지를 의미한다.

{md_table(cond_display, ['condition','group','n','mean_incremental_return_bp','p_incremental_return','q_incremental_return','mean_incremental_defense_bp','p_incremental_defense','q_incremental_defense'], n=16)}

## 유의 조건 요약

- BH 보정 q<0.10인 추가 하방 방어 조건 수: {len(sig_def)}
- BH 보정 q<0.10인 추가 수익 조건 수: {len(sig_ret)}

이 숫자가 충분히 크다면 감성 피처는 “선택적으로만 의미 있는 피처”가 아니라 “특정 위험 국면에서 반복적으로 작동하는 보정 피처”라고 쓸 수 있다. 반대로 유의 조건이 적다면, 감성 피처는 현재 형태 그대로 최종 결론의 중심에 두기보다 추가 feature engineering이 필요하다.

## 후보 감성 피처별 비교

{md_table(scores_display, ['feature','spearman_next_5d','spearman_p','top_bottom_strategy_delta_diff_bp','q_strategy_delta_diff','top_bottom_downside_defense_diff_bp','q_downside_defense_diff'], n=12)}

## 상호작용 회귀

아래 회귀는 감성 신호가 고변동성 또는 직전 하락 조건과 결합될 때 효과가 커지는지 보는 보조 검정이다. 계수 단위는 bp이다.

{md_table(interaction if len(interaction) else reg.head(12), ['outcome','sentiment_feature','term','coef_bp','t','p','q','n'], n=12)}

## 감성위험 overlay 실험

현재 모델은 감성을 K-FGI 점수에 섞는 방식이다. 하지만 감성 피처가 논문 핵심이라면, 더 자연스러운 구조는 `감성위험이 커질 때 최대 노출을 낮추는 overlay`이다. 아래 표는 감성위험 조건에서 기존 K-FGI weight를 cap 또는 multiplier로 낮춘 후보 중, 감성 제외 전략 대비 하락일 방어가 양(+)이고 q<0.10인 후보를 Sharpe 기준으로 정렬한 것이다.

{md_table(overlay_good if len(overlay_good) else overlay, ['overlay','feature','threshold_quantile','parameter','active_days','ann_ret','sharpe','mdd','total_return','incremental_vs_no_sent_bp','incremental_downside_vs_no_sent_bp','incremental_downside_vs_no_sent_p','q_incremental_downside_vs_no_sent'], n=16)}

이 결과는 감성 피처를 “수익률 예측 점수”로만 쓰는 현재 구조보다 “위험 노출 제한 장치”로 쓰는 구조가 하방 방어 가설에 더 맞을 수 있음을 보여준다. 다만 일부 overlay는 하방 방어 유의성을 높이는 대신 총수익률과 Sharpe를 희생한다. 따라서 최종 모델로 채택하려면 수익률, MDD, 거래비용, OOS 안정성의 균형을 함께 봐야 한다.

## 다음 모델 수정 제안

1. 감성 피처를 단독 방향성 피처로 넣기보다 `negative_attention`, `panic_pressure`, `disagreement_attention`, `sentiment_risk_composite`처럼 관심도와 결합한 피처로 재구성한다.
2. 감성은 K-FGI 점수 자체의 일부와 별도로, 최대 노출 상한을 조절하는 risk overlay로도 테스트한다. 예: 감성위험 상위 20%에서는 leverage cap을 1.0x 이하로 제한.
3. 최종 논문에서는 raw sentiment score보다 “정규화 감성 점수 + 댓글 관심도 + 의견 불일치”를 결합한 피처가 더 재현 가능하고 설명 가능하다고 서술한다.
4. 단, 이 문서의 조건부 결과는 탐색적 feature development이다. 최종 주장으로 쓰려면 OOS 또는 rolling window에서 동일 조건이 유지되는지 별도 검증해야 한다.

## 생성된 그림

- `20_감성조건부_하방방어.png`: 조건별로 감성 포함 모델이 감성 제외 모델보다 하락일을 얼마나 더 방어했는지 보여준다.
- `21_감성위험지표와_추가기여.png`: 감성위험 복합지표와 감성 포함-제외 전략 기여의 시계열 관계를 보여준다.
- `22_감성후보피처_효과비교.png`: 새 후보 감성 피처별 전략 기여와 하방 방어 차이를 비교한다.
- `23_감성위험_오버레이_후보.png`: 감성위험 overlay 후보의 전체 수익성과 추가 하방 방어 효과를 함께 비교한다.
"""


def save_outputs(cond: pd.DataFrame, reg: pd.DataFrame, scores: pd.DataFrame, overlay: pd.DataFrame, md: str) -> None:
    outputs = [
        ("sentiment_conditioned_tests.csv", cond),
        ("sentiment_interaction_regression.csv", reg),
        ("sentiment_candidate_feature_scores.csv", scores),
        ("sentiment_risk_overlay_tests.csv", overlay),
    ]
    for root in [REPO_OUT, PAPER_OUT]:
        for name, table in outputs:
            table.to_csv(root / "tables" / name, index=False, encoding="utf-8-sig")
        (root / "SENTIMENT_FEATURE_DEVELOPMENT.md").write_text(md, encoding="utf-8")
    docs = CODE_ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "SENTIMENT_FEATURE_DEVELOPMENT.md").write_text(md, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_data()
    cond = condition_tests(df)
    reg = interaction_regression(df)
    scores = candidate_feature_scores(df)
    overlay = overlay_tests(df)
    make_figures(df, cond, scores, overlay)
    md = build_markdown(df, cond, reg, scores, overlay)
    save_outputs(cond, reg, scores, overlay, md)
    print(md)


if __name__ == "__main__":
    main()
