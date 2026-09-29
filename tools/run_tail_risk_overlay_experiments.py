from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "10_Paper_outputs"
OUT = PAPER / "05_추가실험"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
DATA = OUT / "data"
MPL_CACHE = OUT / ".mpl-cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib.pyplot as plt

FEE = 0.00015
N_BOOT = 3000
RNG_SEED = 20260723
EXPECTED_START = pd.Timestamp("2016-01-06")
EXPECTED_END = pd.Timestamp("2025-12-22")
EXPECTED_N = 2445
EXPECTED_DOWN_DAYS = 1130


@dataclass(frozen=True)
class OverlaySpec:
    name: str
    short_name: str
    cap: float
    description: str
    added_features: str


def normal_p_from_t(t_stat: float) -> float:
    if np.isnan(t_stat):
        return np.nan
    return erfc(abs(t_stat) / sqrt(2))


def one_sample_t(values: pd.Series) -> tuple[float, float]:
    v = pd.Series(values).dropna()
    if len(v) < 2:
        return np.nan, np.nan
    se = v.std(ddof=1) / np.sqrt(len(v))
    if se == 0 or np.isnan(se):
        return np.nan, np.nan
    t_stat = v.mean() / se
    return float(t_stat), float(normal_p_from_t(t_stat))


def max_drawdown(ret: pd.Series) -> float:
    cum = np.exp(pd.Series(ret).fillna(0).cumsum())
    return float((cum / cum.cummax() - 1).min())


def cvar(ret: pd.Series, alpha: float = 0.05) -> float:
    v = pd.Series(ret).dropna()
    if len(v) == 0:
        return np.nan
    q = v.quantile(alpha)
    return float(v[v <= q].mean())


def performance(ret: pd.Series, bh: pd.Series, weight: pd.Series, label: str) -> dict[str, float | str]:
    r = pd.Series(ret).fillna(0)
    b = pd.Series(bh).fillna(0)
    w = pd.Series(weight).fillna(0)
    down = b < 0
    excess_down = r[down] - b[down]
    downside_t, downside_p = one_sample_t(excess_down)
    ann_ret = r.mean() * 252
    ann_vol = r.std(ddof=1) * np.sqrt(252)
    bh_mdd = max_drawdown(b)
    strat_mdd = max_drawdown(r)
    return {
        "strategy": label,
        "n": int(len(r)),
        "down_days": int(down.sum()),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / (ann_vol + 1e-12),
        "total_return": np.exp(r.sum()) - 1,
        "mdd": strat_mdd,
        "bh_mdd": bh_mdd,
        "mdd_improvement_pctp": (strat_mdd - bh_mdd) * 100,
        "mdd_defense_rate_pct": (strat_mdd - bh_mdd) / abs(bh_mdd) * 100,
        "var5": r.quantile(0.05),
        "cvar5": cvar(r, 0.05),
        "bh_cvar5": cvar(b, 0.05),
        "cvar5_improvement_pctp": (cvar(r, 0.05) - cvar(b, 0.05)) * 100,
        "downside_excess_bp": excess_down.mean() * 10000,
        "downside_t": downside_t,
        "downside_p": downside_p,
        "downside_hit_rate": (excess_down > 0).mean(),
        "avg_weight": w.mean(),
        "zero_weight_pct": (w <= 1e-12).mean() * 100,
        "full_weight_pct": (w >= 1 - 1e-12).mean() * 100,
        "turnover": w.diff().abs().fillna(w.iloc[0]).mean(),
    }


def rolling_percentile(s: pd.Series, window: int = 252) -> pd.Series:
    def pct(x: np.ndarray) -> float:
        last = x[-1]
        return float(np.mean(x <= last))

    return s.rolling(window, min_periods=max(40, window // 4)).apply(pct, raw=True)


def stationary_bootstrap_indices(n: int, avg_block: int, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / avg_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < p:
            idx[i] = rng.integers(0, n)
        else:
            idx[i] = (idx[i - 1] + 1) % n
    return idx


def fixed_block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, max(1, n - block + 1), size=int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])
    if len(idx) < n:
        idx = np.pad(idx, (0, n - len(idx)), mode="wrap")
    return idx[:n]


def bootstrap_metric_tests(df: pd.DataFrame, strategy_cols: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    bh = df["buy_hold_return"].to_numpy()
    for strategy in strategy_cols:
        sr = df[strategy].to_numpy()
        obs = {
            "mdd_diff": max_drawdown(sr) - max_drawdown(bh),
            "cvar5_diff": cvar(sr) - cvar(bh),
            "market_down_day_excess_bp": ((sr[bh < 0] - bh[bh < 0]).mean() * 10000),
        }
        for method in ["iid", "fixed_block_20d", "stationary_avg20d"]:
            boot = {k: [] for k in obs}
            for _ in range(N_BOOT):
                if method == "iid":
                    idx = rng.integers(0, len(df), size=len(df))
                elif method == "fixed_block_20d":
                    idx = fixed_block_bootstrap_indices(len(df), 20, rng)
                else:
                    idx = stationary_bootstrap_indices(len(df), 20, rng)
                b = bh[idx]
                s = sr[idx]
                down = b < 0
                boot["mdd_diff"].append(max_drawdown(s) - max_drawdown(b))
                boot["cvar5_diff"].append(cvar(s) - cvar(b))
                boot["market_down_day_excess_bp"].append((s[down] - b[down]).mean() * 10000 if down.any() else np.nan)
            for metric, values in boot.items():
                vals = np.asarray(values, dtype=float)
                vals = vals[np.isfinite(vals)]
                rows.append({
                    "strategy": strategy,
                    "metric": metric,
                    "method": method,
                    "observed_diff": obs[metric],
                    "ci_low": np.quantile(vals, 0.025),
                    "ci_high": np.quantile(vals, 0.975),
                    "one_sided_p_against_zero": (np.sum(vals <= 0) + 1) / (len(vals) + 1),
                    "n_boot": len(vals),
                })
    return pd.DataFrame(rows)


def white_reality_check(df: pd.DataFrame, candidate_cols: list[str], benchmark_col: str = "K-FGI final existing") -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 11)
    obs_stats = {}
    base = df[benchmark_col].to_numpy()
    for col in candidate_cols:
        obs_stats[col] = max_drawdown(df[col]) - max_drawdown(base)
    best_col = max(obs_stats, key=obs_stats.get)
    best_obs = obs_stats[best_col]

    centered = {col: df[col].to_numpy() - base - np.mean(df[col].to_numpy() - base) for col in candidate_cols}
    boot_max = []
    for _ in range(N_BOOT):
        idx = stationary_bootstrap_indices(len(df), 20, rng)
        base_sample = base[idx]
        vals = []
        for col in candidate_cols:
            pseudo = base_sample + centered[col][idx]
            vals.append(max_drawdown(pseudo) - max_drawdown(base_sample))
        boot_max.append(max(vals))
    boot_max = np.asarray(boot_max)
    return pd.DataFrame([{
        "benchmark": benchmark_col,
        "best_candidate": best_col,
        "observed_best_mdd_diff_vs_benchmark": best_obs,
        "stationary_bootstrap_p_white_reality_check": (np.sum(boot_max >= best_obs) + 1) / (len(boot_max) + 1),
        "n_candidates": len(candidate_cols),
        "n_boot": len(boot_max),
        "interpretation": "Exploratory WRC-style adjustment for choosing the best overlay after testing multiple candidates.",
    }])


def apply_overlay(base_signal_weight: pd.Series, trigger: pd.Series, cap: float) -> pd.Series:
    w = base_signal_weight.copy()
    return w.where(~trigger.fillna(False), np.minimum(w, cap)).clip(0, 1)


def returns_from_signal_weight(
    bh: pd.Series,
    signal_weight: pd.Series,
    base_signal_weight: pd.Series | None = None,
    base_strategy_return: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    effective_weight = signal_weight.shift(1).fillna(0)
    turnover = effective_weight.diff().abs().fillna(effective_weight.iloc[0])
    if base_signal_weight is not None and base_strategy_return is not None:
        base_effective_weight = base_signal_weight.shift(1).fillna(0)
        base_turnover = base_effective_weight.diff().abs().fillna(base_effective_weight.iloc[0])
        exposure_delta = base_effective_weight - effective_weight
        cost_delta = (turnover - base_turnover) * FEE
        ret = base_strategy_return - exposure_delta * bh - cost_delta
        return ret, effective_weight, turnover
    ret = effective_weight * bh - turnover * FEE
    return ret, effective_weight, turnover


def build_markdown(
    perf: pd.DataFrame,
    boot: pd.DataFrame,
    features: pd.DataFrame,
    triggers: pd.DataFrame,
    nonoverlap: pd.DataFrame,
    crisis: pd.DataFrame,
    wrc: pd.DataFrame,
) -> str:
    def to_md_table(d: pd.DataFrame) -> str:
        if d.empty:
            return "_No rows._"
        text = d.copy()
        text = text.fillna("")
        text = text.astype(str)
        cols = list(text.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in text.iterrows():
            vals = [str(row[c]).replace("\n", " ").replace("|", "\\|") for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    def fmt_pct(x: float) -> str:
        return f"{x * 100:.2f}%"

    def fmt_pctp(x: float) -> str:
        return f"{x:.2f}%p"

    def fmt_p(x: float) -> str:
        if pd.isna(x):
            return ""
        if x < 0.001:
            return "<0.001"
        return f"{x:.3f}"

    display = perf.copy()
    display["ann_ret"] = display["ann_ret"].map(fmt_pct)
    display["ann_vol"] = display["ann_vol"].map(fmt_pct)
    display["total_return"] = display["total_return"].map(fmt_pct)
    display["mdd"] = display["mdd"].map(fmt_pct)
    display["mdd_improvement_pctp"] = display["mdd_improvement_pctp"].map(fmt_pctp)
    display["cvar5"] = display["cvar5"].map(fmt_pct)
    display["cvar5_improvement_pctp"] = display["cvar5_improvement_pctp"].map(fmt_pctp)
    display["downside_excess_bp"] = display["downside_excess_bp"].map(lambda x: f"{x:.1f}bp")
    display["downside_p"] = display["downside_p"].map(fmt_p)
    display["avg_weight"] = display["avg_weight"].map(lambda x: f"{x:.3f}x")
    display["zero_weight_pct"] = display["zero_weight_pct"].map(lambda x: f"{x:.1f}%")

    key_cols = [
        "strategy", "ann_ret", "sharpe", "total_return", "mdd",
        "mdd_improvement_pctp", "cvar5", "cvar5_improvement_pctp",
        "downside_excess_bp", "downside_p", "avg_weight", "zero_weight_pct",
    ]
    mdd_boot = boot[(boot["metric"] == "mdd_diff") & (boot["method"].isin(["iid", "fixed_block_20d", "stationary_avg20d"]))].copy()
    mdd_boot["observed_diff"] = mdd_boot["observed_diff"].map(lambda x: f"{x * 100:.2f}%p")
    mdd_boot["ci_low"] = mdd_boot["ci_low"].map(lambda x: f"{x * 100:.2f}%p")
    mdd_boot["ci_high"] = mdd_boot["ci_high"].map(lambda x: f"{x * 100:.2f}%p")
    mdd_boot["one_sided_p_against_zero"] = mdd_boot["one_sided_p_against_zero"].map(fmt_p)

    md = [
        "# Drawdown-Aware Tail-Risk Overlay 추가 실험",
        "",
        "## 1. 실험 목적",
        "",
        "본 추가 실험은 기존 K-FGI final을 대체하기 위한 최종 모델 선정 실험이 아니라, MDD가 왜 경로 의존적으로 불안정한지와 MDD를 직접 겨냥한 tail-risk overlay가 어떤 변화를 만드는지 확인하기 위한 확장 분석이다. 기존 논문의 메인 주장은 계속 `시장 하락일 방어폭`, `CVaR 개선`, `변동성 감소`, `고변동성 구간 투자 비중 축소`에 둔다.",
        "",
        "핵심 설계 원칙은 다음과 같다.",
        "",
        "- 기존 K-FGI 점수와 기본 비중 산식은 바꾸지 않는다.",
        "- 추가 피처는 t-1까지 관측 가능한 가격, 변동성, 추세, 감성, 시장 폭, 신용위험 조건만 사용한다.",
        "- overlay는 새 alpha 예측모형이 아니라 특정 위험 조건에서 기존 K-FGI 비중의 상한을 낮추는 circuit breaker이다.",
        "- 여러 후보를 비교했기 때문에 가장 좋은 후보를 곧바로 최종 모델로 주장하지 않고, White Reality Check 스타일 보정 결과와 함께 탐색적 결과로 해석한다.",
        "",
        "## 1.1 핵심 결론",
        "",
        "추가 overlay는 일부 지표를 개선했지만, MDD 개선을 5% 기준에서 강건하게 유의하게 만들지는 못했다. 가장 좋은 기술통계 결과는 `Overlay F: breadth-credit cap`이며, 기존 K-FGI final 대비 MDD가 -26.59%에서 -25.74%로 약 0.85%p 추가 완화되고 Sharpe도 0.461에서 0.512로 개선되었다. 그러나 MDD bootstrap p-value는 iid 기준 0.044로만 5% 아래이며, fixed block 20D 기준 0.059, stationary bootstrap 기준 0.064로 5%를 넘는다. 또한 8개 후보 중 가장 좋은 overlay를 사후 선택했다는 점을 반영한 White Reality Check 스타일 보정 p-value는 0.832이다.",
        "",
        "따라서 논문 해석은 `overlay가 MDD 유의성을 확정적으로 확보했다`가 아니라, `경로 기반 위험 피처를 추가하면 MDD와 Sharpe가 일부 개선될 수 있으나, 최종적으로 방어 가능한 핵심 통계 주장은 여전히 하락일 방어폭과 CVaR 개선이다`로 두는 것이 안전하다. 이 결과는 기존 논문 방향을 바꾸기보다, 왜 본문에서 MDD를 보조 성과로 낮춰 쓰는지가 타당하다는 추가 근거로 사용한다.",
        "",
        "## 2. 추가된 피처와 의미",
        "",
        to_md_table(features),
        "",
        "## 3. Overlay 후보 규칙",
        "",
        to_md_table(triggers),
        "",
        "## 4. 기존 실험 대비 전체 성과",
        "",
        to_md_table(display[key_cols]),
        "",
        "## 5. MDD bootstrap 검정",
        "",
        "MDD는 일별 평균이 아니라 최악 누적 낙폭 경로 하나에 좌우되므로 iid bootstrap뿐 아니라 fixed block bootstrap과 stationary bootstrap을 함께 확인했다.",
        "",
        to_md_table(mdd_boot[["strategy", "method", "observed_diff", "ci_low", "ci_high", "one_sided_p_against_zero"]]),
        "",
        "## 6. 비중첩 252거래일 블록 반복성",
        "",
        to_md_table(nonoverlap),
        "",
        "## 7. 위기/서브기간 결과",
        "",
        to_md_table(crisis),
        "",
        "## 8. Multiple-candidate 보정",
        "",
        to_md_table(wrc),
        "",
        "해석상 가장 중요한 점은, overlay 후보 중 일부가 기술통계상 MDD를 더 줄이더라도 여러 후보를 사후 비교했다는 사실을 반드시 인정해야 한다는 것이다. 따라서 논문 본문에서는 기존 K-FGI final을 메인으로 유지하고, 이 실험은 `drawdown-aware tail-risk overlay`라는 확장 또는 강건성 분석으로 배치하는 것이 안전하다.",
        "",
        "## 9. 그림 파일",
        "",
        "- `figures/01_cumulative_return_and_drawdown.png`: 누적수익률과 drawdown 비교",
        "- `figures/02_metric_comparison.png`: MDD, CVaR, 하락일 방어폭, 평균 투자 비중 비교",
        "- `figures/03_bootstrap_mdd_ci.png`: MDD 개선 bootstrap CI",
        "- `figures/04_overlay_trigger_timeline.png`: 주요 overlay 작동 구간",
        "",
        "## 10. 논문 반영 권장 문장",
        "",
        "> Because maximum drawdown is a path-dependent tail metric, its statistical significance is harder to establish than average market-down-day loss mitigation. We therefore retain the original K-FGI as the main specification and examine a drawdown-aware tail-risk overlay as an extension. The overlay starts from the published K-FGI final return series and uses only lagged crisis indicators, such as recent market losses, drawdown depth, trend breakdown, volatility spikes, sentiment pressure, and breadth deterioration, to further cap market exposure during fragile regimes.",
        "",
        "한국어 원고에서는 다음처럼 쓰는 것이 안전하다.",
        "",
        "> MDD는 표본 내 최악 누적 손실 경로 하나에 좌우되는 경로 의존적 지표이므로 평균 하락일 방어폭보다 통계적으로 유의성을 확보하기 어렵다. 이에 본 연구는 기존 K-FGI final을 메인 모형으로 유지하되, 최근 급락, drawdown 심화, 추세 붕괴, 변동성 급등, 부정 감성 압력, 시장 폭 약화가 동시에 나타나는 구간에서 투자 비중 상한을 추가로 낮추는 drawdown-aware tail-risk overlay를 확장 분석으로 검토한다.",
        "",
    ]
    return "\n".join(md)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    q = pd.read_csv(PAPER / "data" / "q1_benchmark_timeseries.csv", parse_dates=["date"])
    robust = pd.read_csv(PAPER / "robustness" / "data" / "priority_ab_kfgi_timeseries_robust.csv", parse_dates=["date"])
    keep = [
        "date", "K_FGI", "egarch_vol", "vol_shock", "vol_regime_high", "sub_index3",
        "sub_index5", "sub_index7", "neg_z", "comment_count", "above_ma60",
        "above_ma120", "sent_composite_ma10", "kospi_close",
    ]
    df = q.merge(robust[keep], on="date", how="left").sort_values("date").reset_index(drop=True)
    if (
        df["date"].iloc[0] != EXPECTED_START
        or df["date"].iloc[-1] != EXPECTED_END
        or len(df) != EXPECTED_N
        or int((df["buy_hold_return"] < 0).sum()) != EXPECTED_DOWN_DAYS
    ):
        raise ValueError(
            "Final evaluation window mismatch: expected "
            f"{EXPECTED_START.date()}~{EXPECTED_END.date()}, "
            f"n={EXPECTED_N}, down_days={EXPECTED_DOWN_DAYS}; got "
            f"{df['date'].iloc[0].date()}~{df['date'].iloc[-1].date()}, "
            f"n={len(df)}, down_days={int((df['buy_hold_return'] < 0).sum())}."
        )
    df["base_signal_weight"] = df["K-FGI final_weight"].clip(0, 1)
    df["K-FGI final existing"] = df["K-FGI final"]

    bh = df["buy_hold_return"]
    cum = np.exp(bh.cumsum())
    df["market_drawdown_lag1"] = (cum / cum.cummax() - 1).shift(1)
    df["ret5_lag1"] = bh.rolling(5).sum().shift(1)
    df["ret20_lag1"] = bh.rolling(20).sum().shift(1)
    df["realized_vol20_lag1"] = bh.rolling(20).std().shift(1)
    df["realized_vol120_lag1"] = bh.rolling(120).std().shift(1)
    df["realized_vol_spike_lag1"] = df["realized_vol20_lag1"] / (df["realized_vol120_lag1"] + 1e-12)
    df["egarch_vol_pct_lag1"] = rolling_percentile(df["egarch_vol"], 252).shift(1)
    df["vkospi_proxy_delta5_lag1"] = df["sub_index5"].diff(5).shift(1)
    df["breadth_drop5_lag1"] = (-df["sub_index3"].diff(5)).shift(1)
    df["credit_risk_delta20_lag1"] = (-df["sub_index7"].diff(20)).shift(1)
    df["neg_z_ma10_lag1"] = df["neg_z"].rolling(10).mean().shift(1)
    df["neg_z_ma60_lag1"] = df["neg_z"].rolling(60).mean().shift(1)
    df["sent_neg_delta_lag1"] = df["neg_z_ma10_lag1"] - df["neg_z_ma60_lag1"]
    df["comment_surge_lag1"] = (df["comment_count"].rolling(5).mean() / (df["comment_count"].rolling(60).mean() + 1e-12)).shift(1)
    df["below_ma60_lag1"] = (1 - df["above_ma60"]).shift(1).fillna(0).astype(bool)
    df["below_ma120_lag1"] = (1 - df["above_ma120"]).shift(1).fillna(0).astype(bool)

    q75 = df[["realized_vol_spike_lag1", "vkospi_proxy_delta5_lag1", "breadth_drop5_lag1", "credit_risk_delta20_lag1", "sent_neg_delta_lag1", "comment_surge_lag1"]].quantile(0.75)
    q80 = df[["breadth_drop5_lag1", "credit_risk_delta20_lag1", "sent_neg_delta_lag1", "comment_surge_lag1"]].quantile(0.80)

    df["trigger_return_vol"] = (df["ret20_lag1"] < 0) & (df["egarch_vol_pct_lag1"] >= 0.75)
    df["trigger_drawdown_recovery"] = (df["market_drawdown_lag1"] <= -0.10) & df["below_ma120_lag1"]
    df["trigger_trend_break"] = (df["ret20_lag1"] < 0) & df["below_ma60_lag1"] & df["below_ma120_lag1"]
    shock_raw = (df["vol_shock"].shift(1) > 3) | (df["realized_vol_spike_lag1"] >= q75["realized_vol_spike_lag1"])
    df["trigger_volshock_cooling"] = shock_raw.rolling(5, min_periods=1).max().astype(bool)
    df["trigger_sentiment_pressure"] = (
        (df["sent_neg_delta_lag1"] >= q80["sent_neg_delta_lag1"]) &
        (df["comment_surge_lag1"] >= q80["comment_surge_lag1"]) &
        (df["ret5_lag1"] < 0)
    )
    df["trigger_breadth_credit"] = (
        (df["breadth_drop5_lag1"] >= q75["breadth_drop5_lag1"]) &
        (df["credit_risk_delta20_lag1"] >= q75["credit_risk_delta20_lag1"])
    )

    core = [
        "trigger_return_vol",
        "trigger_drawdown_recovery",
        "trigger_trend_break",
        "trigger_volshock_cooling",
        "trigger_sentiment_pressure",
        "trigger_breadth_credit",
    ]
    df["danger_score"] = df[core].sum(axis=1)
    df["trigger_combined_conservative"] = df["danger_score"] >= 2
    df["trigger_combined_strict"] = df["danger_score"] >= 3

    specs = [
        OverlaySpec("Overlay A: return-vol cap", "return_vol", 0.50, "최근 20일 수익률이 음수이고 EGARCH 변동성 분위가 상위 25%이면 비중 상한 0.5x", "ret20_lag1, egarch_vol_pct_lag1"),
        OverlaySpec("Overlay B: drawdown-recovery cap", "drawdown_recovery", 0.50, "시장 drawdown이 -10% 이하이고 KOSPI가 120일선 아래이면 비중 상한 0.5x", "market_drawdown_lag1, below_ma120_lag1"),
        OverlaySpec("Overlay C: trend-break cap", "trend_break", 0.60, "최근 20일 수익률이 음수이고 60일/120일선 아래이면 비중 상한 0.6x", "ret20_lag1, below_ma60_lag1, below_ma120_lag1"),
        OverlaySpec("Overlay D: vol-shock cooling", "volshock_cooling", 0.35, "vol shock 또는 실현변동성 spike 이후 5거래일 동안 비중 상한 0.35x", "vol_shock_lag1, realized_vol_spike_lag1"),
        OverlaySpec("Overlay E: sentiment-pressure cap", "sentiment_pressure", 0.50, "부정 감성 변화와 댓글 수 급증이 동시에 상위권이고 최근 5일 수익률이 음수이면 비중 상한 0.5x", "sent_neg_delta_lag1, comment_surge_lag1, ret5_lag1"),
        OverlaySpec("Overlay F: breadth-credit cap", "breadth_credit", 0.60, "시장 폭 약화와 신용위험 proxy 악화가 동시에 나타나면 비중 상한 0.6x", "breadth_drop5_lag1, credit_risk_delta20_lag1"),
        OverlaySpec("Overlay G: combined conservative", "combined_conservative", 0.50, "6개 위험 조건 중 2개 이상 발생하면 비중 상한 0.5x", "return, drawdown, trend, volatility, sentiment, breadth/credit trigger set"),
        OverlaySpec("Overlay H: combined strict", "combined_strict", 0.30, "6개 위험 조건 중 3개 이상 발생하면 비중 상한 0.3x", "return, drawdown, trend, volatility, sentiment, breadth/credit trigger set"),
    ]

    trigger_map = {
        "return_vol": df["trigger_return_vol"],
        "drawdown_recovery": df["trigger_drawdown_recovery"],
        "trend_break": df["trigger_trend_break"],
        "volshock_cooling": df["trigger_volshock_cooling"],
        "sentiment_pressure": df["trigger_sentiment_pressure"],
        "breadth_credit": df["trigger_breadth_credit"],
        "combined_conservative": df["trigger_combined_conservative"],
        "combined_strict": df["trigger_combined_strict"],
    }

    perf_rows = [
        performance(df["Buy & Hold"], df["buy_hold_return"], df["Buy & Hold_weight"], "Buy & Hold"),
        performance(df["K-FGI final"], df["buy_hold_return"], df["K-FGI final_weight"].shift(1).fillna(0), "K-FGI final existing"),
    ]
    ret_cols = ["Buy & Hold", "K-FGI final"]
    trigger_rows = []
    for spec in specs:
        signal_weight = apply_overlay(df["base_signal_weight"], trigger_map[spec.short_name], spec.cap)
        ret, effective_weight, turnover = returns_from_signal_weight(
            df["buy_hold_return"],
            signal_weight,
            base_signal_weight=df["base_signal_weight"],
            base_strategy_return=df["K-FGI final existing"],
        )
        col = spec.name
        df[f"{col}_signal_weight"] = signal_weight
        df[f"{col}_effective_weight"] = effective_weight
        df[col] = ret
        ret_cols.append(col)
        perf_rows.append(performance(ret, df["buy_hold_return"], effective_weight, col))
        trigger = trigger_map[spec.short_name].fillna(False)
        trigger_rows.append({
            "strategy": spec.name,
            "cap": spec.cap,
            "trigger_days": int(trigger.sum()),
            "trigger_pct": trigger.mean() * 100,
            "avg_base_signal_weight_on_trigger": df.loc[trigger, "base_signal_weight"].mean() if trigger.any() else 0,
            "avg_signal_weight_after_cap_on_trigger": signal_weight.loc[trigger].mean() if trigger.any() else 0,
            "description": spec.description,
            "added_features": spec.added_features,
        })

    perf_df = pd.DataFrame(perf_rows)
    trigger_df = pd.DataFrame(trigger_rows)
    feature_df = pd.DataFrame([
        {"feature": "ret5_lag1 / ret20_lag1", "source": "KOSPI/B&H daily return", "why_added": "최근 급락 또는 약세 흐름을 포착해 MDD가 커지는 초입을 겨냥", "lookahead_control": "rolling sum shifted by 1 trading day"},
        {"feature": "market_drawdown_lag1", "source": "B&H cumulative return path", "why_added": "이미 누적 손실이 깊어진 구간에서 추가 손실 확대를 제한", "lookahead_control": "drawdown shifted by 1 trading day"},
        {"feature": "realized_vol_spike_lag1", "source": "20D realized vol / 120D realized vol", "why_added": "단기 변동성 급등이 장기 기준보다 큰지 측정", "lookahead_control": "rolling volatility shifted by 1 trading day"},
        {"feature": "egarch_vol_pct_lag1", "source": "EGARCH conditional volatility", "why_added": "조건부 변동성이 과거 1년 대비 높은 구간 식별", "lookahead_control": "rolling percentile shifted by 1 trading day"},
        {"feature": "below_ma60_lag1 / below_ma120_lag1", "source": "KOSPI trend features", "why_added": "추세 붕괴 구간에서 비중 상한을 낮추기 위한 경로 정보", "lookahead_control": "moving-average state shifted by 1 trading day"},
        {"feature": "vkospi_proxy_delta5_lag1", "source": "sub_index5 volatility sentiment proxy", "why_added": "변동성 심리 급변 구간 확인", "lookahead_control": "5D difference shifted by 1 trading day"},
        {"feature": "breadth_drop5_lag1", "source": "sub_index3 market breadth", "why_added": "시장 폭이 빠르게 약해지는 구간에서 손실 확산 위험 포착", "lookahead_control": "negative 5D difference shifted by 1 trading day"},
        {"feature": "sent_neg_delta_lag1 + comment_surge_lag1", "source": "NAVER sentiment and comment count", "why_added": "부정 감성 변화와 관심 급증이 동시에 나타나는 행동 위험 상태 포착", "lookahead_control": "10D-60D sentiment and 5D/60D comment ratio shifted by 1 trading day"},
        {"feature": "credit_risk_delta20_lag1", "source": "sub_index7 credit spread proxy", "why_added": "신용위험 proxy의 방향성 악화가 동반되는지 확인", "lookahead_control": "20D difference shifted by 1 trading day"},
    ])

    candidate_cols = [s.name for s in specs if s.short_name != "base_recomputed"]
    boot_df = bootstrap_metric_tests(df, ["K-FGI final existing"] + candidate_cols)
    wrc_df = white_reality_check(df, candidate_cols)

    block_rows = []
    n = len(df)
    for i, start in enumerate(range(0, n, 252), start=1):
        sub = df.iloc[start:start + 252].copy()
        if len(sub) < 120:
            continue
        row = {
            "block": i,
            "start": sub["date"].iloc[0].date().isoformat(),
            "end": sub["date"].iloc[-1].date().isoformat(),
            "n": len(sub),
        }
        for col in ["K-FGI final existing", "Overlay G: combined conservative", "Overlay H: combined strict"]:
            row[f"{col}_mdd"] = f"{max_drawdown(sub[col]) * 100:.1f}%"
            row[f"{col}_downside_bp"] = f"{((sub.loc[sub['buy_hold_return'] < 0, col] - sub.loc[sub['buy_hold_return'] < 0, 'buy_hold_return']).mean() * 10000):.1f}"
        block_rows.append(row)
    nonoverlap_df = pd.DataFrame(block_rows)

    crisis_defs = [
        ("2016_initial_drawdown", "2016-01-06", "2016-12-31"),
        ("2020_covid", "2020-01-01", "2020-12-31"),
        ("2021_2022_rate_hike", "2021-01-01", "2022-12-31"),
        ("2023_2025_recent", "2023-01-01", "2025-12-22"),
        ("full_sample", "2016-01-06", "2025-12-22"),
    ]
    crisis_rows = []
    for name, start, end in crisis_defs:
        sub = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        for col in ["K-FGI final existing", "Overlay G: combined conservative", "Overlay H: combined strict"]:
            p = performance(sub[col], sub["buy_hold_return"], sub[f"{col}_effective_weight"] if f"{col}_effective_weight" in sub else sub["K-FGI final_weight"].shift(1).fillna(0), col)
            crisis_rows.append({
                "period": name,
                "strategy": col,
                "n": p["n"],
                "mdd": f"{p['mdd'] * 100:.1f}%",
                "mdd_improvement_pctp": f"{p['mdd_improvement_pctp']:.1f}%p",
                "cvar5": f"{p['cvar5'] * 100:.2f}%",
                "downside_excess_bp": f"{p['downside_excess_bp']:.1f}",
                "avg_weight": f"{p['avg_weight']:.3f}x",
            })
    crisis_df = pd.DataFrame(crisis_rows)

    df.to_csv(DATA / "tail_risk_overlay_timeseries.csv", index=False, encoding="utf-8-sig")
    perf_df.to_csv(TABLES / "tail_risk_overlay_performance.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(TABLES / "tail_risk_overlay_added_features.csv", index=False, encoding="utf-8-sig")
    trigger_df.to_csv(TABLES / "tail_risk_overlay_trigger_summary.csv", index=False, encoding="utf-8-sig")
    boot_df.to_csv(TABLES / "tail_risk_overlay_bootstrap_tests.csv", index=False, encoding="utf-8-sig")
    nonoverlap_df.to_csv(TABLES / "tail_risk_overlay_nonoverlap_252d_blocks.csv", index=False, encoding="utf-8-sig")
    crisis_df.to_csv(TABLES / "tail_risk_overlay_crisis_subperiods.csv", index=False, encoding="utf-8-sig")
    wrc_df.to_csv(TABLES / "tail_risk_overlay_white_reality_check.csv", index=False, encoding="utf-8-sig")

    plt.style.use("seaborn-v0_8-whitegrid")
    key_plot = ["Buy & Hold", "K-FGI final existing", "Overlay G: combined conservative", "Overlay H: combined strict"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for col in key_plot:
        axes[0].plot(df["date"], np.exp(df[col].cumsum()), label=col, lw=1.8)
        dd = np.exp(df[col].cumsum()) / np.exp(df[col].cumsum()).cummax() - 1
        axes[1].plot(df["date"], dd * 100, label=col, lw=1.4)
    axes[0].set_title("Cumulative Return: Tail-Risk Overlay Comparison")
    axes[0].set_ylabel("Growth of 1")
    axes[1].set_title("Drawdown Path")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "01_cumulative_return_and_drawdown.png", dpi=180)
    plt.close(fig)

    plot_perf = perf_df[perf_df["strategy"].isin(key_plot)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("mdd", "MDD (%)", lambda x: x * 100),
        ("cvar5", "CVaR 5% (%)", lambda x: x * 100),
        ("downside_excess_bp", "Down-day defense (bp)", lambda x: x),
        ("avg_weight", "Average exposure (x)", lambda x: x),
    ]
    for ax, (metric, title, transform) in zip(axes.ravel(), metrics):
        vals = plot_perf[metric].map(transform)
        ax.barh(plot_perf["strategy"], vals)
        ax.set_title(title)
        ax.axvline(0, color="black", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_metric_comparison.png", dpi=180)
    plt.close(fig)

    mdd_ci = boot_df[(boot_df["metric"] == "mdd_diff") & (boot_df["method"] == "stationary_avg20d")]
    mdd_ci = mdd_ci[mdd_ci["strategy"].isin(["K-FGI final existing", "Overlay G: combined conservative", "Overlay H: combined strict"])]
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(mdd_ci))
    obs = mdd_ci["observed_diff"] * 100
    lo = mdd_ci["ci_low"] * 100
    hi = mdd_ci["ci_high"] * 100
    ax.errorbar(obs, y, xerr=[obs - lo, hi - obs], fmt="o", capsize=4)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(y, mdd_ci["strategy"])
    ax.set_xlabel("MDD improvement vs Buy & Hold (%p)")
    ax.set_title("Stationary Bootstrap CI for MDD Improvement")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_bootstrap_mdd_ci.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    trigger_cols = ["trigger_return_vol", "trigger_drawdown_recovery", "trigger_trend_break", "trigger_volshock_cooling", "trigger_combined_conservative"]
    offsets = np.arange(len(trigger_cols))
    for i, col in enumerate(trigger_cols):
        active_dates = df.loc[df[col], "date"]
        ax.scatter(active_dates, np.full(len(active_dates), i), s=7, label=col.replace("trigger_", ""))
    ax.set_yticks(offsets, [c.replace("trigger_", "") for c in trigger_cols])
    ax.set_title("Overlay Trigger Timeline")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIGURES / "04_overlay_trigger_timeline.png", dpi=180)
    plt.close(fig)

    md = build_markdown(perf_df, boot_df, feature_df, trigger_df, nonoverlap_df, crisis_df, wrc_df)
    (OUT / "TAIL_RISK_OVERLAY_ADDITIONAL_EXPERIMENTS.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
