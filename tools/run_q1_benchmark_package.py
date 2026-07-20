from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path.cwd()
OUT = ROOT / "paper_outputs"
TABLE = OUT / "tables"
DATA = OUT / "data"
ROBUST = OUT / "robustness" / "data"

FEE_BP = 15.0
FEE_RATE = FEE_BP / 10000
TRADING_DAYS = 252


def drawdown(r: pd.Series) -> pd.Series:
    wealth = np.exp(r.fillna(0).cumsum())
    return wealth / wealth.cummax() - 1


def perf(label: str, r: pd.Series, bh: pd.Series, weight: pd.Series) -> dict:
    r = pd.Series(r).fillna(0)
    bh = pd.Series(bh).fillna(0)
    weight = pd.Series(weight).fillna(0)
    ann_ret = float(np.exp(r.mean() * TRADING_DAYS) - 1)
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else np.nan
    mdd = float(drawdown(r).min())
    bh_mdd = float(drawdown(bh).min())
    down = bh < 0
    diff = r[down] - bh[down]
    t, p = stats.ttest_1samp(diff.dropna(), 0, alternative="greater")
    return {
        "strategy": label,
        "n": int(len(r)),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "total_return": float(np.exp(r.sum()) - 1),
        "mdd": mdd,
        "bh_mdd": bh_mdd,
        "mdd_improvement_pctp": (abs(bh_mdd) - abs(mdd)) * 100,
        "mdd_defense_rate_pct": ((abs(bh_mdd) - abs(mdd)) / abs(bh_mdd) * 100) if bh_mdd else np.nan,
        "downside_excess_bp": float(diff.mean() * 10000),
        "downside_t": float(t),
        "downside_p": float(p),
        "downside_hit_rate": float((diff > 0).mean()),
        "avg_weight": float(weight.mean()),
        "max_weight": float(weight.max()),
        "turnover": float(weight.diff().abs().fillna(0).mean()),
    }


def strategy_return(ret: pd.Series, raw_weight: pd.Series) -> tuple[pd.Series, pd.Series]:
    weight = raw_weight.clip(0, 1).shift(1).fillna(0)
    turnover = weight.diff().abs().fillna(0)
    cost = turnover * FEE_RATE
    return weight * ret - cost, weight


def kfgi_weight(score: pd.Series, fear: float = 25, greed: float = 65, k_min: float = 0.5, k_max: float = 1.6) -> pd.Series:
    raw = k_min + (score / 100.0) * (k_max - k_min)
    raw = raw.where(score >= fear, raw * 0.25)
    raw = raw.where(score < greed, raw * 0.75)
    return raw.clip(0, 1)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(format(float(val), floatfmt))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load() -> pd.DataFrame:
    robust = pd.read_csv(ROBUST / "priority_ab_kfgi_timeseries_robust.csv", parse_dates=["date"])
    ts = pd.read_csv(DATA / "conservative_kfgi_position_caps_timeseries.csv", parse_dates=["date"])
    final = ts[ts["variant"].eq("lean_sent_composite_ma10__cap_1_0_no_leverage")][
        ["date", "K_FGI", "weight", "strategy_return", "buy_hold_return"]
    ].copy()
    merged = final.merge(robust, on="date", how="left", suffixes=("", "_robust"))
    return merged.sort_values("date").reset_index(drop=True)


def main() -> None:
    TABLE.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    df = load()
    ret = df["buy_hold_return"].fillna(df["target_reg"]).fillna(0)

    strategies: dict[str, tuple[pd.Series, pd.Series]] = {}
    strategies["Buy & Hold"] = (ret, pd.Series(1.0, index=df.index))
    strategies["K-FGI final"] = (df["strategy_return"].fillna(0), df["weight"].fillna(0))

    ma_weight = (df["ma20"] > df["ma60"]).astype(float)
    strategies["MA 20/60 timing"] = strategy_return(ret, ma_weight)

    rv = ret.rolling(60, min_periods=20).std()
    target_daily_vol = 0.10 / np.sqrt(TRADING_DAYS)
    rv_weight = (target_daily_vol / (rv + 1e-9)).clip(0, 1)
    strategies["Realized-vol targeting"] = strategy_return(ret, rv_weight)

    egarch_weight = (target_daily_vol / (df["egarch_vol"] + 1e-9)).clip(0, 1)
    strategies["EGARCH-only vol targeting"] = strategy_return(ret, egarch_weight)

    vk_weight = (df["sub_index5"] / 100.0).clip(0, 1)
    strategies["VKOSPI/sub_index5 only"] = strategy_return(ret, vk_weight)

    market_score = df[[f"sub_index{i}" for i in range(2, 8)]].mean(axis=1)
    strategies["Market-only FGI"] = strategy_return(ret, kfgi_weight(market_score))

    sent_score = df[["sent_composite_ma10"]].rank(pct=True)["sent_composite_ma10"] * 100
    strategies["Sentiment-only index"] = strategy_return(ret, kfgi_weight(sent_score))

    rows = []
    ts_rows = pd.DataFrame({"date": df["date"], "buy_hold_return": ret})
    for label, (r, w) in strategies.items():
        rows.append(perf(label, r, ret, w))
        ts_rows[label] = pd.Series(r).to_numpy()
        ts_rows[f"{label}_weight"] = pd.Series(w).to_numpy()

    bench = pd.DataFrame(rows).sort_values(["mdd_defense_rate_pct", "downside_excess_bp"], ascending=False)
    bench.to_csv(TABLE / "q1_benchmark_performance.csv", index=False, encoding="utf-8-sig")
    ts_rows.to_csv(DATA / "q1_benchmark_timeseries.csv", index=False, encoding="utf-8-sig")

    yearly_rows = []
    ts_rows["year"] = pd.to_datetime(ts_rows["date"]).dt.year
    for year, g in ts_rows.groupby("year"):
        bh = g["buy_hold_return"]
        for label in strategies:
            yearly_rows.append(
                {
                    "year": int(year),
                    **perf(label, g[label], bh, g[f"{label}_weight"] if f"{label}_weight" in g else pd.Series(1, index=g.index)),
                }
            )
    pd.DataFrame(yearly_rows).to_csv(TABLE / "q1_benchmark_yearly_performance.csv", index=False, encoding="utf-8-sig")

    md = "# Q1 보강용 벤치마크 성과표\n\n"
    md += "동일 기간, 동일 거래비용 15bp, 1.0x 상한 기준으로 B&H 외 벤치마크를 추가했다.\n\n"
    md += markdown_table(bench[
        [
            "strategy",
            "total_return",
            "ann_vol",
            "sharpe",
            "mdd",
            "mdd_defense_rate_pct",
            "downside_excess_bp",
            "downside_p",
            "avg_weight",
        ]
    ])
    md += "\n\n## 해석\n\n"
    top = bench.iloc[0]
    kfgi = bench[bench["strategy"].eq("K-FGI final")].iloc[0]
    md += (
        f"- 최종 K-FGI의 MDD 방어율은 {kfgi['mdd_defense_rate_pct']:.1f}%이며, "
        f"시장 하락일 방어폭은 {kfgi['downside_excess_bp']:.1f}bp/day이다.\n"
        "- B&H 단독 비교가 아니라 MA timing, realized-vol targeting, EGARCH-only, "
        "VKOSPI/sub_index5-only, market-only FGI, sentiment-only index와 비교했다.\n"
        f"- MDD 방어율 기준 최상위 전략은 `{top['strategy']}`이며, "
        "최종 논문에서는 K-FGI가 단순 변동성/추세 규칙 대비 어떤 장단점을 갖는지 함께 서술한다.\n"
    )
    (OUT / "Q1_BENCHMARK_RESULTS.md").write_text(md, encoding="utf-8")
    print(TABLE / "q1_benchmark_performance.csv")
    print(OUT / "Q1_BENCHMARK_RESULTS.md")


if __name__ == "__main__":
    main()
