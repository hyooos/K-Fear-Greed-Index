from __future__ import annotations

import math
import unicodedata
from pathlib import Path

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


def perf(r: pd.Series, ref: pd.Series | None = None) -> dict:
    r = r.dropna()
    ann_ret = float(r.mean() * TRADING_DAYS)
    ann_vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    sharpe = ann_ret / (ann_vol + 1e-12)
    cum = np.exp(r.cumsum())
    dd = cum / cum.cummax() - 1
    out = {
        "n": len(r),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "mdd": float(dd.min()),
        "total_return": float(cum.iloc[-1] - 1),
        "positive_day_rate": float((r > 0).mean()),
    }
    if ref is not None:
        a = pd.concat([r.rename("r"), ref.rename("b")], axis=1).dropna()
        ex = a["r"] - a["b"]
        out["mean_excess_bp"] = float(ex.mean() * 10000)
        out["excess_p"] = float(stats.ttest_1samp(ex, 0).pvalue)
        down = a["b"] < 0
        if down.any():
            dex = ex[down]
            out["downside_excess_bp"] = float(dex.mean() * 10000)
            out["downside_p"] = float(stats.ttest_1samp(dex, 0).pvalue)
    return out


def rank01(s: pd.Series) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).rank(pct=True).fillna(0.5)


def make_return(df: pd.DataFrame, weight: pd.Series, fee: float = 0.0015) -> pd.Series:
    weight_lag = weight.shift(1).fillna(0)
    turnover = weight_lag.diff().abs().fillna(weight_lag.iloc[0])
    return weight_lag * df["target_reg"] - turnover * fee


def add_event_stress(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    neg_sent = (-d["sent_composite_ma10"]).clip(lower=0)
    weak_trend = 3 - d["trend_strength"]
    high_vol = d["vol_ratio"].clip(lower=0)
    shock = d["vol_shock"].clip(lower=0)
    fear_kfgi = 100 - d["K_FGI"]
    weak_breadth = 100 - d["sub_index3"]
    event_parts = [
        rank01(high_vol),
        rank01(shock),
        rank01(neg_sent),
        rank01(weak_trend),
        rank01(fear_kfgi),
        rank01(weak_breadth),
    ]
    d["event_stress_proxy"] = pd.concat(event_parts, axis=1).mean(axis=1)
    d["early_unstable_period"] = (d["date"] < pd.Timestamp("2017-01-01")).astype(float)
    return d


def main() -> None:
    df = pd.read_csv(REPO_OUT / "robustness" / "data" / "priority_ab_kfgi_timeseries_robust.csv", parse_dates=["date"])
    df = add_event_stress(df)
    ref = df["target_reg"]
    base_weight = df["weight"]
    rows = []
    candidates = []

    candidates.append(("baseline_current", base_weight, "current weight"))
    candidates.append(("no_leverage_cap_1_0", base_weight.clip(0, 1.0), "global cap 1.0x"))
    candidates.append(("cap_1_2", base_weight.clip(0, 1.2), "global cap 1.2x"))

    for q in [0.70, 0.75, 0.80]:
        thr = df["event_stress_proxy"].quantile(q)
        for mult in [0.4, 0.5, 0.6, 0.7]:
            w = base_weight.where(df["event_stress_proxy"] < thr, base_weight * mult)
            candidates.append((f"event_stress_q{int(q*100)}_mult{mult}", w, f"event stress >= q{q:.2f}, weight x {mult}"))
        for cap in [0.5, 0.7, 1.0]:
            w = base_weight.mask(df["event_stress_proxy"] >= thr, base_weight.clip(0, cap))
            candidates.append((f"event_stress_q{int(q*100)}_cap{cap}", w, f"event stress >= q{q:.2f}, cap {cap}x"))

    # Early instability overlay is intentionally simple and observable by time:
    # first full evaluation year uses lower cap to avoid over-leverage during warm-up.
    for cap in [0.5, 0.7, 1.0]:
        w = base_weight.mask(df["early_unstable_period"].eq(1), base_weight.clip(0, cap))
        candidates.append((f"early_2016_cap{cap}", w, f"date < 2017, cap {cap}x"))

    for label, w, rule in candidates:
        r = make_return(df, w)
        row = {"label": label, "rule": rule, **perf(r, ref)}
        for period, start, end in [
            ("2016", "2016-01-01", "2016-12-31"),
            ("2017", "2017-01-01", "2017-12-31"),
            ("2020_COVID", "2020-02-01", "2020-12-31"),
            ("2022_bear", "2022-01-01", "2022-12-31"),
            ("2025_bull", "2025-01-01", "2025-12-31"),
        ]:
            mask = df["date"].between(start, end)
            p = perf(r[mask], ref[mask])
            row[f"{period}_total_return"] = p["total_return"]
            row[f"{period}_mdd"] = p["mdd"]
            row[f"{period}_sharpe"] = p["sharpe"]
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["sharpe", "mdd"], ascending=[False, False])
    for root in [REPO_OUT, PAPER_OUT]:
        (root / "tables").mkdir(parents=True, exist_ok=True)
        out.to_csv(root / "tables" / "event_regime_overlay_experiments.csv", index=False, encoding="utf-8-sig")
        df[["date", "event_stress_proxy", "K_FGI", "sent_composite_ma10", "vol_ratio", "vol_shock", "trend_strength", "sub_index3", "weight"]].to_csv(
            root / "data" / "event_stress_proxy_timeseries.csv", index=False, encoding="utf-8-sig"
        )
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
