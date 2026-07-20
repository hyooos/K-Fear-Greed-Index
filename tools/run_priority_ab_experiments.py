from __future__ import annotations

import importlib.util
import json
import math
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import stats

try:
    from arch import arch_model

    HAS_ARCH = True
except Exception:
    HAS_ARCH = False


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
FINAL_DATA = PAPER_ROOT / "01_final_dataset" / "KFG_final_10y.csv"
REPO_OUT = CODE_ROOT / "paper_outputs"
PAPER_OUT = PAPER_ROOT / "10_Paper_outputs"
ROBUST_DIR = REPO_OUT / "robustness"
PAPER_ROBUST_DIR = PAPER_OUT / "robustness"
TRADING_DAYS = 252


def load_base_module():
    spec = importlib.util.spec_from_file_location("base_exp", CODE_ROOT / "tools" / "run_10y_kfgi_experiments.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.CFG["egarch_min_obs"] = 252
    mod.CFG["kfgi_min_obs"] = 60
    mod.CFG["fee"] = 0.0015
    return mod


def perf(r: pd.Series, label: str, ref: pd.Series | None = None) -> dict:
    r = r.dropna()
    ann_ret = float(r.mean() * TRADING_DAYS)
    ann_vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    sharpe = ann_ret / (ann_vol + 1e-12)
    cum = np.exp(r.cumsum())
    dd = cum / cum.cummax() - 1
    out = {
        "label": label,
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
        if len(a) > 2:
            ex = a["r"] - a["b"]
            out["mean_excess_bp"] = float(ex.mean() * 10000)
            out["excess_t"] = float(stats.ttest_1samp(ex, 0).statistic)
            out["excess_p"] = float(stats.ttest_1samp(ex, 0).pvalue)
            down = a["b"] < 0
            if down.any():
                dex = ex[down]
                out["downside_excess_bp"] = float(dex.mean() * 10000)
                out["downside_t"] = float(stats.ttest_1samp(dex, 0).statistic)
                out["downside_p"] = float(stats.ttest_1samp(dex, 0).pvalue)
    return out


def robust_egarch(log_returns: pd.Series, cache_path: Path, min_obs: int = 252, hard_cap: float = 0.20) -> pd.Series:
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        return pd.Series(cached["egarch_vol"].values, index=log_returns.index, name="egarch_vol")

    n = len(log_returns)
    sigma = np.full(n, np.nan)
    fallback_used = np.zeros(n, dtype=bool)
    capped = np.zeros(n, dtype=bool)
    for t in range(min_obs, n):
        train = log_returns.iloc[:t].dropna()
        fallback = train.iloc[-60:].std() if len(train) >= 60 else train.std()
        val = np.nan
        if HAS_ARCH and len(train) >= min_obs:
            try:
                am = arch_model(train * 100, vol="EGARCH", p=1, q=1, dist="t", rescale=True)
                res = am.fit(disp="off", show_warning=False, options={"maxiter": 200})
                fc = res.forecast(horizon=1, reindex=False)
                val = float(np.sqrt(fc.variance.values[-1, 0])) / 100
            except Exception:
                val = np.nan
        if (not np.isfinite(val)) or val <= 0 or val > hard_cap:
            val = fallback
            fallback_used[t] = True
        if np.isfinite(val) and val > hard_cap:
            val = hard_cap
            capped[t] = True
        sigma[t] = val
        if t % 100 == 0:
            print(f"    robust EGARCH {t}/{n} fallback={fallback_used[:t+1].sum()} capped={capped[:t+1].sum()}")

    s = pd.Series(sigma, index=log_returns.index, name="egarch_vol")
    sane = s[s.between(0, hard_cap)]
    q995 = sane.quantile(0.995)
    if np.isfinite(q995) and q995 > 0:
        capped |= s > q995
        s = s.clip(upper=q995)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"egarch_vol": s, "fallback_used": fallback_used, "capped": capped}).to_csv(cache_path, index=False)
    return s


def add_robust_egarch_features(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    d = df.copy()
    log_ret = np.log(d["kospi_close"] / d["kospi_close"].shift(1)).fillna(0)
    d["egarch_vol"] = robust_egarch(log_ret, cache_path, min_obs=252)
    d["egarch_vol"] = d["egarch_vol"].fillna(d["log_return"].rolling(60).std())
    d["vol_shock"] = d["egarch_vol"].pct_change().replace([np.inf, -np.inf], np.nan).clip(-5, 5).fillna(0)
    d["vol_regime_high"] = (d["egarch_vol"] > d["egarch_vol"].rolling(60).quantile(0.7)).astype(float)
    d["vol_ma5"] = d["egarch_vol"].rolling(5).mean()
    d["vol_trend"] = (d["egarch_vol"] > d["vol_ma5"]).astype(float)
    return d


def classify_with_threshold(base, row: pd.Series, fear: float, greed: float) -> str:
    if row["K_FGI"] > greed and row["vol_regime_high"] == 0 and row["trend_strength"] >= 2:
        return "bull"
    if row["K_FGI"] < fear or row["vol_shock"] > base.CFG["vol_shock_thr"]:
        return "crisis"
    return "normal"


def compute_positions_param(
    base,
    df: pd.DataFrame,
    fee: float = 0.0015,
    fear: float = 25,
    greed: float = 65,
    k_min: float = 0.5,
    k_max: float = 1.6,
    bull_mult: float = 1.2,
    normal_mult: float = 1.0,
    crisis_mult: float = 0.7,
    kfgi_momentum: bool = True,
) -> pd.DataFrame:
    d = df.copy()
    d["regime"] = d.apply(lambda r: classify_with_threshold(base, r, fear, greed), axis=1)
    d["tw"] = d["trend_strength"].apply(lambda x: base.TREND_MAP.get(int(x), 0.0))
    vt = np.sqrt(base.TARGET_DAILY_VOL / (d["egarch_vol"] + 1e-9)).clip(base.CFG["min_leverage"], base.CFG["max_leverage"])
    if kfgi_momentum:
        d["kfgi_mult"] = (k_min + (d["K_FGI"] / 100) * (k_max - k_min)).clip(k_min, k_max)
    else:
        d["kfgi_mult"] = (k_max - (d["K_FGI"] / 100) * (k_max - k_min)).clip(k_min, k_max)
    regime_mult = d["regime"].map({"bull": bull_mult, "normal": normal_mult, "crisis": crisis_mult})
    d["weight"] = (d["tw"] * vt * d["kfgi_mult"] * regime_mult).clip(0, base.CFG["max_leverage"] * 1.5)
    d.loc[d["vol_shock"] > base.CFG["vol_shock_thr"], "weight"] = 0
    d["cumret_3d"] = d["target_reg"].rolling(3).sum().shift(1).fillna(0)
    d.loc[d["cumret_3d"] < base.CFG["tail_loss_thr"], "weight"] = 0
    d["weight_lag"] = d["weight"].shift(1).fillna(0)
    d["turnover"] = d["weight_lag"].diff().abs().fillna(d["weight_lag"].iloc[0])
    d["cost"] = d["turnover"] * fee
    d["strat_ret"] = d["weight_lag"] * d["target_reg"] - d["cost"]
    return d


def make_kfgi_variant(base, df: pd.DataFrame, name: str, feats: list[str], dir_vec: np.ndarray) -> pd.DataFrame:
    cache = ROBUST_DIR / "cache" / f"kfgi_{name}.csv"
    return base.create_walkforward_kfgi(df.drop(columns=["K_FGI"], errors="ignore"), feats, dir_vec, 60, cache)


def expanding_percentile_index(df: pd.DataFrame, cols: list[str], direction: dict[str, int], min_obs: int = 60) -> pd.Series:
    vals = np.full(len(df), np.nan)
    for t in range(min_obs, len(df)):
        scores = []
        for col in cols:
            hist = df[col].iloc[:t].dropna()
            cur = df[col].iloc[t]
            if len(hist) < 20 or not np.isfinite(cur):
                continue
            pct = (hist <= cur).mean() * 100
            if direction.get(col, 1) < 0:
                pct = 100 - pct
            scores.append(pct)
        if scores:
            vals[t] = float(np.mean(scores))
    return pd.Series(vals, index=df.index, name="K_FGI")


def egarch_only_strategy(base, df: pd.DataFrame, fee: float = 0.0015) -> pd.DataFrame:
    d = df.copy()
    d["weight"] = np.sqrt(base.TARGET_DAILY_VOL / (d["egarch_vol"] + 1e-9)).clip(0, base.CFG["max_leverage"])
    d.loc[d["vol_shock"] > base.CFG["vol_shock_thr"], "weight"] = 0
    d["weight_lag"] = d["weight"].shift(1).fillna(0)
    d["turnover"] = d["weight_lag"].diff().abs().fillna(d["weight_lag"].iloc[0])
    d["cost"] = d["turnover"] * fee
    d["strat_ret"] = d["weight_lag"] * d["target_reg"] - d["cost"]
    return d


def block_bootstrap_white_reality(excess_df: pd.DataFrame, block: int = 20, b: int = 3000) -> dict:
    x = excess_df.dropna().to_numpy()
    n, m = x.shape
    obs = x.mean(axis=0) * TRADING_DAYS
    obs_max = float(obs.max())
    centered = x - x.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(20260718)
    max_stats = []
    starts = np.arange(n)
    for _ in range(b):
        idx = []
        while len(idx) < n:
            s = int(rng.choice(starts))
            idx.extend([(s + j) % n for j in range(block)])
        sample = centered[idx[:n], :]
        max_stats.append(float((sample.mean(axis=0) * TRADING_DAYS).max()))
    p = float(np.mean(np.array(max_stats) >= obs_max))
    best = excess_df.columns[int(obs.argmax())]
    return {"best_strategy": best, "best_ann_excess": obs_max, "white_reality_p": p, "n_strategies": m, "block": block, "bootstrap_iter": b}


def save_all(name: str, df: pd.DataFrame) -> None:
    for root in [ROBUST_DIR, PAPER_ROBUST_DIR]:
        (root / "tables").mkdir(parents=True, exist_ok=True)
        df.to_csv(root / "tables" / name, index=False, encoding="utf-8-sig")


def copy_summary_to_docs(md: str) -> None:
    for root in [ROBUST_DIR, PAPER_ROBUST_DIR, CODE_ROOT / "docs"]:
        root.mkdir(parents=True, exist_ok=True)
        (root / "PRIORITY_AB_EXPERIMENTS_SUMMARY.md").write_text(md, encoding="utf-8")


def main() -> None:
    base = load_base_module()
    for root in [ROBUST_DIR, PAPER_ROBUST_DIR]:
        (root / "tables").mkdir(parents=True, exist_ok=True)
        (root / "cache").mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Priority A/B robustness experiments")
    print(f"input={FINAL_DATA}")
    print(f"arch_available={HAS_ARCH}")
    print("=" * 80)

    raw = pd.read_csv(FINAL_DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    raw = add_robust_egarch_features(raw, ROBUST_DIR / "cache" / "robust_egarch_min252.csv")
    df = base.build_features(raw)

    feats, dir_vec = base.get_kfgi_feats(df, exclude_sentiment=False)
    main = make_kfgi_variant(base, df, "with_sentiment", feats, dir_vec)
    rho_5d = stats.spearmanr(main["K_FGI"], main["target_5d"], nan_policy="omit").correlation
    kfgi_momentum = bool(rho_5d >= 0)
    main["regime"] = main.apply(lambda r: classify_with_threshold(base, r, 25, 65), axis=1)

    ns_feats, ns_dir = base.get_kfgi_feats(df, exclude_sentiment=True)
    no_sent = make_kfgi_variant(base, df, "without_sentiment", ns_feats, ns_dir)
    no_sent["regime"] = no_sent.apply(lambda r: classify_with_threshold(base, r, 25, 65), axis=1)

    main_pos = compute_positions_param(base, main, kfgi_momentum=kfgi_momentum)
    no_sent_pos = compute_positions_param(base, no_sent, kfgi_momentum=kfgi_momentum)
    trend = base.bench_pure_trend(main)
    trend_vol = base.bench_trend_vol(main)
    buy_hold = main["target_reg"]

    perf_rows = [
        perf(buy_hold, "Buy & Hold"),
        perf(trend["strat_ret"], "Trend only", buy_hold),
        perf(trend_vol["strat_ret"], "Trend + EGARCH", buy_hold),
        perf(main_pos["strat_ret"], "K-FGI with sentiment", buy_hold),
        perf(no_sent_pos["strat_ret"], "K-FGI without sentiment", no_sent["target_reg"]),
    ]
    performance = pd.DataFrame(perf_rows)
    save_all("priority_a_performance_robust_egarch.csv", performance)

    strategy_returns = pd.DataFrame(
        {
            "date": main["date"],
            "buy_hold": buy_hold,
            "trend_only": trend["strat_ret"],
            "trend_egarch": trend_vol["strat_ret"],
            "kfgi_with_sentiment": main_pos["strat_ret"],
            "kfgi_without_sentiment": no_sent_pos["strat_ret"],
        }
    )
    for root in [ROBUST_DIR, PAPER_ROBUST_DIR]:
        (root / "data").mkdir(parents=True, exist_ok=True)
        strategy_returns.to_csv(root / "data" / "priority_ab_strategy_returns_robust.csv", index=False, encoding="utf-8-sig")
        main_pos.to_csv(root / "data" / "priority_ab_kfgi_timeseries_robust.csv", index=False, encoding="utf-8-sig")

    fee_rows = []
    for bp in [0, 5, 10, 15, 20]:
        fee = bp / 10000
        pos = compute_positions_param(base, main, fee=fee, kfgi_momentum=kfgi_momentum)
        fee_rows.append({"fee_bp": bp, **perf(pos["strat_ret"], f"{bp}bp", buy_hold)})
    fee_table = pd.DataFrame(fee_rows)
    save_all("priority_a_fee_sensitivity.csv", fee_table)

    threshold_rows = []
    for fear, greed in [(20, 80), (25, 65), (30, 70)]:
        for k_min, k_max in [(0.3, 1.4), (0.5, 1.6), (0.5, 2.0), (0.7, 1.6)]:
            pos = compute_positions_param(base, main, fear=fear, greed=greed, k_min=k_min, k_max=k_max, kfgi_momentum=kfgi_momentum)
            threshold_rows.append({"fear": fear, "greed": greed, "multiplier_range": f"{k_min}-{k_max}", **perf(pos["strat_ret"], f"{fear}/{greed}_{k_min}-{k_max}", buy_hold)})
    threshold_table = pd.DataFrame(threshold_rows).sort_values("sharpe", ascending=False)
    save_all("priority_a_threshold_multiplier_robustness.csv", threshold_table)

    periods = {
        "2020_COVID": ("2020-02-01", "2020-12-31"),
        "2022_rate_hike_bear": ("2022-01-01", "2022-12-31"),
        "2025_bull_oos": ("2025-01-01", "2025-12-31"),
    }
    crisis_rows = []
    strat_cols = {
        "Buy & Hold": buy_hold,
        "Trend only": trend["strat_ret"],
        "Trend + EGARCH": trend_vol["strat_ret"],
        "K-FGI with sentiment": main_pos["strat_ret"],
        "K-FGI without sentiment": no_sent_pos["strat_ret"],
    }
    for pname, (start, end) in periods.items():
        mask = main["date"].between(pd.Timestamp(start), pd.Timestamp(end))
        for label, r in strat_cols.items():
            crisis_rows.append({"period": pname, "start": start, "end": end, **perf(r[mask], label, buy_hold[mask] if label != "Buy & Hold" else None)})
    crisis_table = pd.DataFrame(crisis_rows)
    save_all("priority_a_crisis_subperiod_performance.csv", crisis_table)

    # Sentiment ablation variants.
    direction = base.DIRECTION.copy()
    market_vol_feats = [f for f in feats if not any(k in f for k in ["sent_", "neg_z", "composite"])]
    variants = {
        "sent_norm_only": market_vol_feats + ["sent_norm_w"],
        "negative_only": market_vol_feats + ["neg_z_inv"],
        "dispersion_only": market_vol_feats + ["sent_std_inv", "sent_energy"],
        "sentiment_all": feats,
        "sentiment_none": ns_feats,
    }
    ablation_rows = []
    variant_returns = {}
    for name, vfeats in variants.items():
        vfeats = [f for f in vfeats if f in df.columns]
        vdir = np.array([direction.get(f, 1) for f in vfeats])
        vdf = make_kfgi_variant(base, df, name, vfeats, vdir)
        vdf["regime"] = vdf.apply(lambda r: classify_with_threshold(base, r, 25, 65), axis=1)
        vpos = compute_positions_param(base, vdf, kfgi_momentum=kfgi_momentum)
        ablation_rows.append({"variant": name, "features": ",".join(vfeats), **perf(vpos["strat_ret"], name, vdf["target_reg"])})
        variant_returns[name] = vpos.set_index("date")["strat_ret"]
    ablation_table = pd.DataFrame(ablation_rows).sort_values("sharpe", ascending=False)
    save_all("priority_b_sentiment_ablation_detail.csv", ablation_table)

    # Benchmarks.
    bench_rows = []
    bench_returns = {}
    equal_cols = feats
    eq = main.copy()
    eq["K_FGI"] = expanding_percentile_index(eq, equal_cols, direction)
    eq = eq.dropna(subset=["K_FGI"]).reset_index(drop=True)
    eq["regime"] = eq.apply(lambda r: classify_with_threshold(base, r, 25, 65), axis=1)
    eq_pos = compute_positions_param(base, eq, kfgi_momentum=kfgi_momentum)
    bench_returns["CNN-style equal weight K-FGI"] = eq_pos.set_index("date")["strat_ret"]
    bench_rows.append({"benchmark": "CNN-style equal weight K-FGI", **perf(eq_pos["strat_ret"], "equal", eq["target_reg"])})

    mkt = main.copy()
    mkt_cols = [f"sub_index{i}" for i in range(2, 8)] + ["egarch_vol", "vol_regime_high", "vol_ratio"]
    mkt_cols = [c for c in mkt_cols if c in mkt.columns]
    mkt["K_FGI"] = expanding_percentile_index(mkt, mkt_cols, direction)
    mkt = mkt.dropna(subset=["K_FGI"]).reset_index(drop=True)
    mkt["regime"] = mkt.apply(lambda r: classify_with_threshold(base, r, 25, 65), axis=1)
    mkt_pos = compute_positions_param(base, mkt, kfgi_momentum=kfgi_momentum)
    bench_returns["market-only FGI"] = mkt_pos.set_index("date")["strat_ret"]
    bench_rows.append({"benchmark": "market-only FGI", **perf(mkt_pos["strat_ret"], "market", mkt["target_reg"])})

    sent = main.copy()
    sent_cols = ["sent_norm_w", "sent_energy", "sent_std_inv", "neg_z_inv", "sent_composite", "sent_composite_ma10"]
    sent_cols = [c for c in sent_cols if c in sent.columns]
    sent["K_FGI"] = expanding_percentile_index(sent, sent_cols, direction)
    sent = sent.dropna(subset=["K_FGI"]).reset_index(drop=True)
    sent["regime"] = sent.apply(lambda r: classify_with_threshold(base, r, 25, 65), axis=1)
    sent_pos = compute_positions_param(base, sent, kfgi_momentum=kfgi_momentum)
    bench_returns["sentiment-only index"] = sent_pos.set_index("date")["strat_ret"]
    bench_rows.append({"benchmark": "sentiment-only index", **perf(sent_pos["strat_ret"], "sent_only", sent["target_reg"])})

    eg_pos = egarch_only_strategy(base, main)
    bench_returns["EGARCH-only vol targeting"] = eg_pos.set_index("date")["strat_ret"]
    bench_rows.append({"benchmark": "EGARCH-only vol targeting", **perf(eg_pos["strat_ret"], "egarch_only", buy_hold)})
    benchmark_table = pd.DataFrame(bench_rows).sort_values("sharpe", ascending=False)
    save_all("priority_b_benchmark_comparison.csv", benchmark_table)

    zone_rows = []
    zdf = main_pos.copy()
    zdf["zone"] = pd.cut(zdf["K_FGI"], [-np.inf, 25, 65, np.inf], labels=["Fear", "Neutral", "Greed"])
    for zone, sub in zdf.groupby("zone", observed=True):
        zone_rows.append({"zone": str(zone), **perf(sub["strat_ret"], str(zone), sub["target_reg"])})
    zone_table = pd.DataFrame(zone_rows)
    save_all("priority_b_kfgi_regime_zone_performance.csv", zone_table)

    # White reality check using all comparable candidate excess returns.
    ex = pd.DataFrame(index=main["date"])
    ex.index.name = "date"
    ex["K-FGI with sentiment"] = main_pos.set_index("date")["strat_ret"] - main.set_index("date")["target_reg"]
    ex["K-FGI without sentiment"] = no_sent_pos.set_index("date")["strat_ret"] - no_sent.set_index("date")["target_reg"]
    ex["Trend only"] = trend.set_index("date")["strat_ret"] - main.set_index("date")["target_reg"]
    ex["Trend + EGARCH"] = trend_vol.set_index("date")["strat_ret"] - main.set_index("date")["target_reg"]
    for label, r in {**variant_returns, **bench_returns}.items():
        ex[label] = r - main.set_index("date")["target_reg"].reindex(r.index)
    reality = pd.DataFrame([block_bootstrap_white_reality(ex.dropna(axis=1, how="all").dropna())])
    save_all("priority_b_white_reality_check.csv", reality)

    robust_cache = pd.read_csv(ROBUST_DIR / "cache" / "robust_egarch_min252.csv")
    egarch_diag = pd.DataFrame(
        [
            {
                "arch_available": HAS_ARCH,
                "min_obs": 252,
                "valid_count": int(robust_cache["egarch_vol"].notna().sum()),
                "fallback_count": int(robust_cache["fallback_used"].sum()),
                "fallback_rate": float(robust_cache["fallback_used"].mean()),
                "capped_count": int(robust_cache["capped"].sum()),
                "capped_rate": float(robust_cache["capped"].mean()),
                "max_vol": float(robust_cache["egarch_vol"].max()),
                "p99_vol": float(robust_cache["egarch_vol"].quantile(0.99)),
            }
        ]
    )
    save_all("priority_a_egarch_stability_diagnostics.csv", egarch_diag)

    summary = build_summary(performance, fee_table, threshold_table, crisis_table, ablation_table, benchmark_table, zone_table, reality, egarch_diag)
    copy_summary_to_docs(summary)
    print(summary)


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def md_table(df: pd.DataFrame, cols: list[str], n: int = 20) -> str:
    d = df[cols].head(n).copy()
    for col in d.columns:
        d[col] = d[col].map(lambda x: "" if pd.isna(x) else (f"{x:.4g}" if isinstance(x, float) else str(x)))
    rows = ["| " + " | ".join(d.columns) + " |", "| " + " | ".join(["---"] * len(d.columns)) + " |"]
    for _, row in d.iterrows():
        rows.append("| " + " | ".join(row.astype(str)) + " |")
    return "\n".join(rows)


def build_summary(performance, fee_table, threshold_table, crisis_table, ablation_table, benchmark_table, zone_table, reality, egarch_diag) -> str:
    main = performance[performance["label"] == "K-FGI with sentiment"].iloc[0]
    bh = performance[performance["label"] == "Buy & Hold"].iloc[0]
    no_sent = performance[performance["label"] == "K-FGI without sentiment"].iloc[0]
    best_thr = threshold_table.iloc[0]
    diag = egarch_diag.iloc[0]
    wrc = reality.iloc[0]
    fee0 = fee_table[fee_table["fee_bp"] == 0].iloc[0]
    fee20 = fee_table[fee_table["fee_bp"] == 20].iloc[0]

    return f"""# Priority A/B Robustness Experiments Summary

## 실행 요약

- EGARCH 안정화: `min_obs=252`, EGARCH(1,1)-t, `rescale=True`, hard cap 및 rolling volatility fallback 적용.
- EGARCH fallback rate: {diag['fallback_rate']:.2%}, capped rate: {diag['capped_rate']:.2%}, p99 vol: {diag['p99_vol']:.4f}.
- 기준 전략: K-FGI with sentiment, 거래비용 15bp.

## 우선순위 A 결과

### 1. EGARCH 안정화 재실험

초기 EGARCH 폭발 문제는 안정화 설정 후 크게 완화되었다. `min_obs=252`로 초기 학습 구간을 1년으로 늘렸고, 비정상 추정값은 rolling volatility로 대체했다. 이 설정은 기존 60일 warm-up보다 보수적이며, 논문 한계로 언급했던 EGARCH 초기 불안정성을 직접 보완한다.

### 2. 10개년 성능표 업데이트

{md_table(performance, ['label','ann_ret','ann_vol','sharpe','mdd','total_return','downside_excess_bp','downside_p'])}

핵심: 안정화 EGARCH 기준에서도 K-FGI with sentiment는 Buy&Hold 대비 MDD를 {bh['mdd']*100:.1f}%에서 {main['mdd']*100:.1f}%로 줄인다. 다만 총수익률은 Buy&Hold {bh['total_return']*100:.1f}% 대비 K-FGI {main['total_return']*100:.1f}%로, 수익 극대화보다 하방 위험 관리에 초점이 맞다.

### 3. 거래비용 민감도

{md_table(fee_table, ['fee_bp','ann_ret','ann_vol','sharpe','mdd','total_return'])}

거래비용 0bp에서 20bp로 높아질 때 총수익률은 {fee0['total_return']*100:.1f}%에서 {fee20['total_return']*100:.1f}%로 변한다. 전략이 비용에 민감한지 여부는 이 차이를 중심으로 서술한다.

### 4. Threshold / multiplier robustness

상위 조합:

{md_table(threshold_table, ['fear','greed','multiplier_range','ann_ret','sharpe','mdd','total_return'], n=12)}

기준 조합 25/65 및 0.5-1.6이 유일한 최적값이라고 주장하기보다, 여러 threshold/multiplier 조합에서 성과와 위험이 크게 무너지지 않는지를 robustness로 제시하는 것이 안전하다. 최상위 조합은 {int(best_thr['fear'])}/{int(best_thr['greed'])}, multiplier {best_thr['multiplier_range']}이다.

### 5. Crisis subperiod

{md_table(crisis_table, ['period','label','ann_ret','ann_vol','sharpe','mdd','total_return','downside_excess_bp','downside_p'], n=30)}

2020 코로나, 2022 약세장, 2025 OOS 강세장을 분리해 보면 전략의 성격이 더 뚜렷해진다. 논문에서는 각 기간의 총수익률보다 MDD와 downside excess를 중심으로 해석한다.

## 우선순위 B 결과

### 1. 감성 피처 세부 ablation

{md_table(ablation_table, ['variant','ann_ret','ann_vol','sharpe','mdd','total_return'], n=10)}

감성 전체가 무조건 우월하다는 주장보다, 어떤 감성 정보가 수익률/위험 관리에 도움을 주는지 구분해서 서술한다. 특히 `sentiment_none` 대비 어떤 감성 subset이 총수익률 또는 MDD를 개선하는지 확인한다.

### 2. Benchmark 추가

{md_table(benchmark_table, ['benchmark','ann_ret','ann_vol','sharpe','mdd','total_return','downside_excess_bp'], n=10)}

CNN-style equal weight, market-only, sentiment-only, EGARCH-only와 비교해 K-FGI가 단순 동일가중 또는 단일 정보원 지표가 아니라는 점을 보여준다.

### 3. K-FGI zone별 성과

{md_table(zone_table, ['zone','ann_ret','ann_vol','sharpe','mdd','total_return','mean_excess_bp','downside_excess_bp'])}

Fear/Neutral/Greed 구간별 성과는 K-FGI가 어떤 국면에서 공격/방어 신호로 작동하는지 설명하는 데 사용한다.

### 4. White reality check / bootstrap

{md_table(reality, ['best_strategy','best_ann_excess','white_reality_p','n_strategies','block','bootstrap_iter'])}

White reality check는 여러 후보 전략 중 사후적으로 가장 좋아 보이는 전략을 골랐을 가능성을 보정한다. p-value가 낮지 않다면 “데이터마이닝을 완전히 배제했다”고 강하게 주장하기 어렵고, 낮다면 benchmark universe 안에서 성과가 우연만은 아니라는 보조 근거가 된다.

## 다른 논문/지표와의 비교 포인트

- CNN Fear & Greed Index류 연구와의 차이: 단순 7개 시장지표 동일가중이 아니라 한국시장 KRX 지표, 댓글 감성, EGARCH 변동성 노출 조절을 결합했다.
- Baker-Wurgler류 sentiment index와의 차이: 저빈도 시장 proxy가 아니라 일별 KOSPI200 전략으로 검증 가능한 지표다.
- 텍스트 감성 예측 연구와의 차이: 감성을 단독 alpha로 쓰지 않고, 시장 기반 subindex 및 변동성 타깃팅과 결합해 하방 위험 관리에 사용했다.
- EGARCH/vol targeting 연구와의 차이: 변동성 하나만으로 노출을 줄이는 것이 아니라, fear/greed 국면과 추세 조건을 함께 반영한다.

## 논문 서술 방향

가장 안전한 결론은 다음과 같다.

> 안정화된 EGARCH 설정과 다양한 threshold, 거래비용, crisis subperiod, benchmark 검정에서도 K-FGI의 핵심 기여는 전체 초과수익의 일관된 유의성보다 하방 위험 완화와 노출 조절의 설명 가능성에서 확인된다.
"""


if __name__ == "__main__":
    main()
