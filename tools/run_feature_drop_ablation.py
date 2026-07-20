from __future__ import annotations

import importlib.util
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
CODE_ROOT = ROOT if (ROOT / "tools" / "run_10y_kfgi_experiments.py").exists() else find_child(PROJECT_ROOT, "kfgi_최종")
try:
    PAPER_ROOT = find_child(PROJECT_ROOT, "논문용")
except FileNotFoundError:
    PAPER_ROOT = CODE_ROOT
REPO_OUT = CODE_ROOT / "paper_outputs"
PAPER_OUT = PAPER_ROOT / "10_Paper_outputs" if PAPER_ROOT != CODE_ROOT else REPO_OUT
ROBUST_DATA = REPO_OUT / "robustness" / "data" / "priority_ab_kfgi_timeseries_robust.csv"
ROBUST_RETURNS = REPO_OUT / "robustness" / "data" / "priority_ab_strategy_returns_robust.csv"


def load_base_module():
    spec = importlib.util.spec_from_file_location("base_exp", CODE_ROOT / "tools" / "run_10y_kfgi_experiments.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.CFG["egarch_min_obs"] = 252
    mod.CFG["kfgi_min_obs"] = 60
    mod.CFG["fee"] = 0.0015
    return mod


def load_priority_module():
    spec = importlib.util.spec_from_file_location("priority_exp", CODE_ROOT / "tools" / "run_priority_ab_experiments.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
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
        ex = a["r"] - a["b"]
        if len(ex) > 2:
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


def save_table(df: pd.DataFrame, rel_name: str) -> None:
    for root in [REPO_OUT, PAPER_OUT]:
        out = root / "tables" / rel_name
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")


def main() -> None:
    base = load_base_module()
    priority = load_priority_module()
    df = pd.read_csv(ROBUST_DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    rets = pd.read_csv(ROBUST_RETURNS, parse_dates=["date"])

    full_feats, full_dir = base.get_kfgi_feats(df, exclude_sentiment=False)
    full_direction = {f: int(d) for f, d in zip(full_feats, full_dir)}
    rho_5d = stats.spearmanr(df["K_FGI"], df["target_5d"], nan_policy="omit").correlation
    kfgi_momentum = bool(rho_5d >= 0)
    buy_hold = rets["buy_hold"]

    drop_groups = {
        "baseline_full": [],
        "drop_sent_norm_w": ["sent_norm_w"],
        "drop_sent_energy": ["sent_energy"],
        "drop_sent_std_inv": ["sent_std_inv"],
        "drop_neg_z_inv": ["neg_z_inv"],
        "drop_sent_composite": ["sent_composite"],
        "drop_sent_composite_ma10": ["sent_composite_ma10"],
        "drop_sent_composite_family": ["sent_composite", "sent_composite_ma10", "sent_composite_diff"],
        "drop_negative_family": ["neg_z", "neg_z_inv", "neg_z_ma5", "negative_attention", "panic_pressure"],
        "drop_disagreement_family": ["sent_std", "sent_std_inv", "disagreement_attention"],
        "drop_attention_family": ["effective_n", "comment_count", "heat", "sent_attention", "heat_attention"],
        "drop_raw_sentiment": ["sent_norm_w", "sent_strength_w"],
        "drop_all_sentiment": [
            "sent_norm_w",
            "sent_strength_w",
            "sent_std",
            "neg_z",
            "neg_z_inv",
            "sent_std_inv",
            "sent_energy",
            "sent_norm_ma5",
            "sent_norm_diff",
            "neg_z_ma5",
            "sent_composite",
            "sent_composite_ma10",
            "sent_composite_diff",
            "effective_n",
            "heat",
            "comment_count",
        ],
        "drop_egarch_family": ["egarch_vol", "vol_regime_high", "vol_ratio"],
        "drop_subindex_family": [f"sub_index{i}" for i in range(2, 8)],
    }
    keep_groups = {
        "lean_sent_composite_ma10_only": [f"sub_index{i}" for i in range(2, 8)]
        + ["sent_composite_ma10", "egarch_vol", "vol_regime_high", "vol_ratio"],
        "lean_sent_composite_pair": [f"sub_index{i}" for i in range(2, 8)]
        + ["sent_composite", "sent_composite_ma10", "egarch_vol", "vol_regime_high", "vol_ratio"],
        "lean_sent_core_four": [f"sub_index{i}" for i in range(2, 8)]
        + ["sent_norm_w", "sent_energy", "neg_z_inv", "sent_composite_ma10", "egarch_vol", "vol_regime_high", "vol_ratio"],
        "market_egarch_only": [f"sub_index{i}" for i in range(2, 8)] + ["egarch_vol", "vol_regime_high", "vol_ratio"],
    }

    rows = []
    feature_rows = []
    specs = [(name, None, drops) for name, drops in drop_groups.items()]
    specs += [(name, keep, []) for name, keep in keep_groups.items()]
    for name, keep, drops in specs:
        feats = [f for f in full_feats if f not in set(drops)] if keep is None else keep
        feats = [f for f in feats if f in df.columns]
        if not feats:
            continue
        dir_vec = np.array([full_direction.get(f, base.DIRECTION.get(f, 1)) for f in feats])
        cache = REPO_OUT / "robustness" / "cache" / f"kfgi_drop_{name}.csv"
        vdf = base.create_walkforward_kfgi(df.drop(columns=["K_FGI"], errors="ignore"), feats, dir_vec, 60, cache)
        vdf["regime"] = vdf.apply(lambda r: priority.classify_with_threshold(base, r, 25, 65), axis=1)
        vpos = priority.compute_positions_param(base, vdf, kfgi_momentum=kfgi_momentum)
        ref = vdf["target_reg"].reset_index(drop=True)
        p = perf(vpos["strat_ret"].reset_index(drop=True), name, ref)
        p["dropped_features"] = ",".join([f for f in drops if f in df.columns])
        p["n_kfgi_features"] = len(feats)
        p["kept_features"] = ",".join(feats)
        p["spec_type"] = "drop" if keep is None else "lean_keep"
        rows.append(p)
        for f in feats:
            feature_rows.append({"variant": name, "feature": f, "kept": True})
        for f in drops:
            if f in df.columns:
                feature_rows.append({"variant": name, "feature": f, "kept": False})

    out = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    base_row = out[out["label"] == "baseline_full"].iloc[0]
    for col in ["ann_ret", "ann_vol", "sharpe", "mdd", "total_return", "downside_excess_bp"]:
        if col in out.columns:
            out[f"delta_{col}_vs_baseline"] = out[col] - base_row[col]
    feature_map = pd.DataFrame(feature_rows)
    save_table(out, "feature_drop_ablation.csv")
    save_table(feature_map, "feature_drop_ablation_feature_map.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
