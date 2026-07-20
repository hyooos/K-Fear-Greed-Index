from __future__ import annotations

import importlib.util
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
CODE_ROOT = ROOT if (ROOT / "tools" / "run_10y_kfgi_experiments.py").exists() else find_child(PROJECT_ROOT, "kfgi_최종")
try:
    PAPER_ROOT = find_child(PROJECT_ROOT, "논문용")
except FileNotFoundError:
    PAPER_ROOT = CODE_ROOT

REPO_OUT = CODE_ROOT / "paper_outputs"
ROBUST_DATA = REPO_OUT / "robustness" / "data" / "priority_ab_kfgi_timeseries_robust.csv"
TABLE_DIR = REPO_OUT / "tables"
FIG_DIR = REPO_OUT / "final_figures"


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


def make_return(df: pd.DataFrame, weight: pd.Series, fee: float = 0.0015) -> pd.Series:
    weight_lag = weight.shift(1).fillna(0)
    turnover = weight_lag.diff().abs().fillna(weight_lag.iloc[0])
    return weight_lag * df["target_reg"] - turnover * fee


def max_drawdown(r: pd.Series) -> float:
    wealth = np.exp(r.fillna(0).cumsum())
    return float((wealth / wealth.cummax() - 1).min())


def downside_metrics(r: pd.Series, b: pd.Series, w: pd.Series, label: str, n_features: int, features: list[str]) -> dict:
    a = pd.concat([r.rename("r"), b.rename("b"), w.rename("w")], axis=1).dropna()
    r = a["r"]
    b = a["b"]
    w = a["w"]
    down = b < 0
    dex = (r - b)[down]
    downside_p = stats.ttest_1samp(dex, 0).pvalue if len(dex) > 2 else np.nan
    downside_t = stats.ttest_1samp(dex, 0).statistic if len(dex) > 2 else np.nan
    mdd = max_drawdown(r)
    bh_mdd = max_drawdown(b)
    neg = r[r < 0]
    downside_dev = float(np.sqrt((neg**2).mean()) * math.sqrt(TRADING_DAYS)) if len(neg) else np.nan
    var5 = float(r.quantile(0.05))
    cvar5 = float(r[r <= var5].mean())
    ann_ret = float(r.mean() * TRADING_DAYS)
    ann_vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    return {
        "model": label,
        "n_obs": len(a),
        "n_features": n_features,
        "features": ", ".join(features),
        "total_return_pct": float((np.exp(r.sum()) - 1) * 100),
        "ann_return_pct": ann_ret * 100,
        "ann_vol_pct": ann_vol * 100,
        "sharpe": ann_ret / (ann_vol + 1e-12),
        "sortino": ann_ret / (downside_dev + 1e-12),
        "mdd_pct": mdd * 100,
        "mdd_defense_pctp": (mdd - bh_mdd) * 100,
        "downside_excess_bp": float(dex.mean() * 10000),
        "downside_t": float(downside_t),
        "downside_p": float(downside_p),
        "downside_hit_rate_pct": float((r[down] > b[down]).mean() * 100),
        "var_5_pct": var5 * 100,
        "cvar_5_pct": cvar5 * 100,
        "avg_exposure": float(w.mean()),
        "zero_exposure_pct": float((w <= 0.05).mean() * 100),
        "full_exposure_pct": float((w >= 0.95).mean() * 100),
    }


def add_drawdown_z(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    wealth = np.exp(d["target_reg"].fillna(0).cumsum())
    dd = wealth / wealth.cummax() - 1
    roll_mean = dd.rolling(252, min_periods=60).mean()
    roll_std = dd.rolling(252, min_periods=60).std()
    d["drawdown_z"] = ((dd - roll_mean) / (roll_std + 1e-9)).clip(-5, 5)
    return d


def build_kfgi(base, df: pd.DataFrame, name: str, features: list[str], directions: dict[str, int]) -> pd.DataFrame:
    feats = [f for f in features if f in df.columns]
    dir_vec = np.array([directions.get(f, 1) for f in feats])
    cache = REPO_OUT / "robustness" / "cache" / f"kfgi_compare_{name}.csv"
    out = base.create_walkforward_kfgi(df.drop(columns=["K_FGI"], errors="ignore"), feats, dir_vec, 60, cache)
    out.attrs["features"] = feats
    return out


def final_weight(priority, base, df: pd.DataFrame, cap: float = 1.0) -> pd.Series:
    rho_5d = stats.spearmanr(df["K_FGI"], df["target_5d"], nan_policy="omit").correlation
    kfgi_momentum = bool(rho_5d >= 0)
    pos = priority.compute_positions_param(base, df, kfgi_momentum=kfgi_momentum)
    return pos["weight"].clip(0, cap)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_module()
    priority = load_priority_module()
    df = pd.read_csv(ROBUST_DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = add_drawdown_z(df)

    directions = {
        **{f"sub_index{i}": 1 for i in range(1, 8)},
        "sent_composite_ma10": 1,
        "egarch_vol": -1,
        "vol_regime_high": -1,
        "vol_ratio": -1,
        "mom20": 1,
        "mom60": 1,
        "ma_ratio_20_60": 1,
        "drawdown_z": -1,
    }
    lean = [f"sub_index{i}" for i in range(2, 8)] + ["sent_composite_ma10", "egarch_vol", "vol_regime_high", "vol_ratio"]
    momentum_block = ["mom20", "mom60", "ma_ratio_20_60", "drawdown_z"]
    specs = {
        "Lean K-FGI 10 features": lean,
        "Lean + sub_index1 momentum": ["sub_index1"] + lean,
        "Extended K-FGI + technical momentum": lean + momentum_block,
        "Extended K-FGI + sub_index1 + momentum": ["sub_index1"] + lean + momentum_block,
        "Market-only FGI": [f"sub_index{i}" for i in range(2, 8)],
        "Market-only + sub_index1": [f"sub_index{i}" for i in range(1, 8)],
        "Without sentiment": [f"sub_index{i}" for i in range(2, 8)] + ["egarch_vol", "vol_regime_high", "vol_ratio"],
        "Sentiment-only index": ["sent_composite_ma10"],
    }

    rows = []
    ts_parts = []
    for name, features in specs.items():
        kdf = build_kfgi(base, df, name.lower().replace(" ", "_").replace("+", "plus"), features, directions)
        feats = kdf.attrs["features"]
        weight = final_weight(priority, base, kdf, cap=1.0)
        ret = make_return(kdf, weight)
        rows.append(downside_metrics(ret, kdf["target_reg"], weight, name, len(feats), feats))
        ts_parts.append(
            pd.DataFrame(
                {
                    "date": kdf["date"],
                    "model": name,
                    "K_FGI": kdf["K_FGI"],
                    "weight": weight,
                    "strategy_return": ret,
                    "buy_hold_return": kdf["target_reg"],
                }
            )
        )

    # EGARCH-only benchmark has no K-FGI feature score.
    eg_w = np.sqrt(base.TARGET_DAILY_VOL / (df["egarch_vol"] + 1e-9)).clip(0, 1.0)
    eg_w = eg_w.mask(df["vol_shock"] > base.CFG["vol_shock_thr"], 0.0)
    eg_ret = make_return(df, eg_w)
    rows.append(downside_metrics(eg_ret, df["target_reg"], eg_w, "EGARCH-only volatility targeting", 3, ["egarch_vol", "vol_regime_high", "vol_ratio"]))

    result = pd.DataFrame(rows).sort_values(["mdd_defense_pctp", "downside_excess_bp"], ascending=[False, False])
    ts = pd.concat(ts_parts, ignore_index=True)

    result.to_csv(TABLE_DIR / "subindex1_momentum_model_comparison.csv", index=False)
    ts.to_csv(REPO_OUT / "data" / "subindex1_momentum_model_timeseries.csv", index=False)

    display = result.copy()
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    colors = ["#2D6CDF" if m == "Lean K-FGI 10 features" else "#F59E0B" if "sub_index1" in m or "momentum" in m else "#9CA3AF" for m in display["model"]]
    ax.barh(display["model"], display["mdd_defense_pctp"], color=colors)
    ax.axvline(0, color="#111827", lw=0.9)
    ax.set_xlabel("MDD defense versus Buy & Hold (%p)")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    for y, v in enumerate(display["mdd_defense_pctp"]):
        ax.text(v + (0.25 if v >= 0 else -0.25), y, f"{v:.1f}%p", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "18_subindex1_momentum_MDD_defense_comparison.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ordered = result.sort_values("downside_excess_bp", ascending=False)
    colors = ["#2D6CDF" if m == "Lean K-FGI 10 features" else "#F59E0B" if "sub_index1" in m or "momentum" in m else "#9CA3AF" for m in ordered["model"]]
    ax.barh(ordered["model"], ordered["downside_excess_bp"], color=colors)
    ax.axvline(0, color="#111827", lw=0.9)
    ax.set_xlabel("Market-down-day defense (bp/day)")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    for y, v in enumerate(ordered["downside_excess_bp"]):
        ax.text(v + 0.6, y, f"{v:.1f}bp", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "19_subindex1_momentum_downside_defense_comparison.png", dpi=220)
    plt.close(fig)

    lean_row = result[result["model"] == "Lean K-FGI 10 features"].iloc[0]
    best_mdd = result.iloc[0]
    best_down = result.sort_values("downside_excess_bp", ascending=False).iloc[0]
    md = "# Subindex1 and Momentum Robustness Comparison\n\n"
    md += "본 문서는 최종 lean K-FGI에 `sub_index1` 및 기술적 모멘텀 피처를 추가할 필요가 있는지 검토한 강건성 실험이다. 모든 후보는 동일하게 포지션 상한 1.0x와 기존 tail-risk overlay를 적용하였다.\n\n"
    md += "## 1. 실험 후보\n\n"
    md += "| 후보 | 피처 수 | 설명 |\n| --- | ---: | --- |\n"
    for name, feats in specs.items():
        present = [f for f in feats if f in df.columns]
        md += f"| {name} | {len(present)} | {', '.join(present)} |\n"
    md += "| EGARCH-only volatility targeting | 3 | egarch_vol, vol_regime_high, vol_ratio |\n\n"
    md += "## 2. 성능 비교\n\n"
    cols = ["model", "n_features", "total_return_pct", "sharpe", "sortino", "mdd_pct", "mdd_defense_pctp", "downside_excess_bp", "downside_p", "avg_exposure"]
    md += result[cols].to_markdown(index=False, floatfmt=".3f")
    md += "\n\n"
    md += "## 3. 해석\n\n"
    md += f"- 메인 lean K-FGI는 피처 {int(lean_row['n_features'])}개, MDD {lean_row['mdd_pct']:.1f}%, MDD 방어 {lean_row['mdd_defense_pctp']:.1f}%p, 하락일 방어 {lean_row['downside_excess_bp']:.1f}bp를 보였다.\n"
    md += f"- MDD 방어 기준 최상위 후보는 `{best_mdd['model']}`이며 MDD 방어는 {best_mdd['mdd_defense_pctp']:.1f}%p다.\n"
    md += f"- 하락일 방어폭 기준 최상위 후보는 `{best_down['model']}`이며 하락일 방어는 {best_down['downside_excess_bp']:.1f}bp다.\n"
    md += "- `sub_index1`은 시장 모멘텀 성격이 맞다. 다만 최종 본문에서는 K-FGI 본체의 단순성, 해석 가능성, 감성 기여도 식별을 위해 lean 10개 피처를 메인으로 두고, `sub_index1` 및 기술적 모멘텀 추가 버전은 강건성/부록 실험으로 제시하는 편이 가장 안전하다.\n"
    md += "- 모멘텀 확장 후보가 lean 모델보다 뚜렷하게 우월하지 않다면, 피처를 늘려 과적합 가능성을 키우기보다 lean K-FGI를 유지하는 것이 논문 방어에 유리하다. 반대로 특정 downside 지표에서만 개선된다면, 해당 결과는 '모멘텀 정보가 보완적일 수 있다'는 부록 결과로 해석한다.\n\n"
    md += "## 4. 산출물\n\n"
    md += "- `paper_outputs/tables/subindex1_momentum_model_comparison.csv`\n"
    md += "- `paper_outputs/data/subindex1_momentum_model_timeseries.csv`\n"
    md += "- `paper_outputs/final_figures/18_subindex1_momentum_MDD_defense_comparison.png`\n"
    md += "- `paper_outputs/final_figures/19_subindex1_momentum_downside_defense_comparison.png`\n"

    (REPO_OUT / "SUBINDEX1_MOMENTUM_ROBUSTNESS.md").write_text(md, encoding="utf-8")
    print(result[cols].to_string(index=False))


if __name__ == "__main__":
    main()
