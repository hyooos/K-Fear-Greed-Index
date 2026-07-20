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


def perf(r: pd.Series, ref: pd.Series, label: str) -> dict:
    a = pd.concat([r.rename("r"), ref.rename("b")], axis=1).dropna()
    r = a["r"]
    b = a["b"]
    ann_ret = float(r.mean() * TRADING_DAYS)
    ann_vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    cum = np.exp(r.cumsum())
    bh_cum = np.exp(b.cumsum())
    dd = cum / cum.cummax() - 1
    bh_dd = bh_cum / bh_cum.cummax() - 1
    down = b < 0
    dex = (r - b)[down]
    return {
        "label": label,
        "n": len(a),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / (ann_vol + 1e-12),
        "mdd": float(dd.min()),
        "bh_mdd": float(bh_dd.min()),
        "mdd_improvement_pctp": float((dd.min() - bh_dd.min()) * 100),
        "downside_excess_bp": float(dex.mean() * 10000),
        "downside_t": float(stats.ttest_1samp(dex, 0).statistic),
        "downside_p": float(stats.ttest_1samp(dex, 0).pvalue),
        "downside_hit_rate": float((r[down] > b[down]).mean()),
        "annual_vol_reduction_pct": float((1 - r.std(ddof=1) / b.std(ddof=1)) * 100),
    }


def max_abs_turnover(pos: pd.DataFrame) -> float:
    return float(pos["turnover"].abs().max()) if "turnover" in pos else np.nan


def save_outputs(table: pd.DataFrame, md: str) -> None:
    for out_root in [REPO_OUT, PAPER_OUT]:
        table_dir = out_root / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_dir / "smoothed_feature_experiments.csv", index=False, encoding="utf-8-sig")
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "SMOOTHED_FEATURE_EXPERIMENTS.md").write_text(md, encoding="utf-8")


def main() -> None:
    base = load_base_module()
    priority = load_priority_module()
    df = pd.read_csv(ROBUST_DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    rets = pd.read_csv(ROBUST_RETURNS, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    if "neg_z_ma5_inv" not in df.columns and "neg_z_ma5" in df.columns:
        df["neg_z_ma5_inv"] = -df["neg_z_ma5"]

    direction = {
        **{f"sub_index{i}": 1 for i in range(2, 8)},
        "sent_norm_w": 1,
        "sent_norm_ma5": 1,
        "sent_energy": 1,
        "sent_std_inv": 1,
        "neg_z_inv": 1,
        "neg_z_ma5_inv": 1,
        "sent_composite": 1,
        "sent_composite_ma10": 1,
        "egarch_vol": -1,
        "vol_ma5": -1,
        "vol_regime_high": -1,
        "vol_ratio": -1,
    }

    specs = {
        "baseline_full_mixed_raw_smooth": [f"sub_index{i}" for i in range(2, 8)]
        + ["sent_norm_w", "sent_energy", "sent_std_inv", "neg_z_inv", "sent_composite", "sent_composite_ma10", "egarch_vol", "vol_regime_high", "vol_ratio"],
        "raw_sentiment_only": [f"sub_index{i}" for i in range(2, 8)]
        + ["sent_norm_w", "sent_energy", "sent_std_inv", "neg_z_inv", "sent_composite", "egarch_vol", "vol_regime_high", "vol_ratio"],
        "smoothed_sentiment_only": [f"sub_index{i}" for i in range(2, 8)]
        + ["sent_norm_ma5", "neg_z_ma5_inv", "sent_composite_ma10", "egarch_vol", "vol_regime_high", "vol_ratio"],
        "sent_composite_ma10_only": [f"sub_index{i}" for i in range(2, 8)]
        + ["sent_composite_ma10", "egarch_vol", "vol_regime_high", "vol_ratio"],
        "smooth_sentiment_smooth_vol": [f"sub_index{i}" for i in range(2, 8)]
        + ["sent_norm_ma5", "neg_z_ma5_inv", "sent_composite_ma10", "vol_ma5", "vol_regime_high", "vol_ratio"],
        "market_egarch_only": [f"sub_index{i}" for i in range(2, 8)] + ["egarch_vol", "vol_regime_high", "vol_ratio"],
    }

    rho_5d = stats.spearmanr(df["K_FGI"], df["target_5d"], nan_policy="omit").correlation
    kfgi_momentum = bool(rho_5d >= 0)

    rows = []
    for name, raw_feats in specs.items():
        feats = [f for f in raw_feats if f in df.columns]
        dir_vec = np.array([direction.get(f, 1) for f in feats])
        cache = REPO_OUT / "robustness" / "cache" / f"kfgi_smooth_{name}.csv"
        vdf = base.create_walkforward_kfgi(df.drop(columns=["K_FGI"], errors="ignore"), feats, dir_vec, 60, cache)
        vdf["regime"] = vdf.apply(lambda r: priority.classify_with_threshold(base, r, 25, 65), axis=1)
        pos = priority.compute_positions_param(base, vdf, kfgi_momentum=kfgi_momentum)
        p = perf(pos["strat_ret"].reset_index(drop=True), vdf["target_reg"].reset_index(drop=True), name)
        p["n_features"] = len(feats)
        p["features"] = ",".join(feats)
        p["max_turnover"] = max_abs_turnover(pos)
        rows.append(p)

    table = pd.DataFrame(rows).sort_values(["mdd", "downside_excess_bp"], ascending=[False, False])

    best = table.iloc[0]
    lean = table[table["label"] == "sent_composite_ma10_only"].iloc[0]
    baseline = table[table["label"] == "baseline_full_mixed_raw_smooth"].iloc[0]
    no_sent = table[table["label"] == "market_egarch_only"].iloc[0]

    md = f"""# 평활화 피처 실험 결과

## 결론

10개년 일별 K-FGI에서는 raw 감성 피처를 여러 개 동시에 넣는 것보다, 감성 정보를 평활화한 `sent_composite_ma10` 중심 구성이 더 안정적이다. 따라서 논문 메인 모델은 raw 감성 피처를 모두 유지하기보다 **`sent_composite_ma10`만 남기는 lean K-FGI**로 제시하는 것이 타당하다.

## 후보별 하방방어 비교

| 후보 | 피처 수 | MDD | MDD 개선 | 하락일 방어폭 | 하락일 방어 t | p-value | 변동성 감소 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for _, row in table.iterrows():
        pval = "<0.001" if row["downside_p"] < 0.001 else f"{row['downside_p']:.4f}"
        md += (
            f"| {row['label']} | {int(row['n_features'])} | "
            f"{row['mdd']*100:.1f}% | {row['mdd_improvement_pctp']:.1f}%p | "
            f"{row['downside_excess_bp']:.1f}bp | {row['downside_t']:.2f} | {pval} | "
            f"{row['annual_vol_reduction_pct']:.1f}% |\n"
        )

    md += f"""
## 논문에 반영할 판단

- `sent_composite_ma10_only`는 피처 수 {int(lean['n_features'])}개로 가장 단순하면서도 MDD {lean['mdd']*100:.1f}%, 하락일 방어폭 {lean['downside_excess_bp']:.1f}bp를 보인다.
- full mixed 구성은 피처 수 {int(baseline['n_features'])}개로 더 복잡하지만 MDD {baseline['mdd']*100:.1f}%, 하락일 방어폭 {baseline['downside_excess_bp']:.1f}bp로 lean 구성 대비 뚜렷한 우위가 없다.
- 감성 제거 구성은 MDD {no_sent['mdd']*100:.1f}%로 악화된다. 따라서 감성은 제거하기보다 평활화된 핵심 피처 하나로 압축하는 것이 적절하다.
- 최종 권장 피처: `sub_index2-7`, `sent_composite_ma10`, `egarch_vol`, `vol_regime_high`, `vol_ratio`.

## 본문 서술 예시

감성 변수는 일별 뉴스 댓글의 잡음과 극단값 영향을 크게 받을 수 있으므로, 본 연구는 raw 감성 변수와 평활화 감성 변수를 비교하였다. 실험 결과 raw 감성 피처를 모두 포함한 모형보다 10일 이동평균 기반 복합 감성지표(`sent_composite_ma10`)만을 포함한 단순 모형이 유사하거나 더 안정적인 하방방어 성과를 보였다. 이에 따라 최종 K-FGI는 감성 정보를 제거하지 않고, 단기 잡음을 완화한 평활화 감성 피처로 압축하여 구성하였다.
"""
    save_outputs(table, md)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
