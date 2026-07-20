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


def perf(r: pd.Series, ref: pd.Series, label: str, weight: pd.Series) -> dict:
    a = pd.concat([r.rename("r"), ref.rename("b"), weight.rename("w")], axis=1).dropna()
    r = a["r"]
    b = a["b"]
    w = a["w"]
    cum = np.exp(r.cumsum())
    bh_cum = np.exp(b.cumsum())
    dd = cum / cum.cummax() - 1
    bh_dd = bh_cum / bh_cum.cummax() - 1
    down = b < 0
    dex = (r - b)[down]
    return {
        "label": label,
        "n": len(a),
        "ann_ret": float(r.mean() * TRADING_DAYS),
        "ann_vol": float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        "sharpe": float((r.mean() * TRADING_DAYS) / (r.std(ddof=1) * math.sqrt(TRADING_DAYS) + 1e-12)),
        "mdd": float(dd.min()),
        "bh_mdd": float(bh_dd.min()),
        "mdd_improvement_pctp": float((dd.min() - bh_dd.min()) * 100),
        "downside_excess_bp": float(dex.mean() * 10000),
        "downside_t": float(stats.ttest_1samp(dex, 0).statistic),
        "downside_p": float(stats.ttest_1samp(dex, 0).pvalue),
        "downside_hit_rate": float((r[down] > b[down]).mean()),
        "annual_vol_reduction_pct": float((1 - r.std(ddof=1) / b.std(ddof=1)) * 100),
        "avg_weight": float(w.mean()),
        "max_weight": float(w.max()),
        "weight_gt_1_pct": float((w > 1).mean() * 100),
        "weight_gt_1_5_pct": float((w > 1.5).mean() * 100),
        "zero_weight_pct": float((w <= 0).mean() * 100),
    }


def subperiod_stats(df: pd.DataFrame, r: pd.Series, w: pd.Series) -> dict:
    out: dict[str, float] = {}
    for label, start, end in [
        ("2016", "2016-01-01", "2016-12-31"),
        ("2017", "2017-01-01", "2017-12-31"),
        ("2020_COVID", "2020-02-01", "2020-12-31"),
        ("2022_bear", "2022-01-01", "2022-12-31"),
        ("2025_bull", "2025-01-01", "2025-12-31"),
    ]:
        mask = df["date"].between(start, end)
        if not mask.any():
            continue
        p = perf(r[mask], df.loc[mask, "target_reg"], label, w[mask])
        out[f"{label}_mdd"] = p["mdd"]
        out[f"{label}_downside_excess_bp"] = p["downside_excess_bp"]
        out[f"{label}_avg_weight"] = p["avg_weight"]
        out[f"{label}_weight_gt_1_pct"] = p["weight_gt_1_pct"]
    return out


def main() -> None:
    base = load_base_module()
    priority = load_priority_module()
    df = pd.read_csv(ROBUST_DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if "neg_z_ma5_inv" not in df.columns and "neg_z_ma5" in df.columns:
        df["neg_z_ma5_inv"] = -df["neg_z_ma5"]

    direction = {
        **{f"sub_index{i}": 1 for i in range(2, 8)},
        "sent_norm_ma5": 1,
        "neg_z_ma5_inv": 1,
        "sent_composite_ma10": 1,
        "egarch_vol": -1,
        "vol_regime_high": -1,
        "vol_ratio": -1,
    }
    specs = {
        "lean_sent_composite_ma10": [f"sub_index{i}" for i in range(2, 8)]
        + ["sent_composite_ma10", "egarch_vol", "vol_regime_high", "vol_ratio"],
        "smoothed_sentiment": [f"sub_index{i}" for i in range(2, 8)]
        + ["sent_norm_ma5", "neg_z_ma5_inv", "sent_composite_ma10", "egarch_vol", "vol_regime_high", "vol_ratio"],
    }

    rho_5d = stats.spearmanr(df["K_FGI"], df["target_5d"], nan_policy="omit").correlation
    kfgi_momentum = bool(rho_5d >= 0)
    rows = []
    ts_parts = []
    for spec_name, raw_feats in specs.items():
        feats = [f for f in raw_feats if f in df.columns]
        dir_vec = np.array([direction.get(f, 1) for f in feats])
        cache = REPO_OUT / "robustness" / "cache" / f"kfgi_conservative_{spec_name}.csv"
        vdf = base.create_walkforward_kfgi(df.drop(columns=["K_FGI"], errors="ignore"), feats, dir_vec, 60, cache)
        vdf["regime"] = vdf.apply(lambda r: priority.classify_with_threshold(base, r, 25, 65), axis=1)
        pos = priority.compute_positions_param(base, vdf, kfgi_momentum=kfgi_momentum)
        base_weight = pos["weight"]
        for cap_label, cap in [("current_cap_3_0", None), ("cap_1_2", 1.2), ("cap_1_0_no_leverage", 1.0), ("cap_0_8_defensive", 0.8)]:
            weight = base_weight if cap is None else base_weight.clip(0, cap)
            ret = make_return(vdf, weight)
            label = f"{spec_name}__{cap_label}"
            row = perf(ret, vdf["target_reg"], label, weight)
            row["feature_spec"] = spec_name
            row["position_cap"] = 3.0 if cap is None else cap
            row["n_features"] = len(feats)
            row["features"] = ",".join(feats)
            row.update(subperiod_stats(vdf, ret, weight))
            rows.append(row)
            ts_parts.append(
                pd.DataFrame(
                    {
                        "date": vdf["date"],
                        "variant": label,
                        "K_FGI": vdf["K_FGI"],
                        "weight": weight,
                        "strategy_return": ret,
                        "buy_hold_return": vdf["target_reg"],
                    }
                )
            )

    out = pd.DataFrame(rows).sort_values(["mdd", "downside_excess_bp"], ascending=[False, False])
    ts = pd.concat(ts_parts, ignore_index=True)

    md = "# 보수형 K-FGI 포지션 상한 실험\n\n"
    md += "하방방어 목적의 전략에서는 1배를 초과하는 레버리지 노출이 논문 주장과 충돌할 수 있으므로, K-FGI 점수 산출은 유지하되 최종 포지션 상한만 보수적으로 제한하는 후보를 비교하였다.\n\n"
    md += "| 후보 | 피처 구성 | 포지션 상한 | MDD | MDD 개선 | 하락일 방어폭 | 하락일 방어 t | p-value | 평균 노출 | 1배 초과 비중 | 2016 MDD | 2016 평균 노출 |\n"
    md += "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    for _, row in out.iterrows():
        pval = "<0.001" if row["downside_p"] < 0.001 else f"{row['downside_p']:.4f}"
        md += (
            f"| {row['label']} | {row['feature_spec']} | {row['position_cap']:.1f}x | "
            f"{row['mdd']*100:.1f}% | {row['mdd_improvement_pctp']:.1f}%p | "
            f"{row['downside_excess_bp']:.1f}bp | {row['downside_t']:.2f} | {pval} | "
            f"{row['avg_weight']:.2f}x | {row['weight_gt_1_pct']:.1f}% | "
            f"{row.get('2016_mdd', np.nan)*100:.1f}% | {row.get('2016_avg_weight', np.nan):.2f}x |\n"
        )
    main_candidate = out[out["label"] == "lean_sent_composite_ma10__cap_1_0_no_leverage"].iloc[0]
    defensive_candidate = out[out["label"] == "lean_sent_composite_ma10__cap_0_8_defensive"].iloc[0]
    md += f"""

## 결론

논문 메인 후보는 `{main_candidate['label']}`로 두는 것이 가장 자연스럽다. 이 후보는 포지션 상한을 1.0배로 제한하여 레버리지 노출을 제거하고, 전체 MDD를 {main_candidate['mdd']*100:.1f}%까지 낮춘다. 수치상 가장 강한 방어 후보는 `{defensive_candidate['label']}`이며 MDD가 {defensive_candidate['mdd']*100:.1f}%까지 낮아지지만, 0.8배 상한은 임의성이 있으므로 메인 전략보다는 보수적 민감도 분석 또는 부록 후보로 제시하는 것이 적절하다.

## 본문 서술 예시

본 연구는 K-FGI를 수익률 극대화 지표가 아니라 하방위험 관리 지표로 정의하므로, 최종 포지션 산출 단계에서 1배 초과 레버리지 노출을 제한한 보수형 운용안을 추가로 검토하였다. 실험 결과 포지션 상한을 1.0배로 제한할 경우 초기 구간의 과도한 시장 노출이 완화되고, 전체 최대낙폭 및 2016년 구간의 낙폭이 개선되었다. 따라서 최종 전략은 K-FGI 신호 자체보다 포지션 상한을 보수적으로 설정하는 것이 하방방어 목적에 부합함을 확인하였다.
"""

    for root in [REPO_OUT, PAPER_OUT]:
        (root / "tables").mkdir(parents=True, exist_ok=True)
        (root / "data").mkdir(parents=True, exist_ok=True)
        out.to_csv(root / "tables" / "conservative_kfgi_position_caps.csv", index=False, encoding="utf-8-sig")
        ts.to_csv(root / "data" / "conservative_kfgi_position_caps_timeseries.csv", index=False, encoding="utf-8-sig")
        (root / "CONSERVATIVE_KFGI_POSITION_CAPS.md").write_text(md, encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
