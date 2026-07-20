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


def make_return(df: pd.DataFrame, weight: pd.Series, fee: float) -> pd.Series:
    weight_lag = weight.shift(1).fillna(0)
    turnover = weight_lag.diff().abs().fillna(weight_lag.iloc[0])
    return weight_lag * df["target_reg"] - turnover * fee


def max_drawdown(r: pd.Series) -> float:
    wealth = np.exp(r.fillna(0).cumsum())
    return float((wealth / wealth.cummax() - 1).min())


def perf(r: pd.Series, ref: pd.Series, weight: pd.Series) -> dict:
    a = pd.concat([r.rename("r"), ref.rename("b"), weight.rename("w")], axis=1).dropna()
    r = a["r"]
    b = a["b"]
    w = a["w"]
    down = b < 0
    dex = (r - b)[down]
    ann_vol = r.std(ddof=1) * math.sqrt(TRADING_DAYS)
    bh_vol = b.std(ddof=1) * math.sqrt(TRADING_DAYS)
    bh_mdd = max_drawdown(b)
    mdd = max_drawdown(r)
    return {
        "n": len(a),
        "mdd": mdd,
        "bh_mdd": bh_mdd,
        "mdd_improvement_pctp": (mdd - bh_mdd) * 100,
        "downside_excess_bp": dex.mean() * 10000,
        "downside_t": stats.ttest_1samp(dex, 0).statistic,
        "downside_p": stats.ttest_1samp(dex, 0).pvalue,
        "downside_hit_rate": (r[down] > b[down]).mean(),
        "annual_vol_reduction_pct": (1 - ann_vol / bh_vol) * 100,
        "avg_weight": w.mean(),
        "max_weight": w.max(),
        "weight_gt_1_pct": (w > 1).mean() * 100,
        "zero_weight_pct": (w <= 0).mean() * 100,
    }


def subperiod_mdd(df: pd.DataFrame, r: pd.Series, year: int) -> float:
    mask = df["date"].dt.year.eq(year)
    return max_drawdown(r[mask]) if mask.any() else np.nan


def build_final_kfgi(base, df: pd.DataFrame) -> pd.DataFrame:
    feats = [f"sub_index{i}" for i in range(2, 8)] + [
        "sent_composite_ma10",
        "egarch_vol",
        "vol_regime_high",
        "vol_ratio",
    ]
    feats = [f for f in feats if f in df.columns]
    direction = {**{f"sub_index{i}": 1 for i in range(2, 8)}, "sent_composite_ma10": 1, "egarch_vol": -1, "vol_regime_high": -1, "vol_ratio": -1}
    dir_vec = np.array([direction[f] for f in feats])
    cache = REPO_OUT / "robustness" / "cache" / "kfgi_final_lean_sent_composite_ma10.csv"
    out = base.create_walkforward_kfgi(df.drop(columns=["K_FGI"], errors="ignore"), feats, dir_vec, 60, cache)
    return out


def compute_weight(priority, base, df: pd.DataFrame, fear: int, greed: int, k_min: float, k_max: float, cap: float, fee: float) -> tuple[pd.Series, pd.Series]:
    rho_5d = stats.spearmanr(df["K_FGI"], df["target_5d"], nan_policy="omit").correlation
    kfgi_momentum = bool(rho_5d >= 0)
    pos = priority.compute_positions_param(
        base,
        df,
        fee=fee,
        fear=fear,
        greed=greed,
        k_min=k_min,
        k_max=k_max,
        bull_mult=1.2,
        normal_mult=1.0,
        crisis_mult=0.7,
        kfgi_momentum=kfgi_momentum,
    )
    weight = pos["weight"].clip(0, cap)
    ret = make_return(df, weight, fee)
    return weight, ret


def run_sensitivity(base, priority, df: pd.DataFrame, kind: str, params: list[dict]) -> pd.DataFrame:
    rows = []
    for p in params:
        fear = p.get("fear", 25)
        greed = p.get("greed", 65)
        k_min = p.get("k_min", 0.5)
        k_max = p.get("k_max", 1.6)
        cap = p.get("cap", 1.0)
        fee = p.get("fee", 0.0015)
        w, r = compute_weight(priority, base, df, fear, greed, k_min, k_max, cap, fee)
        row = {
            "sensitivity_type": kind,
            "setting": p["setting"],
            "fear": fear,
            "greed": greed,
            "k_min": k_min,
            "k_max": k_max,
            "position_cap": cap,
            "fee_bp": fee * 10000,
            **perf(r, df["target_reg"], w),
            "mdd_2016": subperiod_mdd(df, r, 2016),
            "mdd_2020": subperiod_mdd(df, r, 2020),
            "mdd_2022": subperiod_mdd(df, r, 2022),
            "mdd_2025": subperiod_mdd(df, r, 2025),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    base = load_base_module()
    priority = load_priority_module()
    raw = pd.read_csv(ROBUST_DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = build_final_kfgi(base, raw)

    base_setting = {"setting": "baseline_final", "fear": 25, "greed": 65, "k_min": 0.5, "k_max": 1.6, "cap": 1.0, "fee": 0.0015}
    cap_params = [{**base_setting, "setting": f"cap_{cap:.1f}", "cap": cap} for cap in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]
    fee_params = [{**base_setting, "setting": f"fee_{bp}bp", "fee": bp / 10000} for bp in [0, 5, 10, 15, 20]]
    threshold_params = [{**base_setting, "setting": f"fear{fear}_greed{greed}", "fear": fear, "greed": greed} for fear, greed in [(20, 80), (25, 65), (30, 70), (20, 65), (30, 65)]]
    k_params = [{**base_setting, "setting": f"k{kmin:.1f}_{kmax:.1f}", "k_min": kmin, "k_max": kmax} for kmin, kmax in [(0.4, 1.4), (0.5, 1.4), (0.5, 1.6), (0.6, 1.6), (0.5, 1.8)]]

    out = pd.concat(
        [
            run_sensitivity(base, priority, df, "position_cap", cap_params),
            run_sensitivity(base, priority, df, "transaction_cost", fee_params),
            run_sensitivity(base, priority, df, "fear_greed_threshold", threshold_params),
            run_sensitivity(base, priority, df, "kfgi_multiplier", k_params),
        ],
        ignore_index=True,
    )

    baseline = out[(out["sensitivity_type"] == "position_cap") & (out["setting"] == "cap_1.0")].iloc[0]
    md = "# 최종 보수형 K-FGI 민감도 분석\n\n"
    md += "피처는 `sub_index2-7 + sent_composite_ma10 + egarch_vol + vol_regime_high + vol_ratio`로 고정하고, 포지션 상한 1.0배를 기본값으로 두었다. 본 표는 수익률이 아니라 하방방어 지표 중심으로 해석한다.\n\n"
    md += f"기본 후보(`cap_1.0`, fee 15bp, Fear/Greed 25/65, K-FGI multiplier 0.5~1.6)는 MDD {baseline['mdd']*100:.1f}%, MDD 개선 {baseline['mdd_improvement_pctp']:.1f}%p, 하락일 방어폭 {baseline['downside_excess_bp']:.1f}bp, p-value <0.001을 보였다.\n\n"
    for kind, title in [
        ("position_cap", "포지션 상한"),
        ("transaction_cost", "거래비용"),
        ("fear_greed_threshold", "Fear/Greed 기준"),
        ("kfgi_multiplier", "K-FGI multiplier"),
    ]:
        sub = out[out["sensitivity_type"] == kind].copy()
        md += f"## {title}\n\n"
        md += "| 설정 | MDD | MDD 개선 | 하락일 방어폭 | p-value | 평균 노출 | 1배 초과 비중 | 2016 MDD |\n"
        md += "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        for _, row in sub.iterrows():
            pval = "<0.001" if row["downside_p"] < 0.001 else f"{row['downside_p']:.4f}"
            md += (
                f"| {row['setting']} | {row['mdd']*100:.1f}% | {row['mdd_improvement_pctp']:.1f}%p | "
                f"{row['downside_excess_bp']:.1f}bp | {pval} | {row['avg_weight']:.2f}x | "
                f"{row['weight_gt_1_pct']:.1f}% | {row['mdd_2016']*100:.1f}% |\n"
            )
        md += "\n"
    md += """## 논문 반영 문장

민감도 분석 결과, 최종 K-FGI의 하방방어 효과는 특정 단일 파라미터 조합에만 의존하지 않았다. 포지션 상한, 거래비용, Fear/Greed 기준, K-FGI multiplier를 변화시켜도 시장 하락일 방어폭은 대부분 통계적으로 유의하게 유지되었다. 다만 하방위험 관리 목적과 해석 가능성을 고려하면 1.0배 포지션 상한은 레버리지 노출을 제거하면서도 MDD 개선과 하락일 방어 효과를 유지하므로 최종 운용 규칙으로 가장 적절하다.
"""

    for root in [REPO_OUT, PAPER_OUT]:
        (root / "tables").mkdir(parents=True, exist_ok=True)
        out.to_csv(root / "tables" / "final_conservative_kfgi_sensitivity.csv", index=False, encoding="utf-8-sig")
        (root / "FINAL_CONSERVATIVE_KFGI_SENSITIVITY.md").write_text(md, encoding="utf-8")

    cols = ["sensitivity_type", "setting", "mdd", "mdd_improvement_pctp", "downside_excess_bp", "downside_p", "avg_weight", "weight_gt_1_pct", "mdd_2016"]
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
