from __future__ import annotations

import math
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


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


def perf(r: pd.Series) -> dict[str, float]:
    r = r.dropna()
    c = np.exp(r.cumsum())
    dd = c / c.cummax() - 1
    return {
        "total_return": float(c.iloc[-1] - 1),
        "mdd": float(dd.min()),
        "sharpe": float((r.mean() * 252) / (r.std(ddof=1) * math.sqrt(252) + 1e-12)),
        "ann_ret": float(r.mean() * 252),
        "ann_vol": float(r.std(ddof=1) * math.sqrt(252)),
    }


def make_return(df: pd.DataFrame, weight: pd.Series, fee: float = 0.0015) -> pd.Series:
    weight_lag = weight.shift(1).fillna(0)
    turnover = weight_lag.diff().abs().fillna(weight_lag.iloc[0])
    return weight_lag * df["target_reg"] - turnover * fee


def main() -> None:
    df = pd.read_csv(REPO_OUT / "robustness" / "data" / "priority_ab_kfgi_timeseries_robust.csv", parse_dates=["date"])
    bh = df["target_reg"]
    base_weight = df["weight"]
    variants = {
        "Buy & Hold": bh,
        "K-FGI current 15bp": df["strat_ret"],
        "K-FGI no-leverage cap 1.0x": make_return(df, base_weight.clip(0, 1.0)),
        "K-FGI cap 1.2x": make_return(df, base_weight.clip(0, 1.2)),
        "K-FGI warm-up first 252d cap 0.5x": make_return(df, base_weight.mask(df.index < 252, base_weight.clip(0, 0.5))),
    }
    periods = {
        "2015Q4-2016": ("2015-10-01", "2016-12-31"),
        "2016": ("2016-01-01", "2016-12-31"),
        "2017": ("2017-01-01", "2017-12-31"),
        "2016-2017": ("2016-01-01", "2017-12-31"),
        "Full": ("1900-01-01", "2100-01-01"),
    }
    rows = []
    for period, (start, end) in periods.items():
        mask = df["date"].between(start, end)
        for name, r in variants.items():
            m = perf(r[mask])
            rows.append({"period": period, "strategy": name, "n": int(mask.sum()), **m})
    out = pd.DataFrame(rows)
    for root in [REPO_OUT, PAPER_OUT]:
        (root / "tables").mkdir(parents=True, exist_ok=True)
        out.to_csv(root / "tables" / "early_drawdown_diagnosis.csv", index=False, encoding="utf-8-sig")

    current_2016 = out[(out["period"] == "2016") & (out["strategy"] == "K-FGI current 15bp")].iloc[0]
    cap_2016 = out[(out["period"] == "2016") & (out["strategy"] == "K-FGI no-leverage cap 1.0x")].iloc[0]
    current_full = out[(out["period"] == "Full") & (out["strategy"] == "K-FGI current 15bp")].iloc[0]
    cap_full = out[(out["period"] == "Full") & (out["strategy"] == "K-FGI no-leverage cap 1.0x")].iloc[0]
    bh_2016 = out[(out["period"] == "2016") & (out["strategy"] == "Buy & Hold")].iloc[0]

    md = f"""# Early Drawdown Diagnosis

## 핵심 결론

2016년에는 현재 K-FGI가 Buy&Hold보다 분명히 나빴다. 이는 숨기면 안 되는 결과다.

- 2016 Buy&Hold: total return {bh_2016['total_return']*100:.1f}%, MDD {bh_2016['mdd']*100:.1f}%
- 2016 current K-FGI: total return {current_2016['total_return']*100:.1f}%, MDD {current_2016['mdd']*100:.1f}%
- 2016 no-leverage cap 1.0x K-FGI: total return {cap_2016['total_return']*100:.1f}%, MDD {cap_2016['mdd']*100:.1f}%

## 왜 이런 일이 생겼나

2016년은 K-FGI 평균이 높고 bull/normal 판정이 많아 시장 노출이 크게 잡힌 구간이다. 하지만 실제로는 2016년 초 낙폭이 있었고, K-FGI가 이 구간에서 방어 신호를 충분히 빨리 내지 못했다. 즉 K-FGI는 모든 구간에서 방어하는 만능 지표가 아니라, 특정 초기/국면 전환 구간에서는 과노출될 수 있다.

## 모델을 수정해야 하나?

수정한다면 가장 방어 가능한 방법은 **max exposure를 1.0x로 제한한 no-leverage K-FGI**를 main 또는 robustness로 제시하는 것이다. 이는 2016만 보고 만든 규칙이 아니라, 논문/실무적으로 자연스러운 제약이다.

- Full current K-FGI: total return {current_full['total_return']*100:.1f}%, MDD {current_full['mdd']*100:.1f}%, Sharpe {current_full['sharpe']:.3f}
- Full no-leverage cap 1.0x K-FGI: total return {cap_full['total_return']*100:.1f}%, MDD {cap_full['mdd']*100:.1f}%, Sharpe {cap_full['sharpe']:.3f}

## 논문 서술 권장

K-FGI는 2016년과 같은 강세 전환/초기 추정 구간에서는 Buy&Hold보다 큰 손실을 낼 수 있었다. 따라서 본 연구는 K-FGI를 초과수익 창출 신호라기보다, 거래비용과 레버리지 제약을 고려한 위험관리형 지표로 해석한다. 추가적으로 no-leverage cap을 적용하면 초기 낙폭 문제가 완화되어 실무 적용 가능성이 높아진다.
"""
    for root in [REPO_OUT, PAPER_OUT, CODE_ROOT / "docs"]:
        (root / "EARLY_DRAWDOWN_DIAGNOSIS.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
