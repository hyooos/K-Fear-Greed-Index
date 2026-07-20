from __future__ import annotations

import math
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
TRADING_DAYS = 252


def ensure_dirs() -> None:
    for root in [REPO_OUT, PAPER_OUT]:
        (root / "tables").mkdir(parents=True, exist_ok=True)
        (root / "figures").mkdir(parents=True, exist_ok=True)


def clean_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_data() -> pd.DataFrame:
    ts = pd.read_csv(REPO_OUT / "robustness" / "data" / "priority_ab_kfgi_timeseries_robust.csv", parse_dates=["date"])
    rets = pd.read_csv(REPO_OUT / "robustness" / "data" / "priority_ab_strategy_returns_robust.csv", parse_dates=["date"])
    df = ts.merge(rets, on="date", how="inner", suffixes=("", "_ret"))
    for col in df.columns:
        if col != "date":
            df[col] = clean_num(df[col])

    attention = np.log1p(df["effective_n"].clip(lower=0))
    comments = np.log1p(df["comment_count"].clip(lower=0))
    df["sent_attention"] = df["sent_norm_w"] * attention
    df["negative_attention"] = df["neg_z"].clip(lower=0) * attention
    df["panic_pressure"] = (-df["sent_norm_w"]).clip(lower=0) * df["sent_strength_w"].clip(lower=0) * attention
    df["disagreement_attention"] = df["sent_std"].clip(lower=0) * attention
    df["heat_attention"] = df["heat"].clip(lower=0) * comments
    df["sent_confidence"] = df["sent_norm_w"].abs() * attention / (df["sent_std"].abs() + 0.05)
    risk_parts = [
        df[c].rank(pct=True)
        for c in ["negative_attention", "panic_pressure", "disagreement_attention", "heat_attention"]
    ]
    df["sentiment_risk_composite"] = pd.concat(risk_parts, axis=1).mean(axis=1)
    df["strategy_delta"] = df["kfgi_with_sentiment"] - df["kfgi_without_sentiment"]
    df["defense_success_down_day"] = ((df["buy_hold"] < 0) & (df["kfgi_with_sentiment"] > df["buy_hold"])).astype(int)
    df["target_5d_down"] = (df["target_5d"] < 0).astype(int)
    df["target_5d_large_drop"] = (df["target_5d"] <= -0.02).astype(int)
    df["target_1d_down"] = (df["target_reg"] < 0).astype(int)
    return df


SENTIMENT_FEATURES = [
    "sent_norm_w",
    "sent_strength_w",
    "sent_std",
    "neg_z",
    "sent_energy",
    "sent_composite",
    "sent_composite_ma10",
    "sent_attention",
    "negative_attention",
    "panic_pressure",
    "disagreement_attention",
    "heat_attention",
    "sent_confidence",
    "sentiment_risk_composite",
    "effective_n",
    "comment_count",
]

MARKET_FEATURES = [
    "sub_index2",
    "sub_index3",
    "sub_index4",
    "sub_index5",
    "sub_index6",
    "sub_index7",
    "egarch_vol",
    "egarch_vol_lag1",
    "egarch_vol_ma5_lag1",
    "vol_shock",
    "vol_shock_lag1",
    "vol_regime_high",
    "vol_regime_lag1",
    "vol_ratio",
    "trend_strength",
    "ma_ratio_5_20",
    "ma_ratio_20_60",
    "ma_ratio_60_120",
    "rsi14",
    "mom5",
    "mom20",
    "mom60",
    "dayofweek",
    "month",
]


FEATURE_GROUP = {f: "감성" for f in SENTIMENT_FEATURES}
FEATURE_GROUP.update({f: "시장/변동성" for f in MARKET_FEATURES})

FEATURE_LABEL = {
    "sent_norm_w": "정규화 감성",
    "sent_strength_w": "감성 강도",
    "sent_std": "의견 불일치",
    "neg_z": "부정 감성",
    "sent_energy": "감성 에너지",
    "sent_composite": "감성 복합",
    "sent_composite_ma10": "감성 복합 10일",
    "sent_attention": "감성 관심",
    "negative_attention": "부정 관심",
    "panic_pressure": "공포 압력",
    "disagreement_attention": "불일치 관심",
    "heat_attention": "댓글 열기",
    "sent_confidence": "감성 신뢰도",
    "sentiment_risk_composite": "감성위험 복합",
    "effective_n": "유효 댓글 수",
    "comment_count": "댓글 수",
    "K_FGI": "K-FGI",
    "egarch_vol": "EGARCH 변동성",
    "vol_ratio": "변동성 비율",
    "vol_shock": "변동성 충격",
    "vol_regime_high": "고변동성 국면",
    "trend_strength": "추세 강도",
    "ma_ratio_5_20": "MA 5/20",
    "ma_ratio_20_60": "MA 20/60",
    "ma_ratio_60_120": "MA 60/120",
    "mom5": "5일 모멘텀",
    "mom20": "20일 모멘텀",
    "mom60": "60일 모멘텀",
}


TARGETS = {
    "target_5d_down": {
        "label": "5거래일 하락위험",
        "filter_down_days": False,
        "positive_meaning": "향후 5거래일 누적수익률이 음수",
    },
    "target_5d_large_drop": {
        "label": "5거래일 -2% 급락위험",
        "filter_down_days": False,
        "positive_meaning": "향후 5거래일 누적수익률이 -2% 이하",
    },
    "defense_success_down_day": {
        "label": "하락일 방어성공",
        "filter_down_days": True,
        "positive_meaning": "B&H 하락일에 K-FGI 손실이 B&H보다 작음",
    },
}


def modeling_frame(df: pd.DataFrame, target_col: str, filter_down_days: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = [f for f in SENTIMENT_FEATURES + MARKET_FEATURES if f in df.columns]
    d = df.copy()
    if filter_down_days:
        d = d[d["buy_hold"] < 0].copy()
    cols = ["date", target_col] + features
    d = d[cols].replace([np.inf, -np.inf], np.nan).dropna().sort_values("date").reset_index(drop=True)
    x = d[features].copy()
    y = d[target_col].astype(int)
    return x, y, d["date"]


def evaluate_elastic_net(x: pd.DataFrame, y: pd.Series, dates: pd.Series, n_splits: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    candidates = [(c, r) for c in [0.05, 0.1, 0.2, 0.5] for r in [0.3, 0.7, 1.0]]
    rows = []
    best = None
    best_score = -np.inf
    for c, ratio in candidates:
        fold_scores = []
        for fold, (tr, te) in enumerate(tscv.split(x), 1):
            if y.iloc[tr].nunique() < 2 or y.iloc[te].nunique() < 2:
                continue
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            l1_ratio=ratio,
                            C=c,
                            class_weight="balanced",
                            max_iter=8000,
                            random_state=42,
                        ),
                    ),
                ]
            )
            model.fit(x.iloc[tr], y.iloc[tr])
            prob = model.predict_proba(x.iloc[te])[:, 1]
            pred = (prob >= 0.5).astype(int)
            auc = roc_auc_score(y.iloc[te], prob)
            ap = average_precision_score(y.iloc[te], prob)
            bacc = balanced_accuracy_score(y.iloc[te], pred)
            fold_scores.append(auc)
            rows.append(
                {
                    "model": "ElasticNet Logistic",
                    "C": c,
                    "l1_ratio": ratio,
                    "fold": fold,
                    "start": dates.iloc[te].min().date(),
                    "end": dates.iloc[te].max().date(),
                    "n_train": len(tr),
                    "n_test": len(te),
                    "positive_rate_test": float(y.iloc[te].mean()),
                    "auc": auc,
                    "average_precision": ap,
                    "balanced_accuracy": bacc,
                }
            )
        if fold_scores and np.mean(fold_scores) > best_score:
            best_score = float(np.mean(fold_scores))
            best = (c, ratio)
    if best is None:
        raise RuntimeError("No valid Elastic Net split")

    c, ratio = best
    final = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=ratio,
                    C=c,
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=42,
                ),
            ),
        ]
    )
    final.fit(x, y)
    coefs = final.named_steps["clf"].coef_[0]
    coef_df = pd.DataFrame(
        {
            "feature": x.columns,
            "feature_label": [FEATURE_LABEL.get(f, f) for f in x.columns],
            "group": [FEATURE_GROUP.get(f, "기타") for f in x.columns],
            "coef_standardized": coefs,
            "abs_coef": np.abs(coefs),
            "selected": np.abs(coefs) > 1e-8,
        }
    ).sort_values("abs_coef", ascending=False)
    metrics = pd.DataFrame(rows)
    metrics_best = metrics[(metrics["C"] == c) & (metrics["l1_ratio"] == ratio)].copy()
    meta = {
        "best_C": c,
        "best_l1_ratio": ratio,
        "cv_auc_mean": float(metrics_best["auc"].mean()),
        "cv_auc_std": float(metrics_best["auc"].std(ddof=1)),
        "cv_ap_mean": float(metrics_best["average_precision"].mean()),
        "cv_balanced_accuracy_mean": float(metrics_best["balanced_accuracy"].mean()),
        "selected_features": int(coef_df["selected"].sum()),
        "sentiment_selected": int(coef_df.query("group == '감성' and selected").shape[0]),
        "market_selected": int(coef_df.query("group == '시장/변동성' and selected").shape[0]),
    }
    return coef_df, metrics, meta


def evaluate_rf_permutation(x: pd.DataFrame, y: pd.Series, dates: pd.Series, n_splits: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    metric_rows = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold, (tr, te) in enumerate(tscv.split(x), 1):
        if y.iloc[tr].nunique() < 2 or y.iloc[te].nunique() < 2:
            continue
        rf = RandomForestClassifier(
            n_estimators=260,
            max_depth=5,
            min_samples_leaf=30,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42 + fold,
            n_jobs=-1,
        )
        rf.fit(x.iloc[tr], y.iloc[tr])
        prob = rf.predict_proba(x.iloc[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        auc = roc_auc_score(y.iloc[te], prob)
        ap = average_precision_score(y.iloc[te], prob)
        bacc = balanced_accuracy_score(y.iloc[te], pred)
        metric_rows.append(
            {
                "model": "Random Forest",
                "fold": fold,
                "start": dates.iloc[te].min().date(),
                "end": dates.iloc[te].max().date(),
                "n_train": len(tr),
                "n_test": len(te),
                "positive_rate_test": float(y.iloc[te].mean()),
                "auc": auc,
                "average_precision": ap,
                "balanced_accuracy": bacc,
            }
        )
        perm = permutation_importance(
            rf,
            x.iloc[te],
            y.iloc[te],
            scoring="roc_auc",
            n_repeats=7,
            random_state=20260718 + fold,
            n_jobs=-1,
        )
        for feature, imp, std in zip(x.columns, perm.importances_mean, perm.importances_std):
            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "feature_label": FEATURE_LABEL.get(feature, feature),
                    "group": FEATURE_GROUP.get(feature, "기타"),
                    "importance_auc_drop": imp,
                    "importance_std": std,
                }
            )
    imp_df = pd.DataFrame(rows)
    agg = (
        imp_df.groupby(["feature", "feature_label", "group"], as_index=False)
        .agg(
            importance_auc_drop=("importance_auc_drop", "mean"),
            importance_std=("importance_auc_drop", "std"),
            positive_fold_rate=("importance_auc_drop", lambda s: float((s > 0).mean())),
        )
        .sort_values("importance_auc_drop", ascending=False)
    )
    metrics = pd.DataFrame(metric_rows)
    meta = {
        "cv_auc_mean": float(metrics["auc"].mean()),
        "cv_auc_std": float(metrics["auc"].std(ddof=1)),
        "cv_ap_mean": float(metrics["average_precision"].mean()),
        "cv_balanced_accuracy_mean": float(metrics["balanced_accuracy"].mean()),
    }
    return agg, metrics, meta


def set_korean_font() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    for name in ["AppleGothic", "NanumGothic", "Malgun Gothic", "Arial Unicode MS"]:
        plt.rcParams["font.family"] = name
        try:
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, "감성")
            plt.close(fig)
            return
        except Exception:
            plt.close("all")


def save_fig(fig: plt.Figure, filename: str) -> None:
    for root in [REPO_OUT, PAPER_OUT]:
        fig.savefig(root / "figures" / filename, dpi=220, bbox_inches="tight")


def plot_importance(elastic: pd.DataFrame, rf: pd.DataFrame, target_name: str, suffix: str) -> None:
    set_korean_font()
    top = elastic.head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = top["group"].map({"감성": "#d93232", "시장/변동성": "#2f6fd6"}).fillna("#999999")
    ax.barh(top["feature_label"], top["coef_standardized"], color=colors)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_xlabel("표준화 계수")
    ax.grid(axis="x", alpha=0.25)
    handles = [
        plt.Line2D([0], [0], color="#d93232", lw=7, label="감성 피처"),
        plt.Line2D([0], [0], color="#2f6fd6", lw=7, label="시장/변동성 피처"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    ax.text(
        0,
        -0.16,
        f"{target_name}: K-FGI 제외 구성요소 분석. 양(+)의 계수는 해당 위험/성공 확률을 높이는 방향, 음(-)의 계수는 낮추는 방향이다.",
        transform=ax.transAxes,
        color="#555555",
        fontsize=10,
    )
    save_fig(fig, f"24_ElasticNet_{suffix}_계수.png")
    plt.close(fig)

    top = rf.head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = top["group"].map({"감성": "#d93232", "시장/변동성": "#2f6fd6"}).fillna("#999999")
    ax.barh(top["feature_label"], top["importance_auc_drop"], color=colors)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_xlabel("Permutation importance: AUC 감소폭")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(handles=handles, loc="lower right", frameon=True)
    ax.text(
        0,
        -0.16,
        f"{target_name}: K-FGI 제외 구성요소 분석. 값이 클수록 해당 피처를 섞었을 때 OOS AUC가 더 많이 하락한다.",
        transform=ax.transAxes,
        color="#555555",
        fontsize=10,
    )
    save_fig(fig, f"25_RF_{suffix}_순열중요도.png")
    plt.close(fig)


def md_table(df: pd.DataFrame, cols: list[str], n: int = 12) -> str:
    d = df[cols].head(n).copy()
    for col in d.columns:
        d[col] = d[col].map(lambda x: "" if pd.isna(x) else (f"{x:.4g}" if isinstance(x, float) else str(x)))
    rows = ["| " + " | ".join(d.columns) + " |", "| " + " | ".join(["---"] * len(d.columns)) + " |"]
    for _, row in d.iterrows():
        rows.append("| " + " | ".join(row.astype(str)) + " |")
    return "\n".join(rows)


def summarize_target(target_col: str, spec: dict, elastic: pd.DataFrame, emeta: dict, rf: pd.DataFrame, rfmeta: dict, n: int, pos_rate: float) -> str:
    sent_elastic = elastic[elastic["group"] == "감성"].head(8)
    sent_rf = rf[rf["group"] == "감성"].head(8)
    top_el = elastic.iloc[0]
    top_rf = rf.iloc[0]
    return f"""## {spec['label']}

- 표본 수: {n:,}, positive rate: {pos_rate:.1%}.
- Elastic Net CV AUC: {emeta['cv_auc_mean']:.3f} ± {emeta['cv_auc_std']:.3f}, AP: {emeta['cv_ap_mean']:.3f}, balanced accuracy: {emeta['cv_balanced_accuracy_mean']:.3f}.
- Elastic Net 선택 피처: 전체 {emeta['selected_features']}개, 감성 {emeta['sentiment_selected']}개, 시장/변동성 {emeta['market_selected']}개.
- Random Forest permutation CV AUC: {rfmeta['cv_auc_mean']:.3f} ± {rfmeta['cv_auc_std']:.3f}, AP: {rfmeta['cv_ap_mean']:.3f}, balanced accuracy: {rfmeta['cv_balanced_accuracy_mean']:.3f}.
- Elastic Net 최상위 피처: `{top_el['feature']}` ({top_el['feature_label']}), 표준화 계수 {top_el['coef_standardized']:.4f}.
- RF 최상위 피처: `{top_rf['feature']}` ({top_rf['feature_label']}), AUC 감소폭 {top_rf['importance_auc_drop']:.4f}.

### Elastic Net 상위 계수

{md_table(elastic, ['feature','feature_label','group','coef_standardized','abs_coef','selected'], n=12)}

### Elastic Net 감성 피처

{md_table(sent_elastic, ['feature','feature_label','coef_standardized','abs_coef','selected'], n=8)}

### Random Forest permutation importance

{md_table(rf, ['feature','feature_label','group','importance_auc_drop','importance_std','positive_fold_rate'], n=12)}

### Random Forest 감성 피처

{md_table(sent_rf, ['feature','feature_label','importance_auc_drop','importance_std','positive_fold_rate'], n=8)}
"""


def build_revision_text(target_summaries: str, target_overview: pd.DataFrame) -> str:
    return f"""# Interpretable Feature Importance and Paper Text Revision

## 실행 목적

기존 LightGBM gain 기반 feature importance는 예측모델 중심의 해석에 가깝다. 본 연구의 수정된 목표는 수익률 예측이 아니라 하락위험 방어이므로, 피처 중요도도 `수익률 예측 성능`보다 `하락위험 및 방어성공 설명력`을 기준으로 다시 산출하였다.

중요한 수정 사항은 **완성된 K-FGI와 K-FGI 구성 피처를 같은 중요도 모델에 동시에 넣지 않는 것**이다. K-FGI는 이미 시장 하위지표, 감성 피처, 변동성 피처가 결합된 최종 지표이므로, 구성요소별 중요도 분석에 K-FGI 자체를 함께 넣으면 중복 계산 문제가 발생한다. 따라서 본 문서의 Elastic Net 및 Random Forest 중요도 분석에서는 `K_FGI`를 제외하고 하위 시장/변동성 피처와 감성 피처만 투입하였다.

본 문서에서는 LightGBM을 메인 설명도구에서 내리고 다음 두 모델을 사용한다.

- Elastic Net Logistic Regression: 표준화 계수의 방향과 크기를 통해 피처가 하락위험 또는 방어성공 확률에 미치는 방향을 해석한다.
- Random Forest + permutation importance: 비선형 관계를 허용하되, 검증 구간에서 피처를 섞었을 때 AUC가 얼마나 하락하는지로 중요도를 계산한다.

## 타깃 정의

{md_table(target_overview, ['target','label','n','positive_rate','elastic_auc','rf_auc','elastic_top_feature','rf_top_feature'], n=10)}

## 결과 요약

{target_summaries}

## 논문 4.3 실험 설계 수정안

### 4.3 실험 설계

본 연구의 구현 환경은 Python 3.10 기반이며, EGARCH 추정에는 `arch`, 피처 엔지니어링 및 전처리에는 `pandas`와 `numpy`, 감성 분석에는 KR-FinBERT-SC 기반 `transformers`, 하락위험 설명모형에는 `scikit-learn`을 활용하였다. 기존 예측 성능 중심의 LightGBM 분석은 보조 robustness로만 사용하고, 본문에서는 Elastic Net Logistic Regression과 Random Forest permutation importance를 통해 감성 피처의 하방위험 설명력을 검증하였다.

### 4.3.1 TimeSeriesSplit 교차 검증

시계열 자료의 순서를 보존하기 위해 TimeSeriesSplit 5-Fold 교차 검증을 적용하였다. 각 fold에서는 과거 구간만을 학습에 사용하고 이후 구간을 검증하여 look-ahead bias를 방지하였다. Elastic Net의 `C`와 `l1_ratio`는 TimeSeriesSplit 검증 AUC가 가장 높은 조합을 선택하였고, Random Forest의 permutation importance도 각 fold의 검증 구간에서만 계산하였다.

### 4.3.2 Out-of-Sample 및 Rolling 검증

단일 2025년 OOS 검증은 강세장 환경의 영향을 크게 받을 수 있으므로, 본 연구는 10개년 전체 구간에 대해 1년 rolling OOS 검증을 병행하였다. 1년 rolling window는 총 2,254개이며, K-FGI는 Buy & Hold 대비 Sharpe 우위 비율은 39.6%에 그쳤으나 MDD 개선 비율은 83.1%로 나타났다. 따라서 본 연구의 검증 초점은 수익률 예측 우위가 아니라 하방위험 완화의 반복성에 있다.

### 4.3.3 거래비용 스트레스 테스트

거래비용은 0bp, 5bp, 10bp, 15bp, 20bp로 변화시키며 성과 민감도를 확인하였다. 15bp를 기준 가정으로 채택한 이유는 전략 성과를 보수적으로 평가하기 위함이다. 15bp 기준 K-FGI with sentiment의 누적수익률은 54.4%, Sharpe는 0.295, MDD는 -37.1%로 나타났다.

### 4.3.4 해석가능 피처 중요도 모형

피처 중요도 분석의 종속변수는 단순 수익률이 아니라 `5거래일 하락위험`, `5거래일 -2% 급락위험`, `하락일 방어성공`으로 정의하였다. Elastic Net Logistic Regression은 표준화 계수와 선택 여부를 통해 피처의 방향성을 확인하기 위해 사용했고, Random Forest permutation importance는 비선형 구조에서도 해당 피처가 OOS 분류 성능에 기여하는지 확인하기 위해 사용하였다. 단, K-FGI는 감성 피처를 포함해 산출된 최종 지표이므로, 구성요소별 기여도 분석에서는 K-FGI 자체를 제외하였다. 완성된 K-FGI의 유효성은 별도의 전략 성과 비교와 감성 포함/제외 ablation을 통해 검증하였다.

## 논문 4.4 감성 피처 기여도 검증 수정안

### 4.4 감성 피처 기여도 검증

본 절에서는 감성 피처의 역할을 수익률 방향 예측이 아니라 하락위험 식별 및 방어성공 설명력 관점에서 검증하였다. 첫째, 감성 피처 포함/제외 ablation을 통해 K-FGI 전략 성과 변화를 비교하였다. 둘째, Elastic Net Logistic Regression과 Random Forest permutation importance를 통해 감성 피처가 하락위험 및 방어성공 타깃에서 선택되거나 중요하게 작동하는지 확인하였다. 셋째, 공포 압력과 고변동성의 상호작용 및 감성위험 overlay 실험을 통해 감성 피처가 위험 국면에서 노출 조절을 보완하는지 분석하였다.

### 4.4.1 Ablation Study 수정안

감성 피처 전체를 제거하면 K-FGI의 누적수익률은 54.4%에서 17.8%로 하락했고, Sharpe는 0.295에서 0.113으로 낮아졌으며, MDD는 -37.1%에서 -45.3%로 악화되었다. 반면 단일 감성 피처만 사용할 경우 sent_norm_only의 누적수익률은 17.3%, negative_only는 15.9%, dispersion_only는 28.8%에 그쳤다. 이는 감성 정보가 단일 방향성 점수보다 감성 강도, 의견 불일치, 부정 감성, 이동평균 등을 결합한 형태에서 더 유용함을 시사한다.

### 4.4.2 해석가능 모델 기반 피처 중요도 수정안

Elastic Net과 Random Forest 분석은 감성 피처가 단독 수익률 예측 변수라기보다 하락위험 및 방어성공 조건을 설명하는 보조 신호임을 보여준다. 특히 `panic_pressure`, `negative_attention`, `disagreement_attention`, `heat_attention`, `sentiment_risk_composite`와 같이 댓글 관심도와 감성 강도 또는 의견 불일치를 결합한 피처는 단순 감성 평균보다 논문 목적에 더 적합하다. 따라서 본 연구에서는 LightGBM gain importance를 본문 핵심 근거로 사용하지 않고, K-FGI를 제외한 구성요소 수준에서 Elastic Net 표준화 계수와 Random Forest permutation importance를 통해 감성 피처의 방향성과 OOS 기여도를 제시한다.

붙여넣기용 문장:

> K-FGI는 감성 피처를 포함하여 산출된 종합 지표이므로, 구성요소별 기여도 분석에서는 K-FGI 자체를 제외하고 하위 시장·변동성·감성 피처만 투입하였다. 이를 통해 완성 지표의 중요도와 구성 피처의 중요도가 중복 계산되는 문제를 방지하였다. 완성된 K-FGI의 효과는 감성 포함 K-FGI와 감성 제외 K-FGI의 전략 성과 비교를 통해 별도로 검증하였다.

### 4.4.3 조건부 감성 검정 수정안

전체 표본에서 감성 포함 전략은 감성 제외 전략 대비 평균 +1.08bp/day의 추가 수익을 보였고(t=2.377, p=0.0175), 감성위험 overlay 실험에서는 댓글 열기 상위 20%에서 노출을 0.7배로 축소할 때 감성 제외 전략 대비 하락일 방어가 +2.54bp/day 개선되었다(q=0.0050). 이는 감성 피처가 모든 시점에서 동일하게 수익률을 예측하기보다, 투자자 관심과 불안이 확대되는 구간에서 노출 조절 신호로 작동함을 시사한다.

## 논문 4.6 파라미터 설정 근거 수정안

### 4.6 파라미터 설정 근거

파라미터 선택은 단순히 Sharpe가 가장 높은 조합을 채택하는 방식이 아니라, 하방위험 방어라는 연구 목적과 해석 가능성, 민감도 분석의 안정성을 함께 고려하였다.

### 4.6.1 EGARCH 안정화 파라미터

EGARCH는 최소 추정 표본을 252거래일로 설정하였다. 이는 1년 거래일에 해당하며, 초기 추정값 폭발을 완화하기 위한 보수적 설정이다. 비정상적으로 큰 추정값은 상한 처리하거나 rolling volatility로 대체하였다. 안정화 이후 fallback rate는 0.30%, capped rate는 0.48%, p99 volatility는 0.0235로 나타났다.

### 4.6.2 K-FGI 국면 임계값

Fear/Greed 기준은 20/80, 25/65, 30/70 조합을 비교하였다. 최상위 Sharpe 조합은 20/80 및 multiplier 0.7-1.6이었으나, 기준 조합 25/65와 0.5-1.6 역시 성과가 크게 무너지지 않았다. 따라서 25/65는 유일한 최적값이라기보다 해석 가능성과 기존 Fear/Greed 문헌 관행을 반영한 기준값으로 제시하고, threshold robustness를 함께 보고한다.

### 4.6.3 거래비용 기준

거래비용은 15bp를 기본값으로 사용하였다. 0bp에서는 K-FGI with sentiment의 누적수익률이 192.2%, Sharpe가 0.728이었으나, 15bp에서는 누적수익률 54.4%, Sharpe 0.295로 낮아졌다. 이는 전략이 거래비용에 민감함을 의미하므로, 논문에서는 15bp 기준을 보수적 검증으로 사용한다.

### 4.6.4 감성위험 overlay 파라미터

감성위험 overlay는 최종 메인 모델이라기보다 개선 실험으로 제시한다. `heat_attention` 상위 20%에서 노출을 0.7배로 축소하는 후보는 전체 누적수익률 43.3%, Sharpe 0.255를 유지하면서 감성 제외 전략 대비 하락일 방어를 +2.54bp/day 개선하였다(q=0.0050). 이는 감성 피처가 K-FGI 점수에 단순 가산되는 것보다 위험 국면에서 노출을 제한하는 방식으로 활용될 때 더 명확한 하방 방어 근거를 제공함을 보여준다.

## 생성된 파일

- `tables/elastic_net_feature_importance_*.csv`
- `tables/random_forest_permutation_importance_*.csv`
- `tables/interpretable_feature_importance_metrics.csv`
- `figures/24_ElasticNet_*_계수.png`
- `figures/25_RF_*_순열중요도.png`
"""


def save_outputs(outputs: dict[str, pd.DataFrame], md: str) -> None:
    for root in [REPO_OUT, PAPER_OUT]:
        for name, df in outputs.items():
            df.to_csv(root / "tables" / name, index=False, encoding="utf-8-sig")
        (root / "FEATURE_IMPORTANCE_MODEL_REVISION.md").write_text(md, encoding="utf-8")
    docs = CODE_ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "FEATURE_IMPORTANCE_MODEL_REVISION.md").write_text(md, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_data()
    outputs: dict[str, pd.DataFrame] = {}
    overview_rows = []
    sections = []
    primary_for_plot = "target_5d_large_drop"

    for target_col, spec in TARGETS.items():
        x, y, dates = modeling_frame(df, target_col, spec["filter_down_days"])
        coef_df, en_metrics, en_meta = evaluate_elastic_net(x, y, dates)
        rf_df, rf_metrics, rf_meta = evaluate_rf_permutation(x, y, dates)
        suffix = target_col.replace("target_", "").replace("defense_", "defense_")
        outputs[f"elastic_net_feature_importance_{target_col}.csv"] = coef_df
        outputs[f"elastic_net_cv_metrics_{target_col}.csv"] = en_metrics
        outputs[f"random_forest_permutation_importance_{target_col}.csv"] = rf_df
        outputs[f"random_forest_cv_metrics_{target_col}.csv"] = rf_metrics
        overview_rows.append(
            {
                "target": target_col,
                "label": spec["label"],
                "n": len(y),
                "positive_rate": float(y.mean()),
                "elastic_auc": en_meta["cv_auc_mean"],
                "rf_auc": rf_meta["cv_auc_mean"],
                "elastic_top_feature": coef_df.iloc[0]["feature"],
                "rf_top_feature": rf_df.iloc[0]["feature"],
                "elastic_selected_features": en_meta["selected_features"],
                "elastic_sentiment_selected": en_meta["sentiment_selected"],
            }
        )
        sections.append(summarize_target(target_col, spec, coef_df, en_meta, rf_df, rf_meta, len(y), float(y.mean())))
        if target_col in [primary_for_plot, "defense_success_down_day"]:
            plot_importance(coef_df, rf_df, spec["label"], suffix)

    overview = pd.DataFrame(overview_rows)
    outputs["interpretable_feature_importance_metrics.csv"] = overview
    md = build_revision_text("\n".join(sections), overview)
    save_outputs(outputs, md)
    print(md)


if __name__ == "__main__":
    main()
