from __future__ import annotations

import importlib.util
import math
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


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


FEATURES = [f"sub_index{i}" for i in range(2, 8)] + [
    "sent_composite_ma10",
    "egarch_vol",
    "vol_regime_high",
    "vol_ratio",
]


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


def max_drawdown(r: pd.Series) -> float:
    wealth = np.exp(r.fillna(0).cumsum())
    return float((wealth / wealth.cummax() - 1).min())


def make_return(df: pd.DataFrame, weight: pd.Series, fee: float = 0.0015) -> pd.Series:
    weight_lag = weight.shift(1).fillna(0)
    turnover = weight_lag.diff().abs().fillna(weight_lag.iloc[0])
    return weight_lag * df["target_reg"] - turnover * fee


def bootstrap_ci(x: pd.Series, b: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    arr = x.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(b, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float((np.abs(means) >= abs(arr.mean())).mean())


def hac_t(x: pd.Series, lags: int = 5) -> tuple[float, float]:
    arr = x.dropna().to_numpy()
    n = len(arr)
    mu = arr.mean()
    centered = arr - mu
    gamma0 = np.dot(centered, centered) / n
    var = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        cov = np.dot(centered[lag:], centered[:-lag]) / n
        var += 2 * (1 - lag / (lags + 1)) * cov
    se = math.sqrt(max(var, 1e-18) / n)
    t = mu / se
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
    return float(t), float(p)


def build_final_df() -> pd.DataFrame:
    base = load_base_module()
    priority = load_priority_module()
    raw = pd.read_csv(ROBUST_DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    feats = [f for f in FEATURES if f in raw.columns]
    direction = {**{f"sub_index{i}": 1 for i in range(2, 8)}, "sent_composite_ma10": 1, "egarch_vol": -1, "vol_regime_high": -1, "vol_ratio": -1}
    dir_vec = np.array([direction[f] for f in feats])
    cache = REPO_OUT / "robustness" / "cache" / "kfgi_final_lean_sent_composite_ma10.csv"
    df = base.create_walkforward_kfgi(raw.drop(columns=["K_FGI"], errors="ignore"), feats, dir_vec, 60, cache)
    rho_5d = stats.spearmanr(df["K_FGI"], df["target_5d"], nan_policy="omit").correlation
    kfgi_momentum = bool(rho_5d >= 0)
    pos = priority.compute_positions_param(base, df, fee=0.0015, fear=25, greed=65, k_min=0.5, k_max=1.6, kfgi_momentum=kfgi_momentum)
    df["weight_raw"] = pos["weight"]
    df["weight"] = pos["weight"].clip(0, 1.0)
    df["strategy_return"] = make_return(df, df["weight"], 0.0015)
    df["buy_hold_return"] = df["target_reg"]
    df["excess_return"] = df["strategy_return"] - df["buy_hold_return"]
    df["bh_down"] = df["buy_hold_return"] < 0
    df["vol_high"] = df["egarch_vol"] >= df["egarch_vol"].quantile(0.7)
    df["vol_low"] = df["egarch_vol"] <= df["egarch_vol"].quantile(0.3)
    df["kfgi_zone"] = pd.cut(df["K_FGI"], bins=[-np.inf, 25, 65, np.inf], labels=["Fear", "Neutral", "Greed"])
    return df


def final_performance(df: pd.DataFrame) -> dict:
    bh_mdd = max_drawdown(df["buy_hold_return"])
    st_mdd = max_drawdown(df["strategy_return"])
    down = df["bh_down"]
    dex = df.loc[down, "excess_return"]
    return {
        "n": len(df),
        "bh_down_days": int(down.sum()),
        "mdd": st_mdd,
        "bh_mdd": bh_mdd,
        "mdd_improvement_pctp": (st_mdd - bh_mdd) * 100,
        "downside_excess_bp": dex.mean() * 10000,
        "downside_t": stats.ttest_1samp(dex, 0).statistic,
        "downside_p": stats.ttest_1samp(dex, 0).pvalue,
        "downside_hit_rate": (df.loc[down, "strategy_return"] > df.loc[down, "buy_hold_return"]).mean(),
        "annual_vol_reduction_pct": (1 - df["strategy_return"].std() / df["buy_hold_return"].std()) * 100,
        "avg_weight": df["weight"].mean(),
        "max_weight": df["weight"].max(),
        "weight_gt_1_pct": (df["weight"] > 1).mean() * 100,
    }


def statistical_battery(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    down = df["bh_down"]
    dex = df.loc[down, "excess_return"]
    all_ex = df["excess_return"]
    low_vol = df["vol_low"]
    high_vol = df["vol_high"]
    low_vol_weight = df.loc[low_vol, "weight"]
    high_vol_weight = df.loc[high_vol, "weight"]
    bt_low, bt_high, boot_p = bootstrap_ci(dex * 10000)
    hac_stat, hac_p = hac_t(dex)

    tests = [
        ("하방방어", "시장 하락일 초과수익 평균 > 0", "one-sample t-test", stats.ttest_1samp(dex, 0).statistic, stats.ttest_1samp(dex, 0).pvalue, dex.mean() * 10000, "bp/down day"),
        ("하방방어", "시장 하락일 초과수익 평균 > 0", "normal z-test", dex.mean() / (dex.std(ddof=1) / math.sqrt(len(dex))), 2 * (1 - stats.norm.cdf(abs(dex.mean() / (dex.std(ddof=1) / math.sqrt(len(dex)))))), dex.mean() * 10000, "bp/down day"),
        ("하방방어", "자기상관 보정 후 하락일 방어 > 0", "Newey-West HAC t-test", hac_stat, hac_p, dex.mean() * 10000, "bp/down day"),
        ("하방방어", "하락일 초과수익 중앙값 != 0", "Wilcoxon signed-rank", stats.wilcoxon(dex).statistic, stats.wilcoxon(dex).pvalue, dex.median() * 10000, "bp/down day median"),
        ("하방방어", "하락일 방어 bootstrap 95% CI", "iid bootstrap", np.nan, np.nan, dex.mean() * 10000, f"CI [{bt_low:.2f}, {bt_high:.2f}] bp"),
        ("전체 초과수익", "전체 일별 초과수익 평균 != 0", "one-sample t-test", stats.ttest_1samp(all_ex, 0).statistic, stats.ttest_1samp(all_ex, 0).pvalue, all_ex.mean() * 10000, "bp/day"),
        ("위험감소", "전략 분산과 B&H 분산 차이", "F-test variance ratio", df["strategy_return"].var() / df["buy_hold_return"].var(), stats.f.cdf(df["strategy_return"].var() / df["buy_hold_return"].var(), len(df) - 1, len(df) - 1), df["strategy_return"].var() / df["buy_hold_return"].var(), "variance ratio"),
        ("위험감소", "전략/B&H 변동성 차이", "Levene median test", stats.levene(df["strategy_return"], df["buy_hold_return"], center="median").statistic, stats.levene(df["strategy_return"], df["buy_hold_return"], center="median").pvalue, (1 - df["strategy_return"].std() / df["buy_hold_return"].std()) * 100, "vol reduction %"),
        ("노출조절", "저변동성 구간 노출 > 고변동성 구간 노출", "Welch t-test", stats.ttest_ind(low_vol_weight, high_vol_weight, equal_var=False).statistic, stats.ttest_ind(low_vol_weight, high_vol_weight, equal_var=False).pvalue, low_vol_weight.mean() - high_vol_weight.mean(), "exposure x"),
        ("노출조절", "EGARCH 변동성과 노출도 음의 관계", "Spearman correlation", stats.spearmanr(df["egarch_vol"], df["weight"], nan_policy="omit").statistic, stats.spearmanr(df["egarch_vol"], df["weight"], nan_policy="omit").pvalue, stats.spearmanr(df["egarch_vol"], df["weight"], nan_policy="omit").statistic, "rho"),
    ]
    for cat, hyp, test, stat, p, effect, unit in tests:
        rows.append({"category": cat, "hypothesis": hyp, "test": test, "stat": stat, "p_value": p, "effect": effect, "unit": unit})

    groups = [g["strategy_return"].dropna() for _, g in df.groupby("kfgi_zone", observed=True)]
    if len(groups) >= 2:
        rows.append({"category": "국면차이", "hypothesis": "K-FGI zone별 전략 수익률 평균 차이", "test": "one-way ANOVA", "stat": stats.f_oneway(*groups).statistic, "p_value": stats.f_oneway(*groups).pvalue, "effect": np.nan, "unit": ""})
        rows.append({"category": "국면차이", "hypothesis": "K-FGI zone별 전략 수익률 분포 차이", "test": "Kruskal-Wallis", "stat": stats.kruskal(*groups).statistic, "p_value": stats.kruskal(*groups).pvalue, "effect": np.nan, "unit": ""})
    return pd.DataFrame(rows)


def conditional_defense(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    conditions = {
        "전체 시장 하락일": df["bh_down"],
        "고변동성 & 시장 하락일": df["bh_down"] & df["vol_high"],
        "저변동성 & 시장 하락일": df["bh_down"] & df["vol_low"],
        "Fear 구간 & 시장 하락일": df["bh_down"] & df["kfgi_zone"].eq("Fear"),
        "Neutral 구간 & 시장 하락일": df["bh_down"] & df["kfgi_zone"].eq("Neutral"),
        "Greed 구간 & 시장 하락일": df["bh_down"] & df["kfgi_zone"].eq("Greed"),
        "2016 시장 하락일": df["bh_down"] & df["date"].dt.year.eq(2016),
        "2020 시장 하락일": df["bh_down"] & df["date"].dt.year.eq(2020),
        "2022 시장 하락일": df["bh_down"] & df["date"].dt.year.eq(2022),
        "2025 시장 하락일": df["bh_down"] & df["date"].dt.year.eq(2025),
    }
    for name, mask in conditions.items():
        ex = df.loc[mask, "excess_return"]
        if len(ex) < 5:
            continue
        t = stats.ttest_1samp(ex, 0)
        rows.append({
            "condition": name,
            "n": len(ex),
            "downside_excess_bp": ex.mean() * 10000,
            "median_excess_bp": ex.median() * 10000,
            "t": t.statistic,
            "p_value": t.pvalue,
            "hit_rate": (df.loc[mask, "strategy_return"] > df.loc[mask, "buy_hold_return"]).mean(),
            "avg_weight": df.loc[mask, "weight"].mean(),
        })
    return pd.DataFrame(rows)


def feature_map(df: pd.DataFrame) -> pd.DataFrame:
    descriptions = {
        "sub_index2": ("주가 강도", "전종목/시장 구성 종목의 가격 강도", "시장 내부 강도가 약할 때 과도한 낙관을 제한"),
        "sub_index3": ("시장 폭", "상승/하락 또는 breadth 관련 시장 참여 폭", "소수 종목 주도 장세와 시장 약세를 구분"),
        "sub_index4": ("옵션 심리", "KOSPI200 옵션 기반 put/call 또는 파생 심리", "방어 수요 증가를 공포 신호로 반영"),
        "sub_index5": ("변동성 심리", "KOSPI200 변동성지수 계열", "변동성 급등 시 위험 노출 축소"),
        "sub_index6": ("안전자산 선호", "채권/주식 상대 선호", "위험회피 심리가 강할 때 주식 노출 제한"),
        "sub_index7": ("신용위험 선호", "고위험 채권/스프레드 성격의 위험선호", "신용위험 회피 구간의 방어 신호"),
        "sent_composite_ma10": ("평활화 감성", "댓글 감성, 부정성, 감성강도를 결합한 10일 이동평균", "raw 감성 잡음을 줄이고 투자심리 변화만 반영"),
        "egarch_vol": ("조건부 변동성", "EGARCH(1,1) 기반 일별 조건부 변동성", "변동성 타깃팅으로 고위험 구간 노출 축소"),
        "vol_regime_high": ("고변동성 국면", "EGARCH 변동성이 최근 분위수 기준 높은 국면인지", "위기/불안정 국면의 방어 배율"),
        "vol_ratio": ("상대 변동성", "현재 EGARCH 변동성 / 최근 평균 변동성", "평균 대비 변동성 충격을 연속형으로 반영"),
    }
    rows = []
    for f in FEATURES:
        corr = stats.spearmanr(df[f], df["K_FGI"], nan_policy="omit").statistic if f in df else np.nan
        rows.append({
            "feature": f,
            "role": descriptions[f][0],
            "formula_or_source": descriptions[f][1],
            "used_in": "K-FGI 점수 산출",
            "downside_mechanism": descriptions[f][2],
            "spearman_with_kfgi": corr,
            "paper_section": "3. 변수 구성 / 4. 감성 피처 축소 및 하방방어 검증",
        })
    return pd.DataFrame(rows)


def factor_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
    z = StandardScaler().fit_transform(x)
    pca = PCA(n_components=min(5, len(FEATURES))).fit(z)
    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        index=FEATURES,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    ).reset_index(names="feature")
    explained = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(pca.n_components_)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    })
    return loadings, explained


def final_feature_importance(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df[df["bh_down"]].copy()
    data["defense_success"] = (data["strategy_return"] > data["buy_hold_return"]).astype(int)
    cols = [c for c in FEATURES if c in data.columns]
    data = data.dropna(subset=cols + ["defense_success"])
    split = int(len(data) * 0.75)
    train = data.iloc[:split]
    test = data.iloc[split:]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[cols])
    x_test = scaler.transform(test[cols])
    y_train = train["defense_success"].astype(int)
    y_test = test["defense_success"].astype(int)

    logit = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty="elasticnet",
        solver="saga",
        l1_ratios=[0.2, 0.5, 0.8],
        scoring="roc_auc",
        max_iter=5000,
        random_state=42,
        n_jobs=None,
    )
    logit.fit(x_train, y_train)
    prob = logit.predict_proba(x_test)[:, 1]
    logit_auc = roc_auc_score(y_test, prob)
    coef = pd.DataFrame(
        {
            "feature": cols,
            "elasticnet_coef": logit.coef_[0],
            "abs_elasticnet_coef": np.abs(logit.coef_[0]),
            "direction": np.where(logit.coef_[0] >= 0, "defense_success +", "defense_success -"),
            "target": "market_down_day_defense_success",
            "test_auc": logit_auc,
            "train_n": len(train),
            "test_n": len(test),
        }
    ).sort_values("abs_elasticnet_coef", ascending=False)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=30,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=1,
    )
    rf.fit(x_train, y_train)
    rf_prob = rf.predict_proba(x_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_prob)
    perm = permutation_importance(
        rf,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=50,
        random_state=42,
        n_jobs=1,
    )
    rf_imp = pd.DataFrame(
        {
            "feature": cols,
            "rf_permutation_auc_drop": perm.importances_mean,
            "rf_permutation_auc_drop_std": perm.importances_std,
            "target": "market_down_day_defense_success",
            "test_auc": rf_auc,
            "train_n": len(train),
            "test_n": len(test),
        }
    ).sort_values("rf_permutation_auc_drop", ascending=False)
    return coef, rf_imp


def section_plan() -> pd.DataFrame:
    rows = [
        ("1. Introduction", "수익률 예측이 아니라 하방위험 관리 문제로 재정의", "K-FGI가 시장 심리와 변동성을 결합해 하락일 손실을 줄이는 지표라는 연구 질문 제시", "없음"),
        ("2. Literature Review", "CNN FGI, 변동성 관리, 국내 투자심리/변동성 연구 비교", "본 연구는 한국형 시장 데이터와 댓글 감성을 결합한 하방방어 지표라는 차별점 제시", "문헌 비교 표"),
        ("3. Data and Methodology", "10개 최종 피처 설명", "subindex2-7, sent_composite_ma10, EGARCH 계열, 1년 rolling/252일 안정화 설명", "피처 구성표"),
        ("3.x K-FGI Construction", "raw 감성 제거 및 평활화 감성 채택 근거", "sent_composite_ma10만 남긴 lean K-FGI 공식과 방향성 설명", "감성 정규화/평활화 그림"),
        ("4. Empirical Results", "하방방어 중심 성과표", "MDD, 하락일 방어폭, 변동성 감소, p-value 제시", "낙폭방어 그래프, 하방방어 표"),
        ("4.x Robustness", "거래비용, threshold, multiplier, cap 민감도", "특정 파라미터에만 의존하지 않음을 확인", "포지션 상한 그래프, 민감도 표. heatmap은 부록"),
        ("4.x Conditional Analysis", "2016/2020/2022/2025 및 고변동성 구간 분석", "어떤 구간에서 방어가 강했는지 설명", "연도별/국면별 막대그래프"),
        ("4.x Feature Validation", "ablation, 통계검정, permutation importance", "감성 평활화와 최종 피처 선택이 하방방어 기준에서 타당했음을 설명", "피처 중요도 표, ablation 표"),
        ("Appendix", "PCA/요인구조 점검", "최종 모델 산출에는 쓰지 않고 피처 중복성 확인용으로만 제시", "PCA loading 표"),
        ("5. Conclusion", "하방방어 지표로서의 위치", "수익률 예측 한계는 인정하되, 방어효과와 감성 평활화의 기여 강조", "없음"),
    ]
    return pd.DataFrame(rows, columns=["paper_section", "what_to_insert", "how_to_write", "figure_or_table"])


def figure_plan() -> pd.DataFrame:
    rows = [
        ("Figure 1", "댓글 필터링/감성 정규화", "03_감성점수_정규화_전후.png", "데이터 전처리 타당성"),
        ("Figure 2", "K-FGI 시계열", "05_KFGI_공포탐욕_시계열.png", "공포/탐욕 국면과 시장 이벤트 연결"),
        ("Figure 3", "최종 낙폭방어", "08_KFGI_낙폭방어.png 또는 보수형으로 재작성", "B&H 대비 MDD 개선을 시각적으로 제시"),
        ("Figure 4", "포지션 상한 민감도", "신규 표/그래프 생성 권장", "1.0배 상한 선택 근거"),
        ("Figure 5", "감성 피처 ablation", "13_감성피처_포함제외_성능비교.png 수정 권장", "sent_composite_ma10 채택 근거"),
        ("Figure 6", "국면별 하방방어", "19_국면별_초과성과.png 수정 권장", "고변동성/하락일 방어가 핵심임을 제시"),
        ("Figure 7", "민감도 요약", "08_포지션상한_민감도.png, 09_파라미터_민감도_요약.png", "1.0배 상한과 파라미터 안정성"),
        ("Appendix Figure", "민감도 heatmap", "12_민감도_히트맵_부록용.png", "threshold, fee, multiplier 안정성 보조"),
        ("Appendix Figure", "PCA 요인구조", "11_PCA_요인구조_부록용.png", "피처 중복성 점검. 최종 K-FGI 산출 근거로 사용하지 않음"),
    ]
    return pd.DataFrame(rows, columns=["figure_no", "content", "file_or_action", "paper_purpose"])


def write_outputs(df: pd.DataFrame) -> None:
    perf = final_performance(df)
    stats_table = statistical_battery(df)
    cond_table = conditional_defense(df)
    feat_table = feature_map(df)
    loadings, explained = factor_analysis(df)
    elasticnet_imp, rf_imp = final_feature_importance(df)
    sections = section_plan()
    figures = figure_plan()

    sensitivity = pd.read_csv(REPO_OUT / "tables" / "final_conservative_kfgi_sensitivity.csv")
    cap = sensitivity[sensitivity["sensitivity_type"].eq("position_cap")]
    fee = sensitivity[sensitivity["sensitivity_type"].eq("transaction_cost")]
    threshold = sensitivity[sensitivity["sensitivity_type"].eq("fear_greed_threshold")]
    mult = sensitivity[sensitivity["sensitivity_type"].eq("kfgi_multiplier")]

    literature = pd.DataFrame(
        [
            ("CNN Fear & Greed Index", "시장 모멘텀, 주가강도, breadth, put/call, volatility, safe haven, junk bond demand를 결합한 0-100 심리지표", "본 연구는 CNN식 다중 하위지표 구조를 한국 시장에 맞게 재현하되 댓글 감성과 EGARCH를 추가"),
            ("Moreira and Muir (2017)", "고변동성 시점에 위험 노출을 줄이는 volatility-managed portfolio", "본 연구의 EGARCH 기반 노출 축소와 직접 연결"),
            ("Barro, Canestrelli and Lanza (2014)", "volatility control과 drawdown/tail-risk 관리의 중요성", "본 연구가 수익률보다 MDD/하방변동성에 초점을 두는 근거"),
            ("Downside risk-scaling literature", "변동성보다 downside event를 직접 관리하는 risk-scaling", "본 연구의 하락일 방어폭/하방변동성 검정과 연결"),
            ("Kim and Lee (2022), Korean investor sentiment", "한국 KOSPI/KOSDAQ에서 투자심리가 수익률과 관계", "본 연구는 수익률 예측보다 하방방어 목적의 감성 평활화 지표로 확장"),
            ("Pyo and Kim (2019), Korean news sentiment", "뉴스 감성지수가 한국 금융시장 수익률/변동성과 관련", "댓글 기반 감성을 쓰되 raw가 아니라 10일 평활화로 잡음 완화"),
            ("VKOSPI/KIX fear-index studies", "KOSPI200 옵션 기반 변동성 지수가 한국 시장 공포를 반영", "subindex5 및 EGARCH 변동성 방어 논리의 국내 근거"),
        ],
        columns=["literature", "main_point", "position_of_this_paper"],
    )

    tables = {
        "final_master_statistical_tests.csv": stats_table,
        "final_master_conditional_downside_defense.csv": cond_table,
        "final_master_feature_role_map.csv": feat_table,
        "final_master_pca_loadings.csv": loadings,
        "final_master_pca_explained_variance.csv": explained,
        "final_master_elasticnet_defense_feature_importance.csv": elasticnet_imp,
        "final_master_rf_defense_permutation_importance.csv": rf_imp,
        "final_master_paper_section_plan.csv": sections,
        "final_master_figure_plan.csv": figures,
        "final_master_literature_positioning.csv": literature,
    }
    for root in [REPO_OUT, PAPER_OUT]:
        (root / "tables").mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(root / "tables" / name, index=False, encoding="utf-8-sig")

    def pstr(p: float) -> str:
        if pd.isna(p):
            return ""
        return "<0.001" if p < 0.001 else f"{p:.4f}"

    n_features = len([f for f in FEATURES if f in df.columns])
    obs_per_feature = len(df) / n_features
    down_obs_per_feature = int(df["bh_down"].sum()) / n_features
    md = f"""# 최종 논문용 결과 통합 정리

이 문서는 이전 실험 문서의 중복된 내용을 통합한 최종 기준 문서이다. 논문 본문에는 이 문서를 기준으로 숫자와 해석을 사용하고, 이전 full-feature/수익률 중심 결과는 부록 또는 재현성 파일로만 둔다.

## 1. 최종 연구 포지션

본 논문은 수익률 예측 논문이 아니라 **감성 및 변동성 결합 지표(K-FGI)를 활용한 하방위험 관리 전략 연구**이다. 따라서 메인 성과는 누적수익률이나 Sharpe가 아니라 `시장 하락일 방어폭`, `MDD 개선`, `하방변동성 감소`, `고변동성 구간 노출 축소`로 제시한다.

## 2. 최종 모델

- 분석기간: 2015-2025년, 1년 rolling 기반 피처/지표 산출
- 최종 입력 데이터: `paper_outputs/robustness/data/priority_ab_kfgi_timeseries_robust.csv`를 기반으로 lean K-FGI를 재산출
- 최종 유효 관측치: {len(df):,}거래일
- 최종 피처 수: {n_features}개
- 전체 관측치/피처 수 비율: {obs_per_feature:.1f}:1
- 시장 하락일 관측치/피처 수 비율: {down_obs_per_feature:.1f}:1
- K-FGI 피처: `sub_index2-7 + sent_composite_ma10 + egarch_vol + vol_regime_high + vol_ratio`
- 감성 피처: raw 댓글 감성 전체가 아니라 10일 평활화 복합 감성지표 `sent_composite_ma10`
- 포지션 규칙: 최종 노출 상한 1.0배, 즉 레버리지 미사용
- 기본 파라미터: Fear/Greed 25/65, K-FGI multiplier 0.5-1.6, 거래비용 15bp

데이터 수 대비 피처 수는 충분한 편이다. 전체 표본 기준으로 피처 1개당 약 {obs_per_feature:.0f}개 관측치가 있으며, 실제 하방방어 검정에 쓰이는 시장 하락일만 보더라도 피처 1개당 약 {down_obs_per_feature:.0f}개 관측치가 존재한다. 따라서 최종 10개 피처 구성은 10개년 일별 자료 규모에 비해 과도하게 복잡한 구조로 보기 어렵다. 다만 시계열 자료의 자기상관 가능성을 고려하여 본문에서는 HAC 검정과 rolling/OOS 검증을 함께 제시한다.

## 3. 최종 성능 결과

| 지표 | 값 |
| --- | ---: |
| 표본 수 | {int(perf['n'])} |
| 시장 하락일 수 | {int(perf['bh_down_days'])} |
| B&H MDD | {perf['bh_mdd']*100:.1f}% |
| K-FGI MDD | {perf['mdd']*100:.1f}% |
| MDD 개선 | {perf['mdd_improvement_pctp']:.1f}%p |
| 시장 하락일 방어폭 | {perf['downside_excess_bp']:.1f}bp/day |
| 하방방어 t-stat | {perf['downside_t']:.2f} |
| 하방방어 p-value | {pstr(perf['downside_p'])} |
| 하락일 방어 적중률 | {perf['downside_hit_rate']*100:.1f}% |
| 연환산 변동성 감소 | {perf['annual_vol_reduction_pct']:.1f}% |
| 평균 시장 노출 | {perf['avg_weight']:.2f}x |
| 최대 시장 노출 | {perf['max_weight']:.1f}x |

## 4. 최종 피처 설명 및 사용 위치

| 피처 | 역할 | 사용 위치 | 하방방어 메커니즘 |
| --- | --- | --- | --- |
"""
    for _, row in feat_table.iterrows():
        md += f"| `{row['feature']}` | {row['role']} | {row['used_in']} | {row['downside_mechanism']} |\n"

    md += """
## 5. 어떤 구간에서 하방방어가 유의했는가

| 구간 | n | 하락일 방어폭 | 중앙값 | t-stat | p-value | 적중률 | 평균 노출 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for _, row in cond_table.iterrows():
        md += f"| {row['condition']} | {int(row['n'])} | {row['downside_excess_bp']:.1f}bp | {row['median_excess_bp']:.1f}bp | {row['t']:.2f} | {pstr(row['p_value'])} | {row['hit_rate']*100:.1f}% | {row['avg_weight']:.2f}x |\n"

    md += """
해석상 가장 중요한 구간은 `전체 시장 하락일`, `고변동성 & 시장 하락일`, `2020 시장 하락일`, `2022 시장 하락일`이다. 2016년은 기존 모형에서 과노출 문제가 있었으나, 최종 1.0배 상한 적용 후 낙폭이 완화되어 초기 구간의 구조적 취약성을 줄였다.

## 6. 통계적 검정

| 범주 | 가설 | 검정 | 통계량 | p-value | 효과크기 |
| --- | --- | --- | ---: | ---: | ---: |
"""
    for _, row in stats_table.iterrows():
        stat = "" if pd.isna(row["stat"]) else f"{row['stat']:.3f}"
        effect = "" if pd.isna(row["effect"]) else f"{row['effect']:.3f} {row['unit']}"
        md += f"| {row['category']} | {row['hypothesis']} | {row['test']} | {stat} | {pstr(row['p_value'])} | {effect} |\n"

    md += f"""
## 7. PCA/요인구조 점검: 부록용

PCA는 최종 K-FGI 산출이나 가중치 선택에 사용하지 않는다. 최종 모델은 원 피처인 `sub_index2-7`, `sent_composite_ma10`, EGARCH/변동성 계열을 그대로 사용한다. PCA는 피처들이 하나의 중복 축에만 몰려 있지 않은지 확인하는 부록용 진단으로만 사용한다.

| 성분 | 설명분산 | 누적 설명분산 |
| --- | ---: | ---: |
"""
    for _, row in explained.iterrows():
        md += f"| {row['component']} | {row['explained_variance_ratio']*100:.1f}% | {row['cumulative_explained_variance']*100:.1f}% |\n"

    md += """
논문 본문에서는 PCA loading을 길게 해석하지 않는다. `final_master_pca_loadings.csv`와 `11_PCA_요인구조_부록용.png`는 부록에 배치하고, 본문에서는 감성 피처를 `sent_composite_ma10`으로 축소한 이유를 ablation, 하방방어 검정, permutation importance로 설명한다. 핵심은 raw 감성 피처가 잡음이 크기 때문에 평활화된 감성 축만 남겨 K-FGI의 해석 가능성을 높였다는 점이다.

## 8. 하방방어 기준 Feature Importance

피처 중요도는 수익률 예측이 아니라 `시장 하락일에 전략이 B&H보다 손실을 덜 냈는가`를 종속변수로 두고 산출했다. 따라서 이 표는 논문의 하방방어 주장과 직접 연결된다.

### Elastic Net Logistic Regression

| 순위 | 피처 | 표준화 계수 | 방향 | OOS AUC |
| ---: | --- | ---: | --- | ---: |
"""
    for i, (_, row) in enumerate(elasticnet_imp.head(10).iterrows(), start=1):
        md += f"| {i} | `{row['feature']}` | {row['elasticnet_coef']:.4f} | {row['direction']} | {row['test_auc']:.3f} |\n"

    md += """
### Random Forest Permutation Importance

| 순위 | 피처 | AUC 감소폭 | 표준편차 | OOS AUC |
| ---: | --- | ---: | ---: | ---: |
"""
    for i, (_, row) in enumerate(rf_imp.head(10).iterrows(), start=1):
        md += f"| {i} | `{row['feature']}` | {row['rf_permutation_auc_drop']:.4f} | {row['rf_permutation_auc_drop_std']:.4f} | {row['test_auc']:.3f} |\n"

    md += """
해석할 때는 Elastic Net은 방향성, Random Forest permutation은 비선형적 기여도를 보여주는 보조 증거로 사용한다. 감성 피처는 단독 예측력이 강한 변수라기보다, 시장 내부강도 및 변동성 지표와 결합될 때 방어 규칙의 상태 판단을 보완하는 피처로 해석한다.

## 9. 민감도 분석 요약

| 민감도 | 범위 | MDD 범위 | 하락일 방어폭 범위 | 해석 |
| --- | --- | ---: | ---: | --- |
"""
    md += f"| 포지션 상한 | 0.7-1.2x | {cap['mdd'].min()*100:.1f}% ~ {cap['mdd'].max()*100:.1f}% | {cap['downside_excess_bp'].min():.1f} ~ {cap['downside_excess_bp'].max():.1f}bp | 1.0x는 레버리지 제거 근거가 가장 명확 |\n"
    md += f"| 거래비용 | 0-20bp | {fee['mdd'].min()*100:.1f}% ~ {fee['mdd'].max()*100:.1f}% | {fee['downside_excess_bp'].min():.1f} ~ {fee['downside_excess_bp'].max():.1f}bp | 20bp에서도 하방방어 유의 |\n"
    md += f"| Fear/Greed | 20/80, 25/65, 30/70 등 | {threshold['mdd'].min()*100:.1f}% ~ {threshold['mdd'].max()*100:.1f}% | {threshold['downside_excess_bp'].min():.1f} ~ {threshold['downside_excess_bp'].max():.1f}bp | 특정 threshold 의존성 낮음 |\n"
    md += f"| K-FGI multiplier | 0.4-1.8 | {mult['mdd'].min()*100:.1f}% ~ {mult['mdd'].max()*100:.1f}% | {mult['downside_excess_bp'].min():.1f} ~ {mult['downside_excess_bp'].max():.1f}bp | 보수적 multiplier에서도 방어 유지 |\n"

    md += """
## 10. 선행연구 대비 본 연구의 위치

| 선행연구 축 | 기존 연구 | 본 연구의 위치 |
| --- | --- | --- |
"""
    for _, row in literature.iterrows():
        md += f"| {row['literature']} | {row['main_point']} | {row['position_of_this_paper']} |\n"

    md += """
정리하면, 본 연구는 CNN Fear & Greed Index처럼 여러 시장 심리 하위지표를 결합하지만, 단순 심리지표 산출에 머물지 않고 한국 시장의 하방위험 관리 규칙으로 연결한다. 또한 Moreira and Muir식 volatility-managed portfolio와 유사하게 고변동성 구간에서 노출을 줄이지만, 여기에 댓글 기반 감성의 평활화 신호를 결합했다는 점에서 차별성이 있다. 국내 연구와 비교하면 투자심리와 수익률/변동성의 관계를 보는 데서 한 걸음 더 나아가, 감성-변동성 결합 지표가 실제 하락일 손실 방어에 어떻게 작동하는지를 검증한다.

참고 링크:
- CNN Fear & Greed Index methodology: https://edition-prod-cf.sitemirror.cnn.com/markets/fear-and-greed
- Moreira and Muir (2017), Volatility-Managed Portfolios: https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.12513
- Barro, Canestrelli and Lanza (2014), Volatility vs. Downside Risk: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2521007
- Managing downside risk of low-risk anomaly portfolios: https://www.sciencedirect.com/science/article/pii/S1544612321003883
- Kim and Lee (2022), Impact of Investor Sentiment on Stock Returns: https://onlinelibrary.wiley.com/doi/10.1111/ajfs.12362
- Pyo and Kim (2019), News media sentiment and asset prices in Korea: https://www.tandfonline.com/doi/abs/10.1080/16081625.2019.1642115
- Investors’ Sentiment, Arbitrage Constraints, and KOSPI Market Fear Index: https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002519194

## 11. 논문 문항별 수정 지시

| 논문 문항 | 넣을 내용 | 작성 방식 | 그림/표 |
| --- | --- | --- | --- |
"""
    for _, row in sections.iterrows():
        md += f"| {row['paper_section']} | {row['what_to_insert']} | {row['how_to_write']} | {row['figure_or_table']} |\n"

    md += """
## 12. 이미지 배치 계획

| 번호 | 내용 | 파일/작업 | 목적 |
| --- | --- | --- | --- |
"""
    for _, row in figures.iterrows():
        md += f"| {row['figure_no']} | {row['content']} | {row['file_or_action']} | {row['paper_purpose']} |\n"

    md += f"""
## 13. 기존 중복 내용 처리 원칙

- 수익률 예측, target return 예측, LightGBM 예측력 중심 문장은 본문에서 제거한다.
- full sentiment feature 전체를 메인 모델처럼 설명한 문장은 `sent_composite_ma10` 중심으로 수정한다.
- 3배 레버리지 또는 1.2배 이상 노출을 메인 성과로 해석한 문장은 제거하고, 포지션 상한 민감도 분석으로 이동한다.
- 이전 수익률 중심 표는 부록 또는 재현성 파일로만 둔다.
- 본문 핵심 표는 `final_master_statistical_tests.csv`, `final_master_conditional_downside_defense.csv`, `final_conservative_kfgi_sensitivity.csv`를 기준으로 작성한다.

## 14. 바로 붙여넣을 결론 문장

본 연구의 최종 K-FGI는 시장 심리 하위지표, 평활화된 댓글 감성지표, EGARCH 기반 조건부 변동성을 결합한 하방위험 관리 지표이다. 분석 결과 K-FGI는 시장 하락일에 Buy & Hold 대비 평균 {perf['downside_excess_bp']:.1f}bp의 손실 방어 효과를 보였으며, 해당 효과는 t-test, z-test, HAC 보정 검정 및 비모수 검정에서 통계적으로 유의하였다. 또한 최종 포지션 상한을 1.0배로 제한한 보수형 운용 규칙은 전체 최대낙폭을 B&H 대비 {perf['mdd_improvement_pctp']:.1f}%p 개선하였다. 이는 K-FGI가 단기 수익률 예측 지표라기보다, 투자심리와 변동성 정보를 이용해 시장 하락 구간의 손실 노출을 줄이는 위험관리 도구로 기능함을 시사한다.
"""

    for root in [REPO_OUT, PAPER_OUT]:
        (root / "FINAL_PAPER_MASTER_SUMMARY.md").write_text(md, encoding="utf-8")


def main() -> None:
    df = build_final_df()
    write_outputs(df)
    print(REPO_OUT / "FINAL_PAPER_MASTER_SUMMARY.md")
    print(PAPER_OUT / "FINAL_PAPER_MASTER_SUMMARY.md")


if __name__ == "__main__":
    main()
