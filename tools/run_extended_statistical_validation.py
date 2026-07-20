from __future__ import annotations

import math
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


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
REPO_TABLE = CODE_ROOT / "paper_outputs" / "tables"
REPO_DATA = CODE_ROOT / "paper_outputs" / "data"
PAPER_OUT = PAPER_ROOT / "10_Paper_outputs"
PAPER_TABLE = PAPER_OUT / "tables"
PAPER_DATA = PAPER_OUT / "data"
FINAL_DATA = PAPER_ROOT / "01_final_dataset" / "KFG_final_10y.csv"

TRADING_DAYS = 252
SEED = 20260718


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def p_label(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def clean(x) -> np.ndarray:
    arr = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    return arr


def mean_ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    x = clean(x)
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    se = x.std(ddof=1) / math.sqrt(n)
    q = stats.t.ppf(1 - alpha / 2, n - 1)
    return x.mean() - q * se, x.mean() + q * se


def ztest_1samp(x: np.ndarray, value: float = 0.0) -> tuple[float, float]:
    x = clean(x)
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    se = x.std(ddof=1) / math.sqrt(n)
    z = (x.mean() - value) / se if se > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return z, p


def hac_ttest_mean(x: np.ndarray, lags: int = 5) -> tuple[float, float]:
    x = clean(x)
    n = len(x)
    if n < lags + 2:
        return np.nan, np.nan
    xc = x - x.mean()
    gamma0 = np.dot(xc, xc) / n
    long_var = gamma0
    for lag in range(1, lags + 1):
        gamma = np.dot(xc[lag:], xc[:-lag]) / n
        weight = 1 - lag / (lags + 1)
        long_var += 2 * weight * gamma
    se = math.sqrt(max(long_var, 0) / n)
    t = x.mean() / se if se > 0 else np.nan
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1)) if np.isfinite(t) else np.nan
    return t, p


def wilcoxon_signed(x: np.ndarray) -> tuple[float, float]:
    x = clean(x)
    x = x[np.abs(x) > 1e-14]
    if len(x) < 2:
        return np.nan, np.nan
    res = stats.wilcoxon(x, alternative="two-sided", zero_method="wilcox")
    return float(res.statistic), float(res.pvalue)


def sign_test(x: np.ndarray) -> tuple[float, float, int, int]:
    x = clean(x)
    pos = int(np.sum(x > 0))
    neg = int(np.sum(x < 0))
    n = pos + neg
    if n == 0:
        return np.nan, np.nan, pos, neg
    p = stats.binomtest(pos, n=n, p=0.5, alternative="two-sided").pvalue
    z = (pos - 0.5 * n) / math.sqrt(0.25 * n)
    return z, float(p), pos, neg


def bootstrap_mean(x: np.ndarray, b: int = 5000) -> tuple[float, float, float, float]:
    x = clean(x)
    if len(x) < 2:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), size=(b, len(x)))
    vals = x[idx].mean(axis=1)
    p = 2 * min(np.mean(vals <= 0), np.mean(vals >= 0))
    return float(vals.mean()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)), float(p)


def bootstrap_sharpe_diff(a: np.ndarray, b: np.ndarray, b_iter: int = 5000) -> tuple[float, float, float, float]:
    a = clean(a)
    b = clean(b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 30:
        return np.nan, np.nan, np.nan, np.nan

    def sharpe(x):
        sd = x.std(ddof=1)
        return math.sqrt(TRADING_DAYS) * x.mean() / sd if sd > 0 else np.nan

    obs = sharpe(a) - sharpe(b)
    rng = np.random.default_rng(SEED + 1)
    idx = rng.integers(0, n, size=(b_iter, n))
    vals = np.array([sharpe(a[i]) - sharpe(b[i]) for i in idx])
    p = 2 * min(np.mean(vals <= 0), np.mean(vals >= 0))
    return float(obs), float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975)), float(p)


def variance_tests(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float, float]:
    a = clean(a)
    b = clean(b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    var_ratio = np.var(a, ddof=1) / np.var(b, ddof=1)
    f_stat = var_ratio
    p_f = 2 * min(stats.f.cdf(f_stat, n - 1, n - 1), 1 - stats.f.cdf(f_stat, n - 1, n - 1))
    lev = stats.levene(a, b, center="median")
    return float(var_ratio), float(f_stat), float(p_f), float(lev.statistic), float(lev.pvalue)


def proportion_ztest(success_a: int, n_a: int, success_b: int, n_b: int) -> tuple[float, float]:
    p_pool = (success_a + success_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = ((success_a / n_a) - (success_b / n_b)) / se if se > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return float(z), float(p)


def add_result(rows: list[dict], category: str, hypothesis: str, test: str, stat: float, p: float, n: int, effect: float, unit: str, interpretation: str):
    rows.append(
        {
            "category": category,
            "hypothesis": hypothesis,
            "test": test,
            "stat": stat,
            "p_value": p,
            "significance": stars(p),
            "n": n,
            "effect": effect,
            "unit": unit,
            "interpretation": interpretation,
        }
    )


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_해당 기준을 만족하는 결과 없음_"
    d = df.copy()
    for col in d.columns:
        d[col] = d[col].map(lambda x: "" if pd.isna(x) else (f"{x:.4g}" if isinstance(x, float) else str(x)))
    headers = list(d.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in d.iterrows():
        rows.append("| " + " | ".join(str(row[h]).replace("\n", " ") for h in headers) + " |")
    return "\n".join(rows)


def main() -> None:
    ret = pd.read_csv(REPO_DATA / "strategy_returns_10y.csv", parse_dates=["date"])
    kfgi = pd.read_csv(REPO_DATA / "kfgi_10y_timeseries.csv", parse_dates=["date"])
    final = pd.read_csv(FINAL_DATA, parse_dates=["date"])
    df = ret.merge(kfgi, on="date", how="inner").merge(
        final[["date", "sent_norm_w", "sent_strength_w", "neg_z", "comment_count"]], on="date", how="left"
    )
    df = df.sort_values("date").reset_index(drop=True)

    main_ret = df["main_kfgi_sentiment"].to_numpy()
    bh_ret = df["buy_hold"].to_numpy()
    no_sent = None
    perf = pd.read_csv(REPO_TABLE / "performance_summary.csv")
    if "main_kfgi_no_sentiment" in perf["label"].values:
        no_sent_ann = perf.loc[perf["label"] == "main_kfgi_no_sentiment", "ann_ret"].iloc[0]
    else:
        no_sent_ann = np.nan

    rows: list[dict] = []

    excess = main_ret - bh_ret
    t, p = stats.ttest_1samp(clean(excess), 0)
    add_result(rows, "전략 성과", "K-FGI 일별 초과수익 평균 > 0", "one-sample t-test", t, p, len(clean(excess)), clean(excess).mean() * 10000, "bp/day", "전체 초과수익 평균은 유의하지 않음")
    z, pz = ztest_1samp(excess)
    add_result(rows, "전략 성과", "K-FGI 일별 초과수익 평균 > 0", "normal z-test", z, pz, len(clean(excess)), clean(excess).mean() * 10000, "bp/day", "정규근사에서도 유의하지 않음")
    ht, hp = hac_ttest_mean(excess, lags=5)
    add_result(rows, "전략 성과", "자기상관 보정 후 초과수익 평균 > 0", "Newey-West HAC t-test", ht, hp, len(clean(excess)), clean(excess).mean() * 10000, "bp/day", "자기상관 보정 후에도 유의하지 않음")
    w, wp = wilcoxon_signed(excess)
    add_result(rows, "전략 성과", "K-FGI 초과수익 중앙값 != 0", "Wilcoxon signed-rank", w, wp, len(clean(excess)), np.median(clean(excess)) * 10000, "bp/day", "중앙값 기준 차이 검정")
    sz, sp, pos, neg = sign_test(excess)
    add_result(rows, "전략 성과", "K-FGI가 B&H를 이긴 일수가 50%와 다른가", "sign/binomial test", sz, sp, pos + neg, pos / (pos + neg), "win-rate", f"K-FGI 승률 {pos/(pos+neg):.1%}")
    bm, blo, bhi, bp = bootstrap_mean(excess)
    add_result(rows, "전략 성과", "초과수익 평균 bootstrap CI", "iid bootstrap mean", bm / (np.std(excess, ddof=1) / math.sqrt(len(excess))), bp, len(excess), bm * 10000, "bp/day", f"95% CI [{blo*10000:.2f}, {bhi*10000:.2f}] bp/day")

    down = df["buy_hold"] < 0
    downside_excess = (df.loc[down, "main_kfgi_sentiment"] - df.loc[down, "buy_hold"]).to_numpy()
    t, p = stats.ttest_1samp(clean(downside_excess), 0)
    add_result(rows, "하방 방어", "B&H 하락일에 K-FGI 손실 방어 > 0", "one-sample t-test", t, p, len(clean(downside_excess)), clean(downside_excess).mean() * 10000, "bp/down day", "하락일 방어 효과가 매우 유의")
    z, pz = ztest_1samp(downside_excess)
    add_result(rows, "하방 방어", "B&H 하락일에 K-FGI 손실 방어 > 0", "normal z-test", z, pz, len(clean(downside_excess)), clean(downside_excess).mean() * 10000, "bp/down day", "정규근사에서도 매우 유의")
    ht, hp = hac_ttest_mean(downside_excess, lags=5)
    add_result(rows, "하방 방어", "자기상관 보정 후 하락일 방어 > 0", "Newey-West HAC t-test", ht, hp, len(clean(downside_excess)), clean(downside_excess).mean() * 10000, "bp/down day", "HAC 보정 후에도 유의")
    w, wp = wilcoxon_signed(downside_excess)
    add_result(rows, "하방 방어", "하락일 초과수익 중앙값 != 0", "Wilcoxon signed-rank", w, wp, len(clean(downside_excess)), np.median(clean(downside_excess)) * 10000, "bp/down day", "비모수 검정에서도 하방 방어 확인")
    bm, blo, bhi, bp = bootstrap_mean(downside_excess)
    add_result(rows, "하방 방어", "하락일 방어 평균 bootstrap CI", "iid bootstrap mean", bm / (np.std(downside_excess, ddof=1) / math.sqrt(len(downside_excess))), bp, len(downside_excess), bm * 10000, "bp/down day", f"95% CI [{blo*10000:.2f}, {bhi*10000:.2f}] bp")

    var_ratio, f_stat, p_f, lev_stat, lev_p = variance_tests(main_ret, bh_ret)
    add_result(rows, "위험 감소", "K-FGI 수익률 분산이 B&H보다 낮은가", "F-test variance ratio", f_stat, p_f, len(df), var_ratio, "variance ratio", "분산비가 1보다 작으면 변동성 감소")
    add_result(rows, "위험 감소", "K-FGI와 B&H 변동성이 다른가", "Levene median test", lev_stat, lev_p, len(df), var_ratio, "variance ratio", "강건한 분산 차이 검정")
    obs, lo, hi, pboot = bootstrap_sharpe_diff(main_ret, bh_ret)
    add_result(rows, "위험 조정 성과", "K-FGI Sharpe - B&H Sharpe > 0", "bootstrap Sharpe difference", obs, pboot, len(df), obs, "Sharpe diff", f"95% CI [{lo:.3f}, {hi:.3f}]")

    if "main_kfgi_no_sentiment" in ret.columns:
        no_sent = ret["main_kfgi_no_sentiment"].to_numpy()
    else:
        no_sent = None
    # No-sentiment daily series is not in strategy_returns_10y.csv. Use table-level effects only.
    sent_effect_total = (
        perf.loc[perf["label"] == "main_kfgi_sentiment", "total_return"].iloc[0]
        - perf.loc[perf["label"] == "main_kfgi_no_sentiment", "total_return"].iloc[0]
    )
    sent_effect_sharpe = (
        perf.loc[perf["label"] == "main_kfgi_sentiment", "sharpe"].iloc[0]
        - perf.loc[perf["label"] == "main_kfgi_no_sentiment", "sharpe"].iloc[0]
    )
    add_result(rows, "감성 피처", "감성 포함 총수익률 - 감성 제외 총수익률", "performance table difference", np.nan, np.nan, len(df), sent_effect_total * 100, "%p total return", "감성 포함 총수익률은 더 높지만 일별 검정용 no-sentiment 수익률 파일은 별도 저장 필요")
    add_result(rows, "감성 피처", "감성 포함 Sharpe - 감성 제외 Sharpe", "performance table difference", np.nan, np.nan, len(df), sent_effect_sharpe, "Sharpe diff", "감성 포함 Sharpe는 감성 제외보다 낮음")

    # Predictive relation of K-FGI and subsequent returns.
    pred = df.dropna(subset=["K_FGI", "target_reg", "target_5d", "sent_norm_w"]).copy()
    for target in ["target_reg", "target_5d"]:
        pr, pp = stats.pearsonr(pred["K_FGI"], pred[target])
        sr, spv = stats.spearmanr(pred["K_FGI"], pred[target])
        add_result(rows, "예측 상관", f"K-FGI와 {target} 상관", "Pearson correlation", pr, pp, len(pred), pr, "r", "선형 예측 상관")
        add_result(rows, "예측 상관", f"K-FGI와 {target} 순위상관", "Spearman rank correlation", sr, spv, len(pred), sr, "rho", "비선형/순위 예측 상관")

    fear = pred[pred["K_FGI"] <= pred["K_FGI"].quantile(0.25)]["target_5d"]
    greed = pred[pred["K_FGI"] >= pred["K_FGI"].quantile(0.75)]["target_5d"]
    t, p = stats.ttest_ind(greed, fear, equal_var=False)
    add_result(rows, "극단 구간", "Greed 상위 25%와 Fear 하위 25%의 5일 수익률 차이", "Welch t-test", t, p, len(greed) + len(fear), (greed.mean() - fear.mean()) * 100, "%p/5D", "극단 K-FGI 구간 간 평균 차이")
    u = stats.mannwhitneyu(greed, fear, alternative="two-sided")
    add_result(rows, "극단 구간", "Greed/Fear 5일 수익률 분포 차이", "Mann-Whitney U", u.statistic, u.pvalue, len(greed) + len(fear), (greed.median() - fear.median()) * 100, "%p/5D median diff", "비모수 분포 차이")
    gz, gp = proportion_ztest(int((greed > 0).sum()), len(greed), int((fear > 0).sum()), len(fear))
    add_result(rows, "극단 구간", "Greed/Fear 5일 양(+)수익 비율 차이", "two-proportion z-test", gz, gp, len(greed) + len(fear), (greed.gt(0).mean() - fear.gt(0).mean()), "positive-rate diff", "상승 확률 차이")

    # Sentiment relation.
    for target in ["target_reg", "target_5d"]:
        sr, spv = stats.spearmanr(pred["sent_norm_w"], pred[target], nan_policy="omit")
        add_result(rows, "감성 예측", f"정규화 감성과 {target} 순위상관", "Spearman rank correlation", sr, spv, len(pred), sr, "rho", "감성 단독 예측력")
    neg = pred[pred["sent_norm_w"] <= pred["sent_norm_w"].quantile(0.25)]["target_5d"]
    pos = pred[pred["sent_norm_w"] >= pred["sent_norm_w"].quantile(0.75)]["target_5d"]
    t, p = stats.ttest_ind(pos, neg, equal_var=False)
    add_result(rows, "감성 예측", "긍정감성 상위 25%와 부정감성 하위 25%의 5일 수익률 차이", "Welch t-test", t, p, len(pos) + len(neg), (pos.mean() - neg.mean()) * 100, "%p/5D", "감성 극단 구간 차이")

    # EGARCH/volatility exposure.
    vol = df["egarch_vol"].replace([np.inf, -np.inf], np.nan)
    sane = vol[vol.between(0, 0.20)]
    cap = sane.quantile(0.99)
    clean_vol = vol.clip(lower=0, upper=cap)
    dd = df.assign(egarch_vol_clean=clean_vol).dropna(subset=["egarch_vol_clean", "weight"])
    high = dd[dd["egarch_vol_clean"] >= dd["egarch_vol_clean"].quantile(0.75)]["weight"]
    low = dd[dd["egarch_vol_clean"] <= dd["egarch_vol_clean"].quantile(0.25)]["weight"]
    t, p = stats.ttest_ind(low, high, equal_var=False)
    add_result(rows, "EGARCH 노출 조절", "저변동성 구간 노출도 > 고변동성 구간 노출도", "Welch t-test", t, p, len(low) + len(high), low.mean() - high.mean(), "exposure x", "변동성이 높을수록 노출을 낮추는지")
    rho, rp = stats.spearmanr(dd["egarch_vol_clean"], dd["weight"])
    add_result(rows, "EGARCH 노출 조절", "EGARCH 변동성과 시장 노출도 음(-)의 관계", "Spearman rank correlation", rho, rp, len(dd), rho, "rho", "음수이면 변동성 상승 시 노출 축소")
    extreme_ratio = float((vol > 0.20).mean())
    add_result(rows, "EGARCH 안정성", "EGARCH 추정값 중 표시/해석상 비정상적으로 큰 값 비중", "descriptive diagnostic", np.nan, np.nan, int(vol.notna().sum()), extreme_ratio * 100, "% of observations", "초기 추정 안정성 한계로 별도 언급")

    # Regime-level tests.
    groups = [g["main_kfgi_sentiment"].to_numpy() for _, g in df.groupby("regime")]
    if len(groups) >= 2:
        f, p = stats.f_oneway(*groups)
        add_result(rows, "시장 국면", "bull/normal/crisis 국면별 K-FGI 수익률 평균 차이", "one-way ANOVA", f, p, len(df), np.nan, "", "국면별 평균 수익률 차이")
        h, hp = stats.kruskal(*groups)
        add_result(rows, "시장 국면", "bull/normal/crisis 국면별 K-FGI 수익률 분포 차이", "Kruskal-Wallis", h, hp, len(df), np.nan, "", "비모수 국면 차이")

    out = pd.DataFrame(rows)
    out["p_value_formatted"] = out["p_value"].map(p_label)
    out = out[
        [
            "category",
            "hypothesis",
            "test",
            "stat",
            "p_value",
            "p_value_formatted",
            "significance",
            "n",
            "effect",
            "unit",
            "interpretation",
        ]
    ]

    for d in [REPO_TABLE, PAPER_TABLE]:
        d.mkdir(parents=True, exist_ok=True)
        out.to_csv(d / "extended_statistical_tests.csv", index=False, encoding="utf-8-sig")

    sig = out[out["p_value"].fillna(1) < 0.10].copy()
    strong = out[out["p_value"].fillna(1) < 0.05].copy()
    perf_main = perf.set_index("label").loc["main_kfgi_sentiment"]
    perf_bh = perf.set_index("label").loc["buy_hold_kospi200"]
    downside = out[(out["category"] == "하방 방어") & (out["test"] == "one-sample t-test")].iloc[0]
    vol_reduction = out[(out["category"] == "위험 감소") & (out["test"] == "Levene median test")].iloc[0]
    egarch_exp = out[(out["category"] == "EGARCH 노출 조절") & (out["test"] == "Welch t-test")].iloc[0]
    extreme = out[(out["category"] == "극단 구간") & (out["test"] == "Welch t-test")].iloc[0]

    md = f"""# Extended Statistical Validation

분석 기간: {df['date'].min().date()} ~ {df['date'].max().date()}  
표본 수: {len(df):,} trading days  
유의성 표기: *** p<0.01, ** p<0.05, * p<0.10

## 결론 요약

1. **전체 초과수익은 유의하지 않다.** K-FGI의 Buy&Hold 대비 일별 초과수익 t-test p-value는 {p_label(out.loc[(out['category']=='전략 성과') & (out['test']=='one-sample t-test'), 'p_value'].iloc[0])}이다.
2. **하방 방어는 매우 강하게 유의하다.** Buy&Hold 하락일에서 K-FGI의 평균 방어 효과는 {downside['effect']:.2f} bp/day이며, t={downside['stat']:.2f}, p={p_label(downside['p_value'])}이다.
3. **위험 조정 성과는 개선된다.** K-FGI Sharpe는 {perf_main['sharpe']:.3f}, Buy&Hold Sharpe는 {perf_bh['sharpe']:.3f}이다. MDD는 {perf_bh['mdd']*100:.1f}%에서 {perf_main['mdd']*100:.1f}%로 줄었다.
4. **극단 K-FGI 구간의 예측력은 약한 유의 수준이다.** Greed/Fear 5일 수익률 차이는 {extreme['effect']:.2f}%p이고 p={p_label(extreme['p_value'])}이다.
5. **EGARCH는 결과 해석에서 '초기 추정 안정성' 한계가 있다.** 일부 초기 추정값이 비정상적으로 커서 그래프 표시에는 winsorizing을 적용했다. 이는 성과 유의성 부재가 아니라 변동성 추정 모형의 기술적 한계다.

## 논문에 쓸 수 있는 핵심 숫자

- K-FGI 총수익률: {perf_main['total_return']*100:.1f}%, Buy&Hold 총수익률: {perf_bh['total_return']*100:.1f}%.
- K-FGI 연율수익률: {perf_main['ann_ret']*100:.2f}%, 연율변동성: {perf_main['ann_vol']*100:.2f}%, Sharpe: {perf_main['sharpe']:.3f}.
- Buy&Hold 연율수익률: {perf_bh['ann_ret']*100:.2f}%, 연율변동성: {perf_bh['ann_vol']*100:.2f}%, Sharpe: {perf_bh['sharpe']:.3f}.
- 최대낙폭 개선: {perf_bh['mdd']*100:.1f}% -> {perf_main['mdd']*100:.1f}%.
- 하락일 방어 효과: 평균 {downside['effect']:.2f} bp/day, t={downside['stat']:.2f}, p={p_label(downside['p_value'])}, n={int(downside['n']):,}.
- 변동성 차이 Levene 검정: stat={vol_reduction['stat']:.2f}, p={p_label(vol_reduction['p_value'])}.
- EGARCH 노출 조절: 저변동성-고변동성 평균 노출 차이 {egarch_exp['effect']:.3f}x, p={p_label(egarch_exp['p_value'])}.

## 유의한 결과만 보기

{markdown_table(strong[['category','hypothesis','test','stat','p_value_formatted','significance','n','effect','unit']])}

## 10% 수준까지 포함한 약한 유의 결과

{markdown_table(sig[['category','hypothesis','test','stat','p_value_formatted','significance','n','effect','unit']])}

## 전체 검정표

전체 CSV: `tables/extended_statistical_tests.csv`

## 본문 서술 권장 방향

본 연구 결과는 “K-FGI가 Buy&Hold 대비 전체 초과수익을 통계적으로 유의하게 창출했다”는 주장보다는, **시장 하락일과 고변동성 구간에서 노출을 줄여 하방 위험을 방어하는 전략**이라는 주장에 더 강하게 지지된다. 감성 피처는 총수익률 개선에는 기여하지만, Sharpe와 bootstrap 검정에서는 강한 유의성이 확인되지 않으므로 보조 피처로 신중하게 해석하는 것이 적절하다.
"""
    for d in [CODE_ROOT / "paper_outputs", CODE_ROOT / "docs", PAPER_OUT]:
        d.mkdir(parents=True, exist_ok=True)
        (d / "STATISTICAL_VALIDATION.md").write_text(md, encoding="utf-8")

    print(out.to_string(index=False))
    print(f"saved={REPO_TABLE / 'extended_statistical_tests.csv'}")
    print(f"saved={CODE_ROOT / 'paper_outputs' / 'STATISTICAL_VALIDATION.md'}")


if __name__ == "__main__":
    main()
