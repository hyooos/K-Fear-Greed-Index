import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy import stats
from config import CFG


def perf(returns, label, ref=None):
    ann_ret = returns.mean() * 252
    ann_vol = returns.std()  * np.sqrt(252) + 1e-9
    sharpe  = ann_ret / ann_vol
    cum     = np.exp(returns.cumsum())
    mdd     = (cum / cum.cummax() - 1).min()
    calmar  = ann_ret / (abs(mdd) + 1e-9)
    total   = np.exp(returns.sum()) - 1
    defense = float("nan")
    if ref is not None:
        dm      = ref < 0
        defense = returns[dm].mean() / (ref[dm].mean() + 1e-9)
    print(f"  [{label:<36}]  "
          f"연수익={ann_ret*100:+.2f}%  "
          f"Sharpe={sharpe:.3f}  "
          f"MDD={mdd*100:.2f}%  "
          f"Calmar={calmar:.2f}  "
          f"누적={total*100:+.2f}%  "
          f"하락방어율={defense:.2f}")
    return dict(ann_ret=ann_ret, sharpe=sharpe, mdd=mdd, calmar=calmar)


def run_statistical_tests(main_df, df, df_ns, rho_5d):
    """전체 통계 검정 실행 (A~E)."""
    print("\n" + "=" * 62)
    print("  통계적 유의성 검정 (강화판)")
    print("=" * 62)

    common_idx = main_df.index.intersection(df.index)
    excess = (main_df.loc[common_idx, "strat_ret"]
              - df.loc[common_idx, "target_reg"]).dropna()

    # [A] 전체 초과수익 t-test
    t_A, p_A = stats.ttest_1samp(excess, 0)
    print(f"""
[A] 전체 기간 초과수익 t-test
    t-stat  : {t_A:.4f}
    p-value : {p_A:.6f}
    결론    : {"p < 0.05 → 유의" if p_A < 0.05 else "p ≥ 0.05 → 유의하지 않음"}""")

    # [B] 하락 구간 Downside t-test
    down_mask   = df.loc[common_idx, "target_reg"] < 0
    excess_down = excess[down_mask[down_mask].index.intersection(excess.index)].dropna()
    t_B, p_B    = stats.ttest_1samp(excess_down, 0)
    print(f"""
[B] 시장 하락 구간 Downside t-test
    대상 기간: 시장 일별 수익률 < 0인 날 ({len(excess_down)}일)
    t-stat   : {t_B:.4f}
    p-value  : {p_B:.6f}
    결론     : {"p < 0.05 → 하락 방어 효과 유의 ★★" if p_B < 0.05 else "p ≥ 0.05 → 유의하지 않음"}""")

    # [C] K-FGI 극단 구간 t-test
    kfgi_q25 = main_df["K_FGI"].quantile(0.25)
    kfgi_q75 = main_df["K_FGI"].quantile(0.75)
    greed_ret = main_df.loc[main_df["K_FGI"] >= kfgi_q75, "target_reg"].dropna()
    fear_ret  = main_df.loc[main_df["K_FGI"] <= kfgi_q25, "target_reg"].dropna()
    t_C, p_C  = stats.ttest_ind(greed_ret, fear_ret, equal_var=False)
    print(f"""
[C] K-FGI 극단 구간 수익률 차이 t-test
    탐욕 구간 (K-FGI≥{kfgi_q75:.1f}, {len(greed_ret)}일) 평균: {greed_ret.mean()*100:+.4f}%
    공포 구간 (K-FGI≤{kfgi_q25:.1f}, {len(fear_ret)}일) 평균: {fear_ret.mean()*100:+.4f}%
    t-stat  : {t_C:.4f}
    p-value : {p_C:.6f}
    결론    : {"p < 0.05 → 유의 ★★" if p_C < 0.05 else "p ≥ 0.05 → 유의하지 않음"}""")

    # [D] 부트스트랩 검정
    print(f"\n[D] 감성 기여 Spearman ρ 부트스트랩 검정 (n=1000)")
    np.random.seed(42)
    n_boot = 1000
    common_ns   = main_df.index.intersection(df_ns.index)
    kfgi_with   = main_df.loc[common_ns, "K_FGI"].values
    kfgi_without= df_ns.loc[common_ns, "K_FGI"].values
    ret_5d_vals = main_df.loc[common_ns, "target_5d"].values
    valid_mask  = (~np.isnan(kfgi_with)
                   & ~np.isnan(kfgi_without)
                   & ~np.isnan(ret_5d_vals))
    kw  = kfgi_with[valid_mask]
    kwo = kfgi_without[valid_mask]
    r5  = ret_5d_vals[valid_mask]
    rho_obs_with    = spearmanr(kw,  r5)[0]
    rho_obs_without = spearmanr(kwo, r5)[0]
    rho_diff_obs    = rho_obs_with - rho_obs_without
    boot_diffs = []
    n_valid = len(kw)
    for _ in range(n_boot):
        idx   = np.random.choice(n_valid, n_valid, replace=True)
        rho_w = spearmanr(kw[idx],  r5[idx])[0]
        rho_wo= spearmanr(kwo[idx], r5[idx])[0]
        boot_diffs.append(rho_w - rho_wo)
    boot_diffs = np.array(boot_diffs)
    p_boot     = float(np.mean(boot_diffs <= 0))
    print(f"    감성 포함 K-FGI ρ: {rho_obs_with:.4f}")
    print(f"    감성 제외 K-FGI ρ: {rho_obs_without:.4f}")
    print(f"    관측 Δρ          : {rho_diff_obs:+.4f}")
    print(f"    부트스트랩 p-value: {p_boot:.4f}")
    print("    → " + ("유의 ★★" if p_boot < 0.05 else "유의하지 않음 (샘플 부족)"))

    # [E] MDD 요약
    ref = df["target_reg"]
    print(f"\n[E] 전략별 MDD 비교 요약")

    return t_B, p_B, t_C, p_C, p_boot


def run_fee_test(df, compute_positions_fn, fee_list=[0.00015, 0.0003, 0.00045]):
    """수수료 스트레스 테스트."""
    print("\n[거래비용 스트레스 테스트]")
    print("-" * 70)
    original_fee = CFG["fee"]
    for fee in fee_list:
        CFG["fee"] = fee
        res = compute_positions_fn(df.copy())
        print(f"\n▶ 수수료 = {fee*100:.3f}% (편도)")
        perf(res["strat_ret"], f"전략 (fee={fee*100:.3f}%)", df["target_reg"])
    CFG["fee"] = original_fee


def period_test(df, compute_positions_fn):
    print("\n[기간별(연도별) 성과 분석]")
    print("-" * 70)
    df = df.copy()
    df["year"] = df["date"].dt.year
    for y in sorted(df["year"].unique()):
        sub = df[df["year"] == y]
        if len(sub) < 50:
            continue
        res = compute_positions_fn(sub.copy())
        print(f"\n▶ {y}년 ({len(sub)}거래일)")
        perf(res["strat_ret"], f"{y}년", sub["target_reg"])


def rolling_oos_test(df, compute_positions_fn):
    print("\n[Rolling OOS 테스트]")
    print("-" * 70)
    years = sorted(df["date"].dt.year.unique())
    for i in range(2, len(years)):
        train_years = years[:i]
        test_year   = years[i]
        test  = df[df["date"].dt.year == test_year]
        if len(test) < 50:
            continue
        test_res = compute_positions_fn(test.copy())
        print(f"\n▶ 훈련: {list(train_years)}  →  테스트: {test_year}년 ({len(test)}일)")
        perf(test_res["strat_ret"], f"OOS {test_year}년", test["target_reg"])