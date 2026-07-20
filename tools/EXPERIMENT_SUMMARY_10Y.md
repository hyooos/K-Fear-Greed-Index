# 10-Year K-FGI x EGARCH Experiment Summary

## Data

- Input: `01_final_dataset/KFG_final_10y.csv`
- Raw rows: 2,689
- Final experiment rows after EGARCH, rolling features, target construction, and K-FGI warm-up: 2,505
- Experiment period: 2015-10-08 to 2025-12-22
- EGARCH package: `arch` available and used
- `sub_index1` policy: kept in source CSV for reproducibility, excluded from K-FGI/modeling features because price-based momentum features (`mom5`, `mom20`, `mom60`, moving-average ratios, RSI) already represent market momentum.

## K-FGI Construction

Included K-FGI features:

- `sub_index2` to `sub_index7`
- sentiment features: `sent_norm_w`, `sent_energy`, `sent_std_inv`, `neg_z_inv`, `sent_composite`, `sent_composite_ma10`
- volatility features: `egarch_vol`, `vol_regime_high`, `vol_ratio`

Walk-forward constrained ridge-style optimization was used. Feature signs were constrained by economic direction:

- greed/risk-on features: non-negative coefficient
- fear/risk-off features: non-positive coefficient

K-FGI direction was automatically selected from 5-day Spearman correlation:

- 1-day Spearman rho: 0.0374, p-value: 0.0612
- 5-day Spearman rho: 0.0088, p-value: 0.6586
- selected direction: momentum

## Main Performance

| Strategy | Annual Return | Volatility | Sharpe | MDD | Total Return |
|---|---:|---:|---:|---:|---:|
| Buy & Hold KOSPI200 | 8.71% | 18.46% | 0.472 | -41.19% | 137.64% |
| Trend only | 7.45% | 12.12% | 0.615 | -21.36% | 109.68% |
| Trend + EGARCH vol | 5.41% | 11.30% | 0.478 | -22.74% | 71.19% |
| Main K-FGI with sentiment | 8.33% | 13.60% | 0.613 | -22.93% | 128.92% |
| Main K-FGI without sentiment | 7.15% | 10.82% | 0.660 | -21.70% | 100.02% |

Interpretation:

- The main K-FGI strategy reduced MDD substantially versus Buy & Hold.
- Adding sentiment improved total return versus the no-sentiment K-FGI version.
- The no-sentiment version had the highest Sharpe among the two main K-FGI variants because of lower volatility.

## OOS 2025

| Period | Annual Return | Sharpe | MDD | Total Return |
|---|---:|---:|---:|---:|
| Train before 2025 | 2.43% | 0.192 | -22.93% | 24.42% |
| Test 2025 | 64.83% | 3.190 | -7.04% | 84.00% |

## Statistical Tests

| Test | Statistic | p-value | n |
|---|---:|---:|---:|
| Overall excess return t-test | -0.0889 | 0.9292 | 2,505 |
| Downside excess return t-test | 14.7737 | 2.19e-45 | 1,161 |
| K-FGI extreme return t-test | 1.8169 | 0.0695 | 1,254 |
| Sentiment bootstrap delta rho | 0.0172 | 0.3010 | 2,445 |

Interpretation:

- Overall excess return is not statistically significant.
- Downside defense is statistically significant.
- K-FGI extreme regime return difference is marginal but not significant at 5%.
- Sentiment improves the observed correlation directionally, but the bootstrap test is not significant.

## Outputs

- `06_10y_experiments/kfgi_10y_timeseries.csv`
- `06_10y_experiments/strategy_returns_10y.csv`
- `06_10y_experiments/tables/performance_summary.csv`
- `06_10y_experiments/tables/lgbm_cv_summary.csv`
- `06_10y_experiments/tables/oos_2025_performance.csv`
- `06_10y_experiments/tables/yearly_performance.csv`
- `06_10y_experiments/tables/regime_performance.csv`
- `06_10y_experiments/tables/fee_stress_test.csv`
- `06_10y_experiments/tables/statistical_tests.csv`
- `06_10y_experiments/tables/sensitivity_analysis.csv`
- `06_10y_experiments/tables/feature_importance.csv`
- `06_10y_experiments/figures/kfgi_egarch_10y_result.png`
- `06_10y_experiments/figures/feature_importance_10y.png`
