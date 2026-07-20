# Rolling OOS and Regime Evaluation

이 평가는 이미 walk-forward 방식으로 산출된 K-FGI 일별 전략 수익률을 사용해, 기간별 안정성과 조건부 성과를 추가 검증한 것이다.

## 핵심 요약

- 1년 rolling 평가창 수: 2,254
- 1년 rolling Sharpe가 Buy&Hold보다 높았던 비율: 39.6%
- 1년 rolling MDD가 Buy&Hold보다 개선된 비율: 83.1%
- 평균 rolling Sharpe 차이: -0.115
- 평균 rolling MDD 개선폭: 6.67%p
- 연도별 OOS Sharpe 기준 K-FGI가 Buy&Hold를 이긴 연도 비율: 40.0%

## 논문에 쓸 수 있는 해석

K-FGI 전략은 전체 기간 초과수익의 유의성보다는, 1년 단위 rolling window와 시장 국면별 조건부 평가에서 위험 관리 성격이 더 뚜렷하게 나타난다. 특히 rolling MDD 개선 비율과 하락일 조건부 방어 효과를 함께 제시하면, 본 전략의 기여가 수익 극대화보다 하방 위험 완화에 있음을 더 명확히 설명할 수 있다.

주의: `B&H 상승/보합일`, `B&H 하락일`처럼 특정 조건만 뽑은 표본의 연율수익률은 경제적 직관과 다르게 과장될 수 있으므로 본문에서는 보조적으로만 사용한다. 해당 조건부 분석에서는 `mean_excess_bp`, `win_rate_vs_bh`, `paired t-test`를 중심으로 해석한다.

## Rolling OOS 요약

| 지표 | 값 |
| --- | --- |
| Rolling Sharpe 우위 비율 | 39.6% |
| Rolling MDD 개선 비율 | 83.1% |
| 평균 Rolling Sharpe 차이 | -0.115 |
| 평균 Rolling MDD 개선폭 | 6.67%p |

## 조건부 성과표

| dimension | bucket | n | ann_ret | ann_vol | sharpe | mdd | mean_excess_bp | excess_p | win_rate_vs_bh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exposure_bucket | 공격(>1x) | 784 | 0.3301 | 0.1905 | 1.733 | -0.1361 | 1.551 | 0.2833 | 0.5128 |
| exposure_bucket | 방어(<0.5x) | 1163 | -0.01143 | 0.06889 | -0.1659 | -0.208 | -1.027 | 0.761 | 0.466 |
| exposure_bucket | 중립(0.5~1x) | 558 | -0.002775 | 0.1479 | -0.01877 | -0.1896 | -0.7095 | 0.6912 | 0.4427 |
| kfgi_zone | Fear(<25) | 296 | 0.003639 | 0.05348 | 0.06804 | -0.0591 | 4.848 | 0.5844 | 0.4696 |
| kfgi_zone | Greed(>65) | 607 | 0.3154 | 0.1852 | 1.703 | -0.151 | 1.095 | 0.6438 | 0.4909 |
| kfgi_zone | Neutral | 1602 | 0.02607 | 0.1239 | 0.2105 | -0.2527 | -1.544 | 0.4034 | 0.4707 |
| market_condition | B&H 상승/보합일 | 1344 | 2.475 | 0.1078 | 22.96 | -0.0007056 | -30.81 | 2.52e-49 | 0.3036 |
| market_condition | B&H 하락일 | 1161 | -0.7169 | 0.1143 | -6.273 | -0.997 | 35.34 | 2.192e-45 | 0.6744 |
| regime | bull | 432 | 0.4432 | 0.2083 | 2.128 | -0.1312 | 0.9532 | 0.6856 | 0.5 |
| regime | crisis | 333 | -0.06702 | 0.07126 | -0.9404 | -0.1488 | 3.44 | 0.6695 | 0.4625 |
| regime | normal | 1740 | 0.04304 | 0.1216 | 0.3538 | -0.2315 | -1.11 | 0.5305 | 0.4718 |
| vol_quartile | Q1 저변동 | 627 | 0.03928 | 0.1254 | 0.3132 | -0.1936 | -1.287 | 0.5319 | 0.4721 |
| vol_quartile | Q2 | 626 | -0.04439 | 0.122 | -0.3639 | -0.2278 | 1.078 | 0.6686 | 0.476 |
| vol_quartile | Q3 | 626 | 0.1739 | 0.1618 | 1.075 | -0.1823 | 4.986 | 0.1351 | 0.5064 |
| vol_quartile | Q4 고변동 | 626 | 0.197 | 0.1309 | 1.505 | -0.1042 | -5.372 | 0.267 | 0.4473 |

## 조건부 검정표

| dimension | test | stat | p_label |
| --- | --- | --- | --- |
| regime | one-way ANOVA | 4.892 | 0.0076 |
| regime | Kruskal-Wallis | 5.864 | 0.0533 |
| kfgi_zone | one-way ANOVA | 3.149 | 0.0431 |
| kfgi_zone | Kruskal-Wallis | 3.222 | 0.1997 |
| vol_quartile | one-way ANOVA | 1.516 | 0.2084 |
| vol_quartile | Kruskal-Wallis | 3.768 | 0.2876 |
| exposure_bucket | one-way ANOVA | 5.011 | 0.0067 |
| exposure_bucket | Kruskal-Wallis | 10.63 | 0.0049 |
| market_condition | B&H 하락일 paired t-test | 14.77 | <0.001 |

## 생성 파일

- `tables/rolling_oos_1y_performance.csv`
- `tables/yearly_oos_detailed_performance.csv`
- `tables/conditional_regime_performance.csv`
- `tables/conditional_regime_tests.csv`
- `figures/17_rolling_OOS_성과안정성.png`
- `figures/18_연도별_OOS_Sharpe.png`
- `figures/19_국면별_초과성과.png`
