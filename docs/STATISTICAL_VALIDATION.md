# Extended Statistical Validation

분석 기간: 2015-10-08 ~ 2025-12-22  
표본 수: 2,505 trading days  
유의성 표기: *** p<0.01, ** p<0.05, * p<0.10

## 결론 요약

1. **전체 초과수익은 유의하지 않다.** K-FGI의 Buy&Hold 대비 일별 초과수익 t-test p-value는 0.9292이다.
2. **하방 방어는 매우 강하게 유의하다.** Buy&Hold 하락일에서 K-FGI의 평균 방어 효과는 35.34 bp/day이며, t=14.77, p=<0.001이다.
3. **위험 조정 성과는 개선된다.** K-FGI Sharpe는 0.613, Buy&Hold Sharpe는 0.472이다. MDD는 -41.2%에서 -22.9%로 줄었다.
4. **극단 K-FGI 구간의 예측력은 약한 유의 수준이다.** Greed/Fear 5일 수익률 차이는 0.34%p이고 p=0.0373이다.
5. **EGARCH는 결과 해석에서 '초기 추정 안정성' 한계가 있다.** 일부 초기 추정값이 비정상적으로 커서 그래프 표시에는 winsorizing을 적용했다. 이는 성과 유의성 부재가 아니라 변동성 추정 모형의 기술적 한계다.

## 논문에 쓸 수 있는 핵심 숫자

- K-FGI 총수익률: 128.9%, Buy&Hold 총수익률: 137.6%.
- K-FGI 연율수익률: 8.33%, 연율변동성: 13.60%, Sharpe: 0.613.
- Buy&Hold 연율수익률: 8.71%, 연율변동성: 18.46%, Sharpe: 0.472.
- 최대낙폭 개선: -41.2% -> -22.9%.
- 하락일 방어 효과: 평균 35.34 bp/day, t=14.77, p=<0.001, n=1,161.
- 변동성 차이 Levene 검정: stat=228.93, p=<0.001.
- EGARCH 노출 조절: 저변동성-고변동성 평균 노출 차이 0.487x, p=<0.001.

## 유의한 결과만 보기

| category | hypothesis | test | stat | p_value_formatted | significance | n | effect | unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 전략 성과 | K-FGI가 B&H를 이긴 일수가 50%와 다른가 | sign/binomial test | -2.458 | 0.0148 | ** | 2505 | 0.4754 | win-rate |
| 하방 방어 | B&H 하락일에 K-FGI 손실 방어 > 0 | one-sample t-test | 14.77 | <0.001 | *** | 1161 | 35.34 | bp/down day |
| 하방 방어 | B&H 하락일에 K-FGI 손실 방어 > 0 | normal z-test | 14.77 | <0.001 | *** | 1161 | 35.34 | bp/down day |
| 하방 방어 | 자기상관 보정 후 하락일 방어 > 0 | Newey-West HAC t-test | 8.187 | <0.001 | *** | 1161 | 35.34 | bp/down day |
| 하방 방어 | 하락일 초과수익 중앙값 != 0 | Wilcoxon signed-rank | 1.605e+05 | <0.001 | *** | 1161 | 12.89 | bp/down day |
| 하방 방어 | 하락일 방어 평균 bootstrap CI | iid bootstrap mean | 14.77 | <0.001 | *** | 1161 | 35.34 | bp/down day |
| 위험 감소 | K-FGI 수익률 분산이 B&H보다 낮은가 | F-test variance ratio | 0.5424 | <0.001 | *** | 2505 | 0.5424 | variance ratio |
| 위험 감소 | K-FGI와 B&H 변동성이 다른가 | Levene median test | 228.9 | <0.001 | *** | 2505 | 0.5424 | variance ratio |
| 예측 상관 | K-FGI와 target_reg 상관 | Pearson correlation | 0.0485 | 0.0152 | ** | 2505 | 0.0485 | r |
| 예측 상관 | K-FGI와 target_5d 상관 | Pearson correlation | 0.04933 | 0.0135 | ** | 2505 | 0.04933 | r |
| 극단 구간 | Greed 상위 25%와 Fear 하위 25%의 5일 수익률 차이 | Welch t-test | 2.084 | 0.0373 | ** | 1254 | 0.3381 | %p/5D |
| EGARCH 노출 조절 | 저변동성 구간 노출도 > 고변동성 구간 노출도 | Welch t-test | 15.4 | <0.001 | *** | 1254 | 0.4866 | exposure x |
| EGARCH 노출 조절 | EGARCH 변동성과 시장 노출도 음(-)의 관계 | Spearman rank correlation | -0.2948 | <0.001 | *** | 2505 | -0.2948 | rho |
| 시장 국면 | bull/normal/crisis 국면별 K-FGI 수익률 평균 차이 | one-way ANOVA | 4.892 | 0.0076 | *** | 2505 |  |  |

## 10% 수준까지 포함한 약한 유의 결과

| category | hypothesis | test | stat | p_value_formatted | significance | n | effect | unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 전략 성과 | K-FGI가 B&H를 이긴 일수가 50%와 다른가 | sign/binomial test | -2.458 | 0.0148 | ** | 2505 | 0.4754 | win-rate |
| 하방 방어 | B&H 하락일에 K-FGI 손실 방어 > 0 | one-sample t-test | 14.77 | <0.001 | *** | 1161 | 35.34 | bp/down day |
| 하방 방어 | B&H 하락일에 K-FGI 손실 방어 > 0 | normal z-test | 14.77 | <0.001 | *** | 1161 | 35.34 | bp/down day |
| 하방 방어 | 자기상관 보정 후 하락일 방어 > 0 | Newey-West HAC t-test | 8.187 | <0.001 | *** | 1161 | 35.34 | bp/down day |
| 하방 방어 | 하락일 초과수익 중앙값 != 0 | Wilcoxon signed-rank | 1.605e+05 | <0.001 | *** | 1161 | 12.89 | bp/down day |
| 하방 방어 | 하락일 방어 평균 bootstrap CI | iid bootstrap mean | 14.77 | <0.001 | *** | 1161 | 35.34 | bp/down day |
| 위험 감소 | K-FGI 수익률 분산이 B&H보다 낮은가 | F-test variance ratio | 0.5424 | <0.001 | *** | 2505 | 0.5424 | variance ratio |
| 위험 감소 | K-FGI와 B&H 변동성이 다른가 | Levene median test | 228.9 | <0.001 | *** | 2505 | 0.5424 | variance ratio |
| 예측 상관 | K-FGI와 target_reg 상관 | Pearson correlation | 0.0485 | 0.0152 | ** | 2505 | 0.0485 | r |
| 예측 상관 | K-FGI와 target_reg 순위상관 | Spearman rank correlation | 0.03741 | 0.0612 | * | 2505 | 0.03741 | rho |
| 예측 상관 | K-FGI와 target_5d 상관 | Pearson correlation | 0.04933 | 0.0135 | ** | 2505 | 0.04933 | r |
| 극단 구간 | Greed 상위 25%와 Fear 하위 25%의 5일 수익률 차이 | Welch t-test | 2.084 | 0.0373 | ** | 1254 | 0.3381 | %p/5D |
| EGARCH 노출 조절 | 저변동성 구간 노출도 > 고변동성 구간 노출도 | Welch t-test | 15.4 | <0.001 | *** | 1254 | 0.4866 | exposure x |
| EGARCH 노출 조절 | EGARCH 변동성과 시장 노출도 음(-)의 관계 | Spearman rank correlation | -0.2948 | <0.001 | *** | 2505 | -0.2948 | rho |
| 시장 국면 | bull/normal/crisis 국면별 K-FGI 수익률 평균 차이 | one-way ANOVA | 4.892 | 0.0076 | *** | 2505 |  |  |
| 시장 국면 | bull/normal/crisis 국면별 K-FGI 수익률 분포 차이 | Kruskal-Wallis | 5.864 | 0.0533 | * | 2505 |  |  |

## 전체 검정표

전체 CSV: `tables/extended_statistical_tests.csv`

## 본문 서술 권장 방향

본 연구 결과는 “K-FGI가 Buy&Hold 대비 전체 초과수익을 통계적으로 유의하게 창출했다”는 주장보다는, **시장 하락일과 고변동성 구간에서 노출을 줄여 하방 위험을 방어하는 전략**이라는 주장에 더 강하게 지지된다. 감성 피처는 총수익률 개선에는 기여하지만, Sharpe와 bootstrap 검정에서는 강한 유의성이 확인되지 않으므로 보조 피처로 신중하게 해석하는 것이 적절하다.
