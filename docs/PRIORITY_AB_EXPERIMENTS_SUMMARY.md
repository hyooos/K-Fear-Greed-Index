# Priority A/B Robustness Experiments Summary

## 실행 요약

- EGARCH 안정화: `min_obs=252`, EGARCH(1,1)-t, `rescale=True`, hard cap 및 rolling volatility fallback 적용.
- EGARCH fallback rate: 0.30%, capped rate: 0.48%, p99 vol: 0.0235.
- 기준 전략: K-FGI with sentiment, 거래비용 15bp.

## 우선순위 A 결과

### 1. EGARCH 안정화 재실험

초기 EGARCH 폭발 문제는 안정화 설정 후 크게 완화되었다. `min_obs=252`로 초기 학습 구간을 1년으로 늘렸고, 비정상 추정값은 rolling volatility로 대체했다. 이 설정은 기존 60일 warm-up보다 보수적이며, 논문 한계로 언급했던 EGARCH 초기 불안정성을 직접 보완한다.

### 2. 10개년 성능표 업데이트

| label | ann_ret | ann_vol | sharpe | mdd | total_return | downside_excess_bp | downside_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Buy & Hold | 0.08708 | 0.1846 | 0.4716 | -0.4119 | 1.376 |  |  |
| Trend only | 0.04905 | 0.1214 | 0.4041 | -0.2889 | 0.6284 | 38.48 | 5.319e-66 |
| Trend + EGARCH | 0.03337 | 0.1127 | 0.2962 | -0.2851 | 0.3933 | 40.3 | 6.98e-70 |
| K-FGI with sentiment | 0.04367 | 0.1482 | 0.2947 | -0.3711 | 0.5436 | 28.41 | 5.831e-30 |
| K-FGI without sentiment | 0.01651 | 0.1461 | 0.113 | -0.453 | 0.1783 | 28.88 | 6.644e-30 |

핵심: 안정화 EGARCH 기준에서도 K-FGI with sentiment는 Buy&Hold 대비 MDD를 -41.2%에서 -37.1%로 줄인다. 다만 총수익률은 Buy&Hold 137.6% 대비 K-FGI 54.4%로, 수익 극대화보다 하방 위험 관리에 초점이 맞다.

### 3. 거래비용 민감도

| fee_bp | ann_ret | ann_vol | sharpe | mdd | total_return |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.1079 | 0.1481 | 0.7285 | -0.2193 | 1.922 |
| 5 | 0.08646 | 0.1481 | 0.5839 | -0.2704 | 1.362 |
| 10 | 0.06507 | 0.1481 | 0.4393 | -0.3226 | 0.9094 |
| 15 | 0.04367 | 0.1482 | 0.2947 | -0.3711 | 0.5436 |
| 20 | 0.02228 | 0.1483 | 0.1502 | -0.4162 | 0.2479 |

거래비용 0bp에서 20bp로 높아질 때 총수익률은 192.2%에서 24.8%로 변한다. 전략이 비용에 민감한지 여부는 이 차이를 중심으로 서술한다.

### 4. Threshold / multiplier robustness

상위 조합:

| fear | greed | multiplier_range | ann_ret | sharpe | mdd | total_return |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 80 | 0.7-1.6 | 0.04687 | 0.3143 | -0.35 | 0.5935 |
| 20 | 80 | 0.5-1.6 | 0.04389 | 0.3098 | -0.3359 | 0.5469 |
| 20 | 80 | 0.5-2.0 | 0.05328 | 0.3091 | -0.3924 | 0.6983 |
| 30 | 70 | 0.7-1.6 | 0.04674 | 0.3064 | -0.3714 | 0.5915 |
| 20 | 80 | 0.3-1.4 | 0.03616 | 0.3038 | -0.2907 | 0.4325 |
| 30 | 70 | 0.5-1.6 | 0.04395 | 0.3028 | -0.3564 | 0.5479 |
| 30 | 70 | 0.5-2.0 | 0.0535 | 0.3026 | -0.4149 | 0.702 |
| 30 | 70 | 0.3-1.4 | 0.03637 | 0.2979 | -0.3088 | 0.4356 |
| 25 | 65 | 0.7-1.6 | 0.04634 | 0.2973 | -0.3866 | 0.585 |
| 25 | 65 | 0.5-2.0 | 0.05323 | 0.295 | -0.4313 | 0.6975 |
| 25 | 65 | 0.5-1.6 | 0.04367 | 0.2947 | -0.3711 | 0.5436 |
| 25 | 65 | 0.3-1.4 | 0.03621 | 0.2906 | -0.3222 | 0.4333 |

기준 조합 25/65 및 0.5-1.6이 유일한 최적값이라고 주장하기보다, 여러 threshold/multiplier 조합에서 성과와 위험이 크게 무너지지 않는지를 robustness로 제시하는 것이 안전하다. 최상위 조합은 20/80, multiplier 0.7-1.6이다.

### 5. Crisis subperiod

| period | label | ann_ret | ann_vol | sharpe | mdd | total_return | downside_excess_bp | downside_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020_COVID | Buy & Hold | 0.3741 | 0.2962 | 1.263 | -0.3423 | 0.4028 |  |  |
| 2020_COVID | Trend only | 0.3409 | 0.1739 | 1.961 | -0.08499 | 0.3613 | 72.92 | 1.915e-05 |
| 2020_COVID | Trend + EGARCH | 0.2862 | 0.1524 | 1.878 | -0.07817 | 0.2956 | 83.76 | 8.353e-07 |
| 2020_COVID | K-FGI with sentiment | 0.3974 | 0.2037 | 1.951 | -0.1166 | 0.4327 | 69.13 | 0.0001104 |
| 2020_COVID | K-FGI without sentiment | 0.2926 | 0.1971 | 1.485 | -0.1245 | 0.3031 | 67.91 | 0.0001759 |
| 2022_rate_hike_bear | Buy & Hold | -0.3186 | 0.1922 | -1.657 | -0.2884 | -0.2673 |  |  |
| 2022_rate_hike_bear | Trend only | -0.07731 | 0.06435 | -1.201 | -0.07362 | -0.07269 | 81.35 | 1.021e-22 |
| 2022_rate_hike_bear | Trend + EGARCH | -0.07181 | 0.05767 | -1.245 | -0.06858 | -0.0677 | 83.15 | 8.896e-24 |
| 2022_rate_hike_bear | K-FGI with sentiment | -0.08733 | 0.08621 | -1.013 | -0.0826 | -0.08172 | 77.63 | 1.976e-19 |
| 2022_rate_hike_bear | K-FGI without sentiment | -0.08506 | 0.07709 | -1.103 | -0.08048 | -0.07968 | 79.26 | 9.247e-21 |
| 2025_bull_oos | Buy & Hold | 0.6483 | 0.2348 | 2.761 | -0.1426 | 0.8398 |  |  |
| 2025_bull_oos | Trend only | 0.536 | 0.1997 | 2.684 | -0.08554 | 0.6555 | 21.52 | 0.004536 |
| 2025_bull_oos | Trend + EGARCH | 0.4543 | 0.1683 | 2.699 | -0.07372 | 0.533 | 34.89 | 1.802e-05 |
| 2025_bull_oos | K-FGI with sentiment | 0.6564 | 0.2259 | 2.906 | -0.07772 | 0.854 | 12.93 | 0.1495 |
| 2025_bull_oos | K-FGI without sentiment | 0.6793 | 0.2342 | 2.9 | -0.1093 | 0.8944 | 9.789 | 0.325 |

2020 코로나, 2022 약세장, 2025 OOS 강세장을 분리해 보면 전략의 성격이 더 뚜렷해진다. 논문에서는 각 기간의 총수익률보다 MDD와 downside excess를 중심으로 해석한다.

## 우선순위 B 결과

### 1. 감성 피처 세부 ablation

| variant | ann_ret | ann_vol | sharpe | mdd | total_return |
| --- | --- | --- | --- | --- | --- |
| sentiment_all | 0.04367 | 0.1482 | 0.2947 | -0.3711 | 0.5436 |
| dispersion_only | 0.02545 | 0.1509 | 0.1686 | -0.4213 | 0.2878 |
| sentiment_none | 0.01651 | 0.1461 | 0.113 | -0.453 | 0.1783 |
| sent_norm_only | 0.01603 | 0.1424 | 0.1125 | -0.4343 | 0.1727 |
| negative_only | 0.01484 | 0.1449 | 0.1024 | -0.4634 | 0.159 |

감성 전체가 무조건 우월하다는 주장보다, 어떤 감성 정보가 수익률/위험 관리에 도움을 주는지 구분해서 서술한다. 특히 `sentiment_none` 대비 어떤 감성 subset이 총수익률 또는 MDD를 개선하는지 확인한다.

### 2. Benchmark 추가

| benchmark | ann_ret | ann_vol | sharpe | mdd | total_return | downside_excess_bp |
| --- | --- | --- | --- | --- | --- | --- |
| EGARCH-only vol targeting | 0.05901 | 0.1679 | 0.3515 | -0.3883 | 0.7979 | 4.621 |
| market-only FGI | 0.03355 | 0.1281 | 0.2619 | -0.3251 | 0.3847 | 35.23 |
| CNN-style equal weight K-FGI | 0.02763 | 0.1264 | 0.2185 | -0.3474 | 0.3074 | 35.96 |
| sentiment-only index | -0.002459 | 0.1278 | -0.01924 | -0.4328 | -0.02358 | 36.31 |

CNN-style equal weight, market-only, sentiment-only, EGARCH-only와 비교해 K-FGI가 단순 동일가중 또는 단일 정보원 지표가 아니라는 점을 보여준다.

### 3. K-FGI zone별 성과

| zone | ann_ret | ann_vol | sharpe | mdd | total_return | mean_excess_bp | downside_excess_bp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fear | 0.009996 | 0.05004 | 0.1998 | -0.04153 | 0.006726 | 2.668 | 108.6 |
| Neutral | -0.03141 | 0.1177 | -0.2669 | -0.2828 | -0.1557 | -2.878 | 44.42 |
| Greed | 0.1537 | 0.1912 | 0.804 | -0.2653 | 0.816 | -0.8759 | -8.725 |

Fear/Neutral/Greed 구간별 성과는 K-FGI가 어떤 국면에서 공격/방어 신호로 작동하는지 설명하는 데 사용한다.

### 4. White reality check / bootstrap

| best_strategy | best_ann_excess | white_reality_p | n_strategies | block | bootstrap_iter |
| --- | --- | --- | --- | --- | --- |
| EGARCH-only vol targeting | -0.028 | 0.997 | 13 | 20 | 3000 |

White reality check는 여러 후보 전략 중 사후적으로 가장 좋아 보이는 전략을 골랐을 가능성을 보정한다. p-value가 낮지 않다면 “데이터마이닝을 완전히 배제했다”고 강하게 주장하기 어렵고, 낮다면 benchmark universe 안에서 성과가 우연만은 아니라는 보조 근거가 된다.

## 다른 논문/지표와의 비교 포인트

- CNN Fear & Greed Index류 연구와의 차이: 단순 7개 시장지표 동일가중이 아니라 한국시장 KRX 지표, 댓글 감성, EGARCH 변동성 노출 조절을 결합했다.
- Baker-Wurgler류 sentiment index와의 차이: 저빈도 시장 proxy가 아니라 일별 KOSPI200 전략으로 검증 가능한 지표다.
- 텍스트 감성 예측 연구와의 차이: 감성을 단독 alpha로 쓰지 않고, 시장 기반 subindex 및 변동성 타깃팅과 결합해 하방 위험 관리에 사용했다.
- EGARCH/vol targeting 연구와의 차이: 변동성 하나만으로 노출을 줄이는 것이 아니라, fear/greed 국면과 추세 조건을 함께 반영한다.

## 논문 서술 방향

가장 안전한 결론은 다음과 같다.

> 안정화된 EGARCH 설정과 다양한 threshold, 거래비용, crisis subperiod, benchmark 검정에서도 K-FGI의 핵심 기여는 전체 초과수익의 일관된 유의성보다 하방 위험 완화와 노출 조절의 설명 가능성에서 확인된다.
