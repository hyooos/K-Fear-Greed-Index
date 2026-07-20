# Interpretable Feature Importance and Paper Text Revision

## 실행 목적

기존 LightGBM gain 기반 feature importance는 예측모델 중심의 해석에 가깝다. 본 연구의 수정된 목표는 수익률 예측이 아니라 하락위험 방어이므로, 피처 중요도도 `수익률 예측 성능`보다 `하락위험 및 방어성공 설명력`을 기준으로 다시 산출하였다.

중요한 수정 사항은 **완성된 K-FGI와 K-FGI 구성 피처를 같은 중요도 모델에 동시에 넣지 않는 것**이다. K-FGI는 이미 시장 하위지표, 감성 피처, 변동성 피처가 결합된 최종 지표이므로, 구성요소별 중요도 분석에 K-FGI 자체를 함께 넣으면 중복 계산 문제가 발생한다. 따라서 본 문서의 Elastic Net 및 Random Forest 중요도 분석에서는 `K_FGI`를 제외하고 하위 시장/변동성 피처와 감성 피처만 투입하였다.

본 문서에서는 LightGBM을 메인 설명도구에서 내리고 다음 두 모델을 사용한다.

- Elastic Net Logistic Regression: 표준화 계수의 방향과 크기를 통해 피처가 하락위험 또는 방어성공 확률에 미치는 방향을 해석한다.
- Random Forest + permutation importance: 비선형 관계를 허용하되, 검증 구간에서 피처를 섞었을 때 AUC가 얼마나 하락하는지로 중요도를 계산한다.

## 타깃 정의

| target | label | n | positive_rate | elastic_auc | rf_auc | elastic_top_feature | rf_top_feature |
| --- | --- | --- | --- | --- | --- | --- | --- |
| target_5d_down | 5거래일 하락위험 | 2505 | 0.4415 | 0.5125 | 0.5161 | vol_shock | egarch_vol |
| target_5d_large_drop | 5거래일 -2% 급락위험 | 2505 | 0.1637 | 0.5109 | 0.4939 | egarch_vol | sub_index3 |
| defense_success_down_day | 하락일 방어성공 | 1161 | 0.5874 | 0.9266 | 0.9116 | trend_strength | sent_composite_ma10 |

## 결과 요약

## 5거래일 하락위험

- 표본 수: 2,505, positive rate: 44.2%.
- Elastic Net CV AUC: 0.512 ± 0.040, AP: 0.472, balanced accuracy: 0.498.
- Elastic Net 선택 피처: 전체 17개, 감성 4개, 시장/변동성 13개.
- Random Forest permutation CV AUC: 0.516 ± 0.020, AP: 0.470, balanced accuracy: 0.506.
- Elastic Net 최상위 피처: `vol_shock` (변동성 충격), 표준화 계수 -0.1264.
- RF 최상위 피처: `egarch_vol` (EGARCH 변동성), AUC 감소폭 0.0062.

### Elastic Net 상위 계수

| feature | feature_label | group | coef_standardized | abs_coef | selected |
| --- | --- | --- | --- | --- | --- |
| vol_shock | 변동성 충격 | 시장/변동성 | -0.1264 | 0.1264 | True |
| egarch_vol_ma5_lag1 | egarch_vol_ma5_lag1 | 시장/변동성 | -0.08523 | 0.08523 | True |
| egarch_vol_lag1 | egarch_vol_lag1 | 시장/변동성 | -0.07293 | 0.07293 | True |
| ma_ratio_60_120 | MA 60/120 | 시장/변동성 | -0.07146 | 0.07146 | True |
| sub_index7 | sub_index7 | 시장/변동성 | 0.05159 | 0.05159 | True |
| rsi14 | rsi14 | 시장/변동성 | 0.03551 | 0.03551 | True |
| vol_shock_lag1 | vol_shock_lag1 | 시장/변동성 | -0.03476 | 0.03476 | True |
| sent_energy | 감성 에너지 | 감성 | -0.02811 | 0.02811 | True |
| sent_std | 의견 불일치 | 감성 | -0.02413 | 0.02413 | True |
| vol_regime_lag1 | vol_regime_lag1 | 시장/변동성 | -0.01862 | 0.01862 | True |
| ma_ratio_20_60 | MA 20/60 | 시장/변동성 | 0.01721 | 0.01721 | True |
| panic_pressure | 공포 압력 | 감성 | 0.009834 | 0.009834 | True |

### Elastic Net 감성 피처

| feature | feature_label | coef_standardized | abs_coef | selected |
| --- | --- | --- | --- | --- |
| sent_energy | 감성 에너지 | -0.02811 | 0.02811 | True |
| sent_std | 의견 불일치 | -0.02413 | 0.02413 | True |
| panic_pressure | 공포 압력 | 0.009834 | 0.009834 | True |
| sent_norm_w | 정규화 감성 | -0.002921 | 0.002921 | True |
| neg_z | 부정 감성 | 0 | 0 | False |
| sent_composite | 감성 복합 | 0 | 0 | False |
| sent_composite_ma10 | 감성 복합 10일 | 0 | 0 | False |
| sent_attention | 감성 관심 | 0 | 0 | False |

### Random Forest permutation importance

| feature | feature_label | group | importance_auc_drop | importance_std | positive_fold_rate |
| --- | --- | --- | --- | --- | --- |
| egarch_vol | EGARCH 변동성 | 시장/변동성 | 0.006165 | 0.005855 | 0.8 |
| sub_index3 | sub_index3 | 시장/변동성 | 0.004227 | 0.001851 | 1 |
| ma_ratio_5_20 | MA 5/20 | 시장/변동성 | 0.002979 | 0.007709 | 0.6 |
| sub_index6 | sub_index6 | 시장/변동성 | 0.002754 | 0.004019 | 0.8 |
| egarch_vol_lag1 | egarch_vol_lag1 | 시장/변동성 | 0.002505 | 0.001311 | 1 |
| neg_z | 부정 감성 | 감성 | 0.002186 | 0.007587 | 0.8 |
| vol_shock_lag1 | vol_shock_lag1 | 시장/변동성 | 0.001987 | 0.003423 | 0.8 |
| egarch_vol_ma5_lag1 | egarch_vol_ma5_lag1 | 시장/변동성 | 0.001494 | 0.003882 | 0.6 |
| sentiment_risk_composite | 감성위험 복합 | 감성 | 0.001126 | 0.001735 | 0.8 |
| ma_ratio_60_120 | MA 60/120 | 시장/변동성 | 0.001063 | 0.01771 | 0.6 |
| negative_attention | 부정 관심 | 감성 | 0.0008425 | 0.001534 | 0.8 |
| sent_confidence | 감성 신뢰도 | 감성 | 0.0007893 | 0.001779 | 0.6 |

### Random Forest 감성 피처

| feature | feature_label | importance_auc_drop | importance_std | positive_fold_rate |
| --- | --- | --- | --- | --- |
| neg_z | 부정 감성 | 0.002186 | 0.007587 | 0.8 |
| sentiment_risk_composite | 감성위험 복합 | 0.001126 | 0.001735 | 0.8 |
| negative_attention | 부정 관심 | 0.0008425 | 0.001534 | 0.8 |
| sent_confidence | 감성 신뢰도 | 0.0007893 | 0.001779 | 0.6 |
| sent_norm_w | 정규화 감성 | 0.0004813 | 0.002342 | 0.8 |
| sent_energy | 감성 에너지 | 0.0003621 | 0.001531 | 0.6 |
| disagreement_attention | 불일치 관심 | 0.0001921 | 0.001207 | 0.4 |
| sent_attention | 감성 관심 | 0.0001398 | 0.0009226 | 0.6 |

## 5거래일 -2% 급락위험

- 표본 수: 2,505, positive rate: 16.4%.
- Elastic Net CV AUC: 0.511 ± 0.061, AP: 0.193, balanced accuracy: 0.509.
- Elastic Net 선택 피처: 전체 38개, 감성 14개, 시장/변동성 24개.
- Random Forest permutation CV AUC: 0.494 ± 0.064, AP: 0.193, balanced accuracy: 0.505.
- Elastic Net 최상위 피처: `egarch_vol` (EGARCH 변동성), 표준화 계수 0.6887.
- RF 최상위 피처: `sub_index3` (sub_index3), AUC 감소폭 0.0067.

### Elastic Net 상위 계수

| feature | feature_label | group | coef_standardized | abs_coef | selected |
| --- | --- | --- | --- | --- | --- |
| egarch_vol | EGARCH 변동성 | 시장/변동성 | 0.6887 | 0.6887 | True |
| sent_attention | 감성 관심 | 감성 | -0.6294 | 0.6294 | True |
| disagreement_attention | 불일치 관심 | 감성 | -0.4911 | 0.4911 | True |
| rsi14 | rsi14 | 시장/변동성 | 0.4899 | 0.4899 | True |
| mom20 | 20일 모멘텀 | 시장/변동성 | -0.4417 | 0.4417 | True |
| vol_ratio | 변동성 비율 | 시장/변동성 | -0.3857 | 0.3857 | True |
| trend_strength | 추세 강도 | 시장/변동성 | -0.3338 | 0.3338 | True |
| sent_confidence | 감성 신뢰도 | 감성 | -0.2967 | 0.2967 | True |
| egarch_vol_ma5_lag1 | egarch_vol_ma5_lag1 | 시장/변동성 | -0.2921 | 0.2921 | True |
| sent_norm_w | 정규화 감성 | 감성 | 0.2813 | 0.2813 | True |
| vol_shock | 변동성 충격 | 시장/변동성 | -0.2805 | 0.2805 | True |
| egarch_vol_lag1 | egarch_vol_lag1 | 시장/변동성 | -0.2653 | 0.2653 | True |

### Elastic Net 감성 피처

| feature | feature_label | coef_standardized | abs_coef | selected |
| --- | --- | --- | --- | --- |
| sent_attention | 감성 관심 | -0.6294 | 0.6294 | True |
| disagreement_attention | 불일치 관심 | -0.4911 | 0.4911 | True |
| sent_confidence | 감성 신뢰도 | -0.2967 | 0.2967 | True |
| sent_norm_w | 정규화 감성 | 0.2813 | 0.2813 | True |
| neg_z | 부정 감성 | 0.1832 | 0.1832 | True |
| panic_pressure | 공포 압력 | 0.1209 | 0.1209 | True |
| effective_n | 유효 댓글 수 | 0.09545 | 0.09545 | True |
| negative_attention | 부정 관심 | -0.08468 | 0.08468 | True |

### Random Forest permutation importance

| feature | feature_label | group | importance_auc_drop | importance_std | positive_fold_rate |
| --- | --- | --- | --- | --- | --- |
| sub_index3 | sub_index3 | 시장/변동성 | 0.006745 | 0.008892 | 0.8 |
| ma_ratio_60_120 | MA 60/120 | 시장/변동성 | 0.006342 | 0.03268 | 0.8 |
| sub_index5 | sub_index5 | 시장/변동성 | 0.00429 | 0.005845 | 0.8 |
| sub_index4 | sub_index4 | 시장/변동성 | 0.003644 | 0.006553 | 0.8 |
| sent_composite_ma10 | 감성 복합 10일 | 감성 | 0.003278 | 0.005992 | 0.6 |
| sent_confidence | 감성 신뢰도 | 감성 | 0.002254 | 0.00177 | 0.8 |
| sent_strength_w | 감성 강도 | 감성 | 0.002133 | 0.005146 | 0.6 |
| egarch_vol_lag1 | egarch_vol_lag1 | 시장/변동성 | 0.00207 | 0.003315 | 0.6 |
| neg_z | 부정 감성 | 감성 | 0.002034 | 0.004319 | 0.8 |
| sent_attention | 감성 관심 | 감성 | 0.00201 | 0.003132 | 0.6 |
| egarch_vol_ma5_lag1 | egarch_vol_ma5_lag1 | 시장/변동성 | 0.001847 | 0.002593 | 0.8 |
| sent_energy | 감성 에너지 | 감성 | 0.001397 | 0.003114 | 0.6 |

### Random Forest 감성 피처

| feature | feature_label | importance_auc_drop | importance_std | positive_fold_rate |
| --- | --- | --- | --- | --- |
| sent_composite_ma10 | 감성 복합 10일 | 0.003278 | 0.005992 | 0.6 |
| sent_confidence | 감성 신뢰도 | 0.002254 | 0.00177 | 0.8 |
| sent_strength_w | 감성 강도 | 0.002133 | 0.005146 | 0.6 |
| neg_z | 부정 감성 | 0.002034 | 0.004319 | 0.8 |
| sent_attention | 감성 관심 | 0.00201 | 0.003132 | 0.6 |
| sent_energy | 감성 에너지 | 0.001397 | 0.003114 | 0.6 |
| sent_norm_w | 정규화 감성 | 0.00124 | 0.003238 | 0.6 |
| sentiment_risk_composite | 감성위험 복합 | 0.0006068 | 0.00139 | 0.8 |

## 하락일 방어성공

- 표본 수: 1,161, positive rate: 58.7%.
- Elastic Net CV AUC: 0.927 ± 0.043, AP: 0.953, balanced accuracy: 0.849.
- Elastic Net 선택 피처: 전체 23개, 감성 5개, 시장/변동성 18개.
- Random Forest permutation CV AUC: 0.912 ± 0.040, AP: 0.944, balanced accuracy: 0.825.
- Elastic Net 최상위 피처: `trend_strength` (추세 강도), 표준화 계수 -1.7658.
- RF 최상위 피처: `sent_composite_ma10` (감성 복합 10일), AUC 감소폭 0.0244.

### Elastic Net 상위 계수

| feature | feature_label | group | coef_standardized | abs_coef | selected |
| --- | --- | --- | --- | --- | --- |
| trend_strength | 추세 강도 | 시장/변동성 | -1.766 | 1.766 | True |
| sent_composite_ma10 | 감성 복합 10일 | 감성 | -1.265 | 1.265 | True |
| egarch_vol_lag1 | egarch_vol_lag1 | 시장/변동성 | 1.055 | 1.055 | True |
| sub_index2 | sub_index2 | 시장/변동성 | -0.3869 | 0.3869 | True |
| sub_index5 | sub_index5 | 시장/변동성 | -0.3074 | 0.3074 | True |
| vol_regime_high | 고변동성 국면 | 시장/변동성 | -0.2755 | 0.2755 | True |
| sub_index6 | sub_index6 | 시장/변동성 | -0.2629 | 0.2629 | True |
| ma_ratio_5_20 | MA 5/20 | 시장/변동성 | -0.2588 | 0.2588 | True |
| ma_ratio_20_60 | MA 20/60 | 시장/변동성 | -0.2355 | 0.2355 | True |
| negative_attention | 부정 관심 | 감성 | -0.157 | 0.157 | True |
| vol_regime_lag1 | vol_regime_lag1 | 시장/변동성 | 0.1563 | 0.1563 | True |
| egarch_vol | EGARCH 변동성 | 시장/변동성 | 0.1203 | 0.1203 | True |

### Elastic Net 감성 피처

| feature | feature_label | coef_standardized | abs_coef | selected |
| --- | --- | --- | --- | --- |
| sent_composite_ma10 | 감성 복합 10일 | -1.265 | 1.265 | True |
| negative_attention | 부정 관심 | -0.157 | 0.157 | True |
| sent_std | 의견 불일치 | 0.04136 | 0.04136 | True |
| panic_pressure | 공포 압력 | 0.01011 | 0.01011 | True |
| sent_norm_w | 정규화 감성 | -0.004607 | 0.004607 | True |
| disagreement_attention | 불일치 관심 | 0 | 0 | False |
| neg_z | 부정 감성 | 0 | 0 | False |
| sent_energy | 감성 에너지 | 0 | 0 | False |

### Random Forest permutation importance

| feature | feature_label | group | importance_auc_drop | importance_std | positive_fold_rate |
| --- | --- | --- | --- | --- | --- |
| sent_composite_ma10 | 감성 복합 10일 | 감성 | 0.02439 | 0.01552 | 1 |
| trend_strength | 추세 강도 | 시장/변동성 | 0.018 | 0.01109 | 1 |
| ma_ratio_5_20 | MA 5/20 | 시장/변동성 | 0.01457 | 0.01055 | 1 |
| mom60 | 60일 모멘텀 | 시장/변동성 | 0.01327 | 0.007244 | 1 |
| rsi14 | rsi14 | 시장/변동성 | 0.009556 | 0.008719 | 0.8 |
| ma_ratio_20_60 | MA 20/60 | 시장/변동성 | 0.005511 | 0.004725 | 1 |
| sub_index2 | sub_index2 | 시장/변동성 | 0.004328 | 0.007544 | 0.8 |
| mom20 | 20일 모멘텀 | 시장/변동성 | 0.002975 | 0.00446 | 0.6 |
| sub_index3 | sub_index3 | 시장/변동성 | 0.001853 | 0.002305 | 0.8 |
| sub_index5 | sub_index5 | 시장/변동성 | 0.001366 | 0.001039 | 1 |
| sub_index4 | sub_index4 | 시장/변동성 | 0.0007773 | 0.0005898 | 0.8 |
| effective_n | 유효 댓글 수 | 감성 | 0.0007492 | 0.001155 | 1 |

### Random Forest 감성 피처

| feature | feature_label | importance_auc_drop | importance_std | positive_fold_rate |
| --- | --- | --- | --- | --- |
| sent_composite_ma10 | 감성 복합 10일 | 0.02439 | 0.01552 | 1 |
| effective_n | 유효 댓글 수 | 0.0007492 | 0.001155 | 1 |
| sent_attention | 감성 관심 | 0.0004151 | 0.0003892 | 0.8 |
| panic_pressure | 공포 압력 | 0.0003856 | 0.0008301 | 0.8 |
| disagreement_attention | 불일치 관심 | 0.000344 | 0.0006101 | 0.8 |
| sent_std | 의견 불일치 | 0.0002962 | 0.0006348 | 0.8 |
| sent_norm_w | 정규화 감성 | 0.0002952 | 0.0006997 | 0.6 |
| sent_strength_w | 감성 강도 | 0.00026 | 0.0007196 | 0.8 |


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
