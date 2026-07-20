# K-FGI Feature Formula and Drop-Ablation Guide

## 1. 표본 및 컬럼 구조

- 최종 robust 데이터: `paper_outputs/robustness/data/priority_ab_kfgi_timeseries_robust.csv`
- 관측치: 2,505 거래일
- 전체 컬럼 수: 73개
- 해석가능 피처 중요도 입력: 40개
  - 감성 피처 16개
  - 시장/변동성 피처 24개
- K-FGI 산출용 기본 피처: 15개
  - 시장 하위지표 6개
  - 감성 피처 6개
  - EGARCH/변동성 피처 3개

주의: `sub_index1`은 원본 데이터에는 남겨두었지만, 현재 K-FGI 산출 및 모델링 피처에서는 제외한다.

## 2. 기본 표기

- 날짜는 \(t\)로 표기한다.
- \(P_t\): KOSPI200 종가 또는 KOSPI 종가
- \(r_t = \log(P_t / P_{t-1})\)
- \(r_{t+1}\): 다음 거래일 로그수익률
- \(\sigma^{EGARCH}_t\): walk-forward EGARCH 조건부 변동성
- \(\text{rankpct}_t(x)\): 해당 시점 이전 과거분포에서 현재 값의 percentile rank
- 모든 최종 피처는 look-ahead bias를 막기 위해 원칙적으로 1일 lag 처리된다.

## 3. 원천 입력 피처

| 컬럼 | 의미 | 수식/출처 | 사용 |
| --- | --- | --- | --- |
| `sub_index1` | 시장 모멘텀 | 외부 산출된 momentum score | 원본 보존, 최종 제외 |
| `sub_index2` | 주가 강도 | KRX/KOSPI 구성 종목 기반 strength score | K-FGI, 중요도 |
| `sub_index3` | 시장 폭 | 상승/하락 breadth 기반 score | K-FGI, 중요도 |
| `sub_index4` | Put/Call | KOSPI200 옵션 put-call 관련 score | K-FGI, 중요도 |
| `sub_index5` | 변동성 지수 | VKOSPI/KOSPI200 변동성 score | K-FGI, 중요도 |
| `sub_index6` | 안전자산 수요 | 국채/주식 상대수요 score | K-FGI, 중요도 |
| `sub_index7` | 위험채권 수요 | junk/safe demand proxy | K-FGI, 중요도 |
| `sent_norm_w` | 정규화 감성 | 댓글 단위 감성점수의 일별 가중 집계 | K-FGI, 중요도 |
| `sent_strength_w` | 감성 강도 | 댓글 감성의 강도 가중 집계 | 파생 피처 |
| `sent_std` | 의견 불일치 | 일별 댓글 감성 점수 표준편차 | 파생 피처 |
| `neg_z` | 부정 감성 | 일별 부정 감성 z-score 계열 | K-FGI, 중요도 |
| `effective_n` | 유효 댓글 수 | 필터링 이후 분석에 사용된 댓글 수 | 중요도 |
| `heat` | 댓글 열기 | 댓글 관심/열기 지표 | 중요도 |
| `comment_count` | 원 댓글 수 | 일별 전체 댓글 수 | 중요도 |

## 4. 수익률 및 타깃 피처

| 컬럼 | 수식 | 설명 |
| --- | --- | --- |
| `log_return` | \(r_t=\log(P_t/P_{t-1})\) | 당일 로그수익률 |
| `log_return_t+1` | \(r_{t+1}\) | 다음 거래일 로그수익률 |
| `target_reg` | \(r_{t+1}\) | 전략 수익률 계산용 다음날 수익률 |
| `target_5d` | \(\sum_{i=1}^{5} r_{t+i}\) | 향후 5거래일 누적 로그수익률 |
| `target_5d_down` | \(1[\text{target\_5d}<0]\) | 5일 하락 여부, 중요도 분석 타깃 |
| `target_5d_large_drop` | \(1[\text{target\_5d}\le -0.02]\) | 5일 -2% 급락 여부, 중요도 분석 타깃 |
| `defense_success_down_day` | \(1[r^{BH}_t<0 \land r^{KFGI}_t>r^{BH}_t]\) | 하락일 방어성공 타깃. 입력 피처가 아니라 결과 변수 |

## 5. 감성 파생 피처

### 5.1 K-FGI 산출에 직접 들어간 감성 피처

| 컬럼 | 수식 | 방향 |
| --- | --- | --- |
| `sent_norm_w` | 원천 감성 정규화 점수 | 높을수록 greed |
| `neg_z_inv` | \(-\text{neg\_z}\) | 부정 감성이 낮을수록 greed |
| `sent_std_inv` | \(-\text{sent\_std}\) | 의견 불일치가 낮을수록 greed |
| `sent_energy` | \(\text{sent\_strength\_w} \times \text{sent\_norm\_w}\) | 강한 긍정 감성일수록 greed |
| `sent_composite` | \(0.4\cdot\text{sent\_norm\_w}+0.3\cdot\text{neg\_z\_inv}+0.3\cdot\text{sent\_energy}\) | 복합 감성 |
| `sent_composite_ma10` | \(MA_{10}(\text{sent\_composite})\) | 감성 복합 10일 이동평균 |

### 5.2 보조 감성 피처

| 컬럼 | 수식 | 용도 |
| --- | --- | --- |
| `sent_norm_ma5` | \(MA_5(\text{sent\_norm\_w})\) | 감성 추세 확인 |
| `sent_norm_diff` | \(\Delta \text{sent\_norm\_w}\) | 감성 변화율 |
| `neg_z_ma5` | \(MA_5(\text{neg\_z})\) | 부정 감성 추세 |
| `sent_composite_diff` | \(\text{sent\_composite}_t-\text{sent\_composite}_{t-5}\) | 복합 감성 변화 |
| `neg_z_vol_ratio` | \(\text{clip}(\text{neg\_z\_inv}/(\text{egarch\_vol\_lag1}+10^{-6}), -10, 10)\) | 부정 감성과 변동성 결합 |

### 5.3 최근 추가한 감성 위험 피처

이 피처들은 K-FGI 산출용 기본 피처가 아니라, 감성의 조건부 기여도와 피처 중요도를 보기 위해 추가한 해석용 피처다.

| 컬럼 | 수식 | 해석 |
| --- | --- | --- |
| `sent_attention` | \(\text{sent\_norm\_w}\times\log(1+\text{effective\_n})\) | 감성 방향에 댓글 관심도를 곱함 |
| `negative_attention` | \(\max(\text{neg\_z},0)\times\log(1+\text{effective\_n})\) | 부정 감성 관심도 |
| `panic_pressure` | \(\max(-\text{sent\_norm\_w},0)\times\max(\text{sent\_strength\_w},0)\times\log(1+\text{effective\_n})\) | 강한 부정 감성과 댓글 관심이 동시에 높을 때 커짐 |
| `disagreement_attention` | \(\max(\text{sent\_std},0)\times\log(1+\text{effective\_n})\) | 의견 불일치가 관심과 함께 커지는 정도 |
| `heat_attention` | \(\max(\text{heat},0)\times\log(1+\text{comment\_count})\) | 댓글 열기와 댓글 수 결합 |
| `sent_confidence` | \(|\text{sent\_norm\_w}|\times\log(1+\text{effective\_n})/(|\text{sent\_std}|+0.05)\) | 의견 불일치가 낮고 감성이 뚜렷할수록 큼 |
| `sentiment_risk_composite` | 평균\((rankpct(\text{negative\_attention}), rankpct(\text{panic\_pressure}), rankpct(\text{disagreement\_attention}), rankpct(\text{heat\_attention}))\) | 부정/공포/불일치/관심을 결합한 감성 위험 지표 |

## 6. EGARCH 및 변동성 피처

| 컬럼 | 수식 | 설명 |
| --- | --- | --- |
| `egarch_vol` | EGARCH(1,1)-t walk-forward \(\sigma_t\) | min_obs=252, fallback/cap 적용 |
| `vol_shock` | \(\text{clip}(\Delta \text{egarch\_vol}/\text{egarch\_vol}_{t-1}, -5, 5)\) | 변동성 충격 |
| `vol_regime_high` | \(1[\text{egarch\_vol}_t>Q_{0.7}(\text{egarch\_vol}_{t-59:t})]\) | 60일 rolling 70% 분위수 이상 |
| `vol_ma5` | \(MA_5(\text{egarch\_vol})\) | 단기 변동성 평균 |
| `vol_trend` | \(1[\text{egarch\_vol}_t>\text{vol\_ma5}_t]\) | 변동성 상승 추세 |
| `vol_ratio` | \(\text{egarch\_vol}_t/(MA_{60}(\text{egarch\_vol})+10^{-9})\) | 장기 평균 대비 변동성 |
| `egarch_vol_lag1` | \(\text{egarch\_vol}_{t-1}\) | 지연 변동성 |
| `egarch_vol_ma5_lag1` | \(\text{vol\_ma5}_{t-1}\) | 지연 5일 변동성 |
| `vol_shock_lag1` | \(\text{vol\_shock}_{t-1}\) | 지연 변동성 충격 |
| `vol_regime_lag1` | \(\text{vol\_regime\_high}_{t-1}\) | 지연 고변동성 국면 |

## 7. 가격/추세 피처

| 컬럼 | 수식 | 설명 |
| --- | --- | --- |
| `ma20` | \(MA_{20}(P_t)\) | 20일 이동평균 |
| `ma60` | \(MA_{60}(P_t)\) | 60일 이동평균 |
| `ma120` | \(MA_{120}(P_t)\) | 120일 이동평균 |
| `ma_ratio_5_20` | \(MA_5(P_t)/(MA_{20}(P_t)+10^{-9})-1\) | 단기/중기 추세 |
| `ma_ratio_20_60` | \(MA_{20}(P_t)/(MA_{60}(P_t)+10^{-9})-1\) | 중기 추세 |
| `ma_ratio_60_120` | \(MA_{60}(P_t)/(MA_{120}(P_t)+10^{-9})-1\) | 장기 추세 |
| `rsi14` | \(100-100/(1+EMA(gain,14)/(EMA(loss,14)+10^{-9}))\) | RSI |
| `mom5` | \(P_t/P_{t-5}-1\) | 5일 모멘텀 |
| `mom20` | \(P_t/P_{t-20}-1\) | 20일 모멘텀 |
| `mom60` | \(P_t/P_{t-60}-1\) | 60일 모멘텀 |
| `above_ma20` | \(1[P_t>MA_{20}(P_t)]\) | 20일선 상회 |
| `above_ma60` | \(1[P_t>MA_{60}(P_t)]\) | 60일선 상회 |
| `above_ma120` | \(1[P_t>MA_{120}(P_t)]\) | 120일선 상회 |
| `trend_strength` | `above_ma20 + above_ma60 + above_ma120` | 0~3 추세 강도 |

## 8. K-FGI 산출 수식

### 8.1 입력 피처

현재 K-FGI with sentiment 입력 피처:

```text
sub_index2, sub_index3, sub_index4, sub_index5, sub_index6, sub_index7,
sent_norm_w, sent_energy, sent_std_inv, neg_z_inv, sent_composite, sent_composite_ma10,
egarch_vol, vol_regime_high, vol_ratio
```

감성 제외 K-FGI 입력 피처:

```text
sub_index2, sub_index3, sub_index4, sub_index5, sub_index6, sub_index7,
egarch_vol, vol_regime_high, vol_ratio
```

### 8.2 방향 제약

| 피처군 | 방향 |
| --- | --- |
| `sub_index2`~`sub_index7` | \(+\) |
| 감성 피처 | \(+\) |
| `egarch_vol`, `vol_regime_high`, `vol_ratio` | \(-\) |

### 8.3 Walk-forward Ridge 제약 회귀

각 시점 \(t\)에서 과거 자료 \(1,\dots,t-1\)만 사용한다.

\[
\hat{\beta}_t =
\arg\min_{\beta}
\sum_{\tau < t}
\left(y_{\tau}^{5D} - X_{\tau}\beta\right)^2
+ \alpha \sum_j \beta_j^2
\]

방향 제약:

\[
\beta_j \ge 0 \quad \text{if direction}_j=+1
\]

\[
\beta_j \le 0 \quad \text{if direction}_j=-1
\]

계수 정규화:

\[
w_{j,t} = \frac{\hat{\beta}_{j,t}}{\sum_k |\hat{\beta}_{k,t}| + 10^{-12}}
\]

raw score:

\[
S_t = \tilde{X}_t w_t
\]

여기서 \(\tilde{X}_t\)는 과거 학습구간으로 표준화하고 \([-3,3]\)으로 clip한 피처다.

최종 K-FGI 0~100 변환:

\[
KFGI_t =
100 \times
\frac{\text{clip}(S_t, P_1(S), P_{99}(S))-P_1(S)}
{P_{99}(S)-P_1(S)+10^{-12}}
\]

## 9. 포지션 및 전략 수식

### 9.1 국면 분류

\[
\text{bull}_t =
1[KFGI_t>65 \land \text{vol\_regime\_high}_t=0 \land \text{trend\_strength}_t\ge 2]
\]

\[
\text{crisis}_t =
1[KFGI_t<25 \lor \text{vol\_shock}_t>2.0]
\]

그 외는 normal.

### 9.2 추세 가중치

```text
trend_strength = 0 -> 0.00
trend_strength = 1 -> 0.35
trend_strength = 2 -> 0.70
trend_strength = 3 -> 1.00
```

### 9.3 변동성 타깃팅

\[
VT_t =
\text{clip}
\left(
\sqrt{
\frac{0.15/\sqrt{252}}{\text{egarch\_vol}_t+10^{-9}}
},
0.1, 2.0
\right)
\]

### 9.4 K-FGI multiplier

현재는 K-FGI와 향후 5일 수익률의 walk-forward 방향성이 양수라서 momentum 방식이다.

\[
M^{KFGI}_t = 0.5 + \frac{KFGI_t}{100}(1.6-0.5)
\]

즉:

\[
M^{KFGI}_t = 0.5 + 1.1 \times KFGI_t/100
\]

### 9.5 regime multiplier

```text
bull   -> 1.2
normal -> 1.0
crisis -> 0.7
```

### 9.6 최종 포지션

\[
weight_t =
\text{clip}
\left(
TW_t \times VT_t \times M^{KFGI}_t \times RM_t,
0,
3.0
\right)
\]

추가 위험 차단:

\[
weight_t = 0 \quad \text{if } \text{vol\_shock}_t > 2.0
\]

\[
weight_t = 0 \quad \text{if } \sum_{i=1}^{3} r_{t-i} < -0.07
\]

실제 전략 수익률은 1일 지연 포지션을 사용한다.

\[
r^{strat}_t =
weight_{t-1} \times r_{t+1}
- |weight_{t-1}-weight_{t-2}| \times fee
\]

기준 거래비용:

\[
fee = 0.0015
\]

## 10. 제거 실험 추천 순서

### 10.1 논문용 우선순위

| 실험 | 제거 피처 | 목적 |
| --- | --- | --- |
| `drop_sent_composite_ma10` | `sent_composite_ma10` | RF 1위 감성 피처가 빠지면 방어성공/성과가 얼마나 흔들리는지 |
| `drop_sent_composite_family` | `sent_composite`, `sent_composite_ma10`, `sent_composite_diff` | 복합 감성 구조 전체의 기여 확인 |
| `drop_negative_family` | `neg_z`, `neg_z_inv`, `neg_z_ma5`, `negative_attention`, `panic_pressure` | 부정/공포 계열 기여 확인 |
| `drop_disagreement_family` | `sent_std`, `sent_std_inv`, `disagreement_attention` | 의견 불일치 계열 기여 확인 |
| `drop_attention_family` | `effective_n`, `comment_count`, `heat`, `sent_attention`, `heat_attention` | 댓글 관심도 계열 기여 확인 |
| `drop_all_sentiment` | 모든 감성 피처 | 기존 `without sentiment`와 대응 |
| `drop_egarch_family` | `egarch_vol`, `vol_regime_high`, `vol_ratio` | 변동성 결합의 필요성 확인 |
| `drop_subindex_family` | `sub_index2`~`sub_index7` | 시장 하위지표 필요성 확인 |

### 10.2 해석 원칙

- 누적수익률만 보지 말고 Sharpe, MDD, 하락일 방어폭을 함께 본다.
- 감성 피처 제거 시 누적수익률은 낮아지지만 하락일 방어폭이 크게 변하지 않을 수 있다. 이 경우 감성은 “평균 하락일 방어폭”보다 “종합 성과와 특정 방어성공 조건”에 기여한다고 해석한다.
- `defense_success_down_day`는 제거할 피처가 아니다. 결과 변수이므로 입력 피처에 넣으면 안 된다.

## 11. 1차 제거 실험 결과

실험 파일:

- `paper_outputs/tables/feature_drop_ablation.csv`
- `paper_outputs/tables/feature_drop_ablation_feature_map.csv`

주의: 이 제거 실험은 variant별 K-FGI를 동일 조건에서 다시 산출하기 위해 K-FGI warm-up 이후 공통 평가 구간을 사용한다. 따라서 표본 수는 2,445일이며, 기존 2,505일 성능표와 숫자가 완전히 같지는 않다. 비교는 이 표 내부의 baseline 대비 증감으로 해석한다.

| variant | 제거/유지 피처 | Sharpe | MDD | 누적수익률 | baseline 대비 Sharpe 변화 | 해석 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `lean_sent_composite_ma10_only` | subindex2-7, EGARCH 3개, `sent_composite_ma10`만 유지 | 0.400 | -29.6% | 63.1% | +0.007 | 감성 피처를 줄인 lean 후보 중 가장 좋음 |
| `baseline_full` | 기존 15개 피처 유지 | 0.393 | -30.8% | 61.6% | 0.000 | 비교 기준 |
| `lean_sent_core_four` | `sent_norm_w`, `sent_energy`, `neg_z_inv`, `sent_composite_ma10`만 유지 | 0.394 | -30.8% | 61.8% | +0.001 | 복잡도를 줄여도 성과 유지 |
| `drop_sent_composite` | `sent_composite`만 제거 | 0.393 | -30.8% | 61.6% | 0.000 | 단일 composite 제거 영향 작음 |
| `drop_sent_energy` | `sent_energy`만 제거 | 0.392 | -30.8% | 61.3% | -0.001 | 영향 작음 |
| `market_egarch_only` | 감성 전체 제거, subindex2-7 + EGARCH만 유지 | 0.330 | -36.6% | 52.7% | -0.063 | 감성 제외 시 성과/MDD 악화 |
| `drop_egarch_family` | `egarch_vol`, `vol_regime_high`, `vol_ratio` 제거 | 0.315 | -28.6% | 40.4% | -0.078 | EGARCH 제거 시 수익률/Sharpe 약화 |
| `drop_sent_composite_ma10` | `sent_composite_ma10`만 제거 | 0.295 | -36.1% | 42.0% | -0.098 | 핵심 감성 평활화 피처 제거 시 크게 악화 |
| `drop_sent_composite_family` | `sent_composite`, `sent_composite_ma10` 제거 | 0.294 | -36.1% | 41.9% | -0.099 | 복합 감성 계열 제거도 유사하게 악화 |

핵심 해석:

1. 감성 피처를 모두 많이 넣을 필요는 없다. `sent_composite_ma10` 하나만 남긴 lean 후보가 baseline보다 Sharpe와 MDD가 소폭 개선된다.
2. `sent_composite_ma10`은 제거 시 성과가 가장 크게 나빠지는 감성 피처다.
3. 단일 raw sentiment보다 복합·평활화된 감성 피처가 더 중요하다.
4. EGARCH 계열은 빼지 않는 편이 낫다. EGARCH를 제거하면 MDD는 일부 좋아지지만 누적수익률과 Sharpe가 크게 낮아져 전략의 변동성 조절 논리가 약해진다.
5. 하락일 방어폭 자체는 일부 제거 실험에서 오히려 커질 수 있으므로, 감성의 기여는 “평균 하락일 방어폭 증가” 하나로만 주장하지 않는다. 논문에서는 Sharpe, MDD, 누적수익률, 방어성공 피처 중요도를 함께 제시한다.

논문에 넣을 수 있는 문장:

> 추가적인 feature drop ablation 결과, subindex2-7과 EGARCH 피처를 유지한 채 감성 피처를 `sent_composite_ma10` 하나로 축소한 lean K-FGI는 Sharpe 0.400, MDD -29.6%, 누적수익률 63.1%로 baseline보다 소폭 개선되었다. 반대로 `sent_composite_ma10`을 제거하면 Sharpe는 0.393에서 0.295로 하락하고 MDD는 -30.8%에서 -36.1%로 악화되었다. 이는 감성 피처 전체가 모두 필요한 것은 아니지만, 복합 감성을 10일 이동평균으로 평활화한 피처가 K-FGI의 하방위험 관리 성과에 핵심적으로 기여함을 보여준다.

## 12. 2016-2017 초기 낙폭 진단

추가 진단 파일:

- `paper_outputs/tables/early_period_exposure_diagnostics.csv`
- `paper_outputs/tables/early_drawdown_diagnosis.csv`

| 기간 | 평균 K-FGI | 평균 weight | weight > 1 비율 | weight > 1.5 비율 | bull 비율 | crisis 비율 | 해석 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2016 | 64.2 | 1.01 | 52.0% | 31.3% | 37.0% | 5.3% | 초기 구간에서 위험 축소보다 공격 노출이 많았음 |
| 2017 | 62.2 | 1.25 | 67.5% | 34.2% | 30.5% | 2.5% | 강세장 추세를 따라 노출이 컸음 |
| 2016-2017 | 63.2 | 1.13 | 59.7% | 32.7% | 33.7% | 3.9% | 초기 2년은 전체기간보다 레버리지 성향이 큼 |
| Full | 57.6 | 0.75 | 35.9% | 14.9% | 23.4% | 7.2% | 전체기간 평균보다 2016-2017 노출이 높음 |

2016 성과:

| 전략 | 2016 누적수익률 | 2016 MDD | Sharpe |
| --- | ---: | ---: | ---: |
| Buy & Hold | 11.0% | -5.2% | 0.839 |
| K-FGI current 15bp | -12.9% | -16.3% | -1.051 |
| K-FGI no-leverage cap 1.0x | -2.7% | -7.9% | -0.302 |
| K-FGI warm-up first 252d cap 0.5x | -0.9% | -4.3% | -0.149 |

해석:

- 2016년 부진은 감성 피처 하나의 문제라기보다, 초기 구간에서 K-FGI와 추세/변동성 조합이 포지션을 크게 허용한 문제에 가깝다.
- 2016년 평균 weight는 1.01이고, weight가 1배를 넘은 날이 52.0%였다.
- 전체기간 평균 weight 0.75, weight > 1 비율 35.9%와 비교하면 초기 구간 노출이 확실히 높다.
- 따라서 논문에서는 2016년을 숨기기보다 “초기 추정 및 레버리지 제약의 필요성”으로 해석하는 것이 안전하다.

논문에 넣을 수 있는 문장:

> 2016년에는 K-FGI 전략이 Buy & Hold보다 낮은 성과를 보였다. 노출 진단 결과, 2016년 평균 포지션은 1.01배였고 전체 거래일의 52.0%에서 1배 초과 노출이 발생하였다. 이는 초기 구간에서 K-FGI가 위험 국면을 충분히 보수적으로 인식하지 못하고 과도한 시장 노출을 허용했기 때문으로 해석된다. 다만 no-leverage cap을 적용하면 2016년 MDD는 -16.3%에서 -7.9%로 완화되어, K-FGI의 실무 적용에는 레버리지 상한 제약이 필요함을 시사한다.

## 13. 통계적 유의성 요약

추가 검정 파일:

- `paper_outputs/tables/extended_statistical_tests.csv`
- `paper_outputs/tables/sentiment_with_without_paired_tests.csv`

### 13.1 전체 초과수익

| 검정 | 효과 | p-value | 해석 |
| --- | ---: | ---: | --- |
| K-FGI 일별 초과수익 t-test | -0.15bp/day | 0.929 | 전체 초과수익 평균은 유의하지 않음 |
| Newey-West HAC t-test | -0.15bp/day | 0.931 | 자기상관 보정 후에도 유의하지 않음 |
| Sharpe bootstrap difference | +0.141 | 0.574 | Sharpe 우위도 통계적으로 유의하지 않음 |

### 13.2 하방 방어

| 검정 | 효과 | p-value | 해석 |
| --- | ---: | ---: | --- |
| B&H 하락일 방어 t-test | +35.34bp/down day | <0.001 | 매우 유의 |
| Newey-West HAC t-test | +35.34bp/down day | <0.001 | 자기상관 보정 후에도 유의 |
| Wilcoxon signed-rank | median +12.89bp/down day | <0.001 | 비모수 검정에서도 유의 |
| bootstrap 95% CI | [30.71, 40.14]bp | <0.001 | 신뢰구간이 0을 넘음 |

### 13.3 감성 포함 vs 감성 제외

| 비교 | 효과 | p-value | 해석 |
| --- | ---: | ---: | --- |
| 전체 일별 with - without | +1.08bp/day | 0.0175 | 감성 포함 전략의 일별 추가 기여는 유의 |
| bootstrap CI | [0.18, 1.96]bp/day | 0.0228 | bootstrap에서도 0보다 큼 |
| B&H 하락일 with - without | -0.47bp/down day | 0.520 | 하락일 평균 방어폭에서 감성 추가 기여는 유의하지 않음 |

해석:

- 통계적으로 가장 강한 주장은 “전체 초과수익”이 아니라 “하락일 방어”다.
- 감성 피처는 전체 일별 성과에는 유의한 추가 기여가 있으나, 하락일 평균 방어폭에서는 유의하지 않다.
- 따라서 감성은 “하락일마다 더 많이 방어한다”기보다, K-FGI 구조 안에서 누적성과, MDD, 방어성공 조건, 피처 중요도를 개선하는 구성요소로 서술한다.
