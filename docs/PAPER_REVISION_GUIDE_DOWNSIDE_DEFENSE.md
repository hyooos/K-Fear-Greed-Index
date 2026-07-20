# Paper Revision Guide: K-FGI Downside Defense Version

## 0. 최종 방향 정리

이번 논문은 **수익률 예측 논문이 아니다.** 본 연구의 중심은 네이버 금융 댓글 감성, KRX 기반 시장 하위지표, EGARCH 조건부 변동성을 결합한 K-FGI를 만들고, 이를 이용해 **시장 하락 구간에서 손실 노출을 줄이는 하방위험 관리 전략**을 검증하는 것이다.

따라서 본문 전체에서 다음 방향을 유지해야 한다.

- 핵심 주장: K-FGI는 수익률을 맞히는 예측모형이 아니라, 투자자 심리와 변동성 정보를 결합해 시장 노출을 조절하는 리스크 관리 지표다.
- 핵심 성과지표: MDD, 하락일 초과방어, downside excess, rolling MDD 개선 비율, crisis subperiod MDD.
- 감성 피처의 역할: 감성은 단독 alpha가 아니라, 시장 기반 지표가 놓칠 수 있는 투자자 반응 강도와 불안 확산을 보완하는 피처다.
- 수익률 예측 관련 표현: 선행연구 비교, K-FGI 신호 유효성의 보조 검정, 한계에서만 제한적으로 사용한다.

논문 제목도 현재 방향과 맞다.

> 감성 및 변동성 결합 지표(K-FGI)를 활용한 주식시장 리스크 관리 전략 연구  
> A Downside Risk Management Framework Using a Sentiment-Volatility Integrated Index (K-FGI)

## 1. 지금 가장 먼저 해야 할 작업

### 1순위: 논문 본문 방향 정리

기존 원고에서 “예측”, “수익률 예측”, “예측력”이라는 표현이 여러 군데 나온다. 전부 지우라는 뜻은 아니지만, 본 연구의 목적처럼 보이는 곳에서는 반드시 수정해야 한다.

바꿀 기준:

- `수익률 예측 모델` -> `하방위험 관리 전략`
- `예측력 검증` -> `신호 유효성 및 하방방어 기여도 검증`
- `피처가 수익률 예측에 기여` -> `피처가 위험 국면 식별과 노출 조절에 기여`
- `초과수익 창출` -> `하락일 손실 완화`, `MDD 축소`, `downside defense`
- `LightGBM 예측모델` -> `해석가능한 하방위험 설명모형`

### 2순위: 3장 Method 수정

현재 원고의 3.5.5 `K-FGI 예측력 검증`, 3.7 `예측 모델: LightGBM 선택 근거`는 지금 논문 방향과 가장 크게 충돌한다.

수정 방향:

- 3.5.5 제목을 `K-FGI 신호 유효성 검증` 또는 `K-FGI와 하방위험 관리 신호`로 변경.
- 향후 5일 수익률 Spearman을 중심 근거로 쓰지 말고, K-FGI가 포지션 조절에 쓰이는 심리/변동성 지표임을 설명.
- 3.7 LightGBM 중심 절은 삭제하거나 부록/보조분석으로 이동.
- 본문에는 Elastic Net Logistic Regression과 Random Forest permutation importance를 사용한 **하락위험/방어성공 피처 중요도 분석**을 넣는다.

### 3순위: 4장 Experiment 수정

4장은 기존 2022~2025년 3개년 실험 수치가 섞여 있으므로 10개년 결과로 전부 교체해야 한다.

반드시 업데이트할 기준:

- 최종 데이터셋: 2015-01-16 ~ 2025-12-29.
- 모델/전략 평가 유효 구간: 2015-10-08 ~ 2025-12-22.
- 유효 표본: 2,505 거래일.
- 1년 rolling window: 252 거래일.
- rolling OOS window 수: 2,254개.

### 4순위: 그림/표 교체

논문에 넣을 그림은 기존 3개년 figure가 아니라 `논문용/10_Paper_outputs/figures`의 10개년 그림을 기준으로 교체한다.

우선적으로 사용할 그림:

- `07_전략별_누적수익률.png`
- `08_KFGI_낙폭방어.png`
- `09_EGARCH_변동성과_시장노출.png`
- `13_감성피처_포함제외_성능비교.png`
- `17_rolling_OOS_성과안정성.png`
- `19_국면별_초과성과.png`
- `20_감성조건부_하방방어.png`
- `23_감성위험_오버레이_후보.png`
- `24_ElasticNet_defense_success_down_day_계수.png`
- `25_RF_defense_success_down_day_순열중요도.png`

교수님 코멘트에 따라 최종 제출용에서는 한글 그림 텍스트를 영어로 바꾸고, 편집 가능한 PPT 파일에 Figure 번호 순서대로 정리하는 작업도 필요하다.

## 2. 논문 섹션별 수정 위치

### 2.1 Title / Abstract

Abstract는 아직 작성 전이므로, 다음 구조로 작성하면 된다.

1. 문제의식: 한국 시장에는 투자자 감성, 변동성 비대칭성, 하락위험 관리를 통합한 지표가 부족하다.
2. 방법: KRX 시장 하위지표, 네이버 댓글 감성, EGARCH 조건부 변동성을 결합해 K-FGI를 구성한다.
3. 검증: 2015~2025년 10개년 자료로 Buy & Hold 대비 MDD, 하락일 방어, crisis subperiod, rolling OOS를 평가한다.
4. 결과: 전체 수익률 우위보다 하방위험 완화에서 더 강한 효과가 나타난다.
5. 기여: 한국형 감성-변동성 결합 지표와 리스크 관리형 포지션 조절 프레임워크를 제안한다.

붙여넣기용 Abstract 초안:

> 본 연구는 한국 주식시장에서 투자자 감성과 조건부 변동성을 결합한 K-FGI(Korean Fear & Greed Index)를 구축하고, 이를 활용한 하방위험 관리 전략을 제안한다. K-FGI는 KRX 기반 시장 하위지표, 네이버 금융 댓글 감성 피처, EGARCH(1,1) 조건부 변동성을 walk-forward 방식으로 결합하여 산출된다. 본 연구는 2015년부터 2025년까지의 10개년 자료를 이용해 K-FGI 기반 동적 포지션 전략이 Buy & Hold 대비 최대 낙폭과 하락일 손실을 완화하는지 검증하였다. 분석 결과, K-FGI는 전체 수익률 우위보다는 시장 하락 및 고변동성 구간에서의 손실 방어 측면에서 더 뚜렷한 효과를 보였다. 특히 1년 rolling 검증에서 MDD 개선 비율은 83.1%로 나타났으며, 2020년 코로나19 및 2022년 금리 인상 약세장 구간에서도 Buy & Hold 대비 낮은 MDD를 기록하였다. 이는 K-FGI가 단기 수익률 예측모형이 아니라 투자자 심리와 변동성 정보를 활용한 리스크 관리 지표로 기능할 수 있음을 시사한다.

### 2.2 Introduction

현재 Introduction에는 머신러닝 주가 예측 연구 이야기가 들어가 있다. 이 부분은 선행연구의 한계로만 쓰고, 본 연구가 예측 모델이 아니라는 점을 명확히 해야 한다.

수정할 문장 방향:

- 기존 연구들은 수익률 예측에 집중했지만, 실제 운용에서는 하락장 손실 방어와 MDD 관리가 중요하다.
- 본 연구는 예측 정확도 자체가 아니라 투자자 심리와 변동성 비대칭성을 활용한 노출 조절 프레임워크를 제안한다.

붙여넣기용 문단:

> 최근 금융 분야에서는 머신러닝 기반 주가 및 변동성 예측 연구가 활발히 수행되고 있으나, 예측 정확도 개선이 실제 투자전략의 하방위험 관리로 직접 이어지는 것은 아니다. 특히 주식시장은 하락 충격이 발생할 때 변동성이 비대칭적으로 확대되고, 투자자 공포가 급격히 확산되면서 손실이 특정 구간에 집중되는 특성을 가진다. 따라서 본 연구는 단기 수익률 예측보다 시장 하락 구간에서의 손실 노출을 줄이는 리스크 관리 문제에 초점을 둔다.

### 2.3 Related Work

Related Work의 큰 구조는 유지하되, 2.4 제목과 마지막 정리를 바꾸는 것이 좋다.

현재 제목:

- `복합 감성 지표 및 머신러닝 기반 예측 연구`

수정 제목:

- `복합 감성 지표와 리스크 관리형 투자전략 연구`

수정 방향:

- CNN FGI, SVR, Random Forest 등은 “수익률 예측 성능” 그 자체보다 “복합 지표의 가중치 학습 가능성” 근거로만 사용.
- 본 연구의 차별점은 예측모형 경쟁이 아니라 K-FGI를 실제 포지션 노출 조절과 MDD 방어에 연결한 점.

붙여넣기용 정리 문단:

> 기존 복합 감성 지표 연구는 주로 향후 수익률 예측 성능을 개선하는 데 초점을 두었다. 그러나 본 연구는 K-FGI를 직접적인 수익률 예측모형으로 사용하기보다, 투자자 심리와 변동성 위험을 함께 반영하여 시장 노출을 조절하는 리스크 관리 지표로 활용한다. 따라서 본 연구의 차별점은 감성 지표의 예측 정확도 자체가 아니라, 감성-변동성 결합 지표가 하락장 손실 완화와 MDD 방어에 기여하는지를 10개년 자료로 검증한다는 데 있다.

### 2.4 Proposed Method 3.5.5

현재 원고의 `3.5.5 K-FGI 예측력 검증`은 제목부터 수정해야 한다.

추천 제목:

> 3.5.5 K-FGI 신호 유효성 및 하방위험 연계

붙여넣기용 문단:

> 본 연구에서 K-FGI는 수익률을 직접 예측하기 위한 최종 모델이 아니라, 투자자 심리와 시장 변동성 정보를 하나의 0~100 지표로 통합하여 포지션 노출 조절에 활용하기 위한 신호다. 따라서 K-FGI의 유효성은 단순한 향후 수익률 상관계수보다, 전략 적용 이후 하락일 손실 완화와 MDD 축소 여부를 통해 평가하는 것이 적절하다. 본 연구는 K-FGI가 산출된 이후 추세 필터와 EGARCH 변동성 타겟팅을 결합하여 최종 투자 비중을 결정하며, K-FGI의 공포/탐욕 구간은 시장 국면별 포지션 배율 조정에 활용된다.

### 2.5 Proposed Method 3.7

기존 `3.7 예측 모델: LightGBM 선택 근거 및 대안 모델 비교`는 지금 논문과 맞지 않는다.

수정 제목:

> 3.7 해석가능한 하방위험 피처 중요도 분석

수정 내용:

- LightGBM은 본문 핵심에서 제외.
- Elastic Net Logistic Regression: 계수 방향과 선택 여부.
- Random Forest permutation importance: 비선형 관계에서 검증 구간 AUC 감소폭.
- 타깃은 `수익률 예측`이 아니라 `하락일 방어성공`, 필요 시 `5거래일 급락위험`.

붙여넣기용 문단:

> 본 연구는 K-FGI 생성 및 포지션 결정에 별도의 예측모형을 직접 사용하지 않는다. 다만 감성 피처와 시장/변동성 피처가 하방위험 관리에 어떤 역할을 하는지 해석하기 위해 보조적인 피처 중요도 분석을 수행한다. 이를 위해 Elastic Net Logistic Regression과 Random Forest permutation importance를 사용하였다. Elastic Net은 표준화 계수를 통해 각 피처가 하락위험 또는 방어성공 확률에 미치는 방향을 해석할 수 있고, Random Forest permutation importance는 검증 구간에서 특정 피처를 섞었을 때 OOS AUC가 얼마나 감소하는지를 측정하여 비선형적 중요도를 평가할 수 있다.

## 3. 4장 실험 설계 수정안

### 3.1 4.3 실험 설계

현재 4.3에는 2022~2025년 754행 기준, LightGBM 하이퍼파라미터, 2025 단일 OOS가 섞여 있다. 10개년 최종 실험 기준으로 다시 써야 한다.

붙여넣기용:

> 본 연구의 실험은 Python 3.10 환경에서 수행되었다. EGARCH 조건부 변동성 추정에는 `arch`, 데이터 전처리와 피처 엔지니어링에는 `pandas`와 `numpy`, 감성 분석에는 KR-FinBERT-SC 기반 `transformers`, 해석가능 피처 중요도 분석에는 `scikit-learn`을 활용하였다. 최종 분석은 2015년부터 2025년까지의 10개년 구간을 대상으로 하며, 2014년 데이터는 rolling 계산의 초기 warm-up 구간으로 활용하였다. 모델 및 전략의 유효 평가 구간은 2015년 10월 8일부터 2025년 12월 22일까지이며, 총 2,505 거래일을 포함한다.

### 3.2 4.3.1 Walk-forward 및 TimeSeriesSplit

붙여넣기용:

> K-FGI와 EGARCH 변동성은 look-ahead bias를 방지하기 위해 walk-forward 방식으로 산출하였다. 즉, t 시점의 지표는 t-1 시점까지의 과거 정보만을 이용하여 계산된다. 피처 중요도 분석에서는 TimeSeriesSplit 5-Fold를 적용하여 시계열 순서를 유지한 상태에서 과거 구간을 학습하고 이후 구간을 검증하였다. 이 절차는 무작위 분할로 인한 미래 정보 유입을 방지하고 실제 운용 환경과 유사한 검증 구조를 제공한다.

### 3.3 4.3.2 Rolling OOS

붙여넣기용:

> 단일 기간 분리 검증은 특정 시장 환경의 영향을 크게 받을 수 있으므로, 본 연구는 252거래일을 하나의 window로 하는 1년 rolling OOS 검증을 수행하였다. 전체 평가 구간에서 총 2,254개의 rolling window를 구성하였으며, 각 window에서 K-FGI 전략과 Buy & Hold의 Sharpe, MDD, 누적수익률을 비교하였다. 이 검증의 목적은 전략이 특정 연도에만 우연히 작동했는지가 아니라, 시간축 전반에서 하방위험 방어 성격이 반복적으로 나타나는지를 확인하는 데 있다.

넣을 수치:

- rolling window 수: 2,254개.
- K-FGI MDD 개선 비율: 83.1%.
- 평균 rolling MDD 개선폭: +6.67%p.
- 중앙값 rolling MDD 개선폭: +4.27%p.
- K-FGI Sharpe 우위 비율: 39.6%.

해석:

> Sharpe 우위 비율은 39.6%로 절반에 미치지 못하지만, MDD 개선 비율은 83.1%로 높게 나타났다. 이는 K-FGI가 수익률 우위 전략이 아니라 하방위험 완화 전략이라는 본 연구의 해석을 뒷받침한다.

### 3.4 4.3.3 거래비용 스트레스 테스트

주의: 기존 원고에는 0.015%에서 0.045%까지로 적혀 있다. 현재 robust 실험은 `0bp, 5bp, 10bp, 15bp, 20bp` 기준으로 정리되어 있다. 문서 내 거래비용 단위를 반드시 통일해야 한다.

현재 사용 가능한 표:

| 거래비용 | 누적수익률 | Sharpe | MDD |
| ---: | ---: | ---: | ---: |
| 0bp | 192.2% | 0.728 | -21.9% |
| 5bp | 136.2% | 0.584 | -27.0% |
| 10bp | 90.9% | 0.439 | -32.3% |
| 15bp | 54.4% | 0.295 | -37.1% |
| 20bp | 24.8% | 0.150 | -41.6% |

붙여넣기용:

> 실제 거래 환경을 고려하기 위해 거래비용을 0bp부터 20bp까지 변화시키며 전략 성과의 민감도를 확인하였다. 거래비용이 증가할수록 누적수익률과 Sharpe는 하락하고 MDD는 악화되는 경향을 보였다. 따라서 본 연구는 15bp를 보수적 기준으로 사용하여 전략 성과를 평가한다. 15bp 기준 K-FGI with sentiment의 누적수익률은 54.4%, Sharpe는 0.295, MDD는 -37.1%로 나타났다.

### 3.5 4.3.4 해석가능 피처 중요도 분석

붙여넣기용:

> 본 연구의 피처 중요도 분석은 수익률 예측이 아니라 하방위험 설명력을 평가하기 위해 수행되었다. 종속변수는 `하락일 방어성공`을 중심으로 정의하였다. 이는 Buy & Hold가 하락한 날 K-FGI 전략의 손실이 Buy & Hold보다 작았는지를 나타내는 이진 변수다. 추가적으로 `5거래일 하락위험`과 `5거래일 -2% 급락위험`도 보조 타깃으로 검토하였다. Elastic Net Logistic Regression은 표준화 계수를 통해 피처의 방향성을 확인하기 위해 사용하였고, Random Forest permutation importance는 검증 구간에서 피처를 섞었을 때 AUC가 얼마나 하락하는지를 통해 비선형 중요도를 측정하기 위해 사용하였다. 단, K-FGI는 감성 피처를 포함해 산출된 최종 지표이므로 구성요소별 중요도 분석에서는 K-FGI 자체를 제외하였다.

넣을 수치:

| 타깃 | 표본 | positive rate | Elastic Net AUC | RF AUC |
| --- | ---: | ---: | ---: | ---: |
| 5거래일 하락위험 | 2,505 | 44.2% | 0.512 | 0.516 |
| 5거래일 -2% 급락위험 | 2,505 | 16.4% | 0.511 | 0.494 |
| 하락일 방어성공 | 1,161 | 58.7% | 0.927 | 0.912 |

해석:

> 단순 하락 또는 급락 여부의 분류 성능은 AUC 0.5 부근으로 높지 않았다. 반면 K-FGI를 제외한 구성요소 기반 하락일 방어성공 타깃에서는 Elastic Net AUC 0.927, Random Forest AUC 0.912로 높은 설명력이 나타났다. 이는 본 연구의 피처 구조가 시장 방향 예측보다 하락일 손실 완화 조건을 설명하는 데 더 적합함을 보여준다.

## 4. 4.4 감성 피처 기여도 검증 수정안

### 4.1 4.4 전체 도입부

현재 4.4는 “예측력과 전략 성과”로 시작한다. 이 표현을 바꿔야 한다.

붙여넣기용:

> 본 절에서는 K-FGI 설계에서 감성 피처를 독립적으로 구성한 것이 하방위험 관리 관점에서 타당한지 검증한다. 이를 위해 첫째, 감성 피처를 제거한 K-FGI와 감성 포함 K-FGI의 전략 성과를 비교하는 ablation study를 수행한다. 둘째, 감성 피처가 하락일 방어성공을 설명하는 데 기여하는지 Elastic Net Logistic Regression과 Random Forest permutation importance를 통해 확인한다. 셋째, 감성위험이 높은 구간에서 노출을 축소하는 overlay 실험을 통해 감성 정보가 위험 국면에서 추가적인 방어 신호로 활용될 수 있는지 검토한다.

### 4.2 4.4.1 Ablation Study

기존 3개년 수치를 10개년 수치로 교체한다.

넣을 수치:

| 전략 | 누적수익률 | Sharpe | MDD |
| --- | ---: | ---: | ---: |
| K-FGI with sentiment | 54.4% | 0.295 | -37.1% |
| K-FGI without sentiment | 17.8% | 0.113 | -45.3% |
| sent_norm_only | 17.3% | 0.113 | -43.4% |
| negative_only | 15.9% | 0.102 | -46.3% |
| dispersion_only | 28.8% | 0.169 | -42.1% |

붙여넣기용:

> 감성 피처 전체를 제거하면 K-FGI의 누적수익률은 54.4%에서 17.8%로 하락했고, Sharpe는 0.295에서 0.113으로 낮아졌으며, MDD는 -37.1%에서 -45.3%로 악화되었다. 단일 감성 피처만 사용하는 경우에도 sent_norm_only, negative_only, dispersion_only 모두 감성 전체 조합보다 낮은 성과를 보였다. 이는 감성 정보가 단일 방향성 점수보다 감성 강도, 의견 불일치, 부정 감성, 이동평균 등을 함께 반영한 조합 형태에서 더 유용함을 시사한다.

주의:

- “감성이 수익률 예측력을 유의하게 향상시켰다”는 문장은 쓰지 않는다.
- “감성 피처가 K-FGI의 위험조정 성과와 MDD를 개선했다”로 쓴다.

### 4.3 4.4.2 해석가능 모델 기반 피처 중요도

붙여넣기용:

> 하락일 방어성공 타깃에서 K-FGI 자체를 제외하고 구성요소만 투입한 결과, Elastic Net Logistic Regression의 TimeSeriesSplit 평균 AUC는 0.927, Random Forest의 평균 AUC는 0.912로 나타났다. Random Forest permutation importance에서는 감성 복합 10일 피처(sent_composite_ma10)가 AUC 감소폭 0.0244로 1위를 기록했으며, 추세 강도(0.0180), MA 5/20(0.0146), 60일 모멘텀(0.0133)이 뒤를 이었다. Elastic Net에서도 sent_composite_ma10의 표준화 계수 절댓값은 1.265로 추세 강도 다음으로 크게 나타났다. 이는 감성 피처가 완성된 K-FGI 내부에서 단순히 묻히는 변수가 아니라, 하락일 방어성공 조건을 설명하는 핵심 구성요소 중 하나임을 보여준다.

넣을 수치:

- 하락일 방어성공 표본: 1,161일.
- positive rate: 58.7%.
- Elastic Net AUC: 0.927.
- RF AUC: 0.912.
- RF permutation importance 상위, K-FGI 제외 구성요소 기준:
  - sent_composite_ma10: AUC 감소폭 0.0244.
  - trend_strength: 0.0180.
  - ma_ratio_5_20: 0.0146.
  - mom60: 0.0133.

### 4.4 4.4.3 감성위험 조건부 분석

붙여넣기용:

> 감성 피처의 효과는 전체 기간에서 균일하게 나타나기보다, 투자자 관심과 불안이 확대되는 조건에서 더 뚜렷하게 나타났다. 전체 표본에서 감성 포함 전략은 감성 제외 전략 대비 평균 +1.08bp/day의 추가 수익을 보였으며(t=2.377, p=0.0175), 공포 압력(panic_pressure) 상위 20%와 하위 20%의 하락일 방어 차이는 +9.86bp/day로 나타났다(q=3.08e-05). 또한 panic_pressure와 고변동성의 상호작용은 K-FGI의 Buy & Hold 대비 초과 방어에 유의한 양의 영향을 보였다(coef=+10.71bp, p=0.00490, q=0.0746). 이는 감성 피처가 평상시 방향 예측보다 고변동성 위험 국면에서 노출 조절 신호를 보완하는 역할을 함을 시사한다.

## 5. 4.5 전략 강건성 검증 수정안

### 5.1 전체 10개년 성과표

메인 표에 넣을 수치:

| 전략 | 연율수익률 | 연율변동성 | Sharpe | MDD | 누적수익률 | 하락일 초과방어 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Buy & Hold | 8.71% | 18.46% | 0.472 | -41.2% | 137.6% | - |
| Trend only | 4.90% | 12.14% | 0.404 | -28.9% | 62.8% | 38.48bp |
| Trend + EGARCH | 3.34% | 11.27% | 0.296 | -28.5% | 39.3% | 40.30bp |
| K-FGI with sentiment | 4.37% | 14.82% | 0.295 | -37.1% | 54.4% | 28.41bp |
| K-FGI without sentiment | 1.65% | 14.61% | 0.113 | -45.3% | 17.8% | 28.88bp |

붙여넣기용 해석:

> 15bp 거래비용을 적용한 10개년 검증에서 K-FGI with sentiment는 Buy & Hold 대비 누적수익률은 낮았으나, MDD를 -41.2%에서 -37.1%로 완화하였다. 또한 감성 피처를 제거한 K-FGI와 비교하면 누적수익률은 17.8%에서 54.4%로, Sharpe는 0.113에서 0.295로 개선되었고, MDD는 -45.3%에서 -37.1%로 축소되었다. 이는 감성 피처가 K-FGI의 리스크 관리 성과를 보완했음을 보여준다.

### 5.2 Crisis Subperiod

이 부분은 논문에서 꼭 살리는 것이 좋다.

#### 2020 COVID

- Buy & Hold: 누적수익률 40.3%, Sharpe 1.263, MDD -34.2%.
- K-FGI with sentiment: 누적수익률 43.3%, Sharpe 1.951, MDD -11.7%.
- 하락일 초과방어: 69.13bp, p=0.000110.

#### 2022 금리 인상/약세장

- Buy & Hold: 누적수익률 -26.7%, Sharpe -1.658, MDD -28.8%.
- K-FGI with sentiment: 누적수익률 -8.2%, Sharpe -1.013, MDD -8.3%.
- 하락일 초과방어: 77.63bp, p=1.98e-19.

#### 2025 강세장/OOS

- Buy & Hold: 누적수익률 84.0%, Sharpe 2.761, MDD -14.3%.
- K-FGI with sentiment: 누적수익률 85.4%, Sharpe 2.906, MDD -7.8%.
- 하락일 초과방어: 12.93bp, p=0.150.

붙여넣기용:

> 위기 구간 분석에서 K-FGI의 하방위험 관리 성격이 더 명확하게 나타났다. 2020년 코로나19 구간에서 K-FGI with sentiment의 MDD는 -11.7%로 Buy & Hold의 -34.2%보다 크게 낮았고, 2022년 금리 인상 약세장에서도 MDD는 -8.3%로 Buy & Hold의 -28.8%를 크게 하회하였다. 이는 K-FGI가 모든 구간에서 더 높은 수익률을 내는 전략이라기보다, 손실이 집중되는 위기 구간에서 노출을 조절해 낙폭을 줄이는 전략임을 보여준다.

### 5.3 Rolling OOS

붙여넣기용:

> 252거래일 기준 rolling OOS 검증 결과, K-FGI의 Sharpe가 Buy & Hold보다 높았던 window는 39.6%에 그쳤다. 그러나 MDD가 개선된 window는 83.1%로 나타났고, 평균 MDD 개선폭은 +6.67%p였다. 이는 K-FGI가 수익률 우위를 일관되게 제공하는 전략은 아니지만, 다양한 시점에서 하방위험을 줄이는 특성이 반복적으로 나타났음을 의미한다.

### 5.4 2016년 초기 구간 한계

2016년 문제는 숨기지 말고 한계 및 개선 방향에 넣는다.

넣을 수치:

- 2016년 Buy & Hold: 누적수익률 11.0%, MDD -5.2%.
- 2016년 K-FGI current 15bp: 누적수익률 -12.9%, MDD -16.3%.
- no-leverage cap 1.0x 적용 시: 누적수익률 -2.7%, MDD -7.9%.

붙여넣기용:

> 다만 초기 구간인 2016년에는 K-FGI가 과도한 노출을 허용하면서 Buy & Hold보다 큰 낙폭을 보였다. 이는 K-FGI가 모든 국면에서 자동으로 방어에 성공하는 만능 지표가 아니라, 초기 추정 구간과 국면 전환기에서는 노출 상한이 필요함을 보여준다. 실제로 no-leverage cap 1.0x를 적용하면 2016년 MDD는 -16.3%에서 -7.9%로 완화된다. 따라서 향후 연구에서는 감성위험 overlay와 레버리지 상한을 결합한 보수적 운용 구조를 추가 검증할 필요가 있다.

## 6. 4.6 파라미터 설정 근거 수정안

### 6.1 EGARCH min_obs=252

붙여넣기용:

> EGARCH 추정의 최소 학습 표본은 252거래일로 설정하였다. 이는 약 1년의 거래일에 해당하며, 초기 추정 구간에서 변동성 추정값이 불안정하게 폭발하는 문제를 완화하기 위한 보수적 설정이다. 비정상적으로 큰 추정값은 상한 처리하거나 rolling volatility로 대체하였다. 안정화 이후 fallback rate는 0.30%, capped rate는 0.48%, p99 volatility는 0.0235로 나타났다.

### 6.2 K-FGI threshold 25/65

현재 결과상 25/65가 유일한 최적이라고 말하면 안 된다. robustness 기준으로 써야 한다.

붙여넣기용:

> K-FGI의 Fear/Greed 기준은 20/80, 25/65, 30/70 조합을 비교하였다. 최상위 Sharpe 조합은 20/80 및 multiplier 0.7-1.6이었으나, 기준 조합인 25/65와 0.5-1.6 역시 성과가 크게 무너지지 않았다. 따라서 25/65는 단일 최적값이라기보다 공포/탐욕 구간을 직관적으로 구분하면서도 민감도 분석에서 안정적으로 작동한 기준값으로 채택하였다.

### 6.3 거래비용 15bp

붙여넣기용:

> 거래비용은 15bp를 기본값으로 사용하였다. 거래비용이 0bp일 때 K-FGI with sentiment의 누적수익률은 192.2%, Sharpe는 0.728이었으나, 15bp에서는 누적수익률 54.4%, Sharpe 0.295로 낮아졌다. 이는 전략이 거래비용에 민감함을 의미하므로, 본 연구에서는 보수적인 비용 가정인 15bp를 기준으로 성과를 제시한다.

### 6.4 감성위험 overlay

이건 최종 메인 모델이라기보다 개선 실험 또는 향후 연구로 두는 편이 안전하다.

붙여넣기용:

> 추가 실험으로 감성위험이 높은 구간에서 노출을 축소하는 overlay를 검토하였다. `heat_attention` 상위 20%에서 기존 weight를 0.7배로 축소할 경우 전체 누적수익률은 43.3%, Sharpe는 0.255를 기록하면서 감성 제외 전략 대비 하락일 방어가 +2.54bp/day 개선되었다(q=0.0050). 이는 감성 피처가 K-FGI 점수에 단순 가산되는 방식보다, 위험 국면에서 노출을 제한하는 방식으로 활용될 때 하방방어 효과가 더 명확해질 수 있음을 시사한다.

## 7. 한계 작성 방향

한계에는 수익률 예측 이야기를 넣을 수 있지만, 본문 전체를 흔들지 않도록 표현해야 한다.

넣을 한계:

1. K-FGI는 Buy & Hold 대비 전체 수익률 우위를 일관되게 제공하지 않는다.
2. 전략의 강점은 수익률 예측이 아니라 MDD 및 하락일 손실 완화에서 확인된다.
3. 2016년 초기 구간에서는 K-FGI가 과노출되어 Buy & Hold보다 큰 낙폭을 보였다.
4. 거래비용에 민감하다.
5. 감성위험 overlay는 유망하지만, 최종 메인 모델로 채택하려면 별도 rolling OOS 재검증이 필요하다.
6. 댓글 데이터는 플랫폼 이용자 특성과 기사 노출 구조의 영향을 받을 수 있다.

붙여넣기용:

> 본 연구의 K-FGI 전략은 전체 기간에서 Buy & Hold 대비 수익률 우위를 일관되게 제공하는 전략은 아니다. 특히 강한 상승장에서는 노출을 축소한 구간이 기회비용으로 작용할 수 있으며, 초기 추정 구간에서는 과노출 문제가 나타날 수 있다. 또한 거래비용 가정에 따라 성과가 크게 달라지므로 실제 운용에서는 비용과 회전율 관리가 중요하다. 그럼에도 불구하고 K-FGI는 여러 crisis subperiod와 rolling OOS 검증에서 MDD 완화와 하락일 손실 방어라는 리스크 관리 목적에 부합하는 결과를 보였다.

## 8. 현재 파일 기준으로 수정할 위치

### 문서

- `논문용/00_docs/PAPER_REVISION_GUIDE_DOWNSIDE_DEFENSE.md`
  - 지금 이 문서의 논문 섹션별 수정 지침.
- `논문용/00_docs/FEATURE_IMPORTANCE_MODEL_REVISION.md`
  - LightGBM을 대체한 Elastic Net/RF 피처 중요도 문서.
- `논문용/10_Paper_outputs/SENTIMENT_FEATURE_DEVELOPMENT.md`
  - 감성 조건부 검정 및 overlay 결과.
- `논문용/10_Paper_outputs/ROLLING_OOS_REGIME_EVALUATION.md`
  - rolling OOS 및 국면별 결과.
- `논문용/10_Paper_outputs/EARLY_DRAWDOWN_DIAGNOSIS.md`
  - 2016년 문제와 no-leverage cap 개선 실험.

### 표

- `논문용/10_Paper_outputs/robustness/tables/priority_a_performance_robust_egarch.csv`
- `논문용/10_Paper_outputs/robustness/tables/priority_a_fee_sensitivity.csv`
- `논문용/10_Paper_outputs/robustness/tables/priority_a_crisis_subperiod_performance.csv`
- `논문용/10_Paper_outputs/robustness/tables/priority_b_sentiment_ablation_detail.csv`
- `논문용/10_Paper_outputs/tables/interpretable_feature_importance_metrics.csv`
- `논문용/10_Paper_outputs/tables/random_forest_permutation_importance_defense_success_down_day.csv`
- `논문용/10_Paper_outputs/tables/elastic_net_feature_importance_defense_success_down_day.csv`

### 그림

- `논문용/10_Paper_outputs/figures/07_전략별_누적수익률.png`
- `논문용/10_Paper_outputs/figures/08_KFGI_낙폭방어.png`
- `논문용/10_Paper_outputs/figures/09_EGARCH_변동성과_시장노출.png`
- `논문용/10_Paper_outputs/figures/13_감성피처_포함제외_성능비교.png`
- `논문용/10_Paper_outputs/figures/17_rolling_OOS_성과안정성.png`
- `논문용/10_Paper_outputs/figures/19_국면별_초과성과.png`
- `논문용/10_Paper_outputs/figures/23_감성위험_오버레이_후보.png`
- `논문용/10_Paper_outputs/figures/25_RF_defense_success_down_day_순열중요도.png`

## 9. 추가 실험 및 시각화 배치 계획

아래 표는 지금까지 추가한 실험과 그래프를 논문 어디에 넣으면 좋은지 정리한 것이다. 본문에는 핵심 그림만 넣고, 세부 robustness 표는 부록 또는 Appendix로 보내는 구성이 가장 깔끔하다.

### 9.1 3장 Method에 넣을 그림/표

| 위치 | 넣을 항목 | 파일 | 목적 |
| --- | --- | --- | --- |
| 3.1 전체 파이프라인 | K-FGI 데이터/모델 파이프라인 | 새로 제작 필요 또는 기존 pipeline figure | 데이터 수집 -> 감성 분석 -> EGARCH -> K-FGI -> 포지션 결정 흐름 설명 |
| 3.4 감성 피처 설계 | 감성점수 정규화 전후 | `03_감성점수_정규화_전후.png` | raw sentiment를 그대로 쓰지 않고 정규화/가중 처리한 이유 제시 |
| 3.4 감성 피처 설계 | 댓글 수와 감성 추세 | `04_댓글수와_감성추세.png` | 댓글 관심도와 감성 신호가 시간에 따라 달라짐을 보여줌 |
| 3.5 K-FGI 산출 | K-FGI 공포/탐욕 시계열 | `05_KFGI_공포탐욕_시계열.png` | 산출된 지표가 0~100 범위에서 공포/탐욕 구간을 형성함을 제시 |
| 3.6 포지션 결정 | 포지션 크기/시장 노출 | `16_포지션크기_시장노출.png` | K-FGI, 추세, 변동성이 최종 노출로 연결되는 구조 설명 |

주의:

- 3장에서는 성과를 주장하기보다 **설계가 어떻게 구성되는지**를 보여주는 그림만 넣는다.
- `05_KFGI_공포탐욕_시계열.png`는 제목 없이 넣고, 캡션에서 “K-FGI time series and fear/greed thresholds”처럼 설명한다.

### 9.2 4.3 실험 설계에 넣을 표

| 위치 | 넣을 항목 | 파일 | 목적 |
| --- | --- | --- | --- |
| 4.3 실험 설계 | 데이터 기간/평가 구간 요약 | 본문 표 새로 작성 | 2014 warm-up, 2015~2025 최종 구간, 2,505 거래일 명시 |
| 4.3 Rolling OOS | rolling OOS 설정표 | `rolling_oos_1y_performance.csv`에서 요약 | 252거래일 window, 2,254개 window 명시 |
| 4.3 거래비용 | 거래비용 설정표 | `priority_a_fee_sensitivity.csv` | 0/5/10/15/20bp 민감도 설계 설명 |
| 4.3 피처 중요도 | 해석가능 모델 타깃 정의표 | `interpretable_feature_importance_metrics.csv` | 하락위험/방어성공 타깃 정의와 AUC 요약 |

본문에는 `interpretable_feature_importance_metrics.csv`의 요약만 넣고, target별 세부 coefficient/permutation 표는 부록으로 보내는 것이 좋다.

### 9.3 4.4 감성 피처 기여도 검증에 넣을 그림/표

이 절이 감성 연구의 핵심이다. 감성 피처가 “수익률 예측 alpha”가 아니라 “하방방어를 보완하는 심리 피처”라는 흐름으로 구성한다.

| 위치 | 넣을 항목 | 파일 | 목적 |
| --- | --- | --- | --- |
| 4.4.1 Ablation Study | 감성 포함/제외 성능 비교 | `13_감성피처_포함제외_성능비교.png` | 감성 전체 포함이 without sentiment보다 성과/MDD를 개선했음을 제시 |
| 4.4.1 Ablation Study | 감성 ablation 세부 표 | `priority_b_sentiment_ablation_detail.csv` | sentiment_all, sentiment_none, sent_norm_only, negative_only, dispersion_only 비교 |
| 4.4.2 해석가능 모델 | Elastic Net 방어성공 계수 | `24_ElasticNet_defense_success_down_day_계수.png` | 방어성공 타깃에서 어떤 피처가 방향성을 갖는지 설명 |
| 4.4.2 해석가능 모델 | K-FGI 제외 RF 방어성공 permutation importance | `25_RF_defense_success_down_day_순열중요도.png` | 구성요소만 투입했을 때 `sent_composite_ma10`이 RF 방어성공 중요도 1위임을 제시 |
| 4.4.3 조건부 감성 | 감성조건부 하방방어 | `20_감성조건부_하방방어.png` | 공포 압력/부정 감성 조건에서 감성 피처의 추가 방어 효과 확인 |
| 4.4.3 조건부 감성 | 감성위험 지표와 추가 기여 | `21_감성위험지표와_추가기여.png` | 감성위험이 높아지는 구간과 감성 포함-제외 기여의 시간적 관계 설명 |
| 4.4.4 개선 실험 | 감성위험 overlay 후보 | `23_감성위험_오버레이_후보.png` | 감성위험을 점수에 섞는 것보다 노출 축소 overlay로 쓰는 개선 방향 제시 |

본문 추천 순서:

1. `13_감성피처_포함제외_성능비교.png`
2. `25_RF_defense_success_down_day_순열중요도.png`  
   단, 본문에서 반드시 “K-FGI 자체를 제외한 구성요소 중요도”라고 설명한다.
3. `20_감성조건부_하방방어.png`
4. `23_감성위험_오버레이_후보.png`

`21_감성위험지표와_추가기여.png`와 `22_감성후보피처_효과비교.png`는 본문이 길어지면 부록으로 보내도 된다.

### 9.4 4.5 전략 성과 및 강건성 검증에 넣을 그림/표

이 절은 논문의 실증 결과 핵심이다. K-FGI가 수익률 예측 전략이 아니라 하방방어 전략이라는 점을 가장 강하게 보여줘야 한다.

| 위치 | 넣을 항목 | 파일 | 목적 |
| --- | --- | --- | --- |
| 4.5.1 전체 성과 | 전략별 누적수익률 | `07_전략별_누적수익률.png` | Buy & Hold, trend, K-FGI 전략의 장기 성과 비교 |
| 4.5.1 전체 성과 | 10개년 성능표 | `priority_a_performance_robust_egarch.csv` | 연율수익률, Sharpe, MDD, 하락일 초과방어 수치 제시 |
| 4.5.2 낙폭 방어 | K-FGI 낙폭방어 | `08_KFGI_낙폭방어.png` | Buy & Hold 대비 drawdown 완화 시각화 |
| 4.5.2 낙폭 방어 | 통계검정 p값 요약 | `14_통계검정_p값_요약.png` | 전체 초과수익보다 하락일 방어가 통계적으로 강함을 제시 |
| 4.5.3 EGARCH 노출 조절 | EGARCH 변동성과 시장 노출 | `09_EGARCH_변동성과_시장노출.png` | 변동성 상승 시 노출 축소 구조 확인 |
| 4.5.4 Rolling OOS | rolling OOS 성과 안정성 | `17_rolling_OOS_성과안정성.png` | 시간축 전체에서 MDD 방어가 반복되는지 제시 |
| 4.5.4 Rolling OOS | 연도별 OOS Sharpe | `18_연도별_OOS_Sharpe.png` | 연도별 성과 편차와 시장환경 의존성 설명 |
| 4.5.5 국면별 성과 | 국면별 초과성과 | `19_국면별_초과성과.png` | 하락/고변동성/국면별 조건부 성과 제시 |
| 4.5.6 Crisis subperiod | crisis subperiod 표 | `priority_a_crisis_subperiod_performance.csv` | 2020, 2022, 2025 구간별 MDD와 downside defense 제시 |

본문 추천 순서:

1. 성과표 + `07_전략별_누적수익률.png`
2. `08_KFGI_낙폭방어.png`
3. `09_EGARCH_변동성과_시장노출.png`
4. `17_rolling_OOS_성과안정성.png`
5. crisis subperiod 표

`18_연도별_OOS_Sharpe.png`, `19_국면별_초과성과.png`는 본문 또는 부록 중 선택 가능하다. 논문 분량이 빡빡하면 19는 본문, 18은 부록을 추천한다.

### 9.5 4.6 파라미터 설정 근거에 넣을 그림/표

파라미터 설정 절은 “우리가 결과가 좋았던 값만 골랐다”는 인상을 줄이면 안 된다. 민감도와 해석 가능성을 함께 보여줘야 한다.

| 위치 | 넣을 항목 | 파일 | 목적 |
| --- | --- | --- | --- |
| 4.6.1 EGARCH 안정화 | EGARCH 안정화 진단표 | `priority_a_egarch_stability_diagnostics.csv` | min_obs=252, fallback/capped rate 근거 |
| 4.6.2 threshold | threshold/multiplier robustness 표 | `priority_a_threshold_multiplier_robustness.csv` | 25/65가 단일 최적이 아니라 안정적 기준임을 제시 |
| 4.6.3 거래비용 | 거래비용 민감도 표 | `priority_a_fee_sensitivity.csv` | 15bp 기준의 보수성 및 비용 민감도 제시 |
| 4.6.4 감성 overlay | 감성위험 overlay 표 | `sentiment_risk_overlay_tests.csv` | overlay는 메인 모델이 아니라 개선 실험/향후 연구로 제시 |

본문에는 각 표의 상위 5~10개 행만 요약하고, 원본 CSV는 Appendix 또는 supplementary material에 둔다.

### 9.6 Limitations / Discussion에 넣을 실험

| 위치 | 넣을 항목 | 파일 | 목적 |
| --- | --- | --- | --- |
| Limitations | 2016 초기 낙폭 진단 | `EARLY_DRAWDOWN_DIAGNOSIS.md`, `early_drawdown_diagnosis.csv` | K-FGI가 모든 구간에서 완벽히 방어하지 못함을 솔직히 제시 |
| Limitations | White reality check | `priority_b_white_reality_check.csv` | 데이터마이닝 가능성을 보수적으로 인정 |
| Future Work | no-leverage cap 1.0x | `early_drawdown_diagnosis.csv` | 실무적 레버리지 상한 개선 방향 |
| Future Work | sentiment risk overlay | `sentiment_risk_overlay_tests.csv` | 감성을 위험 overlay로 발전시키는 후속 연구 방향 |

붙여넣기용 Discussion 문장:

> 본 연구는 K-FGI가 하방위험 완화에 기여할 수 있음을 보였으나, 모든 구간에서 일관되게 우월한 성과를 보인 것은 아니다. 특히 2016년 초기 구간에서는 K-FGI가 과도한 노출을 허용하면서 Buy & Hold보다 큰 낙폭을 기록하였다. 이는 K-FGI를 단독 전략으로 사용하기보다 레버리지 상한, 감성위험 overlay, 거래비용 제약과 함께 운용해야 함을 시사한다.

### 9.7 Appendix / Supplementary에 넣을 항목

본문에 모두 넣기에는 많은 표와 그림은 부록으로 보낸다.

부록 추천 항목:

- `22_감성후보피처_효과비교.png`
- `24_ElasticNet_5d_large_drop_계수.png`
- `25_RF_5d_large_drop_순열중요도.png`
- `elastic_net_feature_importance_target_5d_down.csv`
- `elastic_net_feature_importance_target_5d_large_drop.csv`
- `random_forest_permutation_importance_target_5d_down.csv`
- `random_forest_permutation_importance_target_5d_large_drop.csv`
- `priority_b_benchmark_comparison.csv`
- `priority_b_kfgi_regime_zone_performance.csv`
- `priority_b_white_reality_check.csv`

부록 해석:

- 하락/급락 위험 예측 AUC가 0.5 근처라는 결과는 본문 핵심으로 크게 강조하지 않는다.
- 대신 부록에서 “본 지표는 방향 예측보다 방어성공 설명에 더 적합하다”는 보조 근거로 사용한다.

## 10. 최종 작업 순서

1. **논문 제목/Abstract/Introduction 수정**
   - 하방위험 관리 프레임워크임을 명확히 한다.
   - 수익률 예측 중심 표현은 선행연구 비교로만 남긴다.

2. **3장 Method 수정**
   - 3.5.5 `예측력 검증` 제목 변경.
   - 3.7 LightGBM 절을 Elastic Net/RF 기반 하방위험 피처 중요도 절로 교체.
   - sub_index1 제외 이유와 추세 피처 분리 이유 유지.

3. **4.3 실험 설계 수정**
   - 10개년 기간, 2,505 거래일, 252일 rolling, 2,254 windows 반영.
   - 2022~2025년 754행 기준 문장 제거.

4. **4.4 감성 피처 검증 수정**
   - 감성 포함/제외 ablation.
   - 하락일 방어성공 피처 중요도.
   - 감성위험 overlay를 순서대로 제시.

5. **4.5 전략 강건성 수정**
   - 전체 성능표.
   - crisis subperiod.
   - rolling OOS.
   - 거래비용 민감도.

6. **4.6 파라미터 근거 수정**
   - EGARCH min_obs=252.
   - threshold 25/65는 단일 최적이 아니라 robustness 기준.
   - 15bp 거래비용은 보수적 기준.
   - overlay는 개선 실험/향후 연구.

7. **그림 교체 및 영어화**
   - 논문 삽입 그림을 10개년 결과로 교체.
   - 한글 figure label을 영어로 변환.
   - 편집 가능한 PPT 파일로 Figure 번호 순서대로 정리.

8. **한계/결론 수정**
   - 전체 수익률 우위가 아니라 하방방어 중심.
   - 2016년 초기 구간 문제와 거래비용 민감도 솔직히 명시.

## 11. 논문 핵심 문장

최종적으로 논문 전체를 관통하는 문장은 다음과 같이 잡으면 된다.

> 본 연구의 K-FGI는 단기 수익률을 직접 예측하기 위한 모델이 아니라, 투자자 감성과 조건부 변동성을 결합하여 시장 하락 및 고변동성 구간에서 노출을 조절하는 리스크 관리 지표이다. 2015~2025년 10개년 실증분석 결과, K-FGI는 Buy & Hold 대비 전체 수익률 우위를 일관되게 제공하지는 않았으나, crisis subperiod와 rolling OOS 검증에서 MDD 및 하락일 손실을 완화하는 성격을 보였다. 특히 감성 피처는 단독 alpha라기보다 댓글 기반 투자자 반응 강도와 불안 확산을 반영하여 K-FGI의 하방위험 관리 기능을 보완하는 역할을 수행하였다.
