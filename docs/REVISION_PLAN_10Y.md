# 10-Year K-FGI Paper Revision Plan

## 1. 핵심 방향 전환

이전 원고는 2022~2025년 3개년 실험을 기준으로 작성되어 있으며, K-FGI의 예측력과 전략 성능을 비교적 강하게 주장하는 구조다. 10개년 결과로 확장하면 논문 방향을 다음과 같이 조정해야 한다.

기존 방향:

> K-FGI가 향후 수익률과 유의한 관계를 가지며, 감성 피처가 예측력 향상에 기여한다.

10개년 수정 방향:

> K-FGI는 평균 수익률 예측 신호라기보다, NAVER 금융 댓글 감성과 EGARCH 조건부 변동성을 결합해 KOSPI200 하방 위험을 관리하는 투자심리 지표다.

현재 10개년 결과에서는 전체 초과수익이나 K-FGI와 향후 수익률의 직접 상관이 강하게 유의하지 않다. 반면 downside defense는 매우 강하게 나타나므로 논문 주장은 “수익률 예측”보다 “하방 위험 관리”에 두는 것이 안전하다.

## 2. 반드시 수정해야 하는 내용

### 2.1 분석 기간

이전 원고:

- 2022.11~2025.12 중심
- 관측치 약 754개

10개년 수정:

- 분석 대상 데이터: 2015-01-16 ~ 2025-12-29
- 최종 실험 구간: 2015-10-08 ~ 2025-12-22
- 최종 실험 관측치: 2,505개
- 2013~2014년 데이터는 rolling/warm-up 계산용으로 설명

### 2.2 K-FGI 예측력 주장

이전 원고에는 K-FGI와 향후 5일 수익률 간 Spearman 상관계수 `rho=0.1995`가 유의하다고 작성되어 있다. 10개년 결과에서는 다음처럼 바뀐다.

- 1일 Spearman rho: 0.0374, p=0.0612
- 5일 Spearman rho: 0.0088, p=0.6586

따라서 “K-FGI가 향후 5일 수익률을 유의하게 예측한다”는 문장은 삭제하거나 크게 약화해야 한다.

대체 문장:

> 10개년 표본에서 K-FGI와 향후 수익률 간 단순 상관은 강하게 유의하지 않았다. 그러나 K-FGI를 변동성 및 추세 기반 포지션 조절 규칙과 결합할 경우, Buy & Hold 대비 최대 낙폭을 크게 낮추는 효과가 나타났다.

### 2.3 감성 피처 기여도

이전 원고는 감성 피처가 예측력 향상에 실질적으로 기여한다는 톤이 강하다.

10개년 결과:

- 감성 포함 K-FGI 총수익률: 128.92%
- 감성 제외 K-FGI 총수익률: 100.02%
- 감성 포함 Sharpe: 0.613
- 감성 제외 Sharpe: 0.660
- sentiment bootstrap delta rho p=0.301

따라서 감성 피처는 “통계적으로 강한 예측력”이 아니라 “총수익률 개선에 기여했으나 Sharpe 측면에서는 trade-off가 존재하는 보조 피처”로 써야 한다.

### 2.4 EGARCH 그래프 및 추정 안정성

현재 EGARCH 결과에는 2015~2016 초반 일부 추정값이 비정상적으로 폭발하는 문제가 확인되었다.

확인된 문제:

- 정상 일별 변동성 중앙값: 약 1.0%
- 일부 초기 추정값: `1.34e+152`

해야 할 일:

1. EGARCH `min_obs`를 60에서 252로 늘리는 재실험.
2. 비정상 추정값은 rolling volatility로 대체.
3. figure에는 winsorizing/log scale 여부를 명시.
4. 논문에는 “초기 EGARCH 추정 안정성을 위해 1년 warm-up을 사용했다”는 식으로 설명.

### 2.5 sub_index1 처리

이전 원고의 방향은 맞다. 다만 10개년 논문에서는 더 명확히 써야 한다.

- sub_index1은 원자료에는 보존.
- 최종 K-FGI 입력에서는 제외.
- 이유: 가격 기반 market momentum은 `mom20`, `mom60`, `ma_ratio`, `RSI` 등 추세 피처와 중복.
- 따라서 K-FGI는 투자심리/변동성 중심 지표로 유지.

## 3. 추가 확인해야 할 분석

### 우선순위 A: 꼭 필요

1. EGARCH 안정화 재실험
   - `min_obs=252`
   - 추정값 상한 처리
   - rolling volatility fallback

2. 10개년 성능표 업데이트
   - Buy & Hold
   - Trend only
   - Trend + EGARCH
   - K-FGI with sentiment
   - K-FGI without sentiment

3. 거래비용 민감도
   - 0bp, 5bp, 10bp, 15bp, 20bp
   - 전략 성과가 비용에 얼마나 민감한지 확인

4. Threshold robustness
   - Fear/Greed 기준 20/80, 25/65, 30/70
   - K-FGI multiplier range 비교

5. Crisis subperiod 분석
   - 2020 코로나
   - 2022 금리 인상/약세장
   - 2025 강세장/OOS

### 우선순위 B: 있으면 논문 설득력 상승

1. 감성 피처 세부 ablation
   - 감성 전체 제외
   - `sent_norm_w`만 사용
   - 부정감성 피처만 사용
   - 감성분산/의견일치 피처만 사용

2. Benchmark 추가
   - CNN-style equal weight K-FGI
   - market-only FGI
   - sentiment-only index
   - EGARCH-only vol targeting

3. K-FGI regime별 수익률/낙폭
   - Fear 구간
   - Neutral 구간
   - Greed 구간

4. White reality check 또는 bootstrap
   - 전략 성능이 데이터마이닝 결과가 아닌지 확인

## 4. 논문 구조 수정안

### 1. Introduction

수정 방향:

- 기존 문단은 유지 가능.
- 단, “수익률 예측”보다 “하방 위험 관리”를 전면에 배치.
- 한국 시장 특화 FGI 부재, 댓글 감성의 즉시성, EGARCH의 하락장 대응 필요성을 연결.

추천 핵심 문장:

> 본 연구는 K-FGI를 단기 수익률 예측 신호가 아니라, 투자자 심리와 조건부 변동성을 통합해 시장 노출을 조절하는 위험 관리형 지표로 설계한다.

### 2. Related Work

유지하되 보강:

- CNN Fear & Greed Index
- Baker-Wurgler sentiment index
- BOK News Sentiment Index
- 한국 뉴스 감성지수/감성분석 논문
- 투자심리와 변동성 예측 연구
- EGARCH 및 volatility targeting 연구

추가할 흐름:

> 기존 연구는 시장 기반 심리지표, 텍스트 감성지표, 변동성 모델링을 각각 다루었으나, 한국시장 댓글 감성·복합 FGI·EGARCH 기반 노출 조절을 하나의 10개년 일별 전략으로 통합한 연구는 제한적이다.

### 3. Data

새로 강화해야 함:

- KRX subindex별 데이터 출처
- NAVER 댓글 수집 기준
- 2013~2014년 데이터의 역할
- 2015~2025년 최종 분석 구간
- missing 처리
- 댓글 없는 날 처리
- raw data는 공개하지 않는 이유

### 4. Method

순서:

1. Subindex construction
2. Comment filtering
3. Sentiment score normalization
4. EGARCH volatility estimation
5. K-FGI construction
6. Position rule
7. Validation design

여기서 LightGBM은 투자 의사결정 모델이 아니라 feature importance 분석 도구라고 명확히 써야 한다.

### 5. Results

권장 순서:

1. 데이터/필터링 결과
2. 감성 정규화 근거
3. K-FGI 시계열
4. 전략 성과 비교
5. Drawdown 비교
6. EGARCH exposure 분석
7. Feature importance
8. Weight sensitivity
9. Sentiment ablation
10. Statistical tests
11. OOS 2025

### 6. Discussion

반드시 들어갈 내용:

- 전체 초과수익은 유의하지 않음.
- 하방 방어는 유의함.
- 감성 피처는 총수익률 개선에 기여하지만 Sharpe 개선은 제한적.
- K-FGI는 수익률 예측 모델보다 risk management index로 해석.
- EGARCH 초기 추정 안정성 및 데이터 공개 제한이 한계.

### 7. Conclusion

주장:

> K-FGI는 한국시장 투자심리와 변동성 정보를 결합하여 Buy & Hold 대비 MDD를 크게 낮추는 위험 관리형 지표로 기능한다.

피해야 할 주장:

> K-FGI가 10개년 전체에서 향후 수익률을 강하게 예측한다.

## 5. 현재 원고에서 바로 고쳐야 할 숫자

| 항목 | 기존 원고 | 10개년 수정 |
|---|---:|---:|
| 관측치 | 754개 | 2,505개 |
| K-FGI 기간 | 2022.11~2025.12 | 2015.10~2025.12 |
| Spearman 5D | 0.1995 유의 | 0.0088, p=0.6586 |
| Buy & Hold MDD | -20.67% 수준 | -41.19% |
| K-FGI MDD | -14.87% 수준 | -22.93% |
| K-FGI 총수익률 | 기존 3개년 기준 | 128.92% |
| Buy & Hold 총수익률 | 기존 3개년 기준 | 137.64% |

## 6. 작성 톤

논문 전체 톤은 다음처럼 조정한다.

강한 주장:

- 한국형 복합 투자심리 지표를 구축했다.
- 댓글 감성과 시장 기반 지표를 결합했다.
- EGARCH 변동성을 통해 하방 위험 관리 전략으로 확장했다.
- 10개년 데이터에서 MDD 개선이 확인되었다.

약한 주장:

- 평균 수익률 예측력.
- 감성 피처의 독립적 통계 유의성.
- K-FGI threshold의 보편성.

추천 제목:

> NAVER 금융 댓글 감성과 EGARCH 변동성을 결합한 한국형 Fear & Greed Index 구축 및 KOSPI200 하방 위험 관리 전략

