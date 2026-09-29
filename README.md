## 📉 K-Fear & Greed Index (K-FGI)

### A Downside Risk Management Framework Using a Sentiment-Volatility Integrated Index
> 감성 및 변동성 결합 지표(K-FGI)를 활용한 주식시장 하방위험 관리 전략 연구

---

## Overview

본 프로젝트는 NAVER 금융 댓글 기반 감성 분석, KRX 시장 하위지표, EGARCH(1,1) 변동성 모델링을 결합하여 한국형 투자심리 지표인 **K-Fear & Greed Index (K-FGI)** 를 구축한 프로젝트입니다.

K-FGI를 활용하여 시장 국면(Bull / Normal / Crisis)에 따라 포지션을 조절하고, 시장 하락 구간에서 최대 낙폭(MDD)을 줄이는 리스크 관리 전략을 설계했습니다.

---

## Key Features

- NAVER 금융 댓글 기반 감성 분석
- KRX 기반 시장 하위지표 구성
- EGARCH 기반 조건부 변동성 모델링
- 감성 및 변동성 결합 K-FGI 지표 생성
- Walk-forward 기반 시계열 검증 및 정보 누수 제거 재검증
- Downside Risk(MDD) 중심 리스크 관리 전략
- 감성 피처 및 모멘텀 피처 강건성 검정
- Circuit Breaker Overlay를 활용한 극단 하락 구간 대응

---

## Tech Stack

#### Language

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" height="28"/>
</p>

---

#### Data Processing & Analysis

<p>
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" height="28"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" height="28"/>
  <img src="https://img.shields.io/badge/Scikit Learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" height="28"/>
</p>

---

#### Machine Learning & NLP

<p>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" height="28"/>
  <img src="https://img.shields.io/badge/Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black" height="28"/>
  <img src="https://img.shields.io/badge/LightGBM-02569B?style=flat-square" height="28"/>
  <img src="https://img.shields.io/badge/KR--FinBERT--SC-6A1B9A?style=flat-square" height="28"/>
</p>

---

#### Time Series & Volatility Modeling

<p>
  <img src="https://img.shields.io/badge/EGARCH-1E88E5?style=flat-square" height="28"/>
  <img src="https://img.shields.io/badge/RidgeCV-43A047?style=flat-square" height="28"/>
  <img src="https://img.shields.io/badge/ARCH-5E35B1?style=flat-square" height="28"/>
</p>

---

#### Visualization & Dashboard

<p>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square" height="28"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" height="28"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" height="28"/>
</p>

---

## Pipeline

<img width="1071" height="409" alt="image" src="https://github.com/user-attachments/assets/82221098-5b66-4153-a491-349d6bcb961a" />

---

## Final Model

최종 K-FGI는 기존 하위지표와 감성, 변동성 피처를 모두 사용하는 방식에서 출발했지만, 최종 실험에서는 과적합 가능성을 줄이기 위해 lean feature set을 사용했습니다.

```text
sub_index2
sub_index3
sub_index4
sub_index5
sub_index6
sub_index7
sent_composite_ma10
egarch_vol
vol_regime_high
vol_ratio
```

`sub_index1`은 모멘텀 성격이 강해 원자료 및 강건성 검정에는 남기되, 최종 K-FGI 점수 계산에서는 제외했습니다.

---

## Results

누수를 제거한 파이프라인(`9_Leakfree_experiments/step6`) 기준 결과입니다. 논문의 원래 규칙에서 정보 누수만 막고 거래 시점을 s+2일(단일 지연)로 조정했으며, 평가 기간 2016-01-06 ~ 2025-12-22(2,445거래일) 전체가 표본 밖입니다. KOSPI 200, 편도 거래비용 3bp 기준입니다.

| Metric | Buy & Hold | 같은 노출의 정적 전략 | K-FGI Strategy |
|---|---:|---:|---:|
| Sharpe | 0.501 | 0.501 | **0.522** |
| Total Return | 146.8% | 64.0% | 73.5% |
| MDD | -41.2% | -25.2% | **-27.6%** |
| CVaR (5%, 일별) | -2.73% | -1.50% | -1.76% |
| Average Exposure | 1.00x | 0.55x | 0.55x |
| 하락일 방어 (bp/day) | - | 38.9 | **43.7** |

### Main Findings

- Buy & Hold 대비 MDD **13.6%p 개선**, 다만 블록 부트스트랩 95% 신뢰구간은 [-3.4, 34.7]로 0을 포함
- Buy & Hold 대비 CVaR 개선은 모든 지수(KOSPI 200, KOSPI, KOSDAQ)에서 유의
- 같은 평균 노출(0.55x)의 정적 전략 대비 Sharpe 차이는 +0.02, 신뢰구간 [-0.43, 0.46]으로 **유의하지 않음**
- K-FGI 구성 요소 중 **감성 피처가 가장 유용** (감성만 사용 시 Sharpe 0.631, 세 지수 모두 1위)
- 감성·하위지표를 더해도 변동성·VaR·큰 손실 예측 정확도는 개선되지 않음 (Diebold-Mariano 검정) → 감성의 기여는 예측 정확도보다 노출 신호를 안정적으로 만드는 데서 나옴

### Leak-free Re-experiments

`9_Leakfree_experiments/`에서는 학습 라벨·표준화·EGARCH 추정이 결정 시점 이후 정보를 쓰지 않도록 파이프라인을 다시 구성했습니다.

- 개발 구간(2016-01-06 ~ 2023-06-30)에서만 설정을 선택하고, 봉인 구간(2023-07-01 ~ 2025-12-22)은 한 번만 평가
- 머신러닝(LightGBM, RF, XGBoost)과 강화학습을 포함해 총 482개 설정을 시도했지만, 어떤 설정도 같은 노출의 정적 전략을 통계적으로 이기지 못함
- 변동성 예측력은 EGARCH가 복잡한 모델보다 낫거나 비슷함
- 이전 파이프라인(누수 포함, s+3 지연)의 결과(MDD -26.6%, 개선 14.6%p)는 누수 제거 후 위 표의 수치로 대체됨

자세한 결과는 [`9_Leakfree_experiments/README.md`](9_Leakfree_experiments/README.md)를 참고하세요.

---

## Directory Structure

```bash
.
├── 1_KFGI_subindex/          # KRX 기반 시장 하위지표 생성
├── 2_Naver_crawling/         # NAVER 금융 기사 및 댓글 수집
├── 3_Filtering_final/        # 정치/독성 댓글 필터링
│   └── param_experiments/    # 필터링 파라미터 및 하위 전략 민감도 실험
├── 4_Sentiment_analysis/     # 댓글 감성 분석 및 감성 피처 생성
├── 5_Merge_to_final_csv/     # 하위지표, 감성, 시장 데이터 병합
├── 6_KFGI_weight/            # K-FGI 가중치 산출 및 초기 실험
├── 7_Modeling/               # EGARCH, K-FGI 전략, 성과 평가
├── 8_Dashboard/              # Streamlit 기반 시각화 대시보드
├── 9_Leakfree_experiments/   # 정보 누수를 제거한 재실험 (개발/봉인 구간 분리)
├── tools/                    # 최종 실험 및 논문용 산출물 생성 스크립트
├── .gitignore                # Git 업로드 제외 파일 설정
├── environment.yml           # Conda 환경 설정
├── requirements.txt          # Python 패키지 의존성
└── README.md                 # 프로젝트 소개 문서
```

---

## Run

### Installation

```bash
git clone https://github.com/hyooos/K-Fear-Greed-Index.git

cd K-Fear-Greed-Index

conda env create -f environment.yml

conda activate kfgi
```

이미 환경이 있는 경우:

```bash
pip install -r requirements.txt
```

---

### Streamlit Dashboard

```bash
streamlit run 8_Dashboard/streamlit_app.py
```

---

## Conclusion

본 연구는 NAVER 금융 댓글 기반 감성 분석과 EGARCH(1,1) 변동성 모델링을 결합하여 한국형 투자심리 지표인 K-FGI(K-Fear & Greed Index)를 제안했습니다.

정보 누수를 제거한 재검증에서 K-FGI 기반 전략은 Buy & Hold 대비 최대 낙폭(MDD)을 약 **13.6%p**, 꼬리위험(CVaR)을 유의하게 줄였습니다. 다만 같은 평균 노출의 정적 전략과 비교하면 위험조정 성과 차이는 통계적으로 유의하지 않아, 하방 위험 감소의 상당 부분은 평균 노출 축소에서 나온다고 해석됩니다.

K-FGI 구성 요소 중에서는 감성 피처가 지수·목표 변수와 관계없이 가장 일관되게 유용했으며, 이는 추세와 겹치지 않는 정보를 노출 신호에 더하기 때문으로 보입니다.

---

## Notice

본 저장소는 학술 연구 및 프로젝트 정리를 위한 코드 저장소이며, 투자 조언이 아닙니다.
