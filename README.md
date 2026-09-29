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
- Walk-forward 기반 시계열 검증
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

KOSPI 200, 평가 기간 2016-01-06 ~ 2025-12-22(2,445거래일), 편도 거래비용 3bp 기준입니다.

| Strategy | Total Return | Sharpe | MDD | CVaR 5% | 하락일 방어폭 | 평균 노출 |
|---|---:|---:|---:|---:|---:|---:|
| Buy & Hold | 149.21% | 0.507 | -41.19% | -2.73% | 0.00bp | 1.000 |
| **K-FGI Strategy** | 99.73% | **0.679** | **-20.55%** | -1.65% | **46.34bp** | 0.528 |
| 이동평균 추세 전략 | 86.49% | 0.464 | -32.02% | -2.23% | 37.32bp | 0.581 |
| 실현 변동성 타겟 전략 | 41.09% | 0.336 | -27.08% | -1.59% | 34.53bp | 0.636 |
| EGARCH 변동성 타겟 전략 | 47.42% | 0.382 | -26.20% | -1.55% | 34.60bp | 0.632 |
| 평균 노출 동일 정적 전략 | 61.96% | 0.507 | -24.44% | -1.44% | 40.54bp | 0.528 |

### Main Findings

- Buy & Hold 대비 MDD **-41.19% → -20.55%**, CVaR 5% **-2.73% → -1.65%**
- 시장 하락일 하루 평균 **46.34bp** 손실 감소
- Sharpe **0.679** (Buy & Hold 0.507), 총수익률은 99.73%로 Buy & Hold(149.21%)보다 낮음
- 평균 노출이 같은 정적 전략 대비 Sharpe·MDD 차이는 통계적으로 확인되지 않았고, 하락일 방어폭 차이만 유의
- 성과의 대부분은 노출 조절 단계의 추세 가중에서 나오며, K-FGI 기반 심리 배수는 추가 기여
- 감성 피처를 빼면 Sharpe 0.679 → 0.617, MDD -20.55% → -25.13%로 악화되어 감성 피처가 구성 요소 중 성과에 가장 크게 기여
- K-FGI가 낮은 구간의 이후 5거래일 변동성은 연 16.1%로 높은 구간(12.5%)보다 커서 위험 상태를 구분함

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

본 프로젝트는 시장 하위 지표, NAVER 금융 댓글 감성, EGARCH(1,1) 조건부 변동성을 결합한 한국형 공포·탐욕 지수 K-FGI(K-Fear & Greed Index)를 제안했습니다.

K-FGI를 노출 조절에 적용한 전략은 Buy & Hold 대비 최대 낙폭(MDD)을 -41.19%에서 **-20.55%**로 줄이고, Sharpe 지수를 0.507에서 **0.679**로 높였습니다. 평균 노출이 0.528배로 낮아 개선의 일부는 노출 축소에서 나오며, 같은 노출의 정적 전략과의 Sharpe·MDD 차이는 통계적으로 확인되지 않았습니다.

K-FGI는 수익률 예측 모델이 아니라 시장의 위험 상태를 요약하는 지표로, 감성 피처는 거래를 줄이고 낙폭을 얕게 하는 데 기여했습니다.

---

## Notice

본 저장소는 학술 연구 및 프로젝트 정리를 위한 코드 저장소이며, 투자 조언이 아닙니다.
