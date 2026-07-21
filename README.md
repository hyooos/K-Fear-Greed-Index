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

## Streamlit Dashboard

본 프로젝트는 최종 산출물을 일반 사용자도 직관적으로 확인할 수 있도록 **K-Fear&Greed Index 대시보드**를 제공합니다.

대시보드는 논문용 검정표를 나열하기보다, 실제 Fear & Greed Index를 보는 경험에 가깝게 구성했습니다.

- 현재 K-FGI 점수를 바늘형 계기판으로 표시
- `극단적 공포 / 공포 / 중립 / 탐욕 / 극단적 탐욕` 상태를 한국어로 해석
- 전일, 1주 전, 1개월 전, 1년 전 점수와 현재 상태 비교
- 초기 투자금과 매월 추가 투자금을 입력해 전략별 평가금액 시뮬레이션
- `- / +` 버튼과 쉼표 표기(`100,000,000`)를 지원하는 투자금 입력 UI
- K-FGI 흐름과 전략별 누적 평가금액을 Plotly 차트로 확인
- 연구용 피처 중요도, 통계 검정, 민감도 분석 화면은 제외하여 사용자용 화면에 집중

주요 화면은 `한눈에 보기`, `K-FGI 흐름`, `투자 결과` 세 탭으로 구성되어 있습니다.

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

| Metric | Buy & Hold | K-FGI Strategy |
|---|---:|---:|
| MDD | -41.2% | **-26.6%** |
| MDD Improvement | - | **+14.6%p** |
| Average Exposure | 1.00x | **0.55x** |
| Max Exposure | 1.00x | **1.00x** |
| Volatility Reduction | - | **40.3%** |

### Main Findings

- Buy & Hold 대비 MDD **14.6%p 개선**
- 시장 하락일 평균 **43.1 bp/day** 방어 효과 확인
- 하락일 방어 Hit Rate **68.1%**
- 평균 시장 노출도 **0.55x**로 감소
- 연환산 변동성 **40.3% 감소**
- 감성 + 변동성 결합 구조가 하방 리스크 관리에 유효함을 확인

---

## Directory Structure

```bash
.
├── 1_KFGI_subindex/          # KRX 기반 시장 하위지표 생성
├── 2_Naver_crawling/         # NAVER 금융 기사 및 댓글 수집
├── 3_Filtering_final/        # 정치/독성 댓글 필터링
├── 4_Sentiment_analysis/     # 댓글 감성 분석 및 감성 피처 생성
├── 5_Merge_to_final_csv/     # 하위지표, 감성, 시장 데이터 병합
├── 6_KFGI_weight/            # K-FGI 가중치 산출 및 초기 실험
├── 7_Modeling/               # EGARCH, K-FGI 전략, 성과 평가
├── 8_Dashboard/              # Streamlit 기반 시각화 대시보드
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
git clone https://github.com/your-repository.git

cd your-repository

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

K-FGI 기반 전략은 Buy & Hold 대비 최대 낙폭(MDD)을 약 **14.6%p** 감소시키며, 하락장 방어 성능을 개선했습니다.

특히 Walk-forward 기반 검증 구조를 적용하여 Look-ahead Bias를 최소화하고, 시장 국면(Bull / Normal / Crisis)에 따라 포지션을 동적으로 조절하는 하방 리스크 중심 전략 프레임워크를 구축했다는 점에서 의의가 있습니다.

---

## Notice

본 저장소는 학술 연구 및 프로젝트 정리를 위한 코드 저장소이며, 투자 조언이 아닙니다.
