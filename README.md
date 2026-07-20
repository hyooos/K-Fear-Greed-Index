## 📉 K-Fear & Greed Index (K-FGI)

### A Downside Risk Management Framework Using a Sentiment-Volatility Integrated Index
> 감성 및 변동성 결합 지표(K-FGI)를 활용한 주식시장 리스크 관리 전략 연구

---

## Overview

본 프로젝트는 NAVER 금융 댓글 기반 감성 분석과 EGARCH(1,1) 변동성 모델링을 결합하여 한국형 투자심리 지표인 **K-Fear & Greed Index (K-FGI)** 를 구축한 프로젝트입니다.
K-FGI를 활용하여 시장 국면(Bull / Normal / Crisis)에 따라 포지션을 조절하는 리스크 관리 전략을 설계했습니다.

---

## Key Features

- NAVER 금융 댓글 기반 감성 분석
- EGARCH 기반 변동성 모델링
- 감성 및 변동성 결합 K-FGI 지표 생성
- Walk-forward 기반 시계열 검증
- Downside Risk(MDD) 중심 리스크 관리 전략
- 감성 피처 기여도 검증(Ablation Study) 

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

## Results

| Strategy | Annual Return | Sharpe Ratio | MDD |
|---|---:|---:|---:|
| Buy & Hold | 17.97% | 0.942 | -20.67% |
| K-FGI Strategy | **13.65%** | **0.984** | **-14.59%** |

### Main Findings

- Buy & Hold 대비 MDD **6.08%p 개선**
- 하락 구간에서 손실 방어 성능 강화
- Crisis Regime에서 상대적으로 안정적인 방어 성과 확인
- 감성 피처 제거 시 Sharpe Ratio(0.984 → 0.802) 감소
- 감성 피처 제거 시 MDD(-14.59% → -17.57%) 악화
- 감성 + 변동성 결합 구조의 유효성 검증

---

## Directory Structure

```bash
.
├── 1_KFGI_subindex/
├── 2_Naver_crawling/
├── 3_Filtering_final/
├── 4_Sentiment_analysis/
├── 5_Merge_to_final_csv/
├── 6_KFGI_weight/
├── 7_Modeling/
├── 8_Dashboard/
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Run

### Installation

```bash
git clone https://github.com/your-repository.git

cd your-repository

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

K-FGI 기반 전략은 Buy & Hold 대비 최대 낙폭(MDD)을 약 6.08%p 감소시키며, 하락장 방어 성능을 개선했습니다. 
또한 감성 피처 제거 시 Sharpe Ratio와 MDD 성과가 악화되는 것을 통해 감성 정보가 실제 수익률 예측과 리스크 관리 성과에 유의미하게 기여함을 확인했습니다.

특히 Walk-forward 기반 검증 구조를 적용하여 Look-ahead Bias를 최소화하고, 시장 국면(Bull / Normal / Crisis)에 따라 포지션을 동적으로 조절하는 하방 리스크 중심 전략 프레임워크를 구축했다는 점에서 의의가 있습니다.
