

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
- Walk-forward기반 시계열 검증
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

(사진)

---

## Results

| Strategy | Annual Return | Sharpe | MDD |
|---|---:|---:|---:|
| Buy & Hold | 17.97% | 0.942 | -20.67% |
| K-FGI Strategy | **14.73%** | **1.035** | **-14.87%** |

#### Main Findings

- Buy & Hold 대비 MDD 5.80%p 개선
- Sharpe Ratio 개선
- 하락장 방어 성능 강화
- 감성 피처의 유효성 확인

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

K-FGI는 투자 심리와 변동성을 결합해 Buy & Hold 대비 MDD를 **5.80%p** 줄이고 Sharpe Ratio를 개선하여 **하방 방어**에 강점을 보였습니다.
NAVER 금융 댓글 감성이 수익률 예측에 실질적으로 기여함을 Ablation Study로 검증했으며, 행동재무학과 머신러닝을 한국 시장에 접목한 투자 심리 지표 연구라는 점에서 의의가 있습니다.
