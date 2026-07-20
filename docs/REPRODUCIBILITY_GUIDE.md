# Reproducibility Guide

이 문서는 로컬 `논문용` 데이터가 준비되어 있다는 전제에서 10개년 K-FGI 결과를 다시 생성하는 순서입니다.

## 1. 환경 준비

```bash
conda activate kfgi
```

새 환경을 만들 경우:

```bash
conda env create -f environment.yml
conda activate kfgi
```

## 2. 최종 모델 CSV 생성

```bash
python tools/build_final_model_csv.py
```

입력:

- KRX subindex 1~7
- NAVER 금융 댓글 감성 피처
- KOSPI200 시장 데이터

출력:

- `KFG_final_10y.csv`
- missing report
- raw merged CSV

## 3. 10개년 K-FGI x EGARCH 실험

```bash
python tools/run_10y_kfgi_experiments.py
```

핵심 설정:

- `sub_index1`은 최종 K-FGI/modeling 피처에서 제외
- EGARCH 변동성 피처 사용
- 감성 포함/제외 ablation 비교
- 가중치 민감도 분석
- 2025년 OOS 성능 확인

## 4. 논문용 figure와 dashboard 생성

```bash
MPLCONFIGDIR=/tmp/kfgi_mpl_conda python tools/create_paper_figures.py
```

생성 항목:

- 논문용 figure 15개
- performance/statistical/sensitivity table
- Streamlit dashboard
- GitHub용 `paper_outputs`

## 5. Dashboard 실행

GitHub 저장소 내부 요약 산출물 기준:

```bash
streamlit run 8_Dashboard/streamlit_app.py
```

로컬 논문용 폴더 기준:

```bash
cd /Users/hyowon/Desktop/uni/3-1/기계학습/논문용
conda run -n kfgi streamlit run 8_Dashboard/streamlit_app.py
```

## 6. 결과 해석 방향

K-FGI 전략은 Buy & Hold보다 총수익률이 약간 낮지만, MDD를 크게 낮춥니다. 통계 검정에서도 전체 초과수익보다 하락 구간 방어 성능이 강하게 나타납니다. 따라서 논문에서는 K-FGI를 수익률 예측 모델이 아니라 하방 위험 관리 지표로 해석하는 것이 적절합니다.
