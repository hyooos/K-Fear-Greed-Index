# K-Fear&Greed Index Dashboard

`paper_outputs`의 최종 산출물을 읽어 한국형 Fear & Greed Index와 투자 시뮬레이션을 보여주는 Streamlit 대시보드입니다.

## 화면 구성

- `K-FGI 현재 상태`: 현재 점수를 바늘형 계기판으로 표시합니다.
- `과거와 비교`: 현재, 전일, 1주 전, 1개월 전, 1년 전의 K-FGI 점수와 상태를 비교합니다.
- `투자 시뮬레이션`: 기간, 초기 투자금, 매월 추가 투자금, 전략을 설정합니다.
- `한눈에 보기`: 전략별 누적 평가금액과 최근 20거래일 상태를 확인합니다.
- `K-FGI 흐름`: K-FGI 원 지수와 5거래일 평균 흐름을 확인합니다.
- `투자 결과`: 전략별 최종금액, 손익, 수익률, 위험 대비 점수, 최대 하락폭을 비교합니다.

## UI 방향

- 상태 라벨은 `극단적 공포`, `공포`, `중립`, `탐욕`, `극단적 탐욕`으로 표시합니다.
- 투자금 입력은 `100,000,000`처럼 쉼표가 포함된 형태로 보입니다.
- 투자금 입력 옆의 `- / +` 버튼으로 금액을 빠르게 조정할 수 있습니다.
- 피처 중요도, 통계 검정, 가중치 민감도 같은 연구 검증 화면은 제외했습니다.

## 실행

```bash
conda activate kfgi
streamlit run 8_Dashboard/streamlit_app.py
```

## 입력 파일

- `paper_outputs/data/kfgi_10y_timeseries.csv`
- `paper_outputs/data/strategy_returns_10y.csv`
- `paper_outputs/tables/performance_summary.csv`
