# K-FGI Dashboard

GitHub 저장소 내부 요약 산출물(`paper_outputs`)을 읽는 Streamlit 대시보드입니다.

```bash
conda activate kfgi
streamlit run 8_Dashboard/streamlit_app.py
```

대시보드 입력 파일:

- `paper_outputs/data/kfgi_10y_timeseries.csv`
- `paper_outputs/data/strategy_returns_10y.csv`
- `paper_outputs/tables/*.csv`

원자료나 대용량 중간 산출물 없이도 최종 성능, K-FGI 시계열, 가중치 민감도, 피처 중요도, 검정 결과를 확인할 수 있습니다.
