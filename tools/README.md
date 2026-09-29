# 도구 스크립트

이 폴더에는 최종 K-FGI 논문 결과를 재현하기 위한 실행 스크립트가 들어 있다.

`7_Modeling/`이 재사용 가능한 모델링 함수 모음이라면, `tools/`는 데이터를 불러오고, 모델링 모듈을 호출하고, 강건성 검정을 수행한 뒤, 논문용 표와 그림을 저장하는 실행용 파일이다.

## 최종 재현용 스크립트


| 스크립트 | 목적 | 주요 산출물 |
| --- | --- | --- |
| `create_final_paper_master_md.py` | 최종 요약, 피처 역할표, 통계 검정, 논문 문항별 배치안 생성 | `paper_outputs/FINAL_PAPER_MASTER_SUMMARY.md`, `paper_outputs/tables/final_master_*.csv` |
| `create_final_paper_figures.py` | 최종 논문용 가독성 개선 그림 생성 | `paper_outputs/final_figures/` |
| `run_q1_benchmark_package.py` | Buy & Hold, MA, 변동성 타깃팅, market-only, sentiment-only와 최종 K-FGI 비교 | `paper_outputs/Q1_BENCHMARK_RESULTS.md`, `paper_outputs/tables/q1_benchmark_*.csv` |
| `run_downside_risk_extension.py` | Sortino, Calmar, CVaR, 노출도 분석, bootstrap 검정 | `paper_outputs/DOWNSIDE_RISK_EXTENSION_RESULTS.md` |
| `run_interpretability_economic_layer.py` | 경제적 해석과 특정 날짜 case study 생성 | `paper_outputs/ECONOMIC_INTERPRETATION_EXPLAINABILITY.md` |
| `run_final_conservative_sensitivity.py` | 최종 거래비용, cap, threshold, multiplier 민감도 분석 | `paper_outputs/FINAL_CONSERVATIVE_KFGI_SENSITIVITY.md` |

## 강건성 및 방어용 스크립트

부록 표나 reviewer 대응용 검정에 사용하는 파일이다.

| 스크립트 | 목적 |
| --- | --- |
| `run_subindex1_momentum_comparison.py` | `sub_index1` 또는 기술적 모멘텀 피처 추가가 최종 lean 모델을 개선하는지 확인 |
| `run_circuit_breaker_overlap_analysis.py` | circuit breaker overlay가 K-FGI 본체 피처와 중복되는지 확인 |
| `run_smoothed_feature_experiments.py` | raw 감성 피처와 평활화 복합 감성 피처 비교 |
| `run_feature_drop_ablation.py` | 피처 제거 ablation 실험 |
| `run_interpretable_feature_importance.py` | ElasticNet 및 Random Forest 기반 피처 중요도 분석 |
| `run_oos_regime_evaluation.py` | OOS 및 국면별 성과 평가 |
| `run_conservative_kfgi_variants.py` | 보수형 포지션 상한 후보 비교 |
| `run_priority_ab_experiments.py` | 우선순위 A/B 강건성 실험 묶음 |
| `diagnose_early_drawdown.py` | 2016-2017 초기 구간 낙폭 진단 |
| `run_event_regime_overlay_experiments.py` | 이벤트 스트레스와 감성위험 overlay 실험 |
| `run_sentiment_feature_development.py` | 감성 피처 개발용 탐색 실험 |

## 데이터 준비 스크립트

| 스크립트 | 목적 |
| --- | --- |
| `build_final_model_csv.py` | subindex와 감성 피처를 합쳐 최종 모델 입력 CSV 생성 |
| `run_10y_kfgi_experiments.py` | 10개년 전체 실험의 이전 버전 실행 파일. 추적성과 비교용으로 보존 |

## 그림 생성 보조 스크립트

| 스크립트 | 목적 |
| --- | --- |
| `create_paper_figures.py` | 이전 버전 논문 그림 생성 |
| `redraw_readable_figures.py` | 이전 그림을 더 읽기 쉬운 형태로 다시 그림 |
