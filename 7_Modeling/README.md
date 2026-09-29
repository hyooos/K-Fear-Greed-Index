# 모델링 모듈

이 폴더에는 K-FGI 프로젝트와 대시보드에서 재사용하는 모델링 모듈이 들어 있다.

## 파일 역할

| 파일 | 역할 |
| --- | --- |
| `config.py` | 공통 설정값과 모델링 상수 |
| `egarch_model.py` | EGARCH 변동성 추정 보조 함수 |
| `features.py` | 피처 엔지니어링 보조 함수 |
| `kfgi.py` | K-FGI 점수 산출 보조 함수 |
| `position.py` | 시장 노출도와 전략 포지션 산출 로직 |
| `evaluation.py` | 성과 및 하방위험 평가 지표 |
| `sensitivity.py` | 민감도 분석 보조 함수 |
| `visualization.py` | 시각화 보조 함수 |
| `lgbm_analysis.py` | 이전 탐색용 LightGBM 분석 파일. 최종 피처 중요도 방식은 아님 |

## 최종 모델 기준

이 모델은 수익률 예측 모델이 아니라 **해석 가능한 하방위험 관리 지표**로 사용한다.

최종 lean K-FGI 피처:

```text
sub_index2-7
sent_composite_ma10
egarch_vol
vol_regime_high
vol_ratio
```

최종 운용 제약:

- 노출 상한: `1.0x`
- `vol_shock > 2.0`, 최근 3일 누적수익률 `< -7%` 조건은 K-FGI 점수 피처가 아니라 circuit breaker overlay이다.
