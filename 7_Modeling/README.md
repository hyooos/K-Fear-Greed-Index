# 모델링 모듈

이 폴더에는 K-FGI 프로젝트와 대시보드에서 재사용하는 모델링 모듈이 들어 있다.

여기 있는 파일들은 최종 코드 패키지에 포함되는 것이 맞다. 다만 논문에 들어가는 최종 표, 그림, 강건성 검정 결과는 주로 `tools/`의 실행 스크립트가 이 모듈들을 불러와 생성한다.

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
| `lgbm_analysis.py` | 이전 탐색용 LightGBM 분석 파일. 최종 논문의 메인 피처 중요도 방식은 아님 |

## 최종 논문 기준

최종 논문에서는 이 모델을 수익률 예측 모델이 아니라 **해석 가능한 하방위험 관리 지표**로 설명한다.

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
