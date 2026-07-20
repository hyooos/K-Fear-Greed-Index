# Yearly Event Interpretation and Event-Stress Feature Notes

## 1. 왜 연도별 이벤트 해석이 필요한가

K-FGI의 실증 결과는 모든 연도에서 동일하게 좋지 않다. 따라서 논문에서는 단순히 전체 성과표만 제시하기보다, 각 연도별 시장 이벤트와 K-FGI의 노출 조절이 어떤 관계를 보였는지 설명하는 것이 좋다.

다만 주의할 점이 있다. **사후적으로 알려진 이벤트명을 모델 입력 피처로 직접 넣으면 look-ahead bias 또는 data snooping 문제가 생긴다.** 따라서 이벤트는 본문 해석에 사용하고, 모델 개선은 당시에도 관측 가능한 시장/감성/변동성 피처로 구성한 event-stress proxy를 사용한다.

## 2. 연도별 성과와 이벤트 해석

| 연도 | 주요 이벤트/시장 환경 | K-FGI 결과 | 왜 그런 결과가 나왔는가 | K-FGI/피처 연결 |
| --- | --- | --- | --- | --- |
| 2015 | 중국 경기 둔화와 위안화 평가절하, 신흥국 위험회피 확대 | K-FGI 성과 부진 | 분석 초기 구간이라 EGARCH/K-FGI 추정 안정성이 낮고, warm-up 직후 신호가 불안정 | 초기 추정 안정성, 2014 warm-up 필요성 |
| 2016 | 중국/글로벌 불안 이후 회복, Brexit, 미국 대선 이후 금리상승 기대, 국내 정치 불확실성 | Buy&Hold 대비 크게 부진 | 평균 weight 1.01, weight>1 비율 52.0%, crisis 비율 5.3%로 방어보다 공격 노출이 많았음 | 레버리지 cap 필요성, early instability |
| 2017 | 글로벌 경기 회복, 반도체 호황, KOSPI 강세 | K-FGI도 양호하나 노출이 큼 | 평균 weight 1.25, weight>1 비율 67.5%로 추세를 강하게 따라감 | trend_strength와 K-FGI multiplier가 강세장에 잘 반응 |
| 2018 | 미중 무역분쟁과 Fed 긴축, 글로벌 위험자산 약세 | K-FGI가 Buy&Hold보다 MDD를 완화 | 시장 하락기에서 추세/변동성 피처가 노출을 낮춤 | EGARCH, vol_ratio, trend_strength |
| 2019 | 미중 무역협상 불확실성, 반도체 경기 둔화, 일본 수출규제 | K-FGI는 방어적이나 상승장 일부를 놓침 | 추세 신호가 보수적으로 작동해 수익률 포착력은 제한 | trend only와 유사한 보수성 |
| 2020 | COVID-19 급락과 유동성/반도체·자동차 중심 회복 | MDD 방어가 강하게 작동 | 급락 국면에서 변동성 피처가 노출을 줄이고, 회복 국면에서는 추세가 재진입 | EGARCH vol targeting, vol_shock, trend_strength |
| 2021 | 팬데믹 이후 유동성 장세 후반, 인플레이션/금리 우려 부상 | 성과 약함 | 시장 방향성이 뚜렷하지 않고 노출 조절이 수익률을 크게 개선하지 못함 | 하방방어보다는 횡보장 한계 |
| 2022 | 러시아-우크라이나 전쟁, 인플레이션, BOK/Fed 금리인상, 원화 약세 | K-FGI가 하방방어에 유리 | 주식시장 약세와 고변동성에서 노출 축소가 작동 | EGARCH, vol_regime_high, safe-demand/subindex |
| 2023 | 금리 정점 기대, 반도체 저점 통과 기대, AI 테마 초기 | K-FGI는 일부 상승을 놓침 | 시장 반등이 이어질 때 보수적 포지션이 수익률을 제한 | greed/추세 재진입 속도 점검 필요 |
| 2024 | Corporate Value-up 기대와 실망, 반도체/2차전지 변동성, 정치 불확실성 | K-FGI는 MDD 방어는 하나 수익률은 약함 | 이벤트성 기대와 실망이 반복되며 방향성보다 변동성이 컸음 | event-stress 해석 필요 |
| 2025 | AI/반도체 강세, 외국인 매수, OpenAI-Samsung/SK hynix 이슈, KOSPI 강세 | K-FGI도 강한 성과 | 추세 강도와 K-FGI momentum 구조가 강세장을 따라감 | trend_strength, K-FGI multiplier |

## 3. 2016-2017 부진의 직접 진단

파일:

- `paper_outputs/tables/early_period_exposure_diagnostics.csv`
- `paper_outputs/tables/early_drawdown_diagnosis.csv`

| 기간 | 평균 K-FGI | 평균 weight | weight > 1 | weight > 1.5 | bull 비율 | crisis 비율 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2016 | 64.2 | 1.01 | 52.0% | 31.3% | 37.0% | 5.3% |
| 2017 | 62.2 | 1.25 | 67.5% | 34.2% | 30.5% | 2.5% |
| 2016-2017 | 63.2 | 1.13 | 59.7% | 32.7% | 33.7% | 3.9% |
| Full | 57.6 | 0.75 | 35.9% | 14.9% | 23.4% | 7.2% |

해석:

- 2016-2017의 문제는 K-FGI가 항상 위험을 과소평가했다기보다, 초기 구간에서 포지션 상한이 너무 높게 허용된 문제에 가깝다.
- 2016년은 Buy&Hold가 +11.0%, MDD -5.2%였지만 K-FGI current는 -12.9%, MDD -16.3%였다.
- no-leverage cap 1.0x를 적용하면 2016년 MDD는 -7.9%로 완화된다.
- 따라서 2016년 부진은 감성 피처 제거보다 **초기 안정화 및 레버리지 상한**으로 다루는 것이 더 설득력 있다.

## 4. Event-Stress Proxy 실험

이벤트명을 직접 더미로 넣지 않고, 당시 관측 가능한 피처로 event-stress proxy를 만들었다.

\[
EventStress_t =
\text{mean}\left(
rank(vol\_ratio_t),
rank(max(vol\_shock_t,0)),
rank(max(-sent\_composite\_ma10_t,0)),
rank(3-trend\_strength_t),
rank(100-KFGI_t),
rank(100-sub\_index3_t)
\right)
\]

사용한 요소:

- 고변동성: `vol_ratio`
- 변동성 충격: `vol_shock`
- 부정 복합감성: `-sent_composite_ma10`
- 약한 추세: `3 - trend_strength`
- 공포성 K-FGI: `100 - K_FGI`
- 약한 breadth: `100 - sub_index3`

실험 파일:

- `paper_outputs/tables/event_regime_overlay_experiments.csv`
- `paper_outputs/data/event_stress_proxy_timeseries.csv`

### 4.1 결과 요약

| 후보 | 연율수익률 | Sharpe | MDD | 2016 수익률 | 2016 MDD | 해석 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| current baseline | 4.37% | 0.295 | -37.1% | -12.9% | -16.3% | 기준 |
| event_stress_q70_cap1.0 | 4.39% | 0.296 | -36.7% | -13.2% | -16.7% | 거의 개선 없음 |
| no_leverage_cap_1.0 | 4.85% | 0.413 | -27.8% | -2.7% | -7.9% | 가장 실무적으로 안정 |
| early_2016_cap0.5 | 6.42% | 0.453 | -37.1% | +0.7% | -3.6% | 초기구간 개선은 크지만 사후 기간 제약처럼 보일 수 있음 |

해석:

- 단순 event-stress overlay는 전체 성과 개선이 크지 않다.
- 2016 문제는 이벤트 스트레스 인식 부족보다 레버리지/초기 안정화 문제에 가깝다.
- 논문 본문에서는 event-stress proxy를 “추가 실험”으로 제시하고, 최종 개선 방향은 no-leverage cap 또는 warm-up cap으로 제시하는 것이 더 안전하다.

## 5. 최종 피처/모델 선택 권장

### 5.1 K-FGI 피처 구성

현재 결과상 가장 설득력 있는 방향:

```text
sub_index2-7
+ egarch_vol, vol_regime_high, vol_ratio
+ sent_composite_ma10
```

이 조합은 `lean_sent_composite_ma10_only`로 실험했을 때 baseline보다 Sharpe와 MDD가 소폭 개선되었다.

| 모델 | Sharpe | MDD | 누적수익률 |
| --- | ---: | ---: | ---: |
| baseline_full | 0.393 | -30.8% | 61.6% |
| lean_sent_composite_ma10_only | 0.400 | -29.6% | 63.1% |
| market_egarch_only | 0.330 | -36.6% | 52.7% |
| drop_sent_composite_ma10 | 0.295 | -36.1% | 42.0% |

### 5.2 포지션 제약

2016-2017 문제를 줄이려면:

```text
max leverage cap = 1.0x
```

또는 본문에서는 현 모델을 유지하고, robustness/future work로 no-leverage cap을 제시한다.

## 6. 논문에 넣을 수 있는 문장

> 연도별 성과를 주요 시장 이벤트와 비교한 결과, K-FGI는 2018년 미중 무역분쟁, 2020년 COVID-19 급락, 2022년 금리인상 약세장과 같이 변동성 확대와 시장 하락이 동반된 구간에서 상대적으로 강한 하방방어 효과를 보였다. 반면 2016년에는 K-FGI 평균과 추세 신호가 높게 유지되면서 1배 초과 노출이 빈번하게 발생해 Buy & Hold보다 큰 낙폭을 기록하였다. 이는 K-FGI의 한계가 감성 피처 자체보다는 초기 추정 구간에서의 과도한 레버리지 허용과 관련됨을 시사한다.

> 추가적으로 이벤트명을 직접 더미 변수로 사용하지 않고, 변동성 충격, 약한 breadth, 부정 복합감성, 약한 추세를 결합한 event-stress proxy를 구성해 실험하였다. 그러나 단순 event-stress overlay는 성과 개선이 제한적이었고, no-leverage cap이 2016년 낙폭 완화에 더 효과적이었다. 따라서 본 연구는 사후 이벤트 더미를 도입하기보다, K-FGI의 실무 적용 시 레버리지 상한과 충분한 warm-up 구간을 함께 고려해야 함을 제안한다.

## 7. 참고 출처

- Federal Reserve FEDS Notes: China slowdown/RMB devaluation, U.S. election, U.S.-China trade tension episodes. https://www.federalreserve.gov/econres/notes/feds-notes/u-s-interest-rates-and-emerging-market-currencies-taking-stock-10-years-after-the-taper-tantrum-20231004.html
- Federal Reserve Bank of Boston: financial market implications of the U.S.-China trade war. https://www.bostonfed.org/publications/current-policy-perspectives/2019/financial-market-implications-of-the-trade-war-between-the-united-states-and-china.aspx
- Bank of Korea monetary policy decision, July 13, 2022. https://www.bok.or.kr/eng/bbs/E0000627/view.do?menuNo=400022&nttId=10071643
- Yonhap News, Dec. 3, 2020: KOSPI record high led by chip and auto heavyweights. https://en.yna.co.kr/view/AEN20201203009151320
- Korea Times, Feb. 26, 2024: KOSPI disappointment over Corporate Value-up Program. https://www.koreatimes.co.kr/business/banking-finance/20240226/kospi-dips-on-disappointment-over-steps-to-boost-undervalued-stocks/
- Asia Business Daily, Sep. 30, 2024: Value-Up Index first-day underperformance. https://www.asiae.co.kr/en/print.htm?idxno=2024093018415179187
- Yonhap News, Oct. 2, 2025: KOSPI passes 3,500 on chip rally and OpenAI partnership news. https://en.yna.co.kr/view/AEN20251002010100320
- Yonhap News, Oct. 27, 2025: KOSPI 2025 fastest among G20 indices, semiconductor and foreign buying. https://en.yna.co.kr/view/AEN20251027006952320
