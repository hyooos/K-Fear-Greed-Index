# Early Drawdown Diagnosis

## 핵심 결론

2016년에는 현재 K-FGI가 Buy&Hold보다 분명히 나빴다. 이는 숨기면 안 되는 결과다.

- 2016 Buy&Hold: total return 11.0%, MDD -5.2%
- 2016 current K-FGI: total return -12.9%, MDD -16.3%
- 2016 no-leverage cap 1.0x K-FGI: total return -2.7%, MDD -7.9%

## 왜 이런 일이 생겼나

2016년은 K-FGI 평균이 높고 bull/normal 판정이 많아 시장 노출이 크게 잡힌 구간이다. 하지만 실제로는 2016년 초 낙폭이 있었고, K-FGI가 이 구간에서 방어 신호를 충분히 빨리 내지 못했다. 즉 K-FGI는 모든 구간에서 방어하는 만능 지표가 아니라, 특정 초기/국면 전환 구간에서는 과노출될 수 있다.

## 모델을 수정해야 하나?

수정한다면 가장 방어 가능한 방법은 **max exposure를 1.0x로 제한한 no-leverage K-FGI**를 main 또는 robustness로 제시하는 것이다. 이는 2016만 보고 만든 규칙이 아니라, 논문/실무적으로 자연스러운 제약이다.

- Full current K-FGI: total return 54.4%, MDD -37.1%, Sharpe 0.295
- Full no-leverage cap 1.0x K-FGI: total return 62.0%, MDD -27.8%, Sharpe 0.413

## 논문 서술 권장

K-FGI는 2016년과 같은 강세 전환/초기 추정 구간에서는 Buy&Hold보다 큰 손실을 낼 수 있었다. 따라서 본 연구는 K-FGI를 초과수익 창출 신호라기보다, 거래비용과 레버리지 제약을 고려한 위험관리형 지표로 해석한다. 추가적으로 no-leverage cap을 적용하면 초기 낙폭 문제가 완화되어 실무 적용 가능성이 높아진다.
