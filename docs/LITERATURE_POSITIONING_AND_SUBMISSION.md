# K-FGI Literature Positioning and Submission Strategy

## 1. DOCX 참고문헌에서 확인한 연구 흐름

첨부된 `기존 참고문헌 : 추가 선행연구.docx`의 참고문헌은 크게 네 축으로 정리된다.

### A. 전통적 시장 지표 기반 투자심리 연구

- Shiller (2000)
- Smales (2017)
- Ryu (2012)
- 이정환 (2024)
- Baker and Wurgler (2006)
- Bouteska et al. (2023)

이 축은 투자심리가 직접 관측되지 않기 때문에 시장 기반 proxy로 심리를 측정하는 흐름이다. Baker and Wurgler류의 composite sentiment index, VIX/fear index, 시장 폭, 거래량, IPO, 채권 스프레드 등이 여기에 해당한다.

### B. NLP 기반 금융 감성 분석

- Tetlock (2007)
- Bollen et al. (2011)
- Garcia (2013)
- Araci (2019)
- 박윤신, 김현석 (2024)
- Jung et al. (2024)
- 최보미 외 (2024)
- 네이버 댓글과 주가 관계 연구

이 축은 뉴스, SNS, 댓글, 포럼 텍스트에서 투자심리를 추출하고, 이를 수익률·변동성·거래량·가격 형성과 연결하는 연구다. K-FGI의 NAVER 금융 댓글 기반 감성 점수는 이 흐름에 들어간다.

### C. 변동성 모델링 및 위험 관리

- Nelson (1991)
- 김겨레, 한희준 (2024)
- Harvey et al. (2018)
- Han, Hutan and Ryu (2015)

이 축은 EGARCH/GARCH, 변동성 타깃팅, 위험 국면 식별, 포지션 조절 전략과 연결된다. K-FGI는 단순 감성 지표가 아니라 EGARCH 변동성을 결합해 시장 노출을 조절한다는 점에서 이 축과 직접 연결된다.

### D. 머신러닝 기반 복합 지수 및 투자전략

- Fischer and Krauss (2018)
- Krauss et al. (2017)
- López de Prado (2018)
- 최인실 (2024)
- 권병재 (2023)
- Gómez-Martínez et al. (2025)
- KOSPI 수익률 예측 연구 (2025)

이 축은 머신러닝 기반 예측, 피처 중요도, walk-forward 검증, 투자전략 성과 비교와 연결된다. K-FGI의 LightGBM 피처 중요도, ablation, OOS 검증은 이 흐름에 해당한다.

## 2. K-FGI의 위치

K-FGI는 하나의 문헌 흐름에만 속하지 않고 다음 네 영역의 교차점에 위치한다.

```text
시장 기반 투자심리 지표
        +
NLP 기반 댓글 감성 지표
        +
EGARCH 변동성 기반 위험 관리
        +
머신러닝 기반 전략 검증
```

가장 정확한 포지셔닝은 다음과 같다.

> K-FGI는 CNN Fear & Greed Index와 같은 복합 시장심리 지표를 한국시장에 맞게 재구성하되, NAVER 금융 댓글 기반 감성 피처와 EGARCH 조건부 변동성을 결합하여 KOSPI200 시장 노출을 조절하는 하방 위험 관리 지표다.

따라서 논문 제목/초록에서는 “수익률 예측 모델”보다 아래 표현이 더 안전하다.

- 한국형 투자심리 지표
- 감성-변동성 결합 지표
- 하방 위험 관리 전략
- KOSPI200 노출 조절 전략
- 댓글 기반 투자심리와 시장 기반 fear/greed proxy의 결합

## 3. 기존 지표와의 차별점

| 비교 대상 | 기존 연구/지표 | K-FGI의 차별점 |
|---|---|---|
| CNN Fear & Greed Index | 7개 시장지표를 동일가중으로 결합 | 한국 KRX/KOSPI200 데이터로 재현하고 댓글 감성 및 EGARCH 변동성 추가 |
| Baker-Wurgler sentiment index | 시장 proxy 기반 월별/저빈도 심리지수 | 일별 KOSPI200 전략에 적용 가능한 고빈도 지표 |
| VIX / VKOSPI | 옵션 내재변동성 기반 fear proxy | 변동성 하나가 아니라 시장폭, 옵션, 안전자산, 채권, 댓글 감성까지 결합 |
| BOK News Sentiment Index | 뉴스 기반 경제심리 지수 | 경제 전반 뉴스가 아니라 NAVER 금융 댓글 기반 시장참여자 감성 |
| AAII Sentiment Survey | 개인투자자 설문 기반 주간 심리 | 설문이 아니라 댓글과 시장 데이터를 자동 수집해 일별 지표화 |
| 텍스트 감성 예측 연구 | 뉴스/SNS 감성으로 수익률·변동성 예측 | 감성을 단독 예측 변수로 쓰지 않고 K-FGI와 포지션 조절에 결합 |
| EGARCH/vol targeting 전략 | 변동성만으로 노출 조절 | 심리 국면과 변동성 국면을 함께 고려 |

## 4. 추가로 확인한 유사 연구 및 지표

### 4.1 CNN Fear & Greed Index

CNN Fear & Greed Index는 market momentum, stock price strength, stock price breadth, put/call options, junk bond demand, market volatility, safe haven demand 등 7개 지표를 0~100 점수로 결합한다. K-FGI의 subindex 구조는 이 지표와 가장 직접적으로 연결된다.

K-FGI의 기여는 이를 한국시장 데이터로 재구성하고, 댓글 감성과 EGARCH 변동성 기반 전략 검증까지 확장했다는 점이다.

Source: https://edition-prod-cf.sitemirror.cnn.com/markets/fear-and-greed

### 4.2 한국은행 News Sentiment Index

한국은행의 Machine-Learning-Based News Sentiment Index (NSI)는 인터넷 뉴스 텍스트를 transformer 기반 감성 분류기로 일별 경제심리 지수화한다. K-FGI와 가장 가까운 국내 공식/준공식 선행지표 중 하나다.

차이는 NSI가 경제 전반 뉴스 심리에 초점을 두는 반면, K-FGI는 금융 댓글과 KRX 시장 지표를 결합해 투자전략 및 위험관리에 초점을 둔다는 점이다.

Source: https://www.bok.or.kr/imerEng/bbs/E0002902/view.do?menuNo=600342&nttId=10072851

### 4.3 국내 머신러닝 기반 주식시장 감성지수

`Sentiment Matters in Stock Market: Construction of Sentiment Index Using Machine Learning`은 BERT 기반 뉴스 감성으로 stock market sentiment index를 구축한다. 이 연구는 K-FGI의 국내 NLP 기반 감성지수 선행연구로 넣기 좋다.

K-FGI는 뉴스가 아니라 댓글을 사용하고, 감성을 독립 지표로 끝내지 않고 시장 기반 subindex 및 EGARCH와 결합한다.

Source: https://www.kci.go.kr/kciportal/landing/article.kci?arti_id=ART003162688

### 4.4 한국 뉴스 감성과 자산가격

Pyo and Kim의 `News media sentiment and asset prices in Korea`는 뉴스 기사 감성지수와 한국 금융시장 자산가격의 관계를 분석한다. K-FGI의 “한국시장 텍스트 감성-자산가격 연결” 근거로 활용 가능하다.

Source: https://www.tandfonline.com/doi/abs/10.1080/16081625.2019.1642115

### 4.5 중국 투자자 포럼 기반 시장심리

`Measuring China's Stock Market Sentiment`는 중국 온라인 투자자 포럼 메시지 약 6천만 건을 사용해 텍스트 감성지수와 disagreement index를 구축한다. NAVER 금융 댓글을 사용하는 K-FGI와 데이터 환경이 유사하다.

Source: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3377684

### 4.6 투자심리와 변동성 예측 연구

`Investor sentiment and stock volatility: New evidence`는 partial least squares 기반 투자심리 지수가 주식시장 실현변동성 예측에 유용하다고 제시한다. K-FGI의 EGARCH/변동성 결합 논리와 연결된다.

Source: https://www.sciencedirect.com/science/article/pii/S1057521922000084

### 4.7 신흥시장 투자심리와 변동성

인도 및 말레이시아 시장 연구들은 시장 proxy 기반 sentiment index가 신흥시장 변동성 예측과 관련 있음을 보인다. 한국시장 역시 신흥/아시아 시장 맥락에서 유사하게 포지셔닝할 수 있다.

Sources:

- https://www.sciencedirect.com/science/article/pii/S2214635015000593
- https://doi.org/10.22452/AJAP.vol9no1.2

### 4.8 최신 한국 뉴스 감성 및 상태의존성 연구

2026년 SSRN의 `When Sentiment Matters`는 한국 주식시장에서 뉴스 감성 효과가 극단 가격 움직임 이후 상태의존적으로 나타난다고 보고한다. K-FGI의 “시장 국면별 노출 조절” 논리를 강화하는 최신 참고문헌으로 활용 가능하다.

Source: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6516494

## 5. 논문에서 주장하면 좋은 기여점

강하게 주장해도 되는 부분:

1. CNN Fear & Greed형 복합지표를 한국시장 데이터로 재현 및 확장했다.
2. NAVER 금융 댓글 감성을 시장 기반 subindex와 결합했다.
3. EGARCH 조건부 변동성을 결합해 단순 지표가 아니라 포지션 조절 전략으로 검증했다.
4. 10개년 일별 데이터로 KOSPI200 하방 위험 관리 효과를 검증했다.
5. LightGBM feature importance, ablation, sensitivity analysis로 지표 구성의 설명 가능성을 보강했다.

조심해야 하는 부분:

1. “수익률 예측력이 통계적으로 강하다”는 식으로 쓰면 위험하다.
2. 전체 초과수익 t-test는 유의하지 않기 때문에, 핵심 주장은 “하방 위험 방어”에 두는 것이 안전하다.
3. 감성 피처는 총수익률 개선에는 기여하지만 Sharpe에서는 감성 제외 버전이 더 높기 때문에 “감성이 무조건 우월하다”고 쓰면 안 된다.

추천 문장:

> 본 연구의 K-FGI는 수익률을 직접 예측하기 위한 단일 신호라기보다, 투자심리와 조건부 변동성을 결합하여 시장 노출을 조절하는 위험 관리형 심리 지표로 해석하는 것이 적절하다.

## 6. 투고처 후보

### 6.1 국내 1순위: 금융공학연구

적합도: 높음

이유:

- 금융공학, 머신러닝, 투자전략, 리스크 관리와 잘 맞는다.
- KCI 등재지이고 한국어/영어 모두 가능하다.
- 현재 결과가 “국제 최고 finance journal”보다는 국내 금융공학/응용금융 쪽에 더 잘 맞는다.

Source: https://www.kci.go.kr/kciportal/po/search/poCitaView.kci?sereId=001923

추천 포지셔닝:

> NAVER 금융 댓글 감성과 EGARCH 변동성을 활용한 한국형 Fear & Greed Index 구축 및 KOSPI200 하방 위험 관리 전략

### 6.2 국내 1.5순위: Journal of Derivatives and Quantitative Studies (JDQS)

적합도: 높음

이유:

- JDQS는 derivatives, quantitative finance, risk management, portfolio management, performance measurement 등을 다룬다.
- K-FGI가 옵션/변동성/VKOSPI/put-call/EGARCH를 포함하고 있어 정량금융·위험관리 논문으로 포지셔닝 가능하다.
- Emerald에서 발행되며 open access이고 저자 비용이 없다는 장점이 있다.

Source: https://www.emeraldgrouppublishing.com/journal/jdqs

주의:

- 영어 논문 형태가 더 적합하다.
- 현재 결과를 더 정교하게 만들고 robustness를 보강하면 가능성이 올라간다.

### 6.3 국내 2순위: Korean Journal of Financial Studies / Asia-Pacific Journal of Financial Studies

적합도: 중간~높음

이유:

- Korean Journal of Financial Studies는 한국 금융시장 및 투자, 금융시장, 계량 방법론 논문을 다룬다.
- Asia-Pacific Journal of Financial Studies는 capital markets, investments, quantitative methods를 다루며 SSCI 등재 finance journal이다.

Sources:

- https://submit.e-kjfs.org/about/Author.php
- https://onlinelibrary.wiley.com/page/journal/20416156/homepage/product_detail.htm

주의:

- 이쪽은 기여도와 계량 검증 요구가 더 높다.
- 현재 결과만으로는 “하방 위험 관리” 주장에 대한 강건성 검정, 거래비용 민감도, OOS/rolling robustness, benchmark 확장이 더 필요하다.

### 6.4 국내 2순위: 계량경제학보

적합도: 중간

이유:

- 최근 KCI에 머신러닝 기반 주식시장 감성지수 논문이 게재된 흐름이 있다.
- 감성지수 구축 자체를 강조하면 가능하다.

주의:

- 투자전략/대시보드보다 감성지수 구성 방법론, 계량 검정, 예측력 검증을 더 강화해야 한다.

### 6.5 해외 후보: Journal of Behavioral and Experimental Finance

적합도: 중간

이유:

- behavioral finance, sentiment, asset pricing을 다루는 저널이다.
- 투자심리와 시장 행동을 다루는 K-FGI 주제와 맞다.

Source: https://www.sciencedirect.com/journal/journal-of-behavioral-and-experimental-finance

주의:

- 영어 논문화가 필요하다.
- 현재는 통계적 핵심 결과가 “수익 예측”보다 “하방 방어”이므로, behavioral finance contribution을 더 분명히 해야 한다.

### 6.6 해외 후보: Finance Research Letters

적합도: 중간

이유:

- 짧고 시의성 있는 finance letter에 적합하다.
- “Korean retail comment sentiment + Fear & Greed index + downside risk”로 간결한 contribution을 만들 수 있다.

주의:

- 매우 압축된 논리와 강한 novelty가 필요하다.
- 국내 데이터 특화 연구라면 국제 독자가 왜 관심 가져야 하는지를 “retail-investor-dominated Korean market”으로 설득해야 한다.

### 6.7 해외 후보: International Review of Finance

적합도: 중간

이유:

- Asia-Pacific/emerging market, financial econometrics, risk management, quantitative finance를 다룬다.

Source: https://onlinelibrary.wiley.com/page/journal/14682443/homepage/productinformation.html

주의:

- 현재 프로젝트는 학부/응용 연구 성격이 강하므로, 더 엄밀한 식별 전략과 robustness가 있어야 도전 가능하다.

## 7. 현실적 추천 순서

현재 완성도 기준 추천:

1. 금융공학연구
2. JDQS
3. Korean Journal of Financial Studies
4. 계량경제학보
5. Journal of Behavioral and Experimental Finance
6. Finance Research Letters
7. International Review of Finance

가장 현실적인 투고 전략:

> 1차 목표는 국내 KCI 등재 금융공학/정량금융 저널로 잡고, 논문을 영어로 확장하거나 robustness를 크게 보강한 뒤 JDQS 또는 해외 behavioral finance/finance letters 계열로 확장한다.

## 8. 보강하면 투고 가능성이 올라가는 분석

우선순위 높은 보강:

1. 거래비용 0bp, 5bp, 10bp, 20bp 민감도 표.
2. KOSPI200 Buy&Hold, Trend, Trend+Vol targeting, CNN-style equal weight K-FGI와 비교.
3. 2020 코로나, 2022 금리 인상기, 2025 강세장 등 crisis/event subperiod 분석.
4. K-FGI threshold 25/65 외 20/80, 30/70 robustness.
5. 감성 피처 제외/포함 외에 댓글 수, 부정감성, 감성분산 각각의 ablation.
6. 단순 return prediction보다 downside return, drawdown, volatility forecast 중심의 검정.
7. White reality check 또는 bootstrap 기반 전략 성과 유의성.

논문 주장을 더 안전하게 만드는 문장:

> K-FGI의 예측력은 평균 수익률 차원에서 일관되게 강하게 나타나지는 않지만, 하락 구간의 손실 축소와 변동성 국면에서의 노출 조절 측면에서 경제적 유용성을 보인다.
