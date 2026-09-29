# -*- coding: utf-8 -*-
"""filter_params_{ko,en}.ipynb 생성 — filter_param_lib 의 그림 + 그래프별 설명."""
import nbformat as nbf
from pathlib import Path
import filter_param_lib as L

BASE = Path(__file__).resolve().parent

TXT = {
 "ko": {
  "title": "# 필터링 파라미터 검증 — 왜 이 값일 수밖에 없었는가\n\n"
   "댓글 필터링 파이프라인의 9개 임계값·하이퍼파라미터를 실제 데이터로 스윕한다. "
   "각 절의 목표는 그 값이 **성능 튜닝의 결과가 아니라 데이터 특성상 그 근방일 수밖에 없다**는 것을 보이는 것이다.\n\n"
   "**데이터** — P절(파이프라인 개요)은 최종 논문 표본 **2014-2025 전체**(원천 215만 건, 논문 3.3 표와 동일)를 쓴다. "
   "A~G 파라미터 스윕은 **2014-2025 전체 표본**(독성 필터 입력 141만 건, archive recollect 중간 산출물)에서 수행한다. "
   "F·G는 실제 실행에서 산출된 stock_score·is_stock 컬럼을, D·E는 연도별 4만 건 층화표본을 쓴다. "
   "전략 성과(γ·감성 MA, H·I)는 10년 K-FGI 파이프라인 전 구간 재산출이다.\n\n"
   "**한계** — 감성 확률은 필터 통과 댓글에만 존재하므로, 일별 감성 안정성은 'τ를 더 낮추는(엄격하게) 방향'만 "
   "시뮬레이션한다. 'τ를 높이는(느슨하게) 방향'의 영향은 B-3(가중치 총량 비중)으로 판단한다.",
  "setup": "## 0. 준비 및 데이터 로드\n최초 실행 시 `compute()` 가 약 3~4분 소요되며 `_compute_cache.pkl` 에 캐시된다. 이후 실행은 즉시.",
  "secP": "## P. 필터링 파이프라인 개요  (최종 논문 표본 2014-2025)\n\n"
   "감성 분석에 쓰이는 댓글은 **원천 수집 → 빈 댓글·중복 제거 → 1단계 정치 필터링 → 2단계 독성 필터링 → 3단계 경제 필터링**을 "
   "거친다. 아래 표·그래프는 논문 3.3의 최종 필터링 표(2014-2025 전체)와 동일한 수치이며, "
   "`10_Paper_outputs/04_투고_전략/FINAL_REVISION_MASTER_KR.md` 의 연도별 상세표에서 재산출한 것이다.\n\n"
   "- 1단계 정치 필터링은 정치 키워드(인물·정당·사건명) 정규식 매칭으로 정치 프레임 댓글을 제거한다. "
   "감성 모델이 '재앙·내란' 같은 표현을 시장 부정 감성으로 오해석하는 것을 막는 것이 목적이다.\n"
   "- 제거량이 가장 큰 단계는 3단계 경제 필터링으로, 정치·독성이 아니어도 주식·경제와 무관한 일반 댓글(독성 통과분의 84.9%)을 걸러낸다.\n"
   "- **연도별 원천 댓글 수 차이(P-6)**: ① 수집 기사 수가 2014년 3,651개 → 2018년 이후 ~7,200개(매일 섹션별 상위 10개 기사)로 안정, "
   "② 기사당 평균 수집 댓글 수가 2016년 8개 → 2021년 31개로 증가 — 수집 상한이 아니라 금융 기사당 실제 댓글 증가(중앙값 2→23개, 동학개미·공매도 논란). "
   "크롤러는 기사당 당일 댓글을 전량 수집(네이버 API 한도 ~200개). 2018년(+124%)과 2021년(피크 31만)이 대표적 증가 구간.\n"
   "- **연도별 제거율 차이(P-7)**: ① 빈(삭제) 댓글 비율이 2017-2019년 25~37%로 급등(2018 드루킹 사건 후 네이버 댓글 개편), "
   "② 2019년은 경제 필터 핵심 키워드가 2개뿐이라 최종 잔존율 2.4%로 최저 — DLF 사태·조국 사모펀드·일본 수출규제로 댓글이 비주식 이슈에 쏠려 주식 단어가 DF 0.5% 문턱을 못 넘음.\n\n"
   "> 참고 — A~G 파라미터 스윕은 2014-2025 전체 표본에서 수행한다. F·G는 실제 stock_score·is_stock 컬럼, "
   "D·E는 연도별 4만 건 층화표본이다. 결론(τ·score 임계값의 평탄 구간, γ 무영향)은 표본 크기·연도 범위에 둔감하다. "
   "전략 성과(H·I)는 10년 K-FGI 파이프라인 전 구간 재산출이다.",
  "secA": "## A. 독성 가중치 지수 γ  (현재값 = 2)\n\n"
   "$w_i = (1 - \\text{tox}_i)^{\\gamma}$ — 독성이 높은 댓글일수록 감성 집계 기여도를 낮춘다.\n\n"
   "**왜 2인가**\n"
   "- 전략 성과가 γ에 무감각하다(H-1·H-2). γ 1·2·3에서 Sharpe 0.61–0.62, MDD −22.9% 고정 → 성과로 고른 값이 될 수 없다.\n"
   "- 2는 감쇠 강도의 원칙적 중간값: γ=1은 경계 댓글(tox 0.5)을 0.5로만 낮추고, γ=3은 0.125로 과하게 죽인다. γ=2 → 0.25.\n"
   "- 정보 손실 억제: 유효표본 비율이 γ=2에서 0.85, γ=3에서 0.84 (A-3).",
  "secB": "## B. 독성 hard-drop 임계값 τ  (현재값 = 0.95)\n\n"
   "`keep = tox < τ` — 임계값 이상인 댓글은 완전히 버린다.\n\n"
   "**왜 0.95인가**\n"
   "- 지표가 τ에 구조적으로 불변. γ=2 가중이 이미 고독성 댓글의 발언권을 눌러놨다. tox ≥ 0.95 댓글의 가중치 총합은 "
   "전체의 **0.003%** 뿐이다(B-3). τ를 0.90까지 낮춰도 MAD 0.0002 미만, 0.80까지 낮춰도 0.0004 미만(B-5).\n"
   "- 제거 기준으로서 0.95는 보수적 상한: 모델이 95% 이상 확신하는 것만 버린다(이미 9.3% 제거, B-1).\n"
   "- 더 올리면 명백한 독성이 통과: tox 0.95–0.99 구간은 정치·욕설 표현이 대부분이며 감성 신호가 아니다(B-2).",
  "secC": "## C. 키워드 추출 길이  (현재값 = 2~6글자)\n\n"
   "이 값은 스윕이 아니라 **규칙의 근거**를 제시한다.\n\n"
   "**왜 2~6인가**\n"
   "- 하한 2: 1글자 한글 토큰 중 금융 관련은 '장' 하나뿐. 하한을 3으로 올리면 '기관·외인·환율·수급·시총·국장' 등 2글자 핵심어를 잃는다(C-2).\n"
   "- 상한 6: 7글자 이상은 조사가 붙은 어절이며 짧은 핵심어의 부분문자열로 이미 포착된다. 상한 6 vs 8의 핵심어 집합은 동일(C-2).",
  "secD": "## D. 핵심 키워드 DF 임계값  (실제 실행값 = DF 0.5%)\n\n"
   "**왜 0.5% 부근인가**\n"
   "- DF를 더 내리면 '가장·당장' 같은 일반어가 섞이고, 1% 이상으로 올리면 연도에 따라 핵심 키워드가 0~2개로 줄어 잔존율이 급감한다(D-2, 2019년 참조).\n"
   "- 실제 실행(`3_final_filter.py`)은 `min_doc_ratio=0.005`(DF 0.5%)로 돌았다. F·G의 통과율(15.1%)은 이 실행에서 산출된 stock_score·is_stock 컬럼을 그대로 사용한 것이다.",
  "secF": "## F · G · E. 경제성 점수 임계값 · 핵심≥1 조건 · 10:3 가중치\n\n"
   "점수 = 10·(핵심 키워드 수) + 3·(보조 키워드 수),  통과 = 점수 ≥ 10 **그리고** 핵심 ≥ 1개.\n\n"
   "**왜 이렇게인가**\n"
   "- **score ≥ 10** = 통과율 평탄 구간의 끝. cutoff 3·5·7·10 에서 통과율 15.1% 로 완전히 같고, 13 에서 10.6% 로 급락한다(F-1). "
   "급락은 '핵심어를 정확히 하나 가진 댓글(점수 10)'이 잘리며 생긴다.\n"
   "- **10 : 3** — 보조 점수를 2·5·10 무엇으로 바꿔도 통과율이 동일하다. 핵심 1개=10점이 이미 게이트를 넘기 때문. 보조 점수는 순위용.\n"
   "- **핵심 ≥ 1개** — 이 조건이 없으면 '보조만 4개 이상·핵심 0'인 댓글 약 {aux_only:,}건(통과분의 {aux_pct:.1f}%)이 통과한다.",
  "secHI": "## 전략 성과 (10년 K-FGI 파이프라인 재산출)\n\n"
   "γ 와 감성 이동평균 기간을 바꿔 K-FGI → 포지션 → 전략수익률을 전 구간 재산출한 결과. "
   "코드는 `downstream_strategy_sensitivity.py`, 수치는 `downstream_sensitivity.csv`.\n\n"
   "**핵심** — γ 는 전략 성과에 무영향(H-1·H-2), 감성 이동평균 10일은 명확한 국소 최적(I-1). "
   "즉 τ·γ 는 '근방이면 무엇이든 무방'이고, 감성 MA 는 '10일이어야 하는' 값이다.",
  "concl": "## 결론 — 고정 관행값 vs 우리가 선택한 값\n\n"
   "| 값 | 성격 | 판정 |\n|---|---|---|\n"
   "| τ = 0.95 | 우리 선택 | ✓ 지표·전략이 τ에 구조적 불변 (tox≥0.95 가중치 비중 0.003%) |\n"
   "| γ = 2 | 우리 선택 | ✓ 성과 무영향, 감쇠 강도의 원칙적 중간값 |\n"
   "| score ≥ 10 | 우리 선택 | ✓ 통과율 평탄 구간(3–10)의 끝 |\n"
   "| DF ≥ 0.5% | 우리 선택 | ◐ 일반어 배제; 1%+ 는 연도별 핵심어 0~2개로 급감 |\n"
   "| 핵심 10 : 보조 3 | 우리 선택 | ◐ 보조 점수는 게이트에 무관, 순위용 |\n"
   "| 핵심 ≥ 1개 | 우리 선택 | ✓ 보조 키워드만으로 통과하는 소수 댓글 차단 |\n"
   "| 2 ~ 6 글자 | 방법론 규칙 | ✓ 2글자 금융어 보존 / 긴 어절은 부분문자열로 포착 |\n"
   "| 감성 10일 MA | 우리 선택 | ✓ Sharpe·MDD 모두에서 국소 최적 (근거 필요한 값) |\n"
   "| 252일·60일·70% 분위수 | 금융 관행값 | 튜닝 대상 아님 |",
  "cur": "현재값", "run": "그림 생성",
 },
 "en": {
  "title": "# Filtering Parameter Validation — why these values, not others\n\n"
   "The nine thresholds and hyper-parameters in the comment-filtering pipeline, swept on the real data. "
   "Each section aims to show the value is **forced by the data, not the product of performance tuning**.\n\n"
   "**Data** — Section P (pipeline overview) uses the **full 2014-2025 final sample** (2.15M raw comments, same figures "
   "as the paper's §3.3 table). The A–G parameter sweeps use the **full 2014-2025 sample** (1.41M comments entering the toxicity filter; "
   "F/G from the real stock_score & is_stock columns, D/E from a 40k-per-year stratified sample). "
   "Strategy metrics (γ, sentiment MA — H, I) are re-derived over the full 10-year K-FGI pipeline.\n\n"
   "**Limitation** — sentiment probabilities exist only for kept comments, so daily-sentiment stability is simulated "
   "only for the 'lower τ (stricter)' direction; for the 'raise τ (looser)' direction see B-3 (weight mass).",
  "setup": "## 0. Setup and data load\nThe first `compute()` takes ~3–4 min and is cached to `_compute_cache.pkl`; later runs are instant.",
  "secP": "## P. Filtering pipeline overview  (final 2014-2025 sample)\n\n"
   "Comments used for sentiment analysis pass: **raw crawl → empty/duplicate removal → stage 1 political → stage 2 "
   "toxicity → stage 3 economic**. The table and charts below use the same figures as the paper's §3.3 filtering table "
   "(full 2014-2025), re-derived from the per-year detail table in `10_Paper_outputs/04_투고_전략/FINAL_REVISION_MASTER_KR.md`.\n\n"
   "- Stage 1 political filtering removes politically framed comments by regex-matching political keywords "
   "(politician / party / event names), to stop the sentiment model from reading expressions like '재앙' (catastrophe) "
   "or '내란' (insurrection) as market-negative sentiment.\n"
   "- The largest drop is stage 3 economic filtering (84.9% of the toxicity-passed set), removing ordinary comments unrelated to stocks/economy.\n"
   "- **Why raw counts differ by year (P-6)**: (i) collected articles grew from 3,651 (2014) to ~7,200 from 2018 "
   "(top-10 articles per section per day), and (ii) mean non-empty comments collected per article rose from 8 (2016) to "
   "31 (2021) — not a collection cap but genuine growth in comments per financial article (median 2→23) with the "
   "retail-investing boom. The crawler collects all same-day comments per article (Naver API limit ~200). "
   "2018 (+124%) and 2021 (peak 311k) are the main growth years.\n"
   "- **Why removal rates differ by year (P-7)**: (i) the empty (deleted) comment rate jumped to 25–37% in 2017–2019 "
   "after Naver's 2018 comment overhaul; (ii) 2019 has only 2 economic core keywords, giving the lowest final retention "
   "(2.4%) — the DLF scandal, Cho Kuk affair and Japan export controls pushed comments toward non-stock topics.\n\n"
   "> Note — the A–G parameter sweeps run on the full 2014-2025 sample (1.41M comments); F/G use the real per-comment "
   "directly loadable. Same filter scripts and toxicity model; the conclusions (plateau ranges for τ and score, γ having "
   "no effect) are insensitive to the year range. Strategy metrics (H, I) are re-derived over the full 10-year K-FGI pipeline.",
  "secA": "## A. Toxicity weighting exponent γ  (current = 2)\n\n"
   "$w_i = (1 - \\text{tox}_i)^{\\gamma}$ — a more toxic comment contributes less to the sentiment aggregation.\n\n"
   "**Why 2**\n"
   "- Strategy performance is insensitive to γ (H-1, H-2): Sharpe 0.61–0.62, MDD fixed at −22.9% for γ 1/2/3 — it cannot have been chosen for performance.\n"
   "- 2 is the principled midpoint of decay strength: γ=1 only halves a borderline comment (tox 0.5 → 0.5), γ=3 over-kills it (0.125). γ=2 → 0.25.\n"
   "- Information loss is contained: effective-sample fraction 0.85 at γ=2 vs 0.84 at γ=3 (A-3).",
  "secB": "## B. Toxicity hard-drop threshold τ  (current = 0.95)\n\n"
   "`keep = tox < τ` — comments at or above the threshold are dropped entirely.\n\n"
   "**Why 0.95**\n"
   "- The index is structurally invariant to τ. The γ=2 weight already crushes toxic comments: those with tox ≥ 0.95 hold only **0.004%** of total sentiment weight (B-3). Lowering τ to 0.90 changes the daily index by MAD < 0.0002, to 0.80 by < 0.0004 (B-5).\n"
   "- As a removal criterion, 0.95 is a conservative ceiling — it discards only what the model is ≥ 95% sure of (already 8.6% removed, B-1).\n"
   "- Raising it lets clear toxicity through: the tox 0.95–0.99 band is mostly political invective, not sentiment signal (B-2).",
  "secC": "## C. Keyword extraction length  (current = 2–6 characters)\n\n"
   "This is a **rule justification**, not a sweep.\n\n"
   "**Why 2–6**\n"
   "- Lower bound 2: the only finance-relevant single-character token is '장'. Raising it to 3 loses 2-char core terms (기관/외인/환율/수급/시총/국장) (C-2).\n"
   "- Upper bound 6: tokens of 7+ chars are inflected phrases already captured as substrings of shorter core keywords. The core-keyword set is identical for 6 vs 8 (C-2).",
  "secD": "## D. Core-keyword DF threshold  (real run = DF 0.5%)\n\n"
   "**Why around 0.5%**\n"
   "- Lowering the DF threshold adds keywords but mixes in generic words ('가장', '당장'); raising it is cleaner but loses valid compounds (base rate).\n"
   "- 1% is the balance point, and its pass rate (29%) matches the actual output file `comments_stock_clean`.\n\n"
   "The real run used `min_doc_ratio=0.005` (DF 0.5%); F and G use the stock_score / is_stock columns from that run.",
  "secF": "## F · G · E. Score cutoff · core ≥ 1 condition · 10 : 3 weighting\n\n"
   "score = 10·(core keyword count) + 3·(auxiliary keyword count),  pass = score ≥ 10 **and** core ≥ 1.\n\n"
   "**Why this design**\n"
   "- **score ≥ 10** is the end of the pass-rate plateau. Cutoffs 3·5·7·10 all give 15.1%; at 13 it drops to 10.6% (F-1). The drop is the comments with exactly one core keyword (score 10) being cut.\n"
   "- **10 : 3** — changing the aux weight to 2·5·10 leaves the passing set 100% identical. One core keyword = 10 already clears the gate; the aux weight only ranks.\n"
   "- **core ≥ 1** — without it, ~{aux_only:,} comments ({aux_pct:.1f}% of passers) get through on auxiliary keywords alone.",
  "secHI": "## Strategy performance (10-year K-FGI pipeline, re-derived)\n\n"
   "K-FGI → position → strategy return re-derived over the full period for varying γ and sentiment MA window. "
   "Code: `downstream_strategy_sensitivity.py`; numbers: `downstream_sensitivity.csv`.\n\n"
   "**Key** — γ has no effect on strategy performance (H-1, H-2); the 10-day sentiment MA is a clear local optimum (I-1). "
   "So τ and γ are 'anything in the neighbourhood is fine', while the sentiment MA 'has to be 10 days'.",
  "concl": "## Conclusion — convention vs. our choice\n\n"
   "| Value | Type | Verdict |\n|---|---|---|\n"
   "| τ = 0.95 | our choice | ✓ index & strategy structurally invariant to τ (tox≥0.95 weight mass 0.003%) |\n"
   "| γ = 2 | our choice | ✓ no performance effect; principled midpoint of decay strength |\n"
   "| score ≥ 10 | our choice | ✓ end of the pass-rate plateau (3–10) |\n"
   "| DF ≥ 0.5% | our choice | ◐ excludes generic words; 1%+ collapses to 0–2 core keywords/year |\n"
   "| core 10 : aux 3 | our choice | ◐ aux weight irrelevant to the gate, ranking only |\n"
   "| core ≥ 1 | our choice | ✓ blocks the few comments passing on aux keywords alone |\n"
   "| 2–6 characters | method rule | ✓ keeps 2-char financial terms / long phrases caught as substrings |\n"
   "| 10-day sentiment MA | our choice | ✓ local optimum on Sharpe and MDD (a value that does need a rationale) |\n"
   "| 252-day · 60-day · 70th pct. | finance convention | not tuned |",
  "cur": "current", "run": "generate figure",
 },
}

SECTIONS = [
    ("secP", ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]),
    ("secA", ["TOX", "A1", "A2", "A3", "A4", "A5"]),
    ("secB", ["B1", "B2", "B3", "B4", "B5", "B6"]),
    ("secC", ["C1", "C2"]),
    ("secD", ["D1", "D2", "D3"]),
    ("secF", ["F1", "F2", "E1", "G1"]),
    ("secHI", ["H1", "H2", "I1"]),
]


def build(lang):
    tx = TXT[lang]
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell(tx["title"]))
    c.append(nbf.v4.new_markdown_cell(tx["setup"]))
    c.append(nbf.v4.new_code_cell(
        "%matplotlib inline\n"
        "import filter_param_lib as L\n"
        "import matplotlib.pyplot as plt\n"
        f'LANG = "{lang}"\n'
        "L.use_style(LANG)"))
    c.append(nbf.v4.new_code_cell("tox, kept = L.load_all()"))
    c.append(nbf.v4.new_code_cell("R = L.compute(tox, kept)   # 최초 3~4분, 이후 캐시"))

    # aux_only 수치를 D/F 섹션 텍스트에 채우기 위해 캐시가 있으면 읽는다
    aux_only, aux_pct = 1858, 1.0
    try:
        import pickle
        if L.CACHE.exists():
            RR = pickle.loads(L.CACHE.read_bytes())
            aux_only = RR["G"]["aux_only"]
            aux_pct = aux_only / RR["G"]["n_pass"] * 100
    except Exception:
        pass

    for sec, keys in SECTIONS:
        body = tx[sec].format(aux_only=aux_only, aux_pct=aux_pct) if "{aux" in tx[sec] else tx[sec]
        c.append(nbf.v4.new_markdown_cell(body))
        for k in keys:
            cap = L.LB[lang][k]["cap"]
            c.append(nbf.v4.new_markdown_cell(f"### {k}\n\n{cap}"))
            c.append(nbf.v4.new_code_cell(f'L.make_fig("{k}", R, LANG)'))

    c.append(nbf.v4.new_markdown_cell(tx["concl"]))
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = BASE / f"filter_params_{lang}.ipynb"
    nbf.write(nb, str(out))
    print("wrote", out.name, f"({len(c)} cells)")


def _build_filter():
    for lg in ("ko", "en"):
        build(lg)


# ---------------------------------------------------------------- 전략 민감도 노트북
STX = {
 "ko": {
  "title": "# 전략 성과 민감도 — 독성 가중치 γ · 감성 이동평균 기간\n\n"
   "파라미터를 바꿔가며 **K-FGI → 포지션 → 전략수익률**을 10년 전 구간 재산출하고, "
   "Sharpe · MDD · CVaR(5%) · 하방 변동성 · 하락일 방어율 · Calmar 를 비교한다.\n\n"
   "- 실험 1: γ ∈ {1, 2, 3} — 댓글 → 일별 감성피처 6종 재계산 → `KFG_final` 감성열 치환 → 전 파이프라인 재실행\n"
   "- 실험 2: 감성 복합지표 이동평균 기간 ∈ {5, 10, 20} 거래일 (γ = 2 고정)\n\n"
   "**주의** — EGARCH 조건부 변동성은 수익률만의 함수이므로 캐시"
   "(`08_Modeling/experiment_outputs/cache/egarch_vol_10y.csv`)를 재사용한다. "
   "walk-forward K-FGI 재추정 때문에 6회 실행에 약 8~12분 소요된다. 결과는 `downstream_sensitivity.csv` 에 저장된다.",
  "run": "## 실행\n최초 실행은 8~12분. `downstream_sensitivity.csv` 가 이미 있으면 아래 표·그림 셀만 실행해도 된다.",
  "res": "## 결과",
  "concl": "## 해석\n\n"
   "- **γ 는 전략 성과에 영향이 없다.** γ 1·2·3 에서 Sharpe 0.610 / 0.613 / 0.622, MDD −22.9% 고정. "
   "따라서 γ = 2 는 성과로 고른 값이 아니라 댓글 가중 원칙(경계 독성 절반 감쇠)으로 정한 값이다.\n"
   "- **감성 이동평균 10일은 명확한 국소 최적이다.** 5일(Sharpe 0.55) 은 노이즈, 20일(Sharpe 0.54, MDD −27.3%) 은 반응 지연. "
   "이 값은 τ·γ 와 달리 결과에 민감하므로 반드시 근거가 필요하며, 그 근거가 이 표다.",
 },
 "en": {
  "title": "# Strategy Sensitivity — toxicity weighting γ · sentiment moving-average window\n\n"
   "Re-deriving **K-FGI → position → strategy return** over the full 10 years for varying parameters, "
   "comparing Sharpe · MDD · CVaR(5%) · downside vol · down-day defense · Calmar.\n\n"
   "- Experiment 1: γ ∈ {1, 2, 3} — recompute the 6 daily sentiment features → swap into `KFG_final` → re-run the whole pipeline\n"
   "- Experiment 2: sentiment-composite moving-average window ∈ {5, 10, 20} trading days (γ = 2 fixed)\n\n"
   "**Note** — EGARCH conditional volatility depends only on returns, so its cache "
   "(`08_Modeling/experiment_outputs/cache/egarch_vol_10y.csv`) is reused. The walk-forward K-FGI re-estimation makes "
   "6 runs take ~8–12 min. Results are saved to `downstream_sensitivity.csv`.",
  "run": "## Run\nFirst run takes 8–12 min. If `downstream_sensitivity.csv` already exists, you can run only the table/figure cells below.",
  "res": "## Results",
  "concl": "## Reading\n\n"
   "- **γ has no effect on strategy performance.** Sharpe 0.610 / 0.613 / 0.622 for γ 1/2/3, MDD fixed at −22.9%. "
   "So γ = 2 was chosen on the comment-weighting principle (halve a borderline-toxic comment), not for performance.\n"
   "- **The 10-day sentiment MA is a clear local optimum.** 5 days (Sharpe 0.55) is noisy, 20 days (Sharpe 0.54, MDD −27.3%) lags. "
   "Unlike τ and γ, this value does move the result and therefore needs an explicit rationale — this table is that rationale.",
 },
}


def build_strategy(lang):
    tx = STX[lang]
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell(tx["title"]))
    c.append(nbf.v4.new_code_cell(
        "import pandas as pd, numpy as np\n"
        "import downstream_strategy_sensitivity as D\n"
        "import filter_param_lib as L\n"
        "%matplotlib inline\n"
        "import matplotlib.pyplot as plt\n"
        f'LANG = "{lang}"\n'
        "L.use_style(LANG)"))
    c.append(nbf.v4.new_markdown_cell(tx["run"]))
    c.append(nbf.v4.new_code_cell(
        "kfg = pd.read_csv(D.KFG_FINAL, parse_dates=['date']).sort_values('date').reset_index(drop=True)\n"
        "mkt = kfg[['date','log_return_t+1']].rename(columns={'log_return_t+1':'mkt'}).set_index('date')['mkt']\n"
        "comments = D.load_comments()\n"
        "SENT = ['sent_norm_w','sent_strength_w','sent_std','neg_z','effective_n','heat']\n"
        "def swap_sent(gamma):\n"
        "    ds = D.daily_sentiment(comments, gamma)\n"
        "    m = kfg.drop(columns=SENT).merge(ds, on='date', how='left')\n"
        "    m[SENT] = m[SENT].ffill()\n"
        "    return m"))
    c.append(nbf.v4.new_code_cell(
        "# downstream_sensitivity.csv 가 이미 있으면 재사용한다. 다시 계산하려면 그 파일을 삭제하고 실행.\n"
        "CSV = D.BASE / 'downstream_sensitivity.csv'\n"
        "if CSV.exists():\n"
        "    out = pd.read_csv(CSV)\n"
        "    print('loaded cached', CSV.name)\n"
        "else:\n"
        "    rows = []\n"
        "    for g in [1.0, 2.0, 3.0]:\n"
        "        res = D.run_pipeline(swap_sent(g), sent_ma_window=10)\n"
        "        mm = D.metrics(res['strat_ret'].reset_index(drop=True), mkt.reindex(res['date']).reset_index(drop=True), f'gamma={g:g}')\n"
        "        mm['experiment'], mm['param'] = 'gamma', g; rows.append(mm); print('gamma', g, 'done')\n"
        "    base2 = swap_sent(2.0)\n"
        "    for w in [5, 10, 20]:\n"
        "        res = D.run_pipeline(base2, sent_ma_window=w)\n"
        "        mm = D.metrics(res['strat_ret'].reset_index(drop=True), mkt.reindex(res['date']).reset_index(drop=True), f'sent_MA={w}')\n"
        "        mm['experiment'], mm['param'] = 'sent_ma', w; rows.append(mm); print('sent_MA', w, 'done')\n"
        "    out = pd.DataFrame(rows); out.to_csv(CSV, index=False)\n"
        "out[['experiment','param','sharpe','mdd','cvar5','downside_vol','downday_defense','calmar','total_return']].round(4)"))
    c.append(nbf.v4.new_markdown_cell(tx["res"]))
    for k in ["H1", "H2", "I1"]:
        c.append(nbf.v4.new_markdown_cell(f"### {k}\n\n{L.LB[lang][k]['cap']}"))
        c.append(nbf.v4.new_code_cell(f'L.make_fig("{k}", None, LANG)'))
    c.append(nbf.v4.new_markdown_cell(tx["concl"]))
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = BASE / f"strategy_sensitivity_{lang}.ipynb"
    nbf.write(nb, str(out))
    print("wrote", out.name, f"({len(c)} cells)")


if __name__ == "__main__":
    _build_filter()
    for lg in ("ko", "en"):
        build_strategy(lg)
