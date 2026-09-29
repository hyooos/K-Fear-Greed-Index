# -*- coding: utf-8 -*-
"""
filter_param_lib — 필터링 파라미터 검증 실험의 계산·작도 라이브러리
=================================================================
notebook(filter_params_{ko,en}.ipynb)에서 import 해서 사용한다.

- load_all()          : 2014-2025 archive recollect 데이터 로드 (toxicity_all / classified_stock_comments)
- compute(tox, kept)  : 모든 실험 수치 계산 -> dict R  (pickle 캐시)
- LB[lang][key]       : 그림별 title / xlabel / ylabel / caption
- fig_XX(R, lang)     : 개별 그림(단일 Figure) 생성, 논문용 품질

데이터 한계: 코퍼스는 이미 정치 필터를 통과. toxicity_all 은 독성점수 부여된
제거 이전 전체 댓글이므로 τ(제거) 실험은 실측 가능. 감성확률은 kept 댓글에만
존재하므로 일별 감성 안정성은 'τ 하향(더 엄격)' 방향만 시뮬레이션 가능.
"""
from __future__ import annotations

import glob
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
KFGI = ROOT.parent / "kfgi"
FILT = KFGI / "필터링 데이터"                                  # (구 3년 표본, 미사용)
CACHE = BASE / "_compute_cache.pkl"

# 최종 논문(2014-2025) recollect 파이프라인 중간 산출물 — P (파이프라인 개요) 전용
ARCH = ROOT.parent / "kfgi_최종_local_archive_ignored"
_ARCH_ALT = Path.home() / ".Trash/kfgi_최종_local_archive_ignored"
if not ARCH.exists() and _ARCH_ALT.exists():
    ARCH = _ARCH_ALT
ARCH_FILT = ARCH / "intermediate_outputs/3_Filtering_final"
ARCH_RAW = ARCH / "raw_data/data/NAVER_by_year_recollect_allcomments/comments"
ARCH_ART = ARCH / "raw_data/data/NAVER_by_year_recollect_allcomments/article"

# 필터 파라미터 스윕용 데이터 — 최종 논문 표본(2014-2025) recollect 중간 산출물
TOX_ALL = sorted(glob.glob(str(ARCH_FILT / "recollect_toxicity_filter_model/comments_toxicity_all_*.csv")))
SENT_PROB = sorted(glob.glob(str(ARCH / "intermediate_outputs/4_Sentiment_analysis/"
                                 "recollect_sentiment_scores_model_toxicity/sentiment_with_prob_*.csv")))
# 경제 필터: 실제 실행에서 stock_score / is_stock 가 계산된 파일 (연도별)
CLASSIFIED = sorted(glob.glob(str(ARCH_FILT / "recollect_final_filtered_model_toxicity/*/"
                                  "classified_stock_comments_*.csv")))
KW_ANALYSIS = ARCH_FILT / "recollect_final_filtered_model_toxicity"
# 구 3년 표본(대비용, 미사용)
_TOX_ALL_3Y = sorted(glob.glob(str(FILT / "toxicity_filter/comments_toxicity_all_*.csv")))

# min_doc_ratio: 실제 실행(3_final_filter.py)이 0.005(DF 0.5%)로 돌았음을 archive 로 확인.
CUR = dict(gamma=2.0, tau=0.95, wlen=(2, 6), min_doc_ratio=0.005,
           support_band=0.3, core_pt=10, sup_pt=3, score_cut=10)

STOCK_PATTERNS = [
    r"코스피|코스닥|kospi|kosdaq", r"주식|주가|증시|시장|장",
    r"매수|매도|투자|손절|익절", r"개미|외인|기관|외국인",
    r"삼전|삼성전자|하닉|하이닉스", r"지수|시총|배당|상장",
    r"급등|급락|폭등|폭락|상승|하락", r"수급|거래량|환율|금리",
    r"종목|실적|반도체|전지", r"펀드|연기금|국민연금",
]
_SRE = [re.compile(p) for p in STOCK_PATTERNS]
def is_stock_word(w): return any(r.search(w) for r in _SRE)
def extract_words(t, lo, hi):
    return re.findall(rf"[가-힣]{{{lo},{hi}}}|[a-zA-Z]{{2,10}}", str(t).lower())


# ============================================================ 스타일
SHOW_TITLES = False   # 교수님 지시: 피규어 제목 제거 (캡션에 포함). True 로 두면 제목 표시.

def _T(obj, text, sup=False, **kw):
    """SHOW_TITLES 가 True 일 때만 제목/부제목을 그린다."""
    if not SHOW_TITLES:
        return
    (obj.suptitle if sup else obj.set_title)(text, **kw)

def use_style(lang="ko"):
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,
        "font.family": "AppleGothic" if lang == "ko" else "DejaVu Sans",
        "axes.unicode_minus": False,
        "font.size": 11.5, "axes.titlesize": 13, "axes.titlepad": 12,
        "axes.labelsize": 11.5, "axes.labelpad": 7,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 9.5,
        "axes.grid": True, "grid.alpha": 0.30, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.edgecolor": "#888", "figure.facecolor": "white",
    })

ACC, CURC, OK, RISK = "#c1443c", "#c1443c", "#2e7d55", "#b0433a"

def _ax(fig, key, lang):
    ax = fig.gca() if fig.axes else fig.add_subplot(111)
    L = LB[lang][key]
    ax.set_xlabel(L["x"]); ax.set_ylabel(L["y"]); _T(ax, L["t"])
    return ax

def _curline(ax, x, label):
    ax.axvline(x, color=CURC, ls="--", lw=1.3, label=label, zorder=1)


# ============================================================ 로드 / 집계
def _read(fs, cols=None):
    fr = []
    for f in fs:
        d = pd.read_csv(f, low_memory=False, dtype={"pub_date": "str", "comment_id": "str"})
        fr.append(d[cols] if cols else d)
    return pd.concat(fr, ignore_index=True)

def load_all():
    tox = _read(TOX_ALL)
    tox["text_raw"] = tox["text_raw"].fillna("").astype(str)
    tox["toxicity_score"] = pd.to_numeric(tox["toxicity_score"], errors="coerce")
    tox = tox[tox["toxicity_score"].notna() & (tox["text_raw"].str.strip() != "")].copy()
    tox["weight"] = pd.to_numeric(tox["weight"], errors="coerce")
    tox["year"] = tox["pub_date"].str[:4]
    dt = pd.to_datetime(tox["comment_at"], errors="coerce", utc=True)
    tox["date"] = dt.dt.tz_convert("Asia/Seoul").dt.date
    sp = _read(SENT_PROB, ["comment_id", "p_pos", "p_neg"])
    sp["p_pos"] = pd.to_numeric(sp["p_pos"], errors="coerce")
    sp["p_neg"] = pd.to_numeric(sp["p_neg"], errors="coerce")
    sp = sp.dropna().drop_duplicates("comment_id")
    tox = tox.merge(sp, on="comment_id", how="left")
    tox["sent_norm"] = (tox["p_pos"] - tox["p_neg"]) / (tox["p_pos"] + tox["p_neg"] + 1e-8)
    # 경제 필터 입력·결과: 실제 실행에서 stock_score/is_stock 계산된 파일 (2014-2025)
    kept = _read(CLASSIFIED, ["pub_date", "text_raw", "stock_score", "is_stock"])
    kept["text_raw"] = kept["text_raw"].fillna("").astype(str)
    kept = kept[kept["text_raw"].str.strip() != ""].copy()
    kept["stock_score"] = pd.to_numeric(kept["stock_score"], errors="coerce").fillna(0)
    kept["has_core"] = kept["is_stock"].astype(str).eq("stock")   # is_stock='stock' ⟺ 핵심 키워드 ≥ 1개
    kept["year"] = kept["pub_date"].str[:4]
    print(f"toxicity_all {len(tox):,}  (감성확률 {tox['p_pos'].notna().sum():,})  |  "
          f"classified {len(kept):,}  |  연도 {sorted(tox['year'].unique())}")
    return tox, kept

def _daily(df, w):
    m = df["p_pos"].notna().values
    t = pd.DataFrame({"date": df["date"].values[m],
                      "sw": (df["sent_norm"].values * w)[m], "w": w[m]})
    g = t.groupby("date", sort=True).sum()
    return g["sw"] / (g["w"] + 1e-8)

def _cmp(a, b):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    d = j["a"] - j["b"]
    return dict(corr=float(j["a"].corr(j["b"])), mad=float(d.abs().mean()), maxabs=float(d.abs().max()))


# ============================================================ compute
def compute(tox, kept, use_cache=True):
    if use_cache and CACHE.exists():
        print(f"캐시 로드: {CACHE.name}")
        return pickle.loads(CACHE.read_bytes())

    R = {}
    tv = tox["toxicity_score"].to_numpy(); N = len(tox)
    base_uw = _daily(tox, np.ones(N))
    R["A"] = []
    for g in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        w = np.clip(1 - tv, 0, None) ** g
        st = _cmp(_daily(tox, w), base_uw)
        eff = w.sum() ** 2 / (np.sum(w ** 2) + 1e-8)
        R["A"].append(dict(gamma=g, w30=(1 - .3) ** g, w50=(1 - .5) ** g, w80=(1 - .8) ** g,
                           eff_ratio=eff / N, mad=st["mad"], maxabs=st["maxabs"], corr=st["corr"]))
    R["A_series"] = {g: _daily(tox, np.clip(1 - tv, 0, None) ** g) for g in [0.0, 1.0, 2.0, 3.0, 5.0]}
    R["A_wdist"] = {g: np.clip(1 - tv, 0, None) ** g for g in [1.0, 2.0, 3.0]}

    w_cur = np.clip(1 - tv, 0, None) ** CUR["gamma"]
    base_tau = _daily(tox, w_cur)
    R["B"] = []
    for tau in [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]:
        drop = tv >= tau
        row = dict(tau=tau, dropped=int(drop.sum()), dropped_pct=drop.mean() * 100,
                   remaining=int((~drop).sum()), mad=np.nan, maxabs=np.nan)
        if tau <= 0.95:
            st = _cmp(_daily(tox, np.where(~drop, w_cur, 0.0)), base_tau)
            row.update(mad=st["mad"], maxabs=st["maxabs"])
        R["B"].append(row)
    R["B_bands"] = []
    for lo, hi in [(0.90, 0.95), (0.95, 0.97), (0.97, 0.99), (0.99, 1.01)]:
        sub = tox[(tv >= lo) & (tv < hi)]
        R["B_bands"].append(dict(lo=lo, hi=min(hi, 1.0), n=int(len(sub)),
                                 samples=sub.sort_values("toxicity_score")["text_raw"].head(4).tolist()))
    tot = w_cur.sum()
    R["B_wmass"] = [dict(x=x, pct=float(w_cur[tv >= x].sum() / tot * 100))
                    for x in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]]
    R["B_toxvals"] = tv
    R["B_yearly"] = [dict(year=int(y), pct=float((g["toxicity_score"] >= 0.95).mean() * 100),
                          n=int(len(g)))
                     for y, g in tox.groupby("year")]

    cnt = Counter()
    _hgl = re.compile(r"[가-힣]+")
    for i in range(0, len(tox), 100000):
        cnt.update(_hgl.findall(" ".join(tox["text_raw"].iloc[i:i + 100000].tolist()).lower()))
    R["C"] = []
    for Lc in range(1, 11):
        ws = [w for w in cnt if len(w) == Lc]
        nt = sum(cnt[w] for w in ws); ns = sum(cnt[w] for w in ws if is_stock_word(w))
        R["C"].append(dict(L=Lc, tokens=nt, stock_ratio=(ns / nt if nt else 0.0)))
    R["C_1char"] = sorted(((w, cnt[w]) for w in cnt if len(w) == 1 and is_stock_word(w)), key=lambda x: -x[1])[:8]
    R["C_long"] = sorted(((w, cnt[w]) for w in cnt if len(w) >= 7 and is_stock_word(w)), key=lambda x: -x[1])[:8]
    R["C_window"] = []
    csamp = pd.concat([g.sample(min(len(g), 40000), random_state=0)
                       for _, g in kept.groupby("year")])["text_raw"]
    for lo, hi in [(1, 4), (1, 6), (2, 4), (2, 6), (2, 8), (3, 6)]:
        wdc = Counter()
        for t in csamp:
            for w in set(extract_words(t, lo, hi)):
                wdc[w] += 1
        thr = len(csamp) * CUR["min_doc_ratio"]
        R["C_window"].append(dict(window=f"{lo}-{hi}",
                                  core=sum(1 for w, c in wdc.items() if is_stock_word(w) and c >= thr)))

    R.update(_econ(kept))
    R["N_all"] = N; R["N_kept"] = len(kept)
    R.update(_pipeline())
    CACHE.write_bytes(pickle.dumps(R))
    print(f"캐시 저장: {CACHE.name}")
    return R


def _build_kw(df, mdr, band, lo, hi):
    per = {}
    for y, dfy in df.groupby("year"):
        wdc = Counter()
        for t in dfy["text_raw"]:
            for w in set(extract_words(t, lo, hi)):
                wdc[w] += 1
        thr = len(dfy) * mdr
        core = [(w, c) for w, c in wdc.items() if is_stock_word(w) and c >= thr]
        sup = [(w, c) for w, c in wdc.items() if is_stock_word(w) and thr * band <= c < thr]
        per[y] = (core, sup)
    return per

def _score_keep(df, per, cpt, spt, cut, need_core):
    keep = np.zeros(len(df), bool); sc = np.zeros(len(df)); ch = np.zeros(len(df), bool)
    pos = {ix: i for i, ix in enumerate(df.index)}
    for y, dfy in df.groupby("year"):
        cs = {w for w, _ in per[y][0]}; ss = {w for w, _ in per[y][1]}
        for ix, t in zip(dfy.index, dfy["text_raw"]):
            i = pos[ix]; tl = t.lower()
            hit = any(k in tl for k in cs)
            s = sum(k in tl for k in cs) * cpt + sum(k in tl for k in ss) * spt
            sc[i] = s; ch[i] = hit
            keep[i] = s >= cut and (hit or not need_core)
    return sc, keep, ch

def _pipeline():
    """필터링 단계별 처리 결과 + 정치 키워드 통계 — 최종 논문 표본 2014-2025.

    출처: kfgi_최종_local_archive_ignored/intermediate_outputs/3_Filtering_final/ 의
    recollect 중간 산출물 summary JSON (논문 3.3 필터링 표와 동일 수치).
    """
    import json
    A = ARCH_FILT
    YEARS = list(range(2014, 2026))
    per, kwc, ratios = [], Counter(), []
    for y in YEARS:
        pol = json.loads((A / f"recollect_political_filter/classification_summary_{y}.json").read_text())
        tox = json.loads((A / f"recollect_toxicity_filter_model/toxicity_summary_{y}.json").read_text())
        fin = json.loads((A / f"recollect_final_filtered_model_toxicity/{y}/"
                          f"stock_classification_summary_{y}.json").read_text())
        rc = pd.read_csv(ARCH_RAW / f"comments_{y}.csv", usecols=["news_id", "text_raw"], dtype=str, low_memory=False)
        raw = len(rc)
        raw_ne = int(rc["text_raw"].fillna("").str.strip().ne("").sum())
        n_art = rc["news_id"].nunique()
        try:
            news_list = len(pd.read_csv(next(ARCH_ART.glob(f"*{y}*.csv")), usecols=[0], dtype=str, low_memory=False))
        except Exception:
            news_list = n_art
        po = pd.read_csv(A / f"recollect_political_filter/comments_{y}_political_only.csv",
                         usecols=["matched_keywords"], dtype=str)
        for v in po["matched_keywords"].dropna():
            for m in re.findall(r"'([^']+)'|\"([^\"]+)\"", v):
                kw = m[0] or m[1]
                if kw:
                    kwc[kw] += 1
        try:
            ka = json.loads((A / f"recollect_final_filtered_model_toxicity/{y}/"
                             f"keyword_analysis_result_{y}.json").read_text())
            dfm = {k["word"]: k["percentage"] for k in ka["core_keywords"] + ka["support_keywords"]}
        except Exception:
            dfm = {}
        per.append(dict(year=y, raw=raw, raw_nonempty=raw_ne, n_articles=n_art, news_list=news_list,
                        kw_df={w: dfm.get(w, 0.0) for w in ["공매도", "주식", "금리", "개미들", "주가", "기관"]},
                        empty_rate=(raw - raw_ne) / raw * 100, com_per_article=raw_ne / max(n_art, 1),
                        core_kw=fin.get("core_keywords_count"), stock_ratio=fin.get("stock_ratio"),
                        pol_in=pol["total_comments"],
                        pol_removed=pol["political_comments"], pol_pass=pol["non_political_comments"],
                        tox_removed=tox["dropped"], tox_pass=tox["kept"],
                        econ_removed=fin["other_comments"], final=fin["stock_comments"]))
        ratios.append(dict(year=y, political_ratio=pol["political_ratio"],
                           political_n=pol["political_comments"], base=pol["total_comments"]))
    # 2019 금융 뉴스 주제 분포 (댓글 수 가중) — 왜 주식 단어가 희소했나
    y2019 = {}
    try:
        art = pd.read_csv(ARCH_ART / "news_2019.csv", dtype=str, low_memory=False).drop_duplicates("news_id")
        com = pd.read_csv(ARCH_RAW / "comments_2019.csv", usecols=["news_id", "text_raw"], dtype=str, low_memory=False)
        cc = com[com["text_raw"].fillna("").str.strip().ne("")].groupby("news_id").size()
        art["nc"] = art["news_id"].map(cc).fillna(0).astype(int)
        TOP = {
            "주식·증시": r"코스피|코스닥|증시|공매도|상장|IPO|공모주|청약|주가|주주|배당|시총|거래소|반도체주|바이오주|신라젠|에이치엘비|셀트리온|증권사|펀드런|ETF|외국인.?매도|기관.?매도|개미|동학",
            "은행·DLF·펀드": r"은행|DLF|DLS|파생결합|사모펀드|라임|원금|불완전판매|예금|적금|대출|금감원|저축은행|카드사|보이스피싱|가계부채",
            "조국·정치·검찰": r"조국|정경심|코링크|윤석열|검찰|청와대|법무부|장관|의원|국회|한국당|민주당|정의당|대통령|추미애|나경원|황교안|패스트트랙|선거법|공수처",
            "일본·수출규제": r"일본|아베|수출규제|불매|노재팬|화이트리스트|지소미아|GSOMIA|보복|반도체.?소재|후쿠시마|한일",
            "부동산·건설": r"부동산|아파트|분양|집값|전세|월세|재건축|재개발|주택|임대|건설|시공|GTX|3기신도시",
            "기업·경영·재벌": r"삼성|현대차|SK|LG|롯데|한진|대한항공|아시아나|CJ|이재용|정의선|최태원|구광모|조현|재벌|오너|상속|경영권|총수|갑질|일감|담합",
            "연금·보험·세제·정책": r"연금|보험|건보|건강보험|세금|증세|세제|종부세|양도세|퇴직|국민연금|기초연금|추경|예산|한국은행|금리인하|성장률|경기|GDP",
            "노동·자영업·소비": r"최저임금|52시간|주52|파업|노조|일자리|고용|실업|자영업|소상공인|택배|배달|프랜차이즈|편의점|마트|소비자|리콜|환불|외식|카페",
        }
        tot = int(art["nc"].sum())
        titles = art["title"].fillna("")
        assigned = pd.Series("기타", index=art.index)
        for name, pat in TOP.items():   # 우선순위 순으로 첫 매칭 주제에 배정 (배타적)
            free = assigned.eq("기타")
            hit = titles.str.contains(pat, case=False, regex=True) & free
            assigned[hit] = name
        art["topic"] = assigned
        y2019 = {"total": tot, **{k: int(v) for k, v in art.groupby("topic")["nc"].sum().items()}}
    except Exception as e:
        y2019 = {"error": str(e)}
    return dict(PIPE=per, POL_RATIO=ratios, POL_KW=kwc.most_common(16), SENT_OK=192888, Y2019=y2019)


def _econ(kept):
    """F·G: 2014-2025 전체 실제 stock_score/is_stock 컬럼 사용.
       D·E: 연도별 4만 건 층화표본에서 키워드 재추출·재점수화."""
    sc = kept["stock_score"].to_numpy()
    hc = kept["has_core"].to_numpy()          # is_stock=='stock' ⟺ 핵심 키워드 ≥ 1개
    out = {}
    # ---- F: 경제성 점수 임계값 민감도 (통과 = 점수 ≥ cut AND 핵심 ≥ 1개) ----
    out["F"] = [dict(cut=c, pass_pct=float((hc & (sc >= c)).mean() * 100))
                for c in [3, 5, 7, 10, 13, 15, 20, 30]]
    # ---- G: '핵심 ≥ 1개' 조건 유무 ----
    kw10, k10 = hc & (sc >= 10), sc >= 10
    out["G"] = dict(pass_with=float(kw10.mean() * 100), pass_without=float(k10.mean() * 100),
                    aux_only=int((k10 & ~hc).sum()), n_pass=int(kw10.sum()))
    out["econ_score"] = sc

    # ---- D·E·D_band: 연도별 층화표본 재추출 ----
    lo, hi = CUR["wlen"]
    samp = pd.concat([g.sample(min(len(g), 40000), random_state=0)
                      for _, g in kept.groupby("year")]).reset_index(drop=True)
    bky = _build_kw(samp, CUR["min_doc_ratio"], CUR["support_band"], lo, hi)
    D, D_kw = [], {}
    for pct in [0.5, 1.0, 1.5, 2.0]:
        ky = _build_kw(samp, pct / 100, CUR["support_band"], lo, hi)
        allc = Counter()
        for yv in ky:
            for w, c in ky[yv][0]:
                allc[w] += c
        D_kw[f"{pct}%"] = [w for w, _ in allc.most_common(30)]
        _, kp, _ = _score_keep(samp, ky, CUR["core_pt"], CUR["sup_pt"], CUR["score_cut"], True)
        D.append(dict(DF_pct=pct, core=float(np.mean([len(ky[yv][0]) for yv in ky])),
                      sup=float(np.mean([len(ky[yv][1]) for yv in ky])), pass_pct=float(kp.mean() * 100)))
    out["D"], out["D_kw"] = D, D_kw
    out["E"] = []
    for cp, sp in [(3, 3), (5, 3), (10, 2), (10, 3), (10, 5), (10, 10)]:
        _, kp, _ = _score_keep(samp, bky, cp, sp, CUR["score_cut"], True)
        out["E"].append(dict(core_pt=cp, sup_pt=sp, label=f"{cp}:{sp}", pass_pct=float(kp.mean() * 100)))
    out["D_band"] = []
    for b in [0.1, 0.2, 0.3, 0.5, 0.7]:
        ky = _build_kw(samp, CUR["min_doc_ratio"], b, lo, hi)
        _, kp, _ = _score_keep(samp, ky, CUR["core_pt"], CUR["sup_pt"], CUR["score_cut"], True)
        out["D_band"].append(dict(band=b, band_lo_pct=CUR["min_doc_ratio"] * b * 100,
                                  aux=float(np.mean([len(ky[yv][1]) for yv in ky])),
                                  pass_pct=float(kp.mean() * 100)))
    return out


# ============================================================ 라벨 + 설명
LB = {
 "ko": {
  "P1": dict(t="P-1. 필터링 단계별 처리 결과  (최종 논문 표본 2014-2025 합계)",
            x="", y="",
            cap="원천 수집 댓글 2,150,215건에서 시작한다. 빈 댓글·중복 제거로 정치 필터 입력이 1,575,268건(73.3%)이 되고, "
                "1단계 정치 키워드 필터로 165,288건(입력의 10.5%), 2단계 독성 필터(τ=0.95)로 131,337건(정치 통과분의 9.3%), "
                "3단계 경제 관련성 필터로 1,085,526건(독성 통과분의 84.9%)이 제거된다. 최종 경제/주식 관련 댓글은 "
                "193,117건(원천의 9.0%)이며 이 중 192,888건에 감성 점수가 산출되었다. (논문 3.3 필터링 표와 동일 수치. "
                "연도별 상세표는 Appendix / `10_Paper_outputs/04_투고_전략/FINAL_REVISION_MASTER_KR.md` 참조.)"),
  "P2": dict(t="P-2. 필터링 단계별 잔존 댓글 수 (퍼널, 2014-2025 합계)",
            x="필터링 단계", y="잔존 댓글 수",
            cap="각 막대는 해당 단계를 통과한 뒤 남은 댓글 수. 원천 215만 → 정치 분류 입력 158만 → 정치 통과 141만 → "
                "독성 통과 128만 → 최종 19.3만. 가장 크게 줄어드는 단계는 3단계 경제 필터링으로, 정치·독성이 아니어도 "
                "주식·경제와 무관한 일반 댓글을 대량 제거한다(독성 통과분의 84.9%)."),
  "P3": dict(t="P-3. 1단계 정치 키워드 필터링 결과 (2014-2025)",
            x="", y="",
            cap="좌: 연도별 정치 관련 댓글 비율. 2014-2016년 4~6% 수준에서 2018년 이후 10% 이상, 2025년 15.0%로 "
                "탄핵·계엄 국면에서 높아졌다. 우: 매칭된 정치 키워드 상위 16개(2014-2025 합계). 초기에는 박근혜·근혜, "
                "후기에는 민주당·문재인·재명·이재명·윤석열·석열·탄핵이 상위를 차지한다. 이들 인물·정당·사건어가 "
                "감성 집계에서 정치 프레임 노이즈를 유발하므로 제거한다."),
  "P4": dict(t="P-4. 연도별 필터링 단계 처리 결과 (2014-2025)",
            x="", y="",
            cap="논문 Appendix 표. 원천 댓글은 2018년 이후 연 20만~31만 건으로 크게 늘었고, 최종 경제/주식 관련 댓글은 "
                "시장 이슈(2021 공매도·2023 이후 국장 논쟁)가 있던 해에 3만 건대로 많다. 최종 잔존율은 연 4~13% 범위이며 "
                "전체 평균 9.0%다."),
  "P5": dict(t="P-5. 연도별 필터링 단계별 댓글 구성",
            x="연도", y="댓글 수",
            cap="각 연도 막대를 단계별 제거/유지로 분해한 것. 아래부터 최종 유지(경제/주식 관련), 경제 무관 제거, "
                "독성 제거, 정치 제거, 빈 댓글·중복. 모든 연도에서 '경제 무관 제거'(연회색)가 가장 큰 비중을 차지한다."),
  "P6": dict(t="P-6. 연도별 원천 댓글 수가 다른 이유",
            x="", y="",
            cap="크롤러는 매일 증권·금융 섹션에서 댓글 수 기준 상위 10개 기사(섹션별)를 선정하고, 각 기사의 당일 작성 "
                "댓글을 전량 수집한다(네이버 댓글 API 한도상 기사당 최대 약 200개, 이 한도에 걸리는 기사는 1% 미만). "
                "(좌) 수집 대상 기사 수는 2014년 3,651개에서 2018년 이후 약 7,200개로 늘어 안정된다 — 2014-2016년은 "
                "하루에 댓글 달린 금융 기사가 상위 10개를 채우지 못하는 날이 많아 적게 수집됐고, 2018년부터는 매일 "
                "10개(섹션별)를 모두 채운다. (우) 기사당 평균 수집 댓글 수(비어있지 않은 댓글 기준)는 2016년 8개에서 "
                "2021년 31개로 늘었다 — 수집 상한 때문이 아니라 금융 뉴스 기사당 실제 댓글이 증가했기 때문이다 "
                "(네이버 표시 댓글 수 중앙값 2개(2016) → 23개(2021)). 2020년 동학개미운동·2021년 공매도 재개 논란으로 "
                "금융 뉴스 댓글 참여가 폭증했다. 두 요인이 겹쳐 원천 댓글이 2014년 3.4만 → 2021년 31만으로 약 9배 늘었다."),
  "P8": dict(t="P-8. 2019년 경제 필터가 유독 많이 걸러낸 이유",
            x="", y="",
            cap="(좌) 경제 필터가 뽑는 주요 주식 키워드의 연도별 문서빈도(DF). 점선(0.5%)이 핵심 키워드 문턱이다. "
                "2019년에는 공매도·주식만 문턱을 넘고 금리·개미들·주가·기관은 모두 0.5% 아래로 떨어져 핵심 → 보조로 "
                "강등됐다. 통과 조건이 '점수 ≥ 10 이고 핵심 키워드 ≥ 1개'이므로, 핵심이 2개뿐이면 댓글에 '공매도'나 "
                "'주식'이 없으면 탈락한다 → 최종 잔존율 5.1%(다른 해 12~18%). (우) 2019년 크롤링된 증권·금융 뉴스의 "
                "주제 분포(댓글 수 가중). 은행·DLF 사태, 조국·검찰, 일본 수출규제, 연금·노동이 대부분이고 실제 주식·증시 "
                "기사는 소수다 — 개인 투자 담론(동학개미)은 2020년 3월 이후 시작됐고, 2019년 금융 뉴스 댓글은 "
                "주식시장 얘기가 아니었기 때문에 필터가 그만큼 많이 걸러낸 것이다."),
  "P7": dict(t="P-7. 연도별 빈(삭제) 댓글 비율과 제거율 차이",
            x="", y="",
            cap="(좌) 빈 댓글 비율: 2016년까지 0%에 가깝다가 2017-2019년 25~37%로 급등한 뒤 서서히 22%대로 낮아진다. "
                "2018년 드루킹 댓글조작 사건 이후 네이버가 댓글 시스템을 대폭 개편하고 AI 클린봇을 확대하면서 "
                "삭제된 댓글이 빈 자리(유효 comment_id + 빈 본문)로 남기 시작했고, 2016년 이전의 삭제 댓글은 이미 "
                "완전 삭제되어 수집 자체가 안 된다. (우) 경제 관련성 필터가 뽑는 연도별 핵심 키워드 수: 2019년은 2개로 "
                "가장 적어 최종 잔존율이 2.4%로 최저다. 2019년 금융 뉴스 댓글이 DLF 사태·조국 사모펀드·일본 수출규제 등 "
                "비(非)주식 경제·사회 이슈에 집중되어, 주식 논의 단어들이 연도별 DF 0.5% 문턱을 넘지 못했기 때문이다. "
                "핵심 키워드가 많아지는 2020-2021년(동학개미)은 잔존율도 10% 이상으로 회복한다."),
  "TOX": dict(t="독성 가중치의 설계와 강건성",
            x="", y="",
            cap="(좌) 가중 함수 w = (1 - tox)^γ 를 γ ∈ {0, 1, 2, 3, 5}에 대해 나타낸 것으로, 채택값 γ = 2에서 "
                "독성 점수 0.3, 0.5, 0.8인 댓글의 가중치는 각각 0.49, 0.25, 0.04이다(검은 점). 붉은 세로 점선은 "
                "hard drop 기준 τ = 0.95. (우) 독성 점수가 x 이상인 댓글이 일별 감성 가중치 총합에서 차지하는 비중"
                "(2014-2025년 표본). γ = 2 가중으로 인해 0.95 이상 댓글의 비중은 0.003%에 불과하며, "
                "일별 감성 지표는 hard drop 기준 설정에 사실상 불변이다."),
  "A1": dict(t="A-1. 독성 점수를 가중치로 변환하는 곡선  w = (1 - tox)^γ",
            x="독성 점수  toxicity_score   (0 = 정상,  1 = 확실한 독성)", y="댓글 가중치  weight",
            cap="독성 점수가 높을수록 그 댓글의 감성 집계 기여도(가중치)를 낮춘다. γ가 클수록 곡선이 아래로 휘어 "
                "중간 독성 댓글까지 빠르게 감쇠된다. 현재값 γ=2에서는 독성 0.3 → 가중치 0.49, 독성 0.5 → 0.25, "
                "독성 0.8 → 0.04 이다. γ=1(직선)은 감쇠가 약하고, γ=3 이상은 정보 손실이 크다."),
  "A2": dict(t="A-2. γ 를 바꿀 때 일별 감성지표가 무가중 대비 움직이는 폭",
            x="독성 가중치 지수  γ", y="일별 감성지표 평균절대편차  MAD  (무가중 γ=0 대비)",
            cap="세로축은 각 γ로 계산한 일별 가중감성 시계열이 '가중을 전혀 안 한 경우'와 하루 평균 얼마나 다른지를 "
                "나타낸다. γ=2에서 0.013, γ=5까지 올려도 0.018 로 완만하다. 감성 점수의 범위가 [-1, 1] 임을 감안하면 "
                "이 차이는 매우 작다. 즉 γ를 어디에 두든 일별 감성지표는 거의 같다."),
  "A3": dict(t="A-3. γ 에 따른 유효 표본 비율",
            x="독성 가중치 지수  γ", y="유효표본 / 전체표본   (ΣW)² / (Σ W² · N)",
            cap="가중을 하면 실질적으로 몇 개의 댓글을 쓰는 효과인지를 전체 대비 비율로 나타낸 값(1에 가까울수록 손실 없음). "
                "γ=2에서 0.85, γ=3에서 0.84. 대부분의 댓글이 저독성이라 가중해도 표본 손실은 15% 안쪽이며 γ를 더 키워도 "
                "추가 손실이 작다."),
  "A4": dict(t="A-4. γ 별 월평균 가중감성 시계열과 γ=1~3 폭",
            x="연도·월", y="월평균 가중감성  sent_norm_w",
            cap="위 칸: 실선은 γ=2(채택값)의 월평균 가중감성, 옅은 띠는 γ=1과 γ=3 사이 값의 범위(위 칸 축척에서는 "
                "거의 안 보일 만큼 좁다). 아래 칸: 그 폭(최대-최소)만 따로 확대해 그린 것으로, 전 구간 평균 0.0037, "
                "최댓값도 0.0143에 그친다 — 월평균 가중감성 자체의 변동폭(약 0.6)에 비해 1% 안팎이다. γ를 1~3 사이 "
                "어디로 바꾸어도 감성지표의 움직임이 거의 달라지지 않음을 보여준다."),
  "A5": dict(t="A-5. γ 별 댓글 가중치 분포",
            x="댓글 가중치  weight", y="댓글 수  (로그 스케일)",
            cap="대부분의 댓글은 독성이 낮아 γ와 무관하게 가중치가 1 부근에 몰려 있다. γ가 커질수록 "
                "낮은 가중치(왼쪽 꼬리)의 댓글이 늘어나지만 전체에서 차지하는 비중은 작다."),
  "B1": dict(t="B-1. 독성 hard-drop 임계값 τ 별 실제 제거율",
            x="독성 hard-drop 임계값  τ", y="전체 댓글 중 제거되는 비율 (%)",
            cap="τ 이상인 댓글을 완전히 버린다. 현재값 0.95에서 8.6%(30,313건)가 제거된다. "
                "0.90으로 낮추면 10.6%, 0.99로 올리면 5.1%. 임계값을 조금 옮겨도 제거량은 완만하게 변한다."),
  "B2": dict(t="B-2. 제거되는 댓글의 독성점수 구간별 개수",
            x="독성 점수 구간", y="댓글 수",
            cap="현재 임계값(0.95) 위쪽 구간별로 제거되는 댓글 수. 0.99–1.00 구간에 17,947건으로 가장 많다. "
                "임계값을 0.97·0.99로 올리면 이 구간의 명백한 독성 댓글이 감성 집계에 들어오게 된다."),
  "B3": dict(t="B-3. 독성 tox ≥ x 인 댓글이 차지하는 감성 가중치 총량 비중",
            x="독성 점수 하한  x", y="Σ weight(tox ≥ x) / Σ weight 전체  (%)",
            cap="핵심 그림. γ=2 가중 때문에 고독성 댓글은 가중치가 거의 0이다. 독성 0.95 이상 댓글을 모두 합쳐도 "
                "전체 감성 가중치의 0.003% 에 불과하고, 0.90 이상도 0.015% 뿐이다. 따라서 τ를 어디에 두든 "
                "일별 감성지표와 전략 성과는 사실상 바뀌지 않는다."),
  "B4": dict(t="B-4. 제거 이전 전체 댓글의 독성점수 분포",
            x="독성 점수  toxicity_score", y="댓글 수  (로그 스케일)",
            cap="독성 점수는 0 부근(정상)과 1 부근(확실한 독성)으로 갈리는 양극 분포다. 0.4–0.9 사이 애매한 구간은 "
                "상대적으로 희소하다. 이 때문에 임계값을 이 구간 어디에 두어도 제거 대상이 크게 달라지지 않는다."),
  "B5": dict(t="B-5. τ 를 낮출 때 일별 감성지표의 변화 (τ = 0.95 대비)",
            x="독성 hard-drop 임계값  τ", y="일별 감성지표 MAD  (τ = 0.95 대비)",
            cap="임계값을 0.95에서 0.90까지 낮추면 MAD 0.00014, 0.80까지 낮추면 0.00032로, 더 많은 댓글을 제거해도 "
                "일별 감성지표는 [-1,1] 척도에서 무시할 만큼만 달라진다. (감성확률은 통과 댓글에만 있어 "
                "'더 엄격하게' 방향만 시뮬레이션 가능. '더 느슨하게' 방향은 B-3 의 가중치 비중으로 판단.)"),
  "B6": dict(t="B-6. 연도별 독성 hard-drop(τ = 0.95) 대상 비율",
            x="연도", y="tox ≥ 0.95 비율 (%)",
            cap="2014–2017년 8.7–9.6% 수준이던 hard-drop 대상 비율이 2018–2019년 13.2–13.7%로 일시 상승했다가"
                "(강조), 2020년 이후 다시 9% 안팎으로, 2023–2025년에는 7–8%대로 낮아졌다. 전 기간 최댓값(13.7%, "
                "2018)도 15% 미만이다."),
  "C1": dict(t="C-1. 한글 토큰 길이 분포와 '주식·경제 관련' 비율",
            x="토큰 길이 (글자 수)", y="토큰 등장 횟수  (막대, 로그 스케일)",
            cap="막대는 길이별 토큰 등장 횟수, 빨간 선은 그 길이 토큰 중 주식·경제 패턴에 매칭되는 비율(오른쪽 축). "
                "길이가 길수록 관련성 비율은 높아지지만 개수는 4글자 이후 급감한다. 노란 음영이 현재 추출 범위(2–6글자)."),
  "C2": dict(t="C-2. 키워드 추출 길이창 별 핵심 키워드 수",
            x="키워드 추출 길이창  (최소–최대 글자)", y="핵심 키워드 개수",
            cap="추출 길이 범위를 바꿨을 때 핵심 키워드로 뽑히는 단어 수. 2–6 과 2–8 이 동일하고(상한 확대 효과 없음), "
                "3–6 은 크게 줄어든다(2글자 금융어 상실). 하한 1 은 2와 사실상 같다(1글자 금융어가 거의 없음)."),
  "D1": dict(t="D-1. 핵심 키워드 DF 임계값 별 키워드 수 (연평균)",
            x="핵심 키워드 DF 임계값 (%)", y="키워드 개수 (연평균)",
            cap="문서빈도(DF, 해당 단어가 등장한 댓글 비율)가 임계값 이상이면 핵심, 그 30~100% 구간이면 보조 키워드. "
                "임계값을 올리면 키워드가 빠르게 줄어든다(연평균 핵심 키워드: 0.5% → 약 6개, 1% → 약 2개). "
                "1%에서는 전 연도에서 '공매도·주식·금리'만 남아 '주가·코스피·환율'이 빠진다. 실제 실행값 0.5%. "
                "(연도별 4만 건 층화표본)"),
  "D2": dict(t="D-2. 핵심 키워드 DF 임계값 별 경제 필터 통과율",
            x="핵심 키워드 DF 임계값 (%)", y="경제 필터 통과율 (%)",
            cap="DF 임계값이 통과율에 미치는 영향(층화표본 재추출 기준): 0.5% → 약 13%, 1% → 약 7%, 2% → 약 2%. "
                "1% 이상으로 올리면 연도에 따라 핵심 키워드가 0~2개로 줄어 통과율이 급감한다. 실제 실행값은 0.5%이며, "
                "F·G의 통과율(15.1%)은 이 실행의 실제 stock_score·is_stock 컬럼(2014-2025 전체)을 사용한 값이다."),
  "D3": dict(t="D-3. 보조 키워드 하한 배수 별 보조 키워드 수와 통과율",
            x="보조 키워드 하한 = DF 임계값 × 배수", y="지표값",
            cap="보조 키워드는 'DF 임계값의 (배수)배 ~ DF 임계값' 구간의 단어다. 현재값 0.3 은 DF 0.5% × 0.3 = 0.15% ~ 0.5% 구간. "
                "배수를 0.1로 낮추면 보조 키워드가 급증하지만(희소어까지 포함) 통과율은 거의 변하지 않는다. "
                "핵심 키워드가 이미 통과 여부를 결정하기 때문이며, 0.3 은 보조 키워드를 '자주 쓰이지만 핵심 미만'인 단어로 한정하는 무난한 값이다."),
  "F1": dict(t="F-1. 경제성 점수 임계값 별 통과율  (근거: 평탄 구간)",
            x="경제성 점수 임계값  score cutoff", y="경제 필터 통과율 (%)",
            cap="핵심 그림. 점수 = 10·(핵심 키워드 수) + 3·(보조 키워드 수). 임계값을 3·5·7·10 으로 두면 통과율이 "
                "15.1% 로 완전히 같고(음영 구간), 13 에서 10.6% 로 급락한다. 이 급락은 '핵심 키워드를 정확히 하나만 "
                "가진 댓글(점수 10)' 이 잘려나가면서 생긴다. 10 은 그 평탄 구간의 끝이다."),
  "F2": dict(t="F-2. 댓글의 경제성 점수 분포",
            x="댓글의 경제성 점수  stock_score", y="댓글 수  (로그 스케일)",
            cap="점수 0(비관련)에 큰 봉우리, 점수 10(핵심 키워드 하나)에 뚜렷한 두 번째 봉우리가 있다. "
                "임계값 10 은 이 두 번째 봉우리를 포함하고, 13 은 배제한다."),
  "E1": dict(t="E-1. 핵심 : 보조 점수 가중치 별 경제 필터 통과율",
            x="핵심 점수 : 보조 점수", y="경제 필터 통과율 (%)",
            cap="핵심 1개 = 핵심 점수가 그대로 게이트(임계값 10)를 넘는지가 통과를 결정한다. 핵심 점수가 10이면 "
                "보조 점수를 2·3·5·10 어느 것으로 바꿔도 통과율이 동일하고, 핵심 점수를 5·3 으로 낮추면 "
                "핵심어 1개로는 게이트를 못 넘어 통과율이 급락한다(각각 1.7%·0.7%). 즉 '핵심 10점'은 필수, '보조 3점'은 "
                "통과된 댓글의 순위에만 쓰이는 값이다."),
  "G1": dict(t="G-1. '핵심 키워드 ≥ 1개' 조건의 유무에 따른 통과율",
            x="통과 조건", y="경제 필터 통과율 (%)",
            cap="점수 ≥ 10 만 요구하면 통과율 15.25%, 여기에 '핵심 키워드 최소 1개' 조건을 추가하면 15.10%. "
                "그 차이는 '보조 키워드만 4개 이상·핵심 0개'로 점수 10을 넘긴 댓글들이며, 실측 1,858건(통과분의 1.0%)이다. "
                "이들은 금융 핵심어 없이 사회·정치 맥락에서 '시장·투자' 등이 스치는 글이라, 이 조건이 걸러낸다."),
  "H1": dict(t="H-1. 독성 가중치 지수 γ 에 따른 전략 위험조정성과",
            x="독성 가중치 지수  γ", y="지표값",
            cap="γ 1·2·3 각각으로 댓글 → 일별 감성피처 → K-FGI → 포지션 → 전략수익률을 10년 전 구간 재산출한 결과. "
                "Sharpe 0.610 / 0.613 / 0.622, Calmar 0.36 부근으로 사실상 평평하다. γ=2 는 성과 최적값이 아니다."),
  "H2": dict(t="H-2. γ 에 따른 전략 하방위험 지표",
            x="독성 가중치 지수  γ", y="지표값  (MDD·CVaR 는 음수)",
            cap="MDD −22.9%, CVaR(5%) −2.13%, 하방 변동성 0.114 로 γ 1→3 에서 소수 셋째 자리까지 변화가 없다. "
                "독성 가중치 지수는 전략의 하방위험에 영향을 주지 않는다."),
  "I1": dict(t="I-1. 감성 복합지표 이동평균 기간에 따른 전략 성과",
            x="감성 복합지표 이동평균 기간 (거래일)", y="지표값",
            cap="sent_composite 를 5·10·20 거래일로 평활한 각 경우의 전략 성과. 10일이 Sharpe(0.61)·Calmar(0.36)·"
                "MDD(−22.9%) 모두에서 최적인 뚜렷한 국소 최적점이다. 5일은 노이즈, 20일은 반응 지연으로 나빠진다. "
                "(τ·γ 와 달리 이 값은 결과에 민감하므로 반드시 근거가 필요하다.)"),
 },
 "en": {
  "P1": dict(t="P-1. Filtering results by stage  (final 2014-2025 sample, total)",
            x="", y="",
            cap="Starting from 2,150,215 raw crawled comments. Empty/duplicate removal leaves 1,575,268 (73.3%) as "
                "political-filter input; stage 1 political-keyword filtering removes 165,288 (10.5% of input), stage 2 "
                "toxicity filtering (τ=0.95) removes 131,337 (9.3% of political-passed), and stage 3 economic-relevance "
                "filtering removes 1,085,526 (84.9% of toxicity-passed). The final economy/stock comment set is 193,117 "
                "(9.0% of raw), of which 192,888 were sentiment-scored. (Same figures as the paper's §3.3 filtering table; "
                "per-year detail is in the Appendix / `10_Paper_outputs/04_투고_전략/FINAL_REVISION_MASTER_KR.md`.)"),
  "P2": dict(t="P-2. Comments retained by filtering stage (funnel, 2014-2025 total)",
            x="filtering stage", y="comments retained",
            cap="Each bar is the number of comments left after that stage: 2.15M raw → 1.58M political input → 1.41M "
                "political-passed → 1.28M toxicity-passed → 193k final. The largest drop is stage 3 economic-relevance "
                "filtering (84.9% of the toxicity-passed set), which removes the large volume of ordinary comments "
                "unrelated to stocks/economy."),
  "P3": dict(t="P-3. Stage 1 political-keyword filtering results (2014-2025)",
            x="", y="",
            cap="Left: share of political comments by year — around 4–6% in 2014–2016, above 10% from 2018, and 15.0% in "
                "2025 around the impeachment / martial-law period. Right: the top 16 matched political keywords "
                "(2014-2025 total). Early years are dominated by 박근혜/근혜, later years by 민주당/문재인/재명/이재명/윤석열/석열/탄핵. "
                "These politician/party/event terms inject a political framing bias into the sentiment aggregation, so they are removed."),
  "P4": dict(t="P-4. Filtering results by year and stage (2014-2025)",
            x="", y="",
            cap="Paper appendix table. Raw comment volume grew to 200k–310k per year from 2018; the final "
                "economy/stock comment count is highest (~30k) in years with market controversies (2021 short-selling, "
                "the 2023+ '국장' debate). The final retention rate ranges 4–13% per year, averaging 9.0%."),
  "P5": dict(t="P-5. Comment composition by year and filtering stage",
            x="year", y="number of comments",
            cap="Each year's bar decomposed into removed/retained by stage: from the bottom, final retained "
                "(economy/stock), economic-irrelevant removed, toxicity removed, political removed, empty/duplicate. "
                "In every year the 'economic-irrelevant removed' segment (grey) is the largest."),
  "P6": dict(t="P-6. Why the raw comment count differs by year",
            x="", y="",
            cap="Each day the crawler picks the top 10 articles by comment count in each of the securities and finance "
                "sections, and collects all of that day's comments for each (Naver's comment API returns at most ~200 "
                "per article; fewer than 1% of articles hit this). (Left) collected articles grew from 3,651 (2014) to "
                "~7,200 from 2018 — in 2014–2016 there were often fewer than 10 commented financial articles to fill the "
                "top-10 per section; from 2018 the top-10 is filled every day. (Right) the mean non-empty comments "
                "collected per article rose from 8 (2016) to 31 (2021) — not because of a collection cap but because the "
                "actual comment volume per financial-news article grew (Naver's displayed comment count, median: 2 in "
                "2016 → 23 in 2021), driven by the 2020 retail-investing boom and the 2021 short-selling controversy. "
                "Together, raw comments grew ~9× from 34k (2014) to 311k (2021)."),
  "P8": dict(t="P-8. Why the economic filter removed so much in 2019",
            x="", y="",
            cap="(Left) year-by-year document frequency (DF) of the main stock keywords the economic filter extracts. "
                "The dashed line (0.5%) is the core-keyword threshold. In 2019 only '공매도' (short selling) and '주식' "
                "(stock) clear it; '금리', '개미들', '주가', '기관' all fall below 0.5% and are demoted from core to "
                "auxiliary. Since passage requires 'score ≥ 10 and at least one core keyword', with only two core "
                "keywords a comment must literally contain '공매도' or '주식' → final retention 5.1% (vs 12–18% in "
                "other years). (Right) topic distribution of the securities/finance news crawled in 2019 "
                "(comment-weighted). Bank/DLF mis-selling, the Cho Kuk prosecution, the Japan export controls, pensions "
                "and labour dominate; actual stock-market articles are a minority — retail-investing discourse began "
                "only after March 2020, so 2019 financial-news comments were not about the stock market and the filter "
                "removed them accordingly."),
  "P7": dict(t="P-7. Empty (deleted) comment rate and variation in removal rate by year",
            x="", y="",
            cap="(Left) empty-comment rate: near 0% through 2016, spiking to 25–37% in 2017–2019, then easing to ~22%. "
                "After the 2018 comment-manipulation scandal Naver overhauled its comment system and expanded its AI "
                "filter; deleted comments now remain as empty slots (valid comment_id, empty body), whereas pre-2016 "
                "deletions were fully purged and never collected. (Right) number of core keywords the economic-relevance "
                "filter extracts per year: 2019 has the fewest (2), giving the lowest final retention (2.4%). In 2019, "
                "financial-news comments concentrated on non-stock economic/social issues (the DLF mis-selling scandal, "
                "the Cho Kuk private-equity affair, the Japan export controls), so stock-discussion words fell below the "
                "1% document-frequency threshold. Retention recovers above 10% in 2020–2021 (the retail boom) as more "
                "core keywords appear."),
  "TOX": dict(t="Design and robustness of the toxicity weighting",
            x="", y="",
            cap="(left) The weighting function w = (1 - tox)^γ for γ ∈ {0, 1, 2, 3, 5}; at the chosen γ = 2, comments with "
                "toxicity 0.3, 0.5, 0.8 receive weights 0.49, 0.25, 0.04 respectively (black dots; red dashed line: "
                "hard-drop threshold τ = 0.95). (right) Share of the total daily-sentiment weight contributed by comments "
                "with toxicity ≥ x (2014-2025 sample). Because the γ = 2 weighting already suppresses toxic "
                "comments, those above 0.95 hold only 0.003% of the aggregate weight, so the daily sentiment index is "
                "effectively invariant to the hard-drop threshold."),
  "A1": dict(t="A-1. Curve mapping toxicity score to weight   w = (1 - tox)^γ",
            x="toxicity_score   (0 = clean,  1 = certainly toxic)", y="comment weight",
            cap="A more toxic comment gets a lower weight in the sentiment aggregation. Larger γ bends the curve down, "
                "attenuating even mid-toxicity comments quickly. At the current γ=2: toxicity 0.3 → weight 0.49, "
                "0.5 → 0.25, 0.8 → 0.04. γ=1 (straight line) attenuates too weakly; γ≥3 discards too much."),
  "A2": dict(t="A-2. How far the daily sentiment index moves as γ changes",
            x="toxicity weighting exponent  γ", y="mean abs. deviation of daily sentiment  (vs. unweighted, γ=0)",
            cap="The y-axis is how much the daily weighted-sentiment series differs, per day on average, from the "
                "no-weighting case. It is 0.013 at γ=2 and only 0.018 at γ=5. Given that the sentiment score ranges "
                "over [-1, 1], this is tiny: the daily index is nearly identical for any γ."),
  "A3": dict(t="A-3. Effective sample fraction by γ",
            x="toxicity weighting exponent  γ", y="effective N / total N   (ΣW)² / (Σ W² · N)",
            cap="How many comments the weighting effectively uses, as a fraction of the total (1 = no loss). "
                "0.85 at γ=2, 0.84 at γ=3. Most comments have low toxicity, so weighting costs under 15% of the "
                "sample and raising γ further adds little."),
  "A4": dict(t="A-4. Monthly mean weighted sentiment by γ, and the γ=1-3 gap",
            x="year-month", y="monthly mean weighted sentiment  sent_norm_w",
            cap="Top: the solid line is the monthly weighted sentiment at γ=2 (adopted); the shaded band is the range "
                "spanned by γ=1 and γ=3 (too narrow to see at this scale). Bottom: that same gap (max-min), plotted "
                "on its own axis — averaging 0.0037 and peaking at 0.0143, about 1% of the index's own swing "
                "(~0.6). The index barely moves for any γ between 1 and 3."),
  "A5": dict(t="A-5. Distribution of comment weights by γ",
            x="comment weight", y="number of comments  (log scale)",
            cap="Most comments have low toxicity, so their weight sits near 1 regardless of γ. Larger γ moves more "
                "comments into the low-weight left tail, but they remain a small share of the whole."),
  "B1": dict(t="B-1. Actual removal rate by toxicity hard-drop threshold τ",
            x="toxicity hard-drop threshold  τ", y="share of all comments removed (%)",
            cap="Comments with score ≥ τ are dropped entirely. At the current 0.95, 8.6% (30,313 comments) are "
                "removed; 10.6% at 0.90, 5.1% at 0.99. Moving the threshold changes the removed volume only gradually."),
  "B2": dict(t="B-2. Count of removed comments by toxicity-score band",
            x="toxicity-score band", y="number of comments",
            cap="Removed comments by band above the current threshold. The 0.99–1.00 band is largest (17,947). "
                "Raising the threshold to 0.97 or 0.99 would let the clearly toxic comments of these bands into the "
                "sentiment aggregation."),
  "B3": dict(t="B-3. Weight mass held by comments with toxicity ≥ x",
            x="toxicity-score lower bound  x", y="Σ weight(tox ≥ x) / Σ weight total  (%)",
            cap="The key figure. Because of the γ=2 weighting, highly toxic comments carry almost no weight. All "
                "comments with toxicity ≥ 0.95 together hold just 0.003% of the total sentiment weight, and ≥ 0.90 "
                "only 0.015%. The daily index and strategy performance therefore barely change wherever τ is set."),
  "B4": dict(t="B-4. Toxicity-score distribution of all comments before removal",
            x="toxicity_score", y="number of comments  (log scale)",
            cap="The score is bimodal — near 0 (clean) and near 1 (clearly toxic) — with the ambiguous 0.4–0.9 range "
                "relatively sparse. As a result, the set of removed comments hardly changes wherever the threshold "
                "sits in that range."),
  "B5": dict(t="B-5. Change in the daily sentiment index as τ is lowered (vs. τ = 0.95)",
            x="toxicity hard-drop threshold  τ", y="daily sentiment MAD  (vs. τ = 0.95)",
            cap="Lowering the threshold from 0.95 to 0.90 moves the daily sentiment index by 0.00014 on average, and "
                "to 0.80 by 0.00032 — negligible on the [-1,1] scale even as many more comments are removed. "
                "(Sentiment probabilities exist only for kept comments, so only the 'stricter' direction is "
                "simulable; for the 'looser' direction see B-3.)"),
  "B6": dict(t="B-6. Share of comments subject to hard-drop (τ = 0.95) by year",
            x="year", y="share with tox ≥ 0.95 (%)",
            cap="The hard-drop share ran 8.7-9.6% in 2014-2017, spiked to 13.2-13.7% in 2018-2019 (highlighted), "
                "then fell back to roughly 9% and, by 2023-2025, to 7-8%. Even the all-time peak (13.7%, 2018) "
                "stays under 15%."),
  "C1": dict(t="C-1. Korean token-length distribution and 'finance-related' share",
            x="token length (characters)", y="token occurrences  (bars, log scale)",
            cap="Bars: token occurrences by length. Red line (right axis): the share of those tokens matching the "
                "stock/economy pattern. Longer tokens are more often finance-related, but their count collapses "
                "after 4 characters. The yellow band marks the current extraction range (2–6 characters)."),
  "C2": dict(t="C-2. Core-keyword count by extraction length window",
            x="keyword extraction length window  (min–max characters)", y="number of core keywords",
            cap="Number of words selected as core keywords under each length window. 2–6 equals 2–8 (raising the "
                "upper bound adds nothing), while 3–6 drops sharply (losing 2-character financial terms). Lower "
                "bound 1 is essentially the same as 2 (almost no single-character financial words)."),
  "D1": dict(t="D-1. Keyword count by core-keyword DF threshold (annual mean)",
            x="core-keyword DF threshold (%)", y="number of keywords (annual mean)",
            cap="A word is a core keyword if its document frequency is at least the threshold, and an auxiliary keyword "
                "in the 30–100% band below it. Raising the threshold shrinks the set quickly (annual mean core keywords: "
                "~6 at 0.5%, ~2 at 1%). At 1%, every year keeps only 'short-selling / stock / interest rate', dropping "
                "'share price / KOSPI / exchange rate'. Real run value: 0.5% (40k-per-year stratified sample)."),
  "D2": dict(t="D-2. Economic-filter pass rate by core-keyword DF threshold",
            x="core-keyword DF threshold (%)", y="economic-filter pass rate (%)",
            cap="Effect of the DF threshold on the pass rate (re-extraction on the stratified sample): ~13% at 0.5%, "
                "~7% at 1%, ~2% at 2%. Above 1%, the number of core keywords collapses to 0–2 per year. The real run "
                "used 0.5%; the F/G pass rate (15.1%) uses that run's actual stock_score / is_stock columns (full 2014-2025)."),
  "D3": dict(t="D-3. Auxiliary-keyword count and pass rate by the auxiliary lower-bound multiplier",
            x="auxiliary lower bound = DF threshold × multiplier", y="metric value",
            cap="An auxiliary keyword sits in the band 'DF threshold × multiplier ... DF threshold'. The current 0.3 "
                "means 0.3%–1%. Lowering the multiplier to 0.1 sharply increases the auxiliary-keyword count (adding "
                "rare words) but barely moves the pass rate, because the core keywords already decide "
                "passage. 0.3 keeps auxiliary keywords to 'frequent but below core' words."),
  "F1": dict(t="F-1. Pass rate by economic-relevance score cutoff  (rationale: the plateau)",
            x="economic-relevance score cutoff", y="economic-filter pass rate (%)",
            cap="The key figure. score = 10·(core keyword count) + 3·(auxiliary keyword count). For cutoffs 3, 5, 7, "
                "10 the pass rate is identical at 15.1% (shaded band); at 13 it drops sharply to 10.6%. That drop is "
                "the comments with exactly one core keyword (score 10) being cut. 10 is the end of the plateau."),
  "E1": dict(t="E-1. Economic-filter pass rate by core:auxiliary score weighting",
            x="core score : auxiliary score", y="economic-filter pass rate (%)",
            cap="Passage is decided by whether one core keyword (= the core score) clears the gate (cutoff 10). "
                "With a core score of 10, changing the auxiliary score to 2·3·5·10 leaves the pass rate unchanged; "
                "lowering the core score to 5 or 3 means one core keyword no longer clears the gate and the pass "
                "rate collapses (to 1.7% and 0.7%). So 'core = 10' is essential; 'auxiliary = 3' only ranks the comments that pass."),
  "G1": dict(t="G-1. Pass rate with and without the 'core keyword ≥ 1' condition",
            x="pass condition", y="economic-filter pass rate (%)",
            cap="Requiring only score ≥ 10 gives a 15.25% pass rate; adding 'at least one core keyword' gives 15.10%. "
                "The difference is the comments that cleared score 10 on '4+ auxiliary keywords, 0 core' — 1,858 "
                "(1.0% of passers). These are social/political posts where 'market/investment' appear in passing, without "
                "a financial core keyword, and this condition removes them."),
  "F2": dict(t="F-2. Distribution of comment economic-relevance scores",
            x="comment economic-relevance score  stock_score", y="number of comments  (log scale)",
            cap="A large peak at score 0 (not relevant) and a distinct second peak at score 10 (one core keyword). "
                "Cutoff 10 includes this second peak; cutoff 13 excludes it."),
  "H1": dict(t="H-1. Strategy risk-adjusted performance by toxicity weighting exponent γ",
            x="toxicity weighting exponent  γ", y="metric value",
            cap="Result of re-deriving comment → daily sentiment features → K-FGI → position → strategy return over "
                "the full 10 years for γ = 1, 2, 3. Sharpe 0.610 / 0.613 / 0.622 and Calmar near 0.36 — essentially "
                "flat. γ=2 is not a performance optimum."),
  "H2": dict(t="H-2. Strategy downside-risk metrics by γ",
            x="toxicity weighting exponent  γ", y="metric value  (MDD, CVaR are negative)",
            cap="MDD −22.9%, CVaR (5%) −2.13%, downside volatility 0.114 — unchanged to three decimals from γ 1 to 3. "
                "The toxicity weighting exponent does not affect the strategy's downside risk."),
  "I1": dict(t="I-1. Strategy performance by sentiment-composite moving-average window",
            x="sentiment-composite moving-average window (trading days)", y="metric value",
            cap="Strategy performance when sent_composite is smoothed over 5, 10 or 20 trading days. 10 days is a "
                "clear local optimum on Sharpe (0.61), Calmar (0.36) and MDD (−22.9%). "
                "5 days is noisy, 20 days lags. (Unlike τ and γ, this value does move the result, so it needs an explicit rationale.)"),
 },
}


# ============================================================ 그림
def _new(lang, w=7.6, h=4.6):
    use_style(lang)
    return plt.subplots(figsize=(w, h))


def _pipe_totals(R):
    P = R["PIPE"]
    return {k: sum(p[k] for p in P) for k in
            ["raw", "pol_in", "pol_removed", "pol_pass", "tox_removed", "tox_pass",
             "econ_removed", "final"]}


def fig_P1(R, lang):
    """단계별 처리 결과 표 (2014-2025 합계) — 논문 3.3 필터링 표."""
    T = _pipe_totals(R); raw = T["raw"]; sok = R["SENT_OK"]
    S = ({
     "ko": [("원천 수집 댓글", T["raw"], "", "100.0%"),
            ("정치 필터 분류 입력 (빈 댓글·중복 제거)", T["pol_in"], "", f"{T['pol_in']/raw*100:.1f}%"),
            ("1단계 · 정치 키워드 제거", T["pol_removed"], f"{T['pol_removed']/T['pol_in']*100:.1f}%", ""),
            ("1단계 · 정치 필터 통과", T["pol_pass"], "", f"{T['pol_pass']/raw*100:.1f}%"),
            ("2단계 · 독성 필터 제거 (τ=0.95)", T["tox_removed"], f"{T['tox_removed']/T['pol_pass']*100:.1f}%", ""),
            ("2단계 · 독성 필터 통과", T["tox_pass"], "", f"{T['tox_pass']/raw*100:.1f}%"),
            ("3단계 · 경제 관련성 필터 제거", T["econ_removed"], f"{T['econ_removed']/T['tox_pass']*100:.1f}%", ""),
            ("3단계 · 최종 경제/주식 관련 댓글", T["final"], "", f"{T['final']/raw*100:.1f}%"),
            ("감성 점수 산출 성공", sok, "", f"{sok/raw*100:.1f}%")],
     "en": [("raw crawled comments", T["raw"], "", "100.0%"),
            ("political-filter input (empty & dup removed)", T["pol_in"], "", f"{T['pol_in']/raw*100:.1f}%"),
            ("stage 1 · political keywords removed", T["pol_removed"], f"{T['pol_removed']/T['pol_in']*100:.1f}%", ""),
            ("stage 1 · political filter passed", T["pol_pass"], "", f"{T['pol_pass']/raw*100:.1f}%"),
            ("stage 2 · toxicity removed (τ=0.95)", T["tox_removed"], f"{T['tox_removed']/T['pol_pass']*100:.1f}%", ""),
            ("stage 2 · toxicity filter passed", T["tox_pass"], "", f"{T['tox_pass']/raw*100:.1f}%"),
            ("stage 3 · economic-relevance removed", T["econ_removed"], f"{T['econ_removed']/T['tox_pass']*100:.1f}%", ""),
            ("stage 3 · final economy/stock comments", T["final"], "", f"{T['final']/raw*100:.1f}%"),
            ("sentiment scoring succeeded", sok, "", f"{sok/raw*100:.1f}%")],
    })[lang]
    hdr = (["단계", "댓글 수", "이전 단계 대비 제거율", "최초 입력 대비 잔존율"] if lang == "ko"
           else ["stage", "comments", "removed vs. prev", "retained vs. raw"])
    rows = [[s[0], f"{s[1]:,}", s[2] or "—", s[3] or "—"] for s in S]

    use_style(lang)
    fig, ax = plt.subplots(figsize=(10.8, 3.6))
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=hdr, loc="center", cellLoc="center",
                  colWidths=[0.46, 0.16, 0.19, 0.19])
    tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.55)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#c9ccd3")
        if r == 0:
            cell.set_facecolor("#eef0fb"); cell.set_text_props(weight="bold")
        elif rows[r - 1][0].endswith(("통과", "passed")) or "final" in rows[r - 1][0] or "최종" in rows[r - 1][0]:
            cell.set_facecolor("#f2f6f3")
        if c == 0:
            cell.set_text_props(ha="left"); cell._loc = "left"
    _T(ax, LB[lang]["P1"]["t"], pad=12)
    return fig


def fig_P2(R, lang):
    T = _pipe_totals(R)
    keys = ["raw", "pol_in", "pol_pass", "tox_pass", "final"]
    labs = (["원천 수집", "정치 분류 입력", "1단계 · 정치 통과", "2단계 · 독성 통과", "3단계 · 최종(경제)"]
            if lang == "ko" else
            ["raw", "political input", "stage 1 passed", "stage 2 passed", "stage 3 final"])
    vals = [T[k] for k in keys]
    use_style(lang)
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    colors = ["#5b6472", "#7b8794", "#b48a3a", "#b0433a", "#2e7d55"]
    ax.bar(range(len(vals)), vals, color=colors, width=.62)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:,}\n({v / vals[0] * 100:.1f}%)", (i, v), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labs, rotation=16, ha="right", fontsize=9.5)
    ax.margins(y=0.18)
    L = LB[lang]["P2"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig


def fig_P3(R, lang):
    rr = R["POL_RATIO"]; kw = R["POL_KW"]
    use_style(lang)
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.8), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1.15, 1]})
    yrs = [str(r["year"]) for r in rr]
    vals = [r["political_ratio"] for r in rr]
    ax[0].bar(yrs, vals, color="#b0433a", width=.62)
    for i, v in enumerate(vals):
        ax[0].annotate(f"{v:.1f}", (i, v), xytext=(0, 4), textcoords="offset points",
                       ha="center", fontsize=8.5)
    ax[0].margins(y=0.18)
    for lab in ax[0].get_xticklabels():
        lab.set_rotation(40); lab.set_ha("right")
    _T(ax[0], "연도별 정치 관련 댓글 비율" if lang == "ko" else "Political-comment share by year")
    ax[0].set_ylabel("정치 관련 댓글 비율 (%)" if lang == "ko" else "political-comment share (%)")
    ax[0].set_xlabel("연도" if lang == "ko" else "year")

    words = [w for w, _ in kw][::-1]
    counts = [c for _, c in kw][::-1]
    ax[1].barh(range(len(words)), counts, color="#4c72b0")
    ax[1].set_yticks(range(len(words)))
    ax[1].set_yticklabels(words, fontsize=9.5, fontfamily=["AppleGothic", "DejaVu Sans"])
    for i, c in enumerate(counts):
        ax[1].annotate(f"{c:,}", (c, i), xytext=(4, 0), textcoords="offset points",
                       va="center", fontsize=8)
    _T(ax[1], "정치 키워드 상위 16개 (2014-2025 합계)" if lang == "ko"
                    else "Top 16 political keywords (2014-2025 total)")
    ax[1].set_xlabel("매칭된 댓글 수" if lang == "ko" else "matched comments")
    ax[1].margins(x=0.14)
    _T(fig, LB[lang]["P3"]["t"], sup=True, fontsize=13)
    return fig


def fig_P4(R, lang):
    P = R["PIPE"]
    hdr = (["연도", "원천 댓글", "정치 제거", "정치 통과", "독성 제거", "독성 통과", "경제무관 제거", "최종 댓글", "최종 잔존율"]
           if lang == "ko" else
           ["year", "raw", "political rm", "political pass", "toxicity rm", "toxicity pass",
            "econ-irrel. rm", "final", "final ret."])
    rows = []
    for p in P:
        rows.append([str(p["year"]), f"{p['raw']:,}", f"{p['pol_removed']:,}", f"{p['pol_pass']:,}",
                     f"{p['tox_removed']:,}", f"{p['tox_pass']:,}", f"{p['econ_removed']:,}",
                     f"{p['final']:,}", f"{p['final'] / p['raw'] * 100:.1f}%"])
    T = _pipe_totals(R)
    rows.append([("합계" if lang == "ko" else "total"), f"{T['raw']:,}", f"{T['pol_removed']:,}",
                 f"{T['pol_pass']:,}", f"{T['tox_removed']:,}", f"{T['tox_pass']:,}",
                 f"{T['econ_removed']:,}", f"{T['final']:,}", f"{T['final'] / T['raw'] * 100:.1f}%"])
    use_style(lang)
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=hdr, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8.8); tb.scale(1, 1.42)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#c9ccd3")
        if r == 0:
            cell.set_facecolor("#eef0fb"); cell.set_text_props(weight="bold")
        elif r == len(rows):
            cell.set_facecolor("#e7f2eb"); cell.set_text_props(weight="bold")
    _T(ax, LB[lang]["P4"]["t"], pad=12)
    return fig


def fig_P5(R, lang):
    P = R["PIPE"]
    yrs = [str(p["year"]) for p in P]
    empty = [p["raw"] - p["pol_in"] for p in P]
    polr = [p["pol_removed"] for p in P]
    toxr = [p["tox_removed"] for p in P]
    econr = [p["econ_removed"] for p in P]
    fin = [p["final"] for p in P]
    use_style(lang)
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    segs = ([("최종 유지 (경제/주식)", fin, "#2e7d55"), ("경제 무관 제거", econr, "#9aa2ad"),
             ("독성 제거", toxr, "#b0433a"), ("정치 제거", polr, "#b48a3a"),
             ("빈 댓글·중복", empty, "#5b6472")] if lang == "ko" else
            [("final retained", fin, "#2e7d55"), ("econ-irrelevant removed", econr, "#9aa2ad"),
             ("toxicity removed", toxr, "#b0433a"), ("political removed", polr, "#b48a3a"),
             ("empty / duplicate", empty, "#5b6472")])
    bottom = [0] * len(yrs)
    for name, vals, col in segs:
        ax.bar(yrs, vals, bottom=bottom, color=col, width=.68, label=name)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    for lab in ax.get_xticklabels():
        lab.set_rotation(35); lab.set_ha("right")
    ax.margins(y=0.16)
    L = LB[lang]["P5"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig


def fig_P6(R, lang):
    P = R["PIPE"]; yrs = [str(p["year"]) for p in P]
    use_style(lang)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True)
    ax[0].bar(yrs, [p["news_list"] for p in P], color="#5b6472", width=.62)
    ax[0].bar(yrs, [p["raw"] / 1000 for p in P], color="#b48a3a", width=.62, alpha=.0)  # keep scale
    a0b = ax[0].twinx(); a0b.grid(False)
    a0b.plot(yrs, [p["raw"] for p in P], "o-", color="#b0433a", lw=1.8, label="원천 댓글 수" if lang == "ko" else "raw comments")
    _T(ax[0], "수집 기사 수 (막대) vs 원천 댓글 수 (선)" if lang == "ko"
                    else "collected articles (bars) vs raw comments (line)")
    ax[0].set_ylabel("수집 기사 수" if lang == "ko" else "collected articles")
    a0b.set_ylabel("원천 댓글 수" if lang == "ko" else "raw comments")
    ax[0].set_xlabel("연도" if lang == "ko" else "year")
    a0b.legend(loc="upper left", fontsize=9)

    cpa_all = [p["raw"] / max(p["n_articles"], 1) for p in P]       # 빈 댓글 포함
    cpa_ne = [p["com_per_article"] for p in P]                       # 비어있지 않은 댓글
    ax[1].bar(yrs, cpa_all, color="#c9b48a", width=.62,
              label="빈 댓글 포함" if lang == "ko" else "incl. empty stubs")
    ax[1].bar(yrs, cpa_ne, color="#b48a3a", width=.62,
              label="비어있지 않은 댓글" if lang == "ko" else "non-empty only")
    for i, v in enumerate(cpa_all):
        ax[1].annotate(f"{v:.0f}", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    _T(ax[1], "기사당 평균 수집 댓글 수" if lang == "ko" else "mean collected comments per article")
    ax[1].set_ylabel("기사당 평균 수집 댓글 수" if lang == "ko" else "mean comments per article")
    ax[1].set_xlabel("연도" if lang == "ko" else "year")
    ax[1].legend(fontsize=8.5); ax[1].margins(y=0.16)
    for a in ax:
        for lab in a.get_xticklabels():
            lab.set_rotation(40); lab.set_ha("right")
    _T(fig, LB[lang]["P6"]["t"], sup=True, fontsize=13)
    return fig


def fig_P7(R, lang):
    P = R["PIPE"]; yrs = [str(p["year"]) for p in P]
    use_style(lang)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True)
    er = [p["empty_rate"] for p in P]
    ax[0].bar(yrs, er, color="#5b6472", width=.62)
    for i, v in enumerate(er):
        ax[0].annotate(f"{v:.0f}%", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    ax[0].axvline(3.5, color="#b0433a", ls=":", lw=1.2)
    ax[0].annotate("2018 드루킹 사건\n댓글 시스템 개편" if lang == "ko" else "2018 scandal →\ncomment overhaul",
                   (3.5, max(er) * 0.9), xytext=(6, 0), textcoords="offset points", fontsize=8, color="#b0433a")
    _T(ax[0], "연도별 빈(삭제) 댓글 비율" if lang == "ko" else "empty (deleted) comment rate by year")
    ax[0].set_ylabel("빈 댓글 비율 (%)" if lang == "ko" else "empty comment rate (%)")
    ax[0].set_xlabel("연도" if lang == "ko" else "year"); ax[0].margins(y=0.16)

    ck = [p["core_kw"] or 0 for p in P]
    ret = [p["final"] / p["raw"] * 100 for p in P]
    b = ax[1].bar(yrs, ck, color="#2e7d55", width=.62, label="핵심 키워드 수" if lang == "ko" else "core keywords")
    for i, p in enumerate(P):
        if p["year"] == 2019:
            b[i].set_color("#b0433a")
    a1b = ax[1].twinx(); a1b.grid(False)
    a1b.plot(yrs, ret, "o-", color="#7b4ba3", lw=1.8, label="최종 잔존율 (%)" if lang == "ko" else "final retention (%)")
    _T(ax[1], "연도별 경제필터 핵심 키워드 수 (막대) vs 최종 잔존율 (선)" if lang == "ko"
                    else "core keywords (bars) vs final retention (line) by year")
    ax[1].set_ylabel("핵심 키워드 수" if lang == "ko" else "core keywords")
    a1b.set_ylabel("최종 잔존율 (%)" if lang == "ko" else "final retention (%)")
    ax[1].set_xlabel("연도" if lang == "ko" else "year")
    a1b.legend(loc="upper left", fontsize=9)
    for a in ax:
        for lab in a.get_xticklabels():
            lab.set_rotation(40); lab.set_ha("right")
    _T(fig, LB[lang]["P7"]["t"], sup=True, fontsize=13)
    return fig


def fig_P8(R, lang):
    P = R["PIPE"]; Y = R.get("Y2019", {})
    yrs = [p["year"] for p in P]
    kws = ["공매도", "주식", "금리", "개미들", "주가", "기관"]
    kwl = {"공매도": "공매도" if lang == "ko" else "short-selling", "주식": "주식" if lang == "ko" else "stock",
           "금리": "금리" if lang == "ko" else "rate", "개미들": "개미들" if lang == "ko" else "retail",
           "주가": "주가" if lang == "ko" else "price", "기관": "기관" if lang == "ko" else "institution"}
    use_style(lang)
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.6), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1.25, 1]})
    for w in kws:
        vals = [p["kw_df"].get(w, 0.0) for p in P]
        ax[0].plot(yrs, vals, "o-", lw=1.6, ms=4, label=kwl[w])
    ax[0].axhline(0.5, color="#333", ls="--", lw=1.2,
                  label="핵심 키워드 문턱 (DF 0.5%)" if lang == "ko" else "core threshold (DF 0.5%)")
    ax[0].axvline(2019, color="#b0433a", ls=":", lw=1.4)
    ax[0].set_xticks(yrs)
    for lab in ax[0].get_xticklabels():
        lab.set_rotation(40); lab.set_ha("right")
    ax[0].legend(fontsize=8.5, ncol=2)
    ax[0].set_xlabel("연도" if lang == "ko" else "year")
    ax[0].set_ylabel("문서빈도 DF (%)" if lang == "ko" else "document frequency DF (%)")
    _T(ax[0], "주요 주식 키워드의 연도별 DF" if lang == "ko" else "stock-keyword DF by year")

    if "total" in Y:
        real = sorted(((k, v) for k, v in Y.items() if k not in ("total", "기타")), key=lambda x: x[1])
        etc = [("기타" if lang == "ko" else "other", Y.get("기타", 0))]
        items = etc + real   # 기타를 맨 아래(첫 위치)에, 나머지는 오름차순 → 위로 갈수록 큰 주제
        names = [k for k, _ in items]; vals = [v / Y["total"] * 100 for _, v in items]
        cols = ["#2e7d55" if ("주식" in n or "stock" in n.lower()) else "#9aa2ad" for n in names]
        ax[1].barh(range(len(names)), vals, color=cols)
        ax[1].set_yticks(range(len(names)))
        ax[1].set_yticklabels(names, fontsize=9.5, fontfamily=["AppleGothic", "DejaVu Sans"])
        for i, v in enumerate(vals):
            ax[1].annotate(f"{v:.0f}%", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5)
        ax[1].set_xlabel("2019 금융 뉴스 댓글 비중 (%)" if lang == "ko" else "share of 2019 finance-news comments (%)")
        _T(ax[1], "2019 크롤링 금융 뉴스 주제 (댓글 가중)" if lang == "ko" else "2019 finance-news topics (comment-weighted)")
        ax[1].margins(x=0.16)
    _T(fig, LB[lang]["P8"]["t"], sup=True, fontsize=13)
    return fig


def fig_TOX(R, lang):
    """본문 그림: (a) 감쇠 곡선  (b) 독성 tox>=x 댓글의 감성 가중치 총량 비중."""
    ko = lang == "ko"
    use_style(lang)
    fig, ax = plt.subplots(1, 2, figsize=(13.6, 4.9), constrained_layout=True)
    # (a) decay curves
    xt = np.linspace(0, 1, 200)
    for r in R["A"]:
        ax[0].plot(xt, np.clip(1 - xt, 0, None) ** r["gamma"], lw=1.8, label=f"γ = {r['gamma']:g}")
    for xv, yv in [(.3, .49), (.5, .25), (.8, .04)]:
        ax[0].plot(xv, yv, "o", color="black", ms=5, zorder=6)
        ax[0].annotate(f"{yv:.2f}", (xv, yv), xytext=(8, 6), textcoords="offset points", fontsize=8.5)
    ax[0].axvline(0.95, color=CURC, ls="--", lw=1.2, label="τ = 0.95")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(-.02, 1.03); ax[0].legend(fontsize=8.5, loc="upper right")
    ax[0].set_xlabel("독성 점수  tox   (0 = 정상, 1 = 확실한 독성)" if ko
                     else "toxicity score  tox   (0 = clean, 1 = certainly toxic)")
    ax[0].set_ylabel("댓글 가중치  w" if ko else "comment weight  w")
    # (b) weight mass held by toxic tail
    wm = pd.DataFrame(R["B_wmass"])
    ax[1].plot(wm["x"], wm["pct"], "o-", color=OK, lw=1.8, ms=6)
    for _, r in wm.iterrows():
        if r["x"] in (0.3, 0.5, 0.7, 0.9, 0.95):
            ax[1].annotate(f"{r['pct']:.3f}%", (r["x"], r["pct"]), xytext=(7, 6),
                           textcoords="offset points", fontsize=8.5)
    ax[1].axvline(0.95, color=CURC, ls="--", lw=1.2, label="τ = 0.95")
    ax[1].legend(fontsize=8.5); ax[1].margins(y=0.16)
    ax[1].set_xlabel("독성 점수 하한  x" if ko else "toxicity-score lower bound  x")
    ax[1].set_ylabel("Σw(tox ≥ x) / Σw 전체  (%)" if ko else "Σw(tox ≥ x) / Σw total  (%)")
    _T(fig, LB[lang]["TOX"]["t"], sup=True, fontsize=13)
    return fig


def fig_A1(R, lang):
    fig, ax = _new(lang); xt = np.linspace(0, 1, 200)
    for r in R["A"]:
        ax.plot(xt, np.clip(1 - xt, 0, None) ** r["gamma"], lw=1.8, label=f"γ = {r['gamma']:g}")
    for xv, yv in [(.3, .49), (.5, .25), (.8, .04)]:
        ax.plot(xv, yv, "o", color="black", ms=5, zorder=6)
        ax.annotate(f"γ=2:  {yv:.2f}", (xv, yv), xytext=(10, 6), textcoords="offset points", fontsize=9)
    _curline(ax, 0.95, "τ = 0.95")
    ax.set_xlim(0, 1); ax.set_ylim(-.02, 1.02); ax.legend(loc="upper right", framealpha=.95)
    L = LB[lang]["A1"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_A2(R, lang):
    fig, ax = _new(lang); d = pd.DataFrame(R["A"])
    ax.plot(d.gamma, d.mad, "o-", color=ACC, lw=1.8, ms=6)
    for _, r in d.iterrows():
        ax.annotate(f"{r.mad:.4f}", (r.gamma, r.mad), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5)
    _curline(ax, 2, "현재값" if lang == "ko" else "current")
    ax.set_ylim(bottom=-0.001); ax.legend()
    L = LB[lang]["A2"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_A3(R, lang):
    fig, ax = _new(lang); d = pd.DataFrame(R["A"])
    ax.plot(d.gamma, d.eff_ratio, "o-", color=OK, lw=1.8, ms=6)
    for _, r in d.iterrows():
        ax.annotate(f"{r.eff_ratio:.2f}", (r.gamma, r.eff_ratio), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.5)
    _curline(ax, 2, "현재값" if lang == "ko" else "current"); ax.legend()
    L = LB[lang]["A3"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_A4(R, lang):
    use_style(lang)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.2), height_ratios=[2.4, 1],
                                   sharex=True, constrained_layout=True)
    ms = {}
    for g in [1.0, 2.0, 3.0]:
        s = R["A_series"][g].copy(); s.index = pd.to_datetime(s.index)
        ms[g] = s.resample("MS").mean()
    df = pd.DataFrame(ms).dropna()
    lo, hi = df.min(axis=1), df.max(axis=1)
    line_lbl = "γ = 2 (채택)" if lang == "ko" else "γ = 2 (adopted)"
    ax.fill_between(df.index, lo, hi, color=OK, alpha=0.35, lw=0)
    ax.plot(df.index, df[2.0], color=OK, lw=1.6, label=line_lbl)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=10.5)
    L = LB[lang]["A4"]; ax.set_ylabel(L["y"]); _T(ax, L["t"])

    gap = hi - lo
    gap_lbl = "γ=1~3 최대-최소 폭" if lang == "ko" else "γ=1-3 max-min gap"
    ax2.fill_between(df.index, 0, gap, color=RISK, alpha=0.5, lw=0)
    ax2.plot(df.index, gap, color=RISK, lw=1.0)
    ax2.axhline(gap.mean(), color=RISK, ls="--", lw=1,
                label=(f"평균 {gap.mean():.4f}" if lang == "ko" else f"mean {gap.mean():.4f}"))
    ax2.legend(loc="upper right", fontsize=9, frameon=False)
    ax2.set_ylabel(gap_lbl, fontsize=9.5)
    ax2.set_xlabel(L["x"])

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_xlim(pd.Timestamp("2014-01-01"), pd.Timestamp("2025-12-31"))
    for lab in ax2.get_xticklabels():
        lab.set_rotation(0); lab.set_ha("center")
    return fig

def fig_A5(R, lang):
    fig, ax = _new(lang)
    for g, w in R["A_wdist"].items():
        ax.hist(w, bins=60, histtype="step", lw=1.8, label=f"γ = {g:g}")
    ax.set_yscale("log"); ax.legend()
    L = LB[lang]["A5"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_B1(R, lang):
    fig, ax = _new(lang); d = pd.DataFrame(R["B"])
    ax.plot(d.tau, d.dropped_pct, "o-", color=ACC, lw=1.8, ms=6)
    for _, r in d.iterrows():
        ax.annotate(f"{r.dropped_pct:.1f}%", (r.tau, r.dropped_pct), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.5)
    _curline(ax, 0.95, "현재값" if lang == "ko" else "current"); ax.legend()
    L = LB[lang]["B1"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_B2(R, lang):
    fig, ax = _new(lang)
    b = [x for x in R["B_bands"] if x["lo"] >= 0.95]
    lbl = [f'{x["lo"]:.2f}–{x["hi"]:.2f}' for x in b]
    ax.bar(lbl, [x["n"] for x in b], color=RISK, width=.6)
    for i, x in enumerate(b):
        ax.annotate(f'{x["n"]:,}', (i, x["n"]), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.margins(y=0.15)
    L = LB[lang]["B2"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_B3(R, lang):
    fig, ax = _new(lang); d = pd.DataFrame(R["B_wmass"])
    ax.plot(d.x, d.pct, "o-", color=OK, lw=1.8, ms=6)
    for _, r in d.iterrows():
        if r.x in (0.3, 0.5, 0.7, 0.9, 0.95):
            ax.annotate(f"{r.pct:.3f}%", (r.x, r.pct), xytext=(8, 6), textcoords="offset points",
                        fontsize=8.5)
    _curline(ax, 0.95, "현재값" if lang == "ko" else "current"); ax.legend()
    ax.margins(y=0.15)
    L = LB[lang]["B3"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_B4(R, lang):
    fig, ax = _new(lang)
    ax.hist(R["B_toxvals"], bins=80, color="#5b6472")
    for tau, c in [(0.90, "#999"), (0.95, CURC), (0.97, "#999"), (0.99, "#999")]:
        ax.axvline(tau, ls="--", lw=1.1, color=c)
    ax.annotate("τ = 0.95", (0.95, ax.get_ylim()[1] * 0.5), xytext=(-46, 0),
                textcoords="offset points", color=CURC, fontsize=9)
    ax.set_yscale("log")
    L = LB[lang]["B4"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_B5(R, lang):
    fig, ax = _new(lang, w=8.0)
    d = pd.DataFrame(R["B"]); d = d[d.tau <= 0.95].reset_index(drop=True)
    ax.plot(d.tau, d.mad, "o-", color=ACC, lw=1.8, ms=6)
    for _, r in d.iterrows():
        txt = "0" if r.mad == 0 else f"{r.mad:.5f}"
        dx, dy, ha = (0, 10, "center")
        if r.tau >= 0.95:
            dx, dy, ha = (-10, -16, "right")
        elif r.tau >= 0.93:
            dx, dy, ha = (-6, 11, "right")
        ax.annotate(txt, (r.tau, r.mad), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=8.5)
    _curline(ax, 0.95, "현재값" if lang == "ko" else "current")
    ax.legend(loc="upper right")
    ax.margins(y=0.30)
    ax.set_ylim(bottom=-4e-5)
    L = LB[lang]["B5"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_B6(R, lang):
    fig, ax = _new(lang, w=9.5)
    d = pd.DataFrame(R["B_yearly"]).sort_values("year")
    yrs = [str(y) for y in d.year]
    bars = ax.bar(yrs, d.pct, color="#5b6472", width=.62)
    for b, y, v in zip(bars, d.year, d.pct):
        if y in (2018, 2019):
            b.set_color(RISK)
    for b, v in zip(bars, d.pct):
        ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8.5)
    ax.margins(y=0.18)
    for lab in ax.get_xticklabels():
        lab.set_rotation(30); lab.set_ha("right")
    L = LB[lang]["B6"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_C1(R, lang):
    fig, ax = _new(lang, w=8.4); d = pd.DataFrame(R["C"])
    ax.bar(d.L, d.tokens, color="#4c72b0", width=.62, label="토큰 수" if lang == "ko" else "token count")
    ax.set_yscale("log"); ax.axvspan(1.5, 6.5, color="gold", alpha=.14)
    a2 = ax.twinx(); a2.grid(False)
    a2.plot(d.L, d.stock_ratio, "o-", color=ACC, lw=1.8,
            label="주식·경제 관련 비율" if lang == "ko" else "finance-related share")
    a2.set_ylabel("주식·경제 관련 토큰 비율" if lang == "ko" else "finance-related token share")
    ax.set_xticks(range(1, 11))
    lines = ax.get_legend_handles_labels()[0] + a2.get_legend_handles_labels()[0]
    labs = ax.get_legend_handles_labels()[1] + a2.get_legend_handles_labels()[1]
    ax.legend(lines, labs, loc="upper right")
    L = LB[lang]["C1"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_C2(R, lang):
    fig, ax = _new(lang); d = pd.DataFrame(R["C_window"])
    bars = ax.bar(d.window, d.core, color=OK, width=.6)
    for b, v in zip(bars, d.core):
        ax.annotate(f"{v}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=9.5)
    ax.margins(y=0.16)
    L = LB[lang]["C2"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_D1(R, lang):
    fig, ax = _new(lang); d = pd.DataFrame(R["D"])
    ax.plot(d.DF_pct, d.core, "o-", lw=1.8, ms=6, label="핵심 키워드" if lang == "ko" else "core keywords")
    ax.plot(d.DF_pct, d.sup, "s--", lw=1.8, ms=6, label="보조 키워드" if lang == "ko" else "auxiliary keywords")
    _curline(ax, 0.5, "실제 실행값" if lang == "ko" else "real run"); ax.legend()
    L = LB[lang]["D1"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_D2(R, lang):
    fig, ax = _new(lang); d = pd.DataFrame(R["D"])
    ax.plot(d.DF_pct, d.pass_pct, "o-", color="#7b4ba3", lw=1.8, ms=6)
    for _, r in d.iterrows():
        ax.annotate(f"{r.pass_pct:.1f}%", (r.DF_pct, r.pass_pct), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.5)
    _curline(ax, 0.5, "실제 실행값" if lang == "ko" else "real run"); ax.legend()
    L = LB[lang]["D2"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_D3(R, lang):
    fig, ax = _new(lang, w=8.2); d = pd.DataFrame(R["D_band"])
    ax.plot(d.band_lo_pct, d.aux, "s--", color="#2e7d55", lw=1.8, ms=6,
            label="보조 키워드 수 (연평균)" if lang == "ko" else "aux keyword count (annual mean)")
    a2 = ax.twinx(); a2.grid(False)
    a2.plot(d.band_lo_pct, d.pass_pct, "o-", color="#7b4ba3", lw=1.8, ms=6,
            label="통과율 (%)" if lang == "ko" else "pass rate (%)")
    a2.set_ylabel("경제 필터 통과율 (%)" if lang == "ko" else "economic-filter pass rate (%)")
    a2.set_ylim(0, 45)
    ax.axvline(0.3, color=CURC, ls="--", lw=1.2, label="현재값 (0.3 → 0.3%)" if lang == "ko" else "current (0.3 → 0.3%)")
    h = ax.get_legend_handles_labels(); h2 = a2.get_legend_handles_labels()
    ax.legend(h[0] + h2[0], h[1] + h2[1], loc="center right")
    L = LB[lang]["D3"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_E1(R, lang):
    fig, ax = _new(lang, w=8.0); d = pd.DataFrame(R["E"])
    colors = ["#b0433a" if r.core_pt < 10 else "#4c72b0" for _, r in d.iterrows()]
    bars = ax.bar(d.label, d.pass_pct, color=colors, width=.62)
    for b, v in zip(bars, d.pass_pct):
        ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=9)
    idx = list(d.label).index("10:3")
    bars[idx].set_edgecolor(CURC); bars[idx].set_linewidth(2.2)
    ax.margins(y=0.18)
    L = LB[lang]["E1"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_G1(R, lang):
    fig, ax = _new(lang, w=6.8); g = R["G"]
    labs = (["점수 ≥ 10 만", "점수 ≥ 10\n+ 핵심 ≥ 1개  (현재)"] if lang == "ko"
            else ["score ≥ 10 only", "score ≥ 10\n+ core ≥ 1  (current)"])
    vals = [g["pass_without"], g["pass_with"]]
    bars = ax.bar(labs, vals, color=["#9aa2ad", "#4c72b0"], width=.55)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}%", (b.get_x() + b.get_width() / 2, v), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=9.5)
    diff = g["aux_only"]
    ax.annotate(("차단된 댓글: " if lang == "ko" else "removed: ") + f"{diff:,}"
                + (f"  ({diff / g['n_pass'] * 100:.1f}%)"),
                (0.5, max(vals)), xytext=(0, 20), textcoords="offset points", ha="center",
                fontsize=9, color=RISK)
    ax.margins(y=0.28)
    L = LB[lang]["G1"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_F1(R, lang):
    fig, ax = _new(lang, w=8.2); d = pd.DataFrame(R["F"])
    ax.axvspan(3, 10, color="#e08214", alpha=.12)
    ax.plot(d.cut, d.pass_pct, "o-", color="#e08214", lw=1.8, ms=6)
    for _, r in d.iterrows():
        off, ha = ((0, 9), "center")
        if r.cut == 10:
            off, ha = ((7, -16), "left")     # 현재값 선·이웃 라벨과 겹치지 않게 아래로
        ax.annotate(f"{r.pass_pct:.1f}%", (r.cut, r.pass_pct), xytext=off,
                    textcoords="offset points", ha=ha, fontsize=8.5)
    ax.set_ylim(top=d.pass_pct.max() * 1.25)
    ax.margins(x=0.06)
    _curline(ax, 10, "현재값" if lang == "ko" else "current")
    ax.legend(loc="upper right")
    L = LB[lang]["F1"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_F2(R, lang):
    fig, ax = _new(lang); s = R["econ_score"]; s = s[s <= 130]
    ax.hist(s, bins=45, color="#4c72b0")
    ax.axvline(10, color=CURC, ls="--", lw=1.3, label="현재값" if lang == "ko" else "current")
    ax.set_yscale("log"); ax.legend()
    L = LB[lang]["F2"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def _ds():
    p = BASE / "downstream_sensitivity.csv"
    return pd.read_csv(p) if p.exists() else None

def fig_H1(R, lang):
    fig, ax = _new(lang); d = _ds()
    if d is None: return fig
    g = d[d.experiment == "gamma"]
    ax.plot(g.param, g.sharpe, "o-", lw=1.8, ms=7, label="Sharpe")
    ax.plot(g.param, g.calmar, "s--", lw=1.8, ms=7, label="Calmar")
    for _, r in g.iterrows():
        ax.annotate(f"{r.sharpe:.3f}", (r.param, r.sharpe), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=8.5)
        ax.annotate(f"{r.calmar:.3f}", (r.param, r.calmar), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=8.5)
    ax.axvline(2, color=CURC, ls="--", lw=1.1, alpha=.6, label="현재값" if lang == "ko" else "current")
    ax.legend(loc="center right"); ax.set_ylim(0.28, 0.72)
    ax.set_xticks([1, 2, 3]); ax.margins(x=0.12)
    L = LB[lang]["H1"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_H2(R, lang):
    fig, ax = _new(lang); d = _ds()
    if d is None: return fig
    g = d[d.experiment == "gamma"]
    series = [("MDD", g.mdd, "o-"), ("CVaR 5%", g.cvar5, "s--"),
              ("하방 변동성 (부호반전)" if lang == "ko" else "downside vol (negated)", -g.downside_vol, "^:")]
    for name, ser, sty in series:
        ax.plot(g.param, ser, sty, lw=1.8, ms=6, label=name)
        v = ser.iloc[1]
        ax.annotate(f"{v:.3f}", (2, v), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.5)
    ax.axvline(2, color=CURC, ls="--", lw=1.1, alpha=.6,
               label="현재값" if lang == "ko" else "current")
    ax.legend(loc="center right", framealpha=.95)
    ax.set_xticks([1, 2, 3]); ax.margins(y=0.22)
    L = LB[lang]["H2"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

def fig_I1(R, lang):
    fig, ax = _new(lang, w=8.4); d = _ds()
    if d is None: return fig
    m = d[d.experiment == "sent_ma"].sort_values("param")
    ax.plot(m.param, m.sharpe, "o-", lw=1.8, ms=8, label="Sharpe")
    ax.plot(m.param, m.calmar, "s--", lw=1.8, ms=8, label="Calmar")
    for _, r in m.iterrows():
        dy = 12 if r.param == 10 else 10
        ax.annotate(f"Sharpe {r.sharpe:.3f}\nMDD {r.mdd*100:.1f}%", (r.param, r.sharpe),
                    xytext=(0, dy), textcoords="offset points", ha="center", fontsize=8.5)
    ax.axvline(10, color=CURC, ls="--", lw=1.1, alpha=.6, label="현재값" if lang == "ko" else "current")
    ax.legend(loc="center right")
    ax.set_xticks([5, 10, 20]); ax.set_xlim(3, 22); ax.margins(y=0.28)
    L = LB[lang]["I1"]; ax.set(xlabel=L["x"], ylabel=L["y"]); _T(ax, L["t"]); return fig

FIGS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8",
        "TOX", "A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "B6",
        "C1", "C2", "D1", "D2", "D3", "E1", "F1", "F2", "G1", "H1", "H2", "I1"]

def _render(key, R, lang):
    return globals()[f"fig_{key}"](R, lang)

def make_fig(key, R, lang):
    """노트북 셀용: 그림을 그리고 반환하지 않는다.
    inline 백엔드가 셀 종료 시 pyplot 대기 그림을 딱 1회 표시한다 (중복 방지)."""
    _render(key, R, lang)
    return None

def dump_results(R, path=None):
    """핵심 수치를 results.json 으로 저장."""
    import json
    def js(o):
        if hasattr(o, "item"):
            return o.item()
        if isinstance(o, float) and o != o:
            return None
        return str(o)
    out = dict(
        note="2014-2025 최종 논문 표본. archive recollect 중간 산출물(toxicity_all / classified_stock_comments).",
        n_comments_all=R["N_all"], n_toxicity_kept=R["N_kept"],
        pipeline=R["PIPE"], political_ratio_by_year=R["POL_RATIO"], political_top_keywords=R["POL_KW"],
        A_gamma=R["A"], B_tau=R["B"], B_dropped_bands=[{k: v for k, v in b.items() if k != "samples"} for b in R["B_bands"]],
        B_weight_mass_tail=R["B_wmass"], B_yearly=R["B_yearly"], C_token_len=R["C"], C_1char_stock=R["C_1char"],
        C_long_stock=R["C_long"], C_window=R["C_window"], D_df=R["D"], D_keywords=R["D_kw"],
        D_support_band=R["D_band"], E_points=R["E"], F_score_cut=R["F"], G_core_condition=R["G"],
    )
    p = Path(path or BASE / "results.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=js))
    return p


def save_all(R, lang, outdir=None):
    outdir = Path(outdir or BASE / "figures" / lang)
    outdir.mkdir(parents=True, exist_ok=True)
    for k in FIGS:
        fig = _render(k, R, lang)
        fig.savefig(outdir / f"{k}.png")
        plt.close(fig)
    print(f"[{lang}] {len(FIGS)} figures -> {outdir}/")
