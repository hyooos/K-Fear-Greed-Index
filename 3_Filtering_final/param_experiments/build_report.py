# -*- coding: utf-8 -*-
"""report_{ko,en}.html 생성 — filter_param_lib 의 19개 그림 + 그래프별 설명 임베드."""
import base64
import html
import re
from pathlib import Path

import filter_param_lib as L
import make_notebooks as MN

BASE = Path(__file__).resolve().parent


def b64(lang, key):
    p = BASE / "figures" / lang / f"{key}.png"
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def md(text: str) -> str:
    """아주 작은 markdown -> html (문단/불릿/굵게/코드/인용/헤더)."""
    text = text.replace("$w_i = (1 - \\text{tox}_i)^{\\gamma}$", "<code>w = (1 − tox)^γ</code>")
    text = text.replace("$w_i = (1 - \\text{tox}_i)^{\\gamma}$", "")
    out, ul = [], False
    for raw in text.split("\n"):
        line = raw.rstrip()
        if not line:
            if ul:
                out.append("</ul>"); ul = False
            continue
        line = html.escape(line, quote=False)
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line = re.sub(r"`([^`]+)`", r"<code>\1</code>", line)
        if line.startswith("## "):
            if ul: out.append("</ul>"); ul = False
            out.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("&gt; ") or line.startswith("> "):
            if ul: out.append("</ul>"); ul = False
            out.append(f'<blockquote>{line.split(" ", 1)[1]}</blockquote>')
        elif line.startswith("- "):
            if not ul: out.append("<ul>"); ul = True
            out.append(f"<li>{line[2:]}</li>")
        elif line.startswith("|"):
            continue  # 표는 별도 처리
        else:
            if ul: out.append("</ul>"); ul = False
            out.append(f"<p>{line}</p>")
    if ul: out.append("</ul>")
    return "\n".join(out)


CSS = """
:root{--ground:#f6f7f9;--panel:#fff;--ink:#1b1f27;--ink-soft:#4a5160;--line:#dde0e7;--line-soft:#e9ebf0;
--accent:#3a4db8;--accent-soft:#eef0fb;--ok:#2e7d55;--ok-bg:#e7f2eb;--warn:#a9700c;--warn-bg:#f6eede;--risk:#b0433a;--risk-bg:#f7e6e4;
--mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;--sans:'IBM Plex Sans','Apple SD Gothic Neo','Malgun Gothic',system-ui,sans-serif;
--serif:'Spectral','Apple SD Gothic Neo',Georgia,serif;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--ground:#12141a;--panel:#1a1d25;--ink:#e7e9ee;--ink-soft:#a3a9b8;
--line:#2b2f3a;--line-soft:#232732;--accent:#9aa7ef;--accent-soft:#20243a;--ok:#6fca9b;--ok-bg:#182a22;--warn:#e0b567;--warn-bg:#2c2617;--risk:#e59b92;--risk-bg:#2e1c1a;}}
:root[data-theme="dark"]{--ground:#12141a;--panel:#1a1d25;--ink:#e7e9ee;--ink-soft:#a3a9b8;--line:#2b2f3a;--line-soft:#232732;
--accent:#9aa7ef;--accent-soft:#20243a;--ok:#6fca9b;--ok-bg:#182a22;--warn:#e0b567;--warn-bg:#2c2617;--risk:#e59b92;--risk-bg:#2e1c1a;}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:var(--sans);line-height:1.7;margin:0;-webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:60px 24px 120px}
.col{max-width:740px}
header{border-bottom:1px solid var(--line);padding-bottom:28px;margin-bottom:36px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 12px}
h1{font-family:var(--serif);font-weight:600;font-size:2.3rem;line-height:1.15;margin:0 0 14px;text-wrap:balance}
h2{font-family:var(--serif);font-weight:600;font-size:1.5rem;margin:56px 0 10px;padding-top:8px;border-top:1px solid var(--line-soft)}
h3{font-family:var(--mono);font-size:.82rem;letter-spacing:.06em;color:var(--accent);margin:34px 0 4px;text-transform:uppercase}
p{margin:12px 0;max-width:760px}
ul{max-width:760px}
li{margin:5px 0}
code{font-family:var(--mono);font-size:.85em;background:var(--accent-soft);padding:.1em .38em;border-radius:4px}
strong{font-weight:600}
blockquote{border-left:3px solid var(--warn);background:var(--warn-bg);margin:16px 0;padding:10px 16px;border-radius:0 6px 6px 0;font-size:.94rem}
figure{margin:8px 0 4px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto}
figure img{display:block;width:100%;min-width:640px;height:auto;border-radius:4px}
.cap{font-size:.92rem;color:var(--ink-soft);margin:8px 0 30px;max-width:820px;padding-left:2px;border-left:2px solid var(--line);padding-left:12px}
.lede{font-size:1.05rem;color:var(--ink-soft)}
.tablewrap{overflow-x:auto;margin:22px 0;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;min-width:560px;font-size:.9rem}
th,td{text-align:left;padding:10px 14px;border-bottom:1px solid var(--line-soft);vertical-align:top}
thead th{background:var(--panel);font-family:var(--mono);font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-soft);font-weight:500}
tbody tr:last-child td{border-bottom:none}
.files{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 22px;font-size:.9rem}
.files li{margin:6px 0}
"""

HEAD = {
 "ko": dict(title="필터링 파라미터 검증",
           eyebrow="K-FGI · 3장 필터링 파이프라인",
           h1="왜 이 값일 수밖에 없었는가 — 필터링 파라미터 검증",
           sec_title="검증 절",
           figpref="그림 ", note="아래 각 그래프의 x·y축과 결론 해석은 그림 밑 설명을 참고."),
 "en": dict(title="Filtering Parameter Validation",
           eyebrow="K-FGI · Ch. 3 Filtering pipeline",
           h1="Why these values, not others — parameter validation",
           sec_title="Validation sections",
           figpref="Figure ", note="For each graph, the axes and the takeaway are in the note below it."),
}

VERDICT = {
 "ko": """## 검증 결과 요약
| 값 | 성격 | 판정 |
| --- | --- | --- |
""" + "\n".join([
  "| τ = 0.95 (독성 hard-drop) | 우리 선택 | ✓ 지표·전략이 τ에 구조적 불변 — tox≥0.95 가중치 비중 0.004% |",
  "| γ = 2 (독성 가중치) | 우리 선택 | ✓ 전략 성과 무영향, 감쇠 강도의 원칙적 중간값 |",
  "| score ≥ 10 (경제 필터) | 우리 선택 | ✓ 통과율 평탄 구간(3–10)의 끝, 13에서 급락 |",
  "| DF ≥ 1% (핵심 키워드) | 우리 선택 | ◐ 일반어 배제 / 정당 복합어 보존의 균형점 |",
  "| 핵심 10 : 보조 3 | 우리 선택 | ◐ 보조 점수는 게이트에 무관 (순위용) |",
  "| 핵심 ≥ 1개 조건 | 우리 선택 | ✓ 보조 키워드만으로 통과하는 소수 댓글 차단 |",
  "| 2 ~ 6 글자 (키워드 추출) | 방법론 규칙 | ✓ 2글자 금융어 보존 / 긴 어절은 부분문자열로 포착 |",
  "| 감성 10일 이동평균 | 우리 선택 | ✓ Sharpe·MDD 모두에서 국소 최적 (근거가 필요한 값) |",
  "| 252일 · 60일 · 70% 분위수 | 금융 관행값 | 튜닝 대상 아님 |",
 ]),
 "en": """## Summary of verdicts
| Value | Type | Verdict |
| --- | --- | --- |
""" + "\n".join([
  "| τ = 0.95 (toxicity hard-drop) | our choice | ✓ index & strategy structurally invariant to τ — tox≥0.95 weight mass 0.004% |",
  "| γ = 2 (toxicity weighting) | our choice | ✓ no strategy-performance effect; principled midpoint of decay strength |",
  "| score ≥ 10 (economic filter) | our choice | ✓ end of the pass-rate plateau (3–10), sharp drop at 13 |",
  "| DF ≥ 1% (core keyword) | our choice | ◐ balance of excluding generic / keeping valid compounds |",
  "| core 10 : aux 3 | our choice | ◐ aux weight irrelevant to the gate (ranking only) |",
  "| core ≥ 1 condition | our choice | ✓ blocks the few comments passing on aux keywords alone |",
  "| 2–6 characters (extraction) | method rule | ✓ keeps 2-char financial terms / long phrases caught as substrings |",
  "| 10-day sentiment MA | our choice | ✓ local optimum on Sharpe and MDD (a value that does need a rationale) |",
  "| 252-day · 60-day · 70th pct. | finance convention | not tuned |",
 ]),
}

FILES = {
 "ko": """## 파일
<div class="files"><ul>
<li><code>filter_param_lib.py</code> — 계산·작도 라이브러리 (실제 pre-filter 코퍼스, 2023–2025)</li>
<li><code>filter_params_ko.ipynb</code> · <code>filter_params_en.ipynb</code> — 이 리포트의 노트북판 (그림 + 설명, 실행 완료본)</li>
<li><code>strategy_sensitivity_ko.ipynb</code> · <code>_en.ipynb</code> — 전략 성과 재산출 (γ, 감성 MA)</li>
<li><code>results.json</code> · <code>downstream_sensitivity.csv</code> — 수치 원본 &nbsp;|&nbsp; <code>figures/ko</code>, <code>figures/en</code> — 그림 19종 × 2언어</li>
</ul></div>""",
 "en": """## Files
<div class="files"><ul>
<li><code>filter_param_lib.py</code> — compute + plotting library (real pre-filter corpus, 2023–2025)</li>
<li><code>filter_params_ko.ipynb</code> · <code>filter_params_en.ipynb</code> — notebook version of this report (figures + notes, executed)</li>
<li><code>strategy_sensitivity_ko.ipynb</code> · <code>_en.ipynb</code> — strategy re-derivation (γ, sentiment MA)</li>
<li><code>results.json</code> · <code>downstream_sensitivity.csv</code> — raw numbers &nbsp;|&nbsp; <code>figures/ko</code>, <code>figures/en</code> — 19 figures × 2 languages</li>
</ul></div>""",
}


def table_html(md_table: str) -> str:
    rows = [r for r in md_table.splitlines() if r.strip().startswith("|")]
    head = [c.strip() for c in rows[0].strip("|").split("|")]
    body = rows[2:]
    h = "".join(f"<th>{html.escape(c)}</th>" for c in head)
    trs = []
    for r in body:
        cells = [c.strip() for c in r.strip("|").split("|")]
        trs.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>'


def build(lang):
    H = HEAD[lang]; tx = MN.TXT[lang]
    parts = [f"""<title>{H['title']}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header class="col">
  <p class="eyebrow">{H['eyebrow']}</p>
  <h1>{H['h1']}</h1>
  <div class="lede">{md(tx['title'].split('# ',1)[1].split(chr(10),1)[1])}</div>
</header>
"""]

    # 요약 표
    vt = VERDICT[lang]
    parts.append(f"<h2>{vt.splitlines()[0][3:]}</h2>")
    parts.append(table_html(vt))

    # 섹션 + 그림
    aux_only, aux_pct = 1165, 1.2
    try:
        import pickle
        RR = pickle.loads(L.CACHE.read_bytes())
        aux_only = RR["G"]["aux_only"]; aux_pct = aux_only / RR["G"]["n_pass"] * 100
    except Exception:
        pass

    for sec, keys in MN.SECTIONS:
        body = tx[sec].format(aux_only=aux_only, aux_pct=aux_pct) if "{aux" in tx[sec] else tx[sec]
        parts.append(md(body))
        for k in keys:
            L2 = L.LB[lang][k]
            parts.append(f'<h3>{k}</h3>')
            parts.append(f'<figure><img src="{b64(lang, k)}" alt="{html.escape(L2["t"])}"></figure>')
            parts.append(f'<p class="cap"><strong>{html.escape(L2["t"])}</strong><br>{html.escape(L2["cap"])}</p>')

    parts.append(md(tx["concl"]))
    parts.append(FILES[lang])
    parts.append("</div>")
    (BASE / f"report_{lang}.html").write_text("\n".join(parts))
    print(f"report_{lang}.html  {sum(len(p) for p in parts)//1024} KB")


try:
    import pickle
    L.dump_results(pickle.loads(L.CACHE.read_bytes()))
    print("wrote results.json")
except Exception as e:
    print("results.json skipped:", e)

for lg in ("ko", "en"):
    build(lg)
