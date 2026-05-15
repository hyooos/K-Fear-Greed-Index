import pandas as pd
import re
import os
import json
import numpy as np
from collections import Counter
from pathlib import Path

print("=" * 80)
print("주식/경제 키워드 분석 + 자동 필터링")
print("=" * 80)

# ========== 경로 설정 ==========
BASE_DIR   = Path(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR  = BASE_DIR / "toxicity_filter"   # 2단계 결과 폴더
OUTPUT_DIR = BASE_DIR / "final_filtered"    # 최종 결과 저장 폴더
OUTPUT_DIR.mkdir(exist_ok=True)

YEARS = [2023, 2024, 2025]

# ========== 주식/경제 키워드 패턴 ==========
stock_patterns = [
    r'코스피|코스닥|kospi|kosdaq',
    r'주식|주가|증시|시장|장',
    r'매수|매도|투자|손절|익절',
    r'개미|외인|기관|외국인',
    r'삼전|삼성전자|하닉|하이닉스',
    r'지수|시총|배당|상장',
    r'급등|급락|폭등|폭락|상승|하락',
    r'수급|거래량|환율|금리',
    r'종목|실적|반도체|전지',
    r'펀드|연기금|국민연금',
]

def extract_words(text):
    if pd.isna(text):
        return []
    return re.findall(r'[가-힣]{2,6}|[a-zA-Z]{2,10}', text.lower())

def is_stock_related_word(word):
    return any(re.search(pattern, word) for pattern in stock_patterns)

def score_comment(text, core_kws, support_kws):
    if pd.isna(text):
        return 0
    text_lower = text.lower()
    score  = sum(1 for kw in core_kws    if kw in text_lower) * 10
    score += sum(1 for kw in support_kws if kw in text_lower) * 3
    return score

def classify_comment(text, score, core_kws):
    if score < 10:
        return 'other'
    text_lower = text.lower() if pd.notna(text) else ''
    return 'stock' if any(kw in text_lower for kw in core_kws) else 'other'


# ========== 연도별 실행 ==========
for year in YEARS:
    print(f"\n{'='*80}")
    print(f"{year}년 처리 중...")
    print(f"{'='*80}")

    inp = INPUT_DIR / f"comments_toxicity_kept_{year}.csv"
    if not inp.exists():
        print(f"  ⚠️  파일 없음, 건너뜀: {inp}")
        continue

    # 1. 데이터 로딩
    df = pd.read_csv(inp)
    df_original_len = len(df)
    df = df[df['text_raw'].notna() & (df['text_raw'].str.strip() != '')].copy()
    print(f"원본: {df_original_len:,}개 | 빈 댓글 제거 후: {len(df):,}개")

    # 2. 키워드 자동 추출
    print("\n키워드 자동 추출 중...")
    all_words = []
    for text in df['text_raw']:
        all_words.extend(extract_words(text))
    word_freq = Counter(all_words)

    word_document_count = {}
    for text in df['text_raw']:
        for word in set(extract_words(text)):
            word_document_count[word] = word_document_count.get(word, 0) + 1

    threshold = len(df) * 0.005
    core_keywords    = []
    support_keywords = []

    for word, doc_count in sorted(word_document_count.items(), key=lambda x: x[1], reverse=True):
        if is_stock_related_word(word):
            pct = doc_count / len(df) * 100
            if doc_count >= threshold:
                core_keywords.append((word, doc_count, pct))
            elif doc_count >= threshold * 0.3:
                support_keywords.append((word, doc_count, pct))

    CORE_STOCK_KEYWORDS    = [w for w, _, _ in core_keywords]
    SUPPORT_STOCK_KEYWORDS = [w for w, _, _ in support_keywords]

    print(f"핵심 키워드: {len(core_keywords)}개 | 보조 키워드: {len(support_keywords)}개")
    print("핵심 키워드 목록:")
    for word, _, pct in core_keywords:
        print(f"  '{word}' ({pct:.1f}%)")

    # 3. 점수 계산 및 분류
    df['stock_score'] = df['text_raw'].apply(
        lambda t: score_comment(t, CORE_STOCK_KEYWORDS, SUPPORT_STOCK_KEYWORDS))
    df['is_stock'] = df.apply(
        lambda row: classify_comment(row['text_raw'], row['stock_score'], CORE_STOCK_KEYWORDS), axis=1)

    stock_df = df[df['is_stock'] == 'stock'].copy()
    other_df = df[df['is_stock'] == 'other'].copy()

    print(f"\n분류 결과:")
    print(f"  주식/경제: {len(stock_df):,}개 ({len(stock_df)/len(df)*100:.1f}%)")
    print(f"  기타:      {len(other_df):,}개 ({len(other_df)/len(df)*100:.1f}%)")
    print(f"  평균 점수: {stock_df['stock_score'].mean():.1f} | 중간값: {stock_df['stock_score'].median():.1f}")

    # 4. 샘플 확인
    print(f"\n[주식 관련 샘플 TOP 5]")
    for _, row in stock_df.nlargest(5, 'stock_score').iterrows():
        print(f"  [점수:{row['stock_score']:3.0f}] {str(row['text_raw'])[:80].replace(chr(10),' ')}...")

    # 5. 저장
    year_out = OUTPUT_DIR / str(year)
    year_out.mkdir(exist_ok=True)

    # 키워드 분석 JSON
    keyword_result = {
        'core_keywords':    [{'word': w, 'count': c, 'percentage': p} for w, c, p in core_keywords],
        'support_keywords': [{'word': w, 'count': c, 'percentage': p} for w, c, p in support_keywords],
    }
    with open(year_out / f'keyword_analysis_result_{year}.json', 'w', encoding='utf-8') as f:
        json.dump(keyword_result, f, ensure_ascii=False, indent=2)

    # 전체 (점수/분류 포함)
    df.to_csv(year_out / f'classified_stock_comments_{year}.csv', index=False, encoding='utf-8-sig')

    # 주식 관련 전체
    stock_df.to_csv(year_out / f'comments_final_stock_only_{year}.csv', index=False, encoding='utf-8-sig')

    # 주식 관련 클린 ⭐
    stock_df.drop(columns=['stock_score', 'is_stock'], errors='ignore').to_csv(
        year_out / f'comments_stock_clean_{year}.csv', index=False, encoding='utf-8-sig')

    # 기타
    other_df.to_csv(year_out / f'comments_other_{year}.csv', index=False, encoding='utf-8-sig')

    # 통계 요약 JSON
    summary = {
        'total_comments':        len(df),
        'stock_comments':        len(stock_df),
        'other_comments':        len(other_df),
        'stock_ratio':           len(stock_df) / len(df) * 100,
        'avg_score':             float(stock_df['stock_score'].mean()),
        'median_score':          float(stock_df['stock_score'].median()),
        'core_keywords_count':   len(core_keywords),
        'support_keywords_count':len(support_keywords),
    }
    with open(year_out / f'stock_classification_summary_{year}.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {year}년 저장 완료 → final_filtered/{year}/")
    print(f"  ⭐ comments_stock_clean_{year}.csv = 최종 클린 데이터")

print(f"\n{'='*80}")
print("전체 완료!")
print("폴더 구조:")
print("  final_filtered/")
print("  ├── 2023/  comments_stock_clean_2023.csv ⭐ ...")
print("  ├── 2024/  comments_stock_clean_2024.csv ⭐ ...")
print("  └── 2025/  comments_stock_clean_2025.csv ⭐ ...")
print(f"{'='*80}")