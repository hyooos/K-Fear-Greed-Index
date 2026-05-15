import pandas as pd
import numpy as np
import re
import os
import json
from collections import Counter

print("=" * 80)
print("정치 키워드 기반 댓글 분류 분석")
print("=" * 80)

# ========== 경로 설정 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트 폴더
OUTPUT_DIR = os.path.join(BASE_DIR, 'political_filter')
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = [2023, 2024, 2025]

# ========== 정치 키워드 패턴 ==========
politics_patterns = [
    r'윤석열|석열|윤통|용산|이재명|재명|개딸|한동훈|동훈|뚜껑|조국|문재인|재앙|박근혜|근혜|이명박|MB',
    r'민주당|더불어민주당|국민의힘|국힘|국짐|정의당|개혁신당|조국혁신당|좌파|우파|좌빨|수구|빨갱이|종북|토착왜구',
    r'탄핵|계엄|특검|비상계엄|내란|반역|구속|체포|영장|기소|검찰|독재|관권선거|부정선거|공천',
    r'의원|국회의원|국회|여당|야당|거대야당|당대표|원내대표|장관|차관|국무총리|대통령실|방통위|권익위',
    r'친문|친명|친윤|비윤|반윤|개헌|정권교체|심판|지지자|촛불집회|태극기부대|정치인|정치질'
]

def contains_political_keywords(text):
    if pd.isna(text):
        return False
    text_lower = text.lower()
    for pattern in politics_patterns:
        if re.search(pattern, text_lower):
            return True
    return False

def get_matched_keywords(text):
    if pd.isna(text):
        return []
    text_lower = text.lower()
    matched = []
    for pattern in politics_patterns:
        for keyword in pattern.split('|'):
            if keyword in text_lower:
                matched.append(keyword)
    return list(set(matched))

# ========== 연도별 실행 ==========
for year in YEARS:
    print(f"\n{'=' * 80}")
    print(f"{year}년 처리 중...")
    print(f"{'=' * 80}")

    # 1. 데이터 로딩
    input_file = os.path.join(BASE_DIR, f'comments_merged_{year}.csv')
    df = pd.read_csv(input_file)

    df_original_len = len(df)
    df = df[df['text_raw'].notna() & (df['text_raw'].str.strip() != '')].copy()
    print(f"전체 댓글 수: {df_original_len:,}개")
    print(f"빈 댓글 제거 후: {len(df):,}개")

    # 2. 분류 실행
    df['is_political'] = df['text_raw'].apply(contains_political_keywords)
    df['matched_keywords'] = df['text_raw'].apply(get_matched_keywords)
    df['keyword_count'] = df['matched_keywords'].apply(len)

    political_comments = df[df['is_political']].copy()
    non_political_comments = df[~df['is_political']].copy()

    print(f"정치 관련 댓글: {len(political_comments):,}개 ({len(political_comments)/len(df)*100:.1f}%)")
    print(f"비정치 댓글: {len(non_political_comments):,}개 ({len(non_political_comments)/len(df)*100:.1f}%)")

    # 3. 키워드 통계
    all_matched_keywords = []
    for keywords in political_comments['matched_keywords']:
        all_matched_keywords.extend(keywords)
    keyword_freq = Counter(all_matched_keywords)

    print(f"\n가장 많이 매칭된 정치 키워드 TOP 10:")
    for keyword, count in keyword_freq.most_common(10):
        pct = count / len(political_comments) * 100
        print(f"  {keyword:15s}: {count:5,}회 (정치 댓글의 {pct:5.1f}%)")

    # 4. 결과 저장
    # 전체 분류 결과
    df[['news_id', 'comment_id', 'text_raw', 'is_political', 'matched_keywords',
        'keyword_count', 'like_count', 'dislike_count', 'comment_at']].to_csv(
        os.path.join(OUTPUT_DIR, f'comments_{year}_classified.csv'),
        index=False, encoding='utf-8-sig')

    # 정치 댓글만
    political_comments[['news_id', 'comment_id', 'text_raw', 'matched_keywords',
        'keyword_count', 'like_count', 'dislike_count', 'comment_at']].to_csv(
        os.path.join(OUTPUT_DIR, f'comments_{year}_political_only.csv'),
        index=False, encoding='utf-8-sig')

    # 정치 제거된 클린 데이터 ⭐ → 다음 단계(독성 필터링) 입력으로 사용
    non_political_comments.drop(
        columns=['is_political', 'matched_keywords', 'keyword_count'], errors='ignore'
    ).to_csv(
        os.path.join(OUTPUT_DIR, f'comments_political_removed_{year}.csv'),
        index=False, encoding='utf-8-sig')

    # 통계 요약 JSON
    summary = {
        'total_comments': len(df),
        'political_comments': len(political_comments),
        'non_political_comments': len(non_political_comments),
        'political_ratio': len(political_comments) / len(df) * 100,
        'top_10_keywords': dict(keyword_freq.most_common(10)),
    }
    with open(os.path.join(OUTPUT_DIR, f'classification_summary_{year}.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {year}년 저장 완료 → political_filter/ 폴더")

print(f"\n{'=' * 80}")
print("전체 완료! political_filter/ 폴더 확인하세요")
print("⭐ 다음 단계 입력 파일: comments_political_removed_2023/2024/2025.csv")
print(f"{'=' * 80}")