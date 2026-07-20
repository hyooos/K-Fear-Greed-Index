import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd


BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

POLITICS_PATTERNS = [
    r"윤석열|석열|윤통|용산|이재명|재명|개딸|한동훈|동훈|뚜껑|조국|문재인|재앙|박근혜|근혜|이명박|MB",
    r"민주당|더불어민주당|국민의힘|국힘|국짐|정의당|개혁신당|조국혁신당|좌파|우파|좌빨|수구|빨갱이|종북|토착왜구",
    r"탄핵|계엄|특검|비상계엄|내란|반역|구속|체포|영장|기소|검찰|독재|관권선거|부정선거|공천",
    r"의원|국회의원|국회|여당|야당|거대야당|당대표|원내대표|장관|차관|국무총리|대통령실|방통위|권익위",
    r"친문|친명|친윤|비윤|반윤|개헌|정권교체|심판|지지자|촛불집회|태극기부대|정치인|정치질",
]


def parse_years(value: str) -> list[int]:
    years: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def contains_political_keywords(text: str) -> bool:
    if pd.isna(text):
        return False
    text_lower = str(text).lower()
    return any(re.search(pattern, text_lower) for pattern in POLITICS_PATTERNS)


def get_matched_keywords(text: str) -> list[str]:
    if pd.isna(text):
        return []
    text_lower = str(text).lower()
    matched: list[str] = []
    for pattern in POLITICS_PATTERNS:
        for keyword in pattern.split("|"):
            if keyword.lower() in text_lower:
                matched.append(keyword)
    return sorted(set(matched))


def input_path_for_year(input_dir: Path, year: int) -> Path:
    candidates = [
        input_dir / f"comments_{year}.csv",
        input_dir / f"comments_merged_{year}.csv",
        BASE_DIR / f"comments_merged_{year}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def run_year(year: int, input_dir: Path, output_dir: Path) -> dict:
    print(f"\n{'=' * 80}")
    print(f"{year}년 정치 키워드 필터링")
    print(f"{'=' * 80}")

    input_file = input_path_for_year(input_dir, year)
    if not input_file.exists():
        print(f"파일 없음, 건너뜀: {input_file}")
        return {"year": year, "skipped": True}

    df = pd.read_csv(input_file)
    original_count = len(df)
    df = df[df["text_raw"].notna() & (df["text_raw"].astype(str).str.strip() != "")].copy()
    print(f"입력: {input_file}")
    print(f"전체 댓글 수: {original_count:,}개")
    print(f"빈 댓글 제거 후: {len(df):,}개")

    df["is_political"] = df["text_raw"].apply(contains_political_keywords)
    df["matched_keywords"] = df["text_raw"].apply(get_matched_keywords)
    df["keyword_count"] = df["matched_keywords"].apply(len)

    political_comments = df[df["is_political"]].copy()
    non_political_comments = df[~df["is_political"]].copy()

    all_matched_keywords: list[str] = []
    for keywords in political_comments["matched_keywords"]:
        all_matched_keywords.extend(keywords)
    keyword_freq = Counter(all_matched_keywords)

    political_ratio = len(political_comments) / len(df) * 100 if len(df) else 0.0
    print(f"정치 관련 댓글: {len(political_comments):,}개 ({political_ratio:.1f}%)")
    print(f"비정치 댓글: {len(non_political_comments):,}개")

    output_dir.mkdir(parents=True, exist_ok=True)
    df[
        [
            "news_id",
            "comment_id",
            "text_raw",
            "is_political",
            "matched_keywords",
            "keyword_count",
            "like_count",
            "dislike_count",
            "comment_at",
            "pub_date",
        ]
    ].to_csv(output_dir / f"comments_{year}_classified.csv", index=False, encoding="utf-8-sig")

    political_comments[
        [
            "news_id",
            "comment_id",
            "text_raw",
            "matched_keywords",
            "keyword_count",
            "like_count",
            "dislike_count",
            "comment_at",
            "pub_date",
        ]
    ].to_csv(output_dir / f"comments_{year}_political_only.csv", index=False, encoding="utf-8-sig")

    non_political_comments.drop(
        columns=["is_political", "matched_keywords", "keyword_count"],
        errors="ignore",
    ).to_csv(output_dir / f"comments_political_removed_{year}.csv", index=False, encoding="utf-8-sig")

    summary = {
        "year": year,
        "input_file": str(input_file),
        "total_comments": int(len(df)),
        "political_comments": int(len(political_comments)),
        "non_political_comments": int(len(non_political_comments)),
        "political_ratio": float(political_ratio),
        "top_10_keywords": dict(keyword_freq.most_common(10)),
    }
    with open(output_dir / f"classification_summary_{year}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {output_dir / f'comments_political_removed_{year}.csv'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: remove political comments.")
    parser.add_argument("--years", default="2014-2025")
    parser.add_argument("--input-dir", default="data/NAVER_by_year/comments")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "political_filter"))
    args = parser.parse_args()

    print("=" * 80)
    print("1단계: 정치 키워드 기반 댓글 제거")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    summaries = [
        run_year(year, Path(args.input_dir), output_dir)
        for year in parse_years(args.years)
    ]
    with open(output_dir / "political_filter_summary_all.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print("\n전체 완료")
    print(f"다음 단계 입력 폴더: {output_dir}")


if __name__ == "__main__":
    main()
