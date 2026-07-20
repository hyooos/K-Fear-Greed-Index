import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd


BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

STOCK_PATTERNS = [
    r"코스피|코스닥|kospi|kosdaq",
    r"주식|주가|증시|시장|장",
    r"매수|매도|투자|손절|익절",
    r"개미|외인|기관|외국인",
    r"삼전|삼성전자|하닉|하이닉스",
    r"지수|시총|배당|상장",
    r"급등|급락|폭등|폭락|상승|하락",
    r"수급|거래량|환율|금리",
    r"종목|실적|반도체|전지",
    r"펀드|연기금|국민연금",
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


def extract_words(text: str) -> list[str]:
    if pd.isna(text):
        return []
    return re.findall(r"[가-힣]{2,6}|[a-zA-Z]{2,10}", str(text).lower())


def is_stock_related_word(word: str) -> bool:
    return any(re.search(pattern, word) for pattern in STOCK_PATTERNS)


def score_comment(text: str, core_keywords: list[str], support_keywords: list[str]) -> int:
    if pd.isna(text):
        return 0
    text_lower = str(text).lower()
    score = sum(1 for kw in core_keywords if kw in text_lower) * 10
    score += sum(1 for kw in support_keywords if kw in text_lower) * 3
    return score


def classify_comment(text: str, score: int, core_keywords: list[str]) -> str:
    if score < 10:
        return "other"
    text_lower = str(text).lower() if pd.notna(text) else ""
    return "stock" if any(kw in text_lower for kw in core_keywords) else "other"


def run_year(year: int, input_dir: Path, output_dir: Path, min_doc_ratio: float) -> dict | None:
    print(f"\n{'=' * 80}")
    print(f"{year}년 주식/경제 최종 필터링")
    print(f"{'=' * 80}")

    inp = input_dir / f"comments_toxicity_kept_{year}.csv"
    if not inp.exists():
        print(f"파일 없음, 건너뜀: {inp}")
        return None

    df = pd.read_csv(inp)
    original_count = len(df)
    df = df[df["text_raw"].notna() & (df["text_raw"].astype(str).str.strip() != "")].copy()
    print(f"원본: {original_count:,}개 | 빈 댓글 제거 후: {len(df):,}개")

    word_document_count: dict[str, int] = {}
    for text in df["text_raw"]:
        for word in set(extract_words(text)):
            word_document_count[word] = word_document_count.get(word, 0) + 1

    threshold = len(df) * min_doc_ratio
    core_keywords: list[tuple[str, int, float]] = []
    support_keywords: list[tuple[str, int, float]] = []

    for word, doc_count in sorted(word_document_count.items(), key=lambda x: x[1], reverse=True):
        if not is_stock_related_word(word):
            continue
        pct = doc_count / len(df) * 100 if len(df) else 0.0
        if doc_count >= threshold:
            core_keywords.append((word, doc_count, pct))
        elif doc_count >= threshold * 0.3:
            support_keywords.append((word, doc_count, pct))

    core_words = [word for word, _, _ in core_keywords]
    support_words = [word for word, _, _ in support_keywords]
    print(f"핵심 키워드: {len(core_words)}개 | 보조 키워드: {len(support_words)}개")

    df["stock_score"] = df["text_raw"].apply(lambda text: score_comment(text, core_words, support_words))
    df["is_stock"] = df.apply(
        lambda row: classify_comment(row["text_raw"], row["stock_score"], core_words),
        axis=1,
    )

    stock_df = df[df["is_stock"] == "stock"].copy()
    other_df = df[df["is_stock"] == "other"].copy()
    stock_ratio = len(stock_df) / len(df) * 100 if len(df) else 0.0
    print(f"주식/경제: {len(stock_df):,}개 ({stock_ratio:.1f}%)")
    print(f"기타: {len(other_df):,}개")

    year_out = output_dir / str(year)
    year_out.mkdir(parents=True, exist_ok=True)

    keyword_result = {
        "core_keywords": [{"word": w, "count": int(c), "percentage": float(p)} for w, c, p in core_keywords],
        "support_keywords": [{"word": w, "count": int(c), "percentage": float(p)} for w, c, p in support_keywords],
    }
    with open(year_out / f"keyword_analysis_result_{year}.json", "w", encoding="utf-8") as f:
        json.dump(keyword_result, f, ensure_ascii=False, indent=2)

    df.to_csv(year_out / f"classified_stock_comments_{year}.csv", index=False, encoding="utf-8-sig")
    stock_df.to_csv(year_out / f"comments_final_stock_only_{year}.csv", index=False, encoding="utf-8-sig")
    stock_df.drop(columns=["stock_score", "is_stock"], errors="ignore").to_csv(
        year_out / f"comments_stock_clean_{year}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    other_df.to_csv(year_out / f"comments_other_{year}.csv", index=False, encoding="utf-8-sig")

    summary = {
        "year": year,
        "total_comments": int(len(df)),
        "stock_comments": int(len(stock_df)),
        "other_comments": int(len(other_df)),
        "stock_ratio": float(stock_ratio),
        "avg_score": float(stock_df["stock_score"].mean()) if len(stock_df) else 0.0,
        "median_score": float(stock_df["stock_score"].median()) if len(stock_df) else 0.0,
        "core_keywords_count": len(core_words),
        "support_keywords_count": len(support_words),
        "top_core_keywords": core_words[:20],
    }
    with open(year_out / f"stock_classification_summary_{year}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {year_out / f'comments_stock_clean_{year}.csv'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: keep stock/economy-related comments.")
    parser.add_argument("--years", default="2014-2025")
    parser.add_argument("--input-dir", default=str(BASE_DIR / "toxicity_filter"))
    parser.add_argument("--output-dir", default=str(BASE_DIR / "final_filtered"))
    parser.add_argument("--min-doc-ratio", type=float, default=0.005)
    args = parser.parse_args()

    print("=" * 80)
    print("3단계: 주식/경제 키워드 분석 + 최종 필터링")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    summaries = []
    for year in parse_years(args.years):
        summary = run_year(year, Path(args.input_dir), output_dir, args.min_doc_ratio)
        if summary:
            summaries.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "final_filter_summary_all.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print("\n전체 완료")
    print(f"최종 출력 폴더: {output_dir}")


if __name__ == "__main__":
    main()
