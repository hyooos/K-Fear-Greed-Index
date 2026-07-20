import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

TOXIC_PATTERNS = {
    "direct_abuse": [
        r"병신|븅신|ㅄ|ㅂㅅ|등신|개새|개쉐|새끼|새퀴|ㅅㄲ|씹|ㅆㅂ|시발|씨발|ㅅㅂ|좆|존나|졸라",
        r"꺼져|닥쳐|죽어|뒤져|뒤진|디져|미친놈|미친년|또라이|돌아이|정신병|정병",
    ],
    "hate_slur": [
        r"틀딱|맘충|한남|한녀|김치녀|된장녀|급식충|좌좀|일베|메갈|워마드",
        r"짱깨|쪽바리|왜놈|흑형|깜둥|조센징",
    ],
    "sexual": [
        r"보지|자지|섹스|강간|따먹|걸레|창녀|성괴",
    ],
}

SEVERITY = {
    "direct_abuse": 0.55,
    "hate_slur": 0.65,
    "sexual": 0.75,
}


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


def compile_patterns() -> dict[str, re.Pattern]:
    return {
        name: re.compile("|".join(f"(?:{pattern})" for pattern in patterns), flags=re.IGNORECASE)
        for name, patterns in TOXIC_PATTERNS.items()
    }


def score_text(text: str, patterns: dict[str, re.Pattern]) -> tuple[float, list[str]]:
    if pd.isna(text) or not str(text).strip():
        return 1.0, ["empty"]

    value = str(text)
    reasons: list[str] = []
    score = 0.0
    for name, pattern in patterns.items():
        matches = pattern.findall(value)
        if not matches:
            continue
        reasons.append(name)
        score += SEVERITY[name]
        if len(matches) >= 2:
            score += 0.12

    if len(value) <= 3 and score > 0:
        score += 0.10
    return min(score, 1.0), reasons


def summarize(df: pd.DataFrame, hard_tau: float, gamma: float) -> dict:
    total = len(df)
    empty = int(df["is_empty"].sum())
    kept = int(df["keep"].sum())
    dropped = int((~df["keep"]).sum())
    return {
        "total_rows": total,
        "empty_rows": empty,
        "non_empty_rows": total - empty,
        "mode": "fast_keyword_weight",
        "hard_tau": hard_tau,
        "gamma": gamma,
        "kept": kept,
        "dropped": dropped,
        "mean_toxicity": float(df.loc[~df["is_empty"], "toxicity_score"].mean()) if total - empty else 0.0,
        "dropped_ratio": float(dropped / total * 100) if total else 0.0,
    }


def run_year(
    year: int,
    input_dir: Path,
    output_dir: Path,
    patterns: dict[str, re.Pattern],
    *,
    hard_tau: float,
    gamma: float,
    overwrite: bool,
) -> dict | None:
    print(f"\n{'=' * 80}")
    print(f"{year}년 빠른 독성 키워드 필터링")
    print(f"{'=' * 80}")

    inp = input_dir / f"comments_political_removed_{year}.csv"
    kept_path = output_dir / f"comments_toxicity_kept_{year}.csv"
    if kept_path.exists() and not overwrite:
        print(f"이미 있음, 건너뜀: {kept_path}")
        return None
    if not inp.exists():
        print(f"파일 없음, 건너뜀: {inp}")
        return None

    df = pd.read_csv(inp)
    print(f"로드 완료: {len(df):,}행")
    df["text_raw"] = df["text_raw"].fillna("").astype(str)
    df["is_empty"] = df["text_raw"].str.strip().eq("")

    scored = df["text_raw"].apply(lambda text: score_text(text, patterns))
    df["toxicity_score"] = scored.apply(lambda item: item[0]).clip(0.0, 1.0)
    df["toxicity_reason"] = scored.apply(lambda item: "|".join(item[1]))
    df["weight"] = (1.0 - df["toxicity_score"]).clip(lower=0.0) ** gamma
    df["keep"] = (~df["is_empty"]) & (df["toxicity_score"] < hard_tau)
    df.loc[df["is_empty"], "weight"] = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"comments_toxicity_all_{year}.csv", index=False, encoding="utf-8-sig")
    df[df["keep"]].copy().to_csv(kept_path, index=False, encoding="utf-8-sig")
    df[~df["keep"]].copy().to_csv(
        output_dir / f"comments_toxicity_dropped_{year}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = summarize(df, hard_tau, gamma)
    summary["year"] = year
    summary["reason_counts"] = (
        df.loc[~df["keep"], "toxicity_reason"]
        .replace("", "unknown")
        .value_counts()
        .head(20)
        .to_dict()
    )
    with open(output_dir / f"toxicity_summary_{year}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"kept: {summary['kept']:,}행")
    print(f"dropped: {summary['dropped']:,}행")
    print(f"저장 완료: {kept_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2 fast fallback: keyword-based toxicity filter.")
    parser.add_argument("--years", default="2014-2025")
    parser.add_argument("--input-dir", default=str(BASE_DIR / "political_filter"))
    parser.add_argument("--output-dir", default=str(BASE_DIR / "toxicity_filter_fast"))
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--hard-tau", type=float, default=0.95)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("2단계: 빠른 독성 키워드 필터링")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    patterns = compile_patterns()
    summaries = []
    for year in parse_years(args.years):
        summary = run_year(
            year,
            Path(args.input_dir),
            output_dir,
            patterns,
            hard_tau=args.hard_tau,
            gamma=args.gamma,
            overwrite=args.overwrite,
        )
        if summary:
            summaries.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "toxicity_summary_all.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print("\n전체 완료")
    print(f"다음 단계 입력 폴더: {output_dir}")


if __name__ == "__main__":
    main()
