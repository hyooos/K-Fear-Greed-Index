import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


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


def load_year(input_dir: Path, year: int) -> pd.DataFrame | None:
    path = input_dir / f"sentiment_with_prob_{year}.csv"
    if not path.exists():
        print(f"파일 없음, 건너뜀: {path}")
        return None

    print(f"{year} -> {path.name} 로드")
    df = pd.read_csv(path)
    df["comment_at"] = pd.to_datetime(df["comment_at"], errors="coerce", utc=True)
    df = df[df["comment_at"].notna()].copy()
    df["comment_at"] = df["comment_at"].dt.tz_convert("Asia/Seoul")
    df["date"] = df["comment_at"].dt.date
    return df


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / (series.std() + 1e-8)


def make_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "weight" not in df.columns:
        df["weight"] = 1.0
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0).clip(lower=0.0)

    df["sent_raw"] = df["p_pos"] - df["p_neg"]
    df["sent_norm"] = (df["p_pos"] - df["p_neg"]) / (df["p_pos"] + df["p_neg"] + 1e-8)
    df["sent_raw_weighted"] = df["sent_raw"] * df["weight"]
    df["sent_norm_weighted"] = df["sent_norm"] * df["weight"]
    df["p_pos_weighted"] = df["p_pos"] * df["weight"]
    df["p_neg_weighted"] = df["p_neg"] * df["weight"]

    grouped = df.groupby("date", sort=True)
    daily = grouped.agg(
        comment_count=("comment_id", "count"),
        weight_sum=("weight", "sum"),
        weight_sq_sum=("weight", lambda x: (x**2).sum()),
        sent_raw_mean=("sent_raw", "mean"),
        sent_norm_mean=("sent_norm", "mean"),
        sent_std=("sent_raw", "std"),
        p_pos_mean=("p_pos", "mean"),
        p_neu_mean=("p_neu", "mean"),
        p_neg_mean=("p_neg", "mean"),
    )

    daily["sent_raw_mean_w"] = grouped["sent_raw_weighted"].sum() / (daily["weight_sum"] + 1e-8)
    daily["sent_norm_w"] = grouped["sent_norm_weighted"].sum() / (daily["weight_sum"] + 1e-8)
    daily["pos_mean_w"] = grouped["p_pos_weighted"].sum() / (daily["weight_sum"] + 1e-8)
    daily["neg_mean_w"] = grouped["p_neg_weighted"].sum() / (daily["weight_sum"] + 1e-8)
    daily["sent_strength_w"] = daily["pos_mean_w"] + daily["neg_mean_w"]
    daily["neg_score"] = daily["neg_mean_w"]

    roll_mean = daily["neg_score"].rolling(60, min_periods=20).mean()
    roll_std = daily["neg_score"].rolling(60, min_periods=20).std()
    daily["neg_z"] = (daily["neg_score"] - roll_mean) / (roll_std + 1e-8)

    daily["effective_n"] = (daily["weight_sum"] ** 2) / (daily["weight_sq_sum"] + 1e-8)
    daily["log_effective_n"] = np.log1p(daily["effective_n"])
    daily["heat"] = zscore(daily["sent_strength_w"]) * zscore(daily["log_effective_n"])

    return daily.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate comment-level sentiment into daily features.")
    parser.add_argument("--years", default="2014-2025")
    parser.add_argument("--input-dir", default=str(BASE_DIR / "recollect_sentiment_scores"))
    parser.add_argument("--output-dir", default=str(BASE_DIR / "recollect_sentiment_final"))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    summaries = []
    for year in parse_years(args.years):
        df = load_year(input_dir, year)
        if df is None:
            continue
        frames.append(df)
        summaries.append({"year": year, "rows": int(len(df))})
        print(f"{year}: {len(df):,}행")

    if not frames:
        raise SystemExit("감성 점수 파일이 없습니다.")

    df_all = pd.concat(frames, ignore_index=True).sort_values("comment_at").reset_index(drop=True)
    print(f"\n전체 병합: {len(df_all):,}행")

    daily = make_daily(df_all)
    daily_out = daily.dropna(subset=["sent_norm_w"]).reset_index(drop=True)

    out = output_dir / "daily_sentiment.csv"
    daily_out.to_csv(out, index=False, encoding="utf-8-sig")
    with open(output_dir / "daily_sentiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(input_dir),
                "rows": summaries,
                "comment_rows": int(len(df_all)),
                "daily_rows": int(len(daily_out)),
                "output_file": str(out),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n저장 완료: {out} ({len(daily_out):,}일)")


if __name__ == "__main__":
    main()
