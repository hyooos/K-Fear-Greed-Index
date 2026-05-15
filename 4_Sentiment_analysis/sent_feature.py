import pandas as pd
import numpy as np
import os
from pathlib import Path

BASE_DIR   = Path(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR  = BASE_DIR / "sentiment_scores"
OUTPUT_DIR = BASE_DIR / "sentiment_final"
OUTPUT_DIR.mkdir(exist_ok=True)

YEARS = [2023, 2024, 2025]  # 2022 데이터 없으므로 제거

'''
연도별 감성 점수 파일 로드
'''
def load_year(year):
    path = INPUT_DIR / f"sentiment_with_prob_{year}.csv"
    if not path.exists():
        raise ValueError(f"{year} 파일이 없습니다: {path}")

    print(f"{year} → {path.name} 로드")
    df = pd.read_csv(path)

    df["comment_at"] = pd.to_datetime(df["comment_at"], utc=True)
    df["comment_at"] = df["comment_at"].dt.tz_convert("Asia/Seoul")
    df["date"] = df["comment_at"].dt.date

    return df

dfs = {}
for y in YEARS:
    dfs[y] = load_year(y)
    print(f"{y}: {len(dfs[y]):,}행")

df_all = pd.concat(dfs.values()).sort_values("date").reset_index(drop=True)
print(f"\n전체 병합: {len(df_all):,}행")

'''
감성 지표 계산 및 일별 집계
'''
def make_daily(df):
    df = df.copy()

    df["sent_raw"]          = df["p_pos"] - df["p_neg"]
    df["sent_raw_weighted"] = df["sent_raw"] * df["weight"]
    df["p_pos_weighted"]    = df["p_pos"] * df["weight"]
    df["p_neg_weighted"]    = df["p_neg"] * df["weight"]

    grouped = df.groupby("date")

    daily = grouped.agg(
        weight_sum    =("weight", "sum"),
        weight_sq_sum =("weight", lambda x: (x**2).sum()),
        sent_std      =("sent_raw", "std"),
    )

    daily["pos_mean_w"] = grouped["p_pos_weighted"].sum() / daily["weight_sum"]
    daily["neg_mean_w"] = grouped["p_neg_weighted"].sum() / daily["weight_sum"]

    daily["sent_norm_w"]     = ((daily["pos_mean_w"] - daily["neg_mean_w"]) /
                                (daily["pos_mean_w"] + daily["neg_mean_w"] + 1e-8))
    daily["sent_strength_w"] = daily["pos_mean_w"] + daily["neg_mean_w"]

    daily = daily.sort_index()
    daily["neg_score"] = daily["neg_mean_w"]

    roll_mean = daily["neg_score"].rolling(60).mean()
    roll_std  = daily["neg_score"].rolling(60).std()
    daily["neg_z"] = (daily["neg_score"] - roll_mean) / (roll_std + 1e-8)

    daily["effective_n"]     = (daily["weight_sum"]**2) / (daily["weight_sq_sum"] + 1e-8)
    daily["log_effective_n"] = np.log1p(daily["effective_n"])

    def zscore(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    daily["heat"] = zscore(daily["sent_strength_w"]) * zscore(daily["log_effective_n"])

    daily = daily.reset_index()
    return daily[["date", "sent_norm_w", "sent_strength_w", "sent_std", "neg_z", "effective_n", "heat"]]


daily_all = make_daily(df_all)
daily_all = daily_all.dropna().reset_index(drop=True)

out = OUTPUT_DIR / "daily_sentiment.csv"
daily_all.to_csv(out, index=False)
print(f"\n✓ 저장 완료 → sentiment_final/daily_sentiment.csv ({len(daily_all):,}일)")