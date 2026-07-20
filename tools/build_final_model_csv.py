from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("/Users/hyowon/Desktop/uni/3-1/기계학습/data")
SENTIMENT_PATH = Path(
    "/Users/hyowon/Desktop/uni/3-1/기계학습/kfgi_최종/"
    "4_Sentiment_analysis/recollect_sentiment_final_model_toxicity/daily_sentiment.csv"
)

OUT_RAW = DATA_DIR / "KFG_final_10y_raw.csv"
OUT_MODEL = DATA_DIR / "KFG_final_10y.csv"
OUT_MISSING = DATA_DIR / "KFG_final_10y_missing_report.csv"


def read_csv_with_date(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        if "일자" in df.columns:
            df = df.rename(columns={"일자": "date"})
        else:
            raise ValueError(f"{path} does not contain a date column")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def select_subindex(path: Path, source_col: str, target_col: str) -> pd.DataFrame:
    df = read_csv_with_date(path)
    missing = {"date", source_col} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df[["date", source_col]].rename(columns={source_col: target_col})


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subindex_specs = [
        ("subindex1/sub_index1_momentum_2015_2025.csv", "Momentum_Score", "sub_index1"),
        ("subindex2/sub_index2_strength_2015_2025.csv", "Strength_Score", "sub_index2"),
        ("subindex3/sub_index3_breadth_2015_2025.csv", "osc_score_0_100", "sub_index3"),
        ("subindex4/sub_index4_pcratio_2015_2025.csv", "score_pcr", "sub_index4"),
        ("subindex5/sub_index5_volatility_2015_2025.csv", "Score_Fear", "sub_index5"),
        ("subindex6/sub_index6_safedemand_2015_2025.csv", "score_safe_demand", "sub_index6"),
        ("subindex7/sub_index7_junk_2015_2025.csv", "Final_JunkIndex", "sub_index7"),
    ]

    merged = None
    for rel_path, source_col, target_col in subindex_specs:
        df = select_subindex(DATA_DIR / rel_path, source_col, target_col)
        merged = df if merged is None else merged.merge(df, on="date", how="left")

    price = read_csv_with_date(DATA_DIR / "subindex1/subindex1_kospi200_2013_2025.csv")
    price = price[["date", "종가"]].rename(columns={"종가": "kospi200_close"})

    sentiment_cols = [
        "date",
        "comment_count",
        "sent_norm_w",
        "sent_strength_w",
        "sent_std",
        "neg_z",
        "effective_n",
        "heat",
    ]
    sentiment = read_csv_with_date(SENTIMENT_PATH)
    missing_sent_cols = set(sentiment_cols) - set(sentiment.columns)
    if missing_sent_cols:
        raise ValueError(f"{SENTIMENT_PATH} missing columns: {sorted(missing_sent_cols)}")
    sentiment = sentiment[sentiment_cols]

    final = (
        merged.merge(price, on="date", how="left")
        .merge(sentiment, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )

    sent_feature_cols = [
        "sent_norm_w",
        "sent_strength_w",
        "sent_std",
        "neg_z",
        "effective_n",
        "heat",
    ]
    final[sent_feature_cols] = final[sent_feature_cols].ffill()
    final["comment_count"] = final["comment_count"].fillna(0)

    final["kospi_close"] = final["kospi200_close"]
    final["log_return"] = np.log(final["kospi200_close"]).diff()
    final["log_return_t+1"] = final["log_return"].shift(-1)

    column_order = [
        "date",
        "sub_index1",
        "sub_index2",
        "sub_index3",
        "sub_index4",
        "sub_index5",
        "sub_index6",
        "sub_index7",
        "sent_norm_w",
        "sent_strength_w",
        "sent_std",
        "neg_z",
        "effective_n",
        "heat",
        "comment_count",
        "kospi200_close",
        "kospi_close",
        "log_return",
        "log_return_t+1",
    ]
    final = final[column_order]

    required_model_cols = [
        "sub_index1",
        "sub_index2",
        "sub_index3",
        "sub_index4",
        "sub_index5",
        "sub_index6",
        "sub_index7",
        "sent_norm_w",
        "sent_strength_w",
        "sent_std",
        "neg_z",
        "effective_n",
        "heat",
        "kospi_close",
        "log_return",
        "log_return_t+1",
    ]
    model = final.dropna(subset=required_model_cols).reset_index(drop=True)

    missing_report = (
        final.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    return final, model, missing_report


def main() -> None:
    raw, model, missing_report = build_dataset()
    raw.to_csv(OUT_RAW, index=False)
    model.to_csv(OUT_MODEL, index=False)
    missing_report.to_csv(OUT_MISSING, index=False)

    print(f"raw: {OUT_RAW}")
    print(f"  rows={len(raw):,}, range={raw['date'].min().date()}~{raw['date'].max().date()}")
    print(f"model: {OUT_MODEL}")
    print(
        f"  rows={len(model):,}, range={model['date'].min().date()}~{model['date'].max().date()}"
    )
    print(f"missing_report: {OUT_MISSING}")
    print(missing_report.to_string(index=False))


if __name__ == "__main__":
    main()
