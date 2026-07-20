import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "snunlp/KR-FinBert-SC"


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


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def label_id(model, label: str) -> int:
    label2id = {str(k).lower(): int(v) for k, v in model.config.label2id.items()}
    if label in label2id:
        return label2id[label]
    fallback = {"negative": 0, "neutral": 1, "positive": 2}
    return fallback[label]


@torch.no_grad()
def get_probabilities(
    texts: list[str],
    tokenizer,
    model,
    *,
    batch_size: int,
    max_length: int,
    device: str,
    pos_id: int,
    neu_id: int,
    neg_id: int,
) -> tuple[list[float], list[float], list[float]]:
    all_pos: list[float] = []
    all_neu: list[float] = []
    all_neg: list[float] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment"):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=1).detach().cpu()
        all_pos.extend(probs[:, pos_id].tolist())
        all_neu.extend(probs[:, neu_id].tolist())
        all_neg.extend(probs[:, neg_id].tolist())

    return all_pos, all_neu, all_neg


def input_path_for_year(input_dir: Path, year: int) -> Path:
    return input_dir / str(year) / f"comments_stock_clean_{year}.csv"


def run_year(
    year: int,
    input_dir: Path,
    output_dir: Path,
    tokenizer,
    model,
    *,
    batch_size: int,
    max_length: int,
    device: str,
    pos_id: int,
    neu_id: int,
    neg_id: int,
    overwrite: bool,
) -> dict | None:
    print(f"\n{'=' * 80}")
    print(f"{year}년 감성 분석")
    print(f"{'=' * 80}")

    inp = input_path_for_year(input_dir, year)
    out = output_dir / f"sentiment_with_prob_{year}.csv"
    if out.exists() and not overwrite:
        print(f"이미 있음, 건너뜀: {out}")
        return None
    if not inp.exists():
        print(f"파일 없음, 건너뜀: {inp}")
        return None

    df = pd.read_csv(inp)
    before = len(df)
    df["comment_at"] = pd.to_datetime(df["comment_at"], errors="coerce")
    df = df[df["comment_at"].notna()].copy()
    df = df[df["comment_at"].dt.year == year].copy()
    if "keep" in df.columns:
        df = df[df["keep"].astype(str).str.lower().isin(["true", "1"])].copy()
    if "is_empty" in df.columns:
        df = df[~df["is_empty"].astype(str).str.lower().isin(["true", "1"])].copy()
    df = df.dropna(subset=["text_raw"]).copy()
    df = df[df["text_raw"].astype(str).str.strip() != ""].copy()
    df.reset_index(drop=True, inplace=True)

    print(f"입력: {inp}")
    print(f"필터 전: {before:,}행 | 감성 분석 대상: {len(df):,}행")

    p_pos, p_neu, p_neg = get_probabilities(
        df["text_raw"].astype(str).tolist(),
        tokenizer,
        model,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        pos_id=pos_id,
        neu_id=neu_id,
        neg_id=neg_id,
    )

    df["p_pos"] = p_pos
    df["p_neu"] = p_neu
    df["p_neg"] = p_neg
    df["sentiment_raw"] = df["p_pos"] - df["p_neg"]
    df["sentiment_norm"] = (
        (df["p_pos"] - df["p_neg"]) / (df["p_pos"] + df["p_neg"] + 1e-8)
    ).clip(-1.0, 1.0)
    df["sentiment_label"] = df[["p_neg", "p_neu", "p_pos"]].idxmax(axis=1).str.replace("p_", "", regex=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    summary = {
        "year": year,
        "input_file": str(inp),
        "output_file": str(out),
        "input_rows": int(before),
        "scored_rows": int(len(df)),
        "mean_sentiment_raw": float(df["sentiment_raw"].mean()) if len(df) else 0.0,
        "mean_sentiment_norm": float(df["sentiment_norm"].mean()) if len(df) else 0.0,
        "label_counts": df["sentiment_label"].value_counts().to_dict(),
    }
    with open(output_dir / f"sentiment_summary_{year}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {out}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score KR-FinBERT sentiment for filtered NAVER comments.")
    parser.add_argument("--years", default="2014-2025")
    parser.add_argument(
        "--input-dir",
        default="3_Filtering_final/recollect_final_filtered_fast_toxicity",
    )
    parser.add_argument("--output-dir", default=str(BASE_DIR / "recollect_sentiment_scores"))
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = args.device or default_device()
    print("=" * 80)
    print("4단계: KR-FinBERT 감성 점수 계산")
    print(f"model: {args.model_name}")
    print(f"device: {device}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.to(device).eval()

    pos_id = label_id(model, "positive")
    neu_id = label_id(model, "neutral")
    neg_id = label_id(model, "negative")
    print(f"label ids: negative={neg_id}, neutral={neu_id}, positive={pos_id}")

    summaries = []
    for year in parse_years(args.years):
        summary = run_year(
            year,
            Path(args.input_dir),
            Path(args.output_dir),
            tokenizer,
            model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            pos_id=pos_id,
            neu_id=neu_id,
            neg_id=neg_id,
            overwrite=args.overwrite,
        )
        if summary:
            summaries.append(summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "sentiment_summary_all.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print("\n전체 완료")
    print(f"출력 폴더: {output_dir}")


if __name__ == "__main__":
    main()
