import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "jinkyeongk/kcELECTRA-toxic-detector"


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


@torch.no_grad()
def predict_toxicity_binary(
    texts: list[str],
    tokenizer,
    model,
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> list[float]:
    scores: list[float] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring toxicity"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        if logits.shape[-1] == 1:
            probs = torch.sigmoid(logits).squeeze(-1)
        else:
            probs = torch.softmax(logits, dim=-1)[:, 1]
        scores.extend(probs.detach().cpu().tolist())
    return scores


def summarize(df: pd.DataFrame, mode: str, tau: float, hard_tau: float, gamma: float) -> dict:
    total = len(df)
    empty = int(df["is_empty"].sum())
    kept = int(df["keep"].sum())
    dropped = int((~df["keep"]).sum())
    return {
        "total_rows": total,
        "empty_rows": empty,
        "non_empty_rows": total - empty,
        "mode": mode,
        "tau": tau,
        "hard_tau": hard_tau,
        "gamma": gamma,
        "kept": kept,
        "dropped": dropped,
        "mean_toxicity": float(df.loc[~df["is_empty"], "toxicity_score"].mean()) if total - empty else 0.0,
    }


def run_year(
    year: int,
    input_dir: Path,
    output_dir: Path,
    tokenizer,
    model,
    *,
    batch_size: int,
    max_length: int,
    mode: str,
    tau: float,
    gamma: float,
    hard_tau: float,
    device: str,
    overwrite: bool,
) -> Optional[dict]:
    print(f"\n{'=' * 80}")
    print(f"{year}년 독성 필터링")
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

    idx = df.index[~df["is_empty"]].tolist()
    texts = df.loc[idx, "text_raw"].tolist()
    df["toxicity_score"] = 0.0
    if texts:
        scores = predict_toxicity_binary(
            texts,
            tokenizer,
            model,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        df.loc[idx, "toxicity_score"] = scores
    df["toxicity_score"] = df["toxicity_score"].fillna(0.0).clip(0.0, 1.0)

    if mode == "drop":
        df["weight"] = 1.0
        df["keep"] = (~df["is_empty"]) & (df["toxicity_score"] < tau)
    else:
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

    summary = summarize(df, mode, tau, hard_tau, gamma)
    summary["year"] = year
    with open(output_dir / f"toxicity_summary_{year}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"kept: {summary['kept']:,}행")
    print(f"dropped: {summary['dropped']:,}행")
    print(f"저장 완료: {kept_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: apply toxicity filter.")
    parser.add_argument("--years", default="2014-2025")
    parser.add_argument("--input-dir", default=str(BASE_DIR / "political_filter"))
    parser.add_argument("--output-dir", default=str(BASE_DIR / "toxicity_filter"))
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--mode", choices=["drop", "weight"], default="weight")
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--hard-tau", type=float, default=0.95)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = args.device or default_device()
    print("=" * 80)
    print("2단계: 독성 필터링")
    print(f"model: {args.model_name}")
    print(f"device: {device}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.to(device).eval()

    output_dir = Path(args.output_dir)
    summaries = []
    for year in parse_years(args.years):
        summary = run_year(
            year,
            Path(args.input_dir),
            output_dir,
            tokenizer,
            model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            mode=args.mode,
            tau=args.tau,
            gamma=args.gamma,
            hard_tau=args.hard_tau,
            device=device,
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
