# apply_toxicity_navert.py
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# ========== 경로 설정 ==========
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR  = BASE_DIR / "political_filter"   # 1단계 결과 폴더
OUTPUT_DIR = BASE_DIR / "toxicity_filter"    # 2단계 결과 저장 폴더
OUTPUT_DIR.mkdir(exist_ok=True)

YEARS = [2023, 2024, 2025]

MODEL_NAME  = "jinkyeongk/kcELECTRA-toxic-detector"
BATCH_SIZE  = 32
MAX_LENGTH  = 256
MODE        = "weight"   # "drop" or "weight"
TAU         = 0.90       # drop 모드 기준
GAMMA       = 2.0        # weight 모드 감쇠
HARD_TAU    = 0.95       # weight 모드 hard drop 기준


# ========== 독성 점수 계산 ==========
@torch.no_grad()
def predict_toxicity_binary(
    texts: List[str],
    model_name: str,
    batch_size: int = 32,
    max_length: int = 256,
    device: Optional[str] = None,
) -> List[float]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device).eval()

    scores: List[float] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring toxicity"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        if logits.shape[-1] == 1:
            p = torch.sigmoid(logits).squeeze(-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            p = probs[:, 1]
        scores.extend(p.detach().cpu().tolist())
    return scores


def summarize_thresholds(df, mode, tau, hard_tau):
    total = len(df)
    empty = int(df["is_empty"].sum())
    print("\n==================== SUMMARY ====================")
    print(f"total rows: {total} | empty: {empty} | non-empty: {total - empty}")
    print(f"mode: {mode}")
    if mode == "drop":
        print(f"drop rule: empty OR toxicity >= {tau}")
        print(f"kept: {int(df['keep'].sum())} | dropped: {int((~df['keep']).sum())}")
    else:
        print(f"weight=(1-tox)^{GAMMA}, hard_drop >= {hard_tau}")
        print(f"hard dropped: {int((~df['keep']).sum())}")
    print("=================================================\n")


def print_bins(df):
    bins   = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
    labels = ["[0,.2)","[.2,.4)","[.4,.6)","[.6,.8)","[.8,.9)","[.9,.95)","[.95,1]"]
    tmp = df[~df["is_empty"]].copy()
    if len(tmp) == 0:
        return
    tmp["tox_bin"] = pd.cut(tmp["toxicity_score"], bins=bins, labels=labels,
                            include_lowest=True, right=False)
    g = tmp.groupby("tox_bin", dropna=False).agg(
        n=("toxicity_score", "size"),
        mean_tox=("toxicity_score", "mean"),
        mean_weight=("weight", "mean"),
        kept=("keep", "sum"),
    ).reset_index()
    print("\n[TOXICITY BINS]")
    print(g.to_string(index=False))


# ========== 연도별 실행 ==========
for year in YEARS:
    print(f"\n{'='*80}")
    print(f"{year}년 독성 필터링 시작...")
    print(f"{'='*80}")

    inp = INPUT_DIR / f"comments_political_removed_{year}.csv"
    if not inp.exists():
        print(f"  ⚠️  파일 없음, 건너뜀: {inp}")
        continue

    df = pd.read_csv(inp)
    print(f"로드 완료: {len(df):,}행")

    # 빈 텍스트 처리
    df["text_raw"] = df["text_raw"].fillna("").astype(str)
    df["is_empty"] = df["text_raw"].str.strip().eq("")

    # 독성 점수 계산 (비어있지 않은 것만)
    idx   = df.index[~df["is_empty"]].tolist()
    texts = df.loc[idx, "text_raw"].tolist()
    df["toxicity_score"] = 0.0
    if len(texts) > 0:
        scores = predict_toxicity_binary(texts, MODEL_NAME, BATCH_SIZE, MAX_LENGTH)
        df.loc[idx, "toxicity_score"] = scores
    df["toxicity_score"] = df["toxicity_score"].fillna(0.0).clip(0.0, 1.0)

    # weight / keep 계산
    if MODE == "drop":
        df["weight"] = 1.0
        df["keep"]   = (~df["is_empty"]) & (df["toxicity_score"] < TAU)
    else:
        df["weight"] = (1.0 - df["toxicity_score"]).clip(lower=0.0) ** GAMMA
        df["keep"]   = (~df["is_empty"]) & (df["toxicity_score"] < HARD_TAU)
        df.loc[df["is_empty"], "weight"] = 0.0

    # 저장
    df.to_csv(OUTPUT_DIR / f"comments_toxicity_all_{year}.csv",
              index=False, encoding="utf-8-sig")

    kept_df    = df[df["keep"]].copy()
    dropped_df = df[~df["keep"]].copy()

    kept_df.to_csv(OUTPUT_DIR / f"comments_toxicity_kept_{year}.csv",
                   index=False, encoding="utf-8-sig")
    dropped_df.to_csv(OUTPUT_DIR / f"comments_toxicity_dropped_{year}.csv",
                      index=False, encoding="utf-8-sig")

    print(f"✓ kept:    {len(kept_df):,}행  →  comments_toxicity_kept_{year}.csv")
    print(f"✓ dropped: {len(dropped_df):,}행  →  comments_toxicity_dropped_{year}.csv")

    summarize_thresholds(df, MODE, TAU, HARD_TAU)
    print_bins(df)

print(f"\n{'='*80}")
print("전체 완료! toxicity_filter/ 폴더 확인하세요")
print("⭐ 다음 단계 입력 파일: comments_toxicity_kept_2023/2024/2025.csv")
print(f"{'='*80}")