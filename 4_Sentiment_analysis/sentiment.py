# compute_sentiment.py
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
from pathlib import Path

# ========== 경로 설정 ==========
BASE_DIR  = Path(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR  = BASE_DIR / "final_filtered"   # 3단계 결과 폴더
OUTPUT_DIR = BASE_DIR / "sentiment_scores"
OUTPUT_DIR.mkdir(exist_ok=True)

YEARS = [2023, 2024, 2025]

# ========== 모델 로드 (한 번만) ==========
model_name = "snunlp/KR-FinBert-SC"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()

neg_id = model.config.label2id["negative"]
neu_id = model.config.label2id["neutral"]
pos_id = model.config.label2id["positive"]

# ========== 감성 확률 계산 함수 ==========
@torch.no_grad()
def get_probabilities(texts, batch_size=64, max_length=64):
    all_pos, all_neu, all_neg = [], [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="감성 분석"):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        ).to(device)
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=1)
        all_pos.extend(probs[:, pos_id].cpu().numpy())
        all_neu.extend(probs[:, neu_id].cpu().numpy())
        all_neg.extend(probs[:, neg_id].cpu().numpy())
    return all_pos, all_neu, all_neg

# ========== 연도별 실행 ==========
for year in YEARS:
    print(f"\n{'='*60}")
    print(f"{year}년 감성 분석 시작...")
    print(f"{'='*60}")

    # 입력 파일: final_filtered/2023/comments_stock_clean_2023.csv
    inp = INPUT_DIR / str(year) / f"comments_stock_clean_{year}.csv"
    if not inp.exists():
        print(f"  ⚠️  파일 없음, 건너뜀: {inp}")
        continue

    # 데이터 로드 및 필터링
    df = pd.read_csv(inp)
    df["comment_at"] = pd.to_datetime(df["comment_at"])
    df = df[df["comment_at"].dt.year == year]
    df = df[(df["is_empty"] == 0) & (df["keep"] == 1)]
    df = df.dropna(subset=["text_raw"])
    df = df[df["text_raw"].str.strip() != ""]
    df.reset_index(drop=True, inplace=True)
    print(f"댓글 수: {len(df):,}개")

    # 감성 분석
    p_pos, p_neu, p_neg = get_probabilities(df["text_raw"].tolist())

    df["p_pos"] = p_pos
    df["p_neu"] = p_neu
    df["p_neg"] = p_neg
    df["sentiment_raw"] = df["p_pos"] - df["p_neg"]

    # 저장
    out = OUTPUT_DIR / f"sentiment_with_prob_{year}.csv"
    df.to_csv(out, index=False)
    print(f"✓ {year}년 저장 완료 → sentiment_scores/sentiment_with_prob_{year}.csv")

print(f"\n{'='*60}")
print("전체 완료! sentiment_scores/ 폴더 확인하세요")
print(f"{'='*60}")