import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ── 데이터 로드 ──
#2024, 2025 파일도 있으면 아래처럼 합치면 됨
df = pd.concat([
    pd.read_csv("sentiment_with_prob_2023.csv"),
    pd.read_csv("sentiment_with_prob_2024.csv"),
    pd.read_csv("sentiment_with_prob_2025.csv"),
], ignore_index=True)
# df = pd.read_csv("sentiment_with_prob_2023.csv")

# keep=True인 행만 (필터링 통과한 댓글)
df = df[df["keep"] == True].copy()

# ── 재정규화 후 점수 계산 ──
df["sent_norm"] = (df["p_pos"] - df["p_neg"]) / (df["p_pos"] + df["p_neg"] + 1e-8)

# ── 시각화 ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("감성 점수 재정규화 전/후 분포 비교", fontsize=13, fontweight="bold")

bins = np.linspace(-1, 1, 80)

# 왼쪽: 재정규화 전 (sentiment_raw = p_pos - p_neg)
axes[0].hist(df["sentiment_raw"], bins=bins, color="#5B8DB8", edgecolor="none", alpha=0.85)
axes[0].set_title(
    f"재정규화 전  $S = P_{{pos}} - P_{{neg}}$\n"
    f"std = {df['sentiment_raw'].std():.3f},  "
    f"IQR = [{df['sentiment_raw'].quantile(0.25):.4f}, {df['sentiment_raw'].quantile(0.75):.4f}]",
    fontsize=10
)
axes[0].set_xlabel("감성 점수")
axes[0].set_ylabel("count")
axes[0].axvline(0, color="gray", lw=0.8, ls="--")

# 오른쪽: 재정규화 후
axes[1].hist(df["sent_norm"], bins=bins, color="#E07B54", edgecolor="none", alpha=0.85)
axes[1].set_title(
    f"재정규화 후  $S = (P_{{pos}} - P_{{neg}}) / (P_{{pos}} + P_{{neg}})$\n"
    f"std = {df['sent_norm'].std():.3f},  "
    f"IQR = [{df['sent_norm'].quantile(0.25):.4f}, {df['sent_norm'].quantile(0.75):.4f}]",
    fontsize=10
)
axes[1].set_xlabel("감성 점수")
axes[1].set_ylabel("count")
axes[1].axvline(0, color="gray", lw=0.8, ls="--")

plt.tight_layout()
plt.savefig("sentiment_renorm_comparison.png", dpi=150, bbox_inches="tight")
plt.show()