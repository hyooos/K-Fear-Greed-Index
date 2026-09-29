import glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

ROOT = Path(__file__).resolve().parents[2]
FS = sorted(glob.glob(str(ROOT / "05_Sentiment_analysis" / "data") + "/"
                      "recollect_sentiment_scores_model_toxicity/sentiment_with_prob_*.csv"))
d = pd.concat([pd.read_csv(f, usecols=["p_pos", "p_neg"]) for f in FS], ignore_index=True)
for c in ("p_pos", "p_neg"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna()

raw = (d["p_pos"] - d["p_neg"]).values
norm = ((d["p_pos"] - d["p_neg"]) / (d["p_pos"] + d["p_neg"] + 1e-8)).values
bins = np.linspace(-1, 1, 80)

LAB = {
    "ko": dict(x="감성 점수  (-1: 완전 부정  ~  +1: 완전 긍정)", y="댓글 수",
              l="재정규화 전   S = p_pos - p_neg",
              r="재정규화 후   S = (p_pos - p_neg) / (p_pos + p_neg)"),
    "en": dict(x="sentiment score  (-1: fully negative  ~  +1: fully positive)", y="number of comments",
              l="before   S = p_pos - p_neg",
              r="after   S = (p_pos - p_neg) / (p_pos + p_neg)"),
}

OUTDIR = str(ROOT / "10_Paper_outputs" / "figures")
for lang in ("ko", "en"):
    L = LAB[lang]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].hist(raw, bins=bins, color="#5B8DB8", alpha=.9)
    ax[0].axvline(0, color="#888", lw=.9, ls="--")
    ax[0].set_xlabel(L["x"]); ax[0].set_ylabel(L["y"])
    ax[1].hist(norm, bins=bins, color="#E07B54", alpha=.9)
    ax[1].axvline(0, color="#888", lw=.9, ls="--")
    ax[1].set_xlabel(L["x"]); ax[1].set_ylabel(L["y"])
    fig.tight_layout()
    out = f"{OUTDIR}/Figure_sent_renorm_{lang}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)

print(f"\nN = {len(d):,}")
print(f"전  std={raw.std():.4f}  IQR=[{np.percentile(raw,25):.4f}, {np.percentile(raw,75):.4f}]  median={np.median(raw):.4f}")
print(f"후  std={norm.std():.4f}  IQR=[{np.percentile(norm,25):.4f}, {np.percentile(norm,75):.4f}]  median={np.median(norm):.4f}")
