import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr
from config import CFG, KFGI_FEATS_BASE, DIRECTION


def get_kfgi_feats_and_dirvec(df, exclude_sentiment=False):
    feats = [f for f in KFGI_FEATS_BASE if f in df.columns]
    if exclude_sentiment:
        feats = [f for f in feats
                 if not any(k in f for k in ["sent_", "neg_z", "composite"])]
    dir_vec = np.array([DIRECTION.get(f, 1) for f in feats])
    return feats, dir_vec


def create_walkforward_kfgi(df, feats, dir_vec, min_obs):
    """
    Walk-forward 방식으로 K-FGI 생성.
    각 시점 t에서 [0, t-1] 구간만 사용하여 RidgeCV 학습.
    결과를 p1~p99 구간 기준으로 0~100 스케일링.
    """
    df = df.copy()
    n  = len(df)
    raw_scores = np.full(n, np.nan)

    for t in range(min_obs, n):
        X_tr = df[feats].iloc[:t].values
        y_tr = df["target_5d"].iloc[:t].values
        mask = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr)
        X_tr, y_tr = X_tr[mask], y_tr[mask]
        if len(X_tr) < 30:
            continue
        try:
            sc    = StandardScaler()
            X_s   = sc.fit_transform(X_tr)
            X_s   = np.clip(X_s, -3, 3)
            ridge = RidgeCV(alphas=np.logspace(-2, 2, 10))
            ridge.fit(X_s, y_tr)
            coef  = ridge.coef_ * dir_vec
            w     = coef / (np.sum(np.abs(coef)) + 1e-12)
            x_t   = df[feats].iloc[t:t+1].values
            if np.isnan(x_t).any():
                continue
            raw_scores[t] = float(
                sc.transform(x_t).clip(-3, 3) @ w
            )
        except Exception:
            continue

    valid = raw_scores[~np.isnan(raw_scores)]
    if len(valid) == 0:
        raise ValueError("K-FGI 유효 추정값 없음")
    p1, p99 = np.percentile(valid, 1), np.percentile(valid, 99)
    kfgi = 100 * (np.clip(raw_scores, p1, p99) - p1) / (p99 - p1 + 1e-12)
    df["K_FGI"] = kfgi

    before = len(df)
    df = df.dropna(subset=["K_FGI"]).reset_index(drop=True)
    print(f"    K-FGI: {before}→{len(df)}행  "
          f"(시작일={df['date'].min().date()})")
    return df


def compute_kfgi_stats(df):
    rho_1d, p1d = spearmanr(df["K_FGI"], df["target_reg"])
    rho_5d, p5d = spearmanr(df["K_FGI"], df["target_5d"])
    print(f"    K-FGI ↔ 1일 수익률  ρ={rho_1d:.4f} (p={p1d:.4f})")
    print(f"    K-FGI ↔ 5일 수익률  ρ={rho_5d:.4f} (p={p5d:.4f})")
    kfgi_momentum = (rho_5d > 0)
    kdir = "순방향(모멘텀)" if kfgi_momentum else "역발상(공포→매수)"
    print(f"    ★ K-FGI 방향 자동 결정: {kdir}")
    return rho_1d, rho_5d, kfgi_momentum, kdir