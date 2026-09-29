"""
다운스트림 전략 성과 민감도  (γ, 감성 이동평균 window)
=====================================================

목적
  파라미터를 바꿔가며 K-FGI -> 포지션 -> 전략 수익률을 다시 산출하고
  Sharpe / MDD / CVaR(5%) / Downside vol / 하락일 방어율 / Calmar 를 비교한다.

  실험 1  gamma in {1, 2, 3}   : 독성 가중치 w=(1-tox)^gamma
          -> 댓글 -> 일별 감성피처 6종 재계산 -> KFG_final 치환 -> 전 파이프라인 재실행
  실험 2  sent_composite MA window in {5, 10, 20}
          -> 감성 이동평균 기간만 바꿔 재실행 (gamma=2 고정)

주의
  - EGARCH 변동성은 수익률만의 함수이므로 캐시( experiment_outputs/cache/egarch_vol_10y.csv )
    를 재사용한다. gamma / MA 변경과 무관.
  - 08_Modeling/scripts/run_10y_kfgi_experiments.py 의 로직을 그대로 옮겨왔다
    (build_features / walk-forward Ridge(양·음 제약) / compute_positions).
"""
from __future__ import annotations

import glob
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.bbox"] = "tight"

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

COMMENT_GLOB = ROOT / "05_Sentiment_analysis/data/recollect_sentiment_scores_model_toxicity/sentiment_with_prob_*.csv"
KFG_FINAL = ROOT / "06_Merge_to_final_csv/KFG_final_10y.csv"
EGARCH_CACHE = ROOT / "08_Modeling/experiment_outputs/cache/egarch_vol_10y.csv"

CFG = dict(egarch_min_obs=60, kfgi_min_obs=60, fee=0.00015, target_vol_ann=0.15,
           max_leverage=2.0, min_leverage=0.1, vol_shock_thr=2.0, tail_loss_thr=-0.07)
TARGET_DAILY_VOL = CFG["target_vol_ann"] / np.sqrt(252)
TREND_MAP = {0: 0.0, 1: 0.35, 2: 0.70, 3: 1.0}

KFGI_FEATS_BASE = [f"sub_index{i}" for i in range(2, 8)] + [
    "sent_norm_w", "sent_energy", "sent_std_inv", "neg_z_inv",
    "sent_composite", "sent_composite_ma10", "egarch_vol", "vol_regime_high", "vol_ratio"]
DIRECTION = {**{f"sub_index{i}": 1 for i in range(2, 8)},
             "sent_norm_w": 1, "sent_energy": 1, "sent_std_inv": 1, "neg_z_inv": 1,
             "sent_composite": 1, "sent_composite_ma10": 1,
             "egarch_vol": -1, "vol_regime_high": -1, "vol_ratio": -1}


# --------------------------------------------------------- 댓글 -> 일별 감성피처
def load_comments() -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(COMMENT_GLOB)))], ignore_index=True)
    df["text_raw"] = df["text_raw"].fillna("").astype(str)
    df = df[df["text_raw"].str.strip() != ""].copy()
    dt = pd.to_datetime(df["comment_at"], errors="coerce", utc=True)
    df = df[dt.notna()].copy()
    df["date"] = pd.to_datetime(dt[dt.notna()].dt.tz_convert("Asia/Seoul").dt.date)
    df["p_pos"] = df["p_pos"].astype(float); df["p_neg"] = df["p_neg"].astype(float)
    df["sent_raw"] = df["p_pos"] - df["p_neg"]
    df["sent_norm"] = (df["p_pos"] - df["p_neg"]) / (df["p_pos"] + df["p_neg"] + 1e-8)
    df["comment_id"] = df.get("comment_id", pd.Series(range(len(df))))
    return df


def daily_sentiment(df: pd.DataFrame, gamma: float) -> pd.DataFrame:
    """sent_feature.make_daily 와 동일하게 6개 피처 산출."""
    d = df.copy()
    w = np.clip(1.0 - d["toxicity_score"].astype(float), 0, None) ** gamma
    d["weight"] = w
    d["sent_raw_w"] = d["sent_raw"] * w
    d["sent_norm_w_"] = d["sent_norm"] * w
    d["p_pos_w"] = d["p_pos"] * w
    d["p_neg_w"] = d["p_neg"] * w
    g = d.groupby("date", sort=True)
    daily = g.agg(comment_count=("comment_id", "count"),
                  weight_sum=("weight", "sum"),
                  weight_sq_sum=("weight", lambda x: float((x ** 2).sum())),
                  sent_std=("sent_raw", "std"))
    daily["sent_norm_w"] = g["sent_norm_w_"].sum() / (daily["weight_sum"] + 1e-8)
    pos_w = g["p_pos_w"].sum() / (daily["weight_sum"] + 1e-8)
    neg_w = g["p_neg_w"].sum() / (daily["weight_sum"] + 1e-8)
    daily["sent_strength_w"] = pos_w + neg_w
    neg_score = neg_w
    rm = neg_score.rolling(60, min_periods=20).mean()
    rs = neg_score.rolling(60, min_periods=20).std()
    daily["neg_z"] = (neg_score - rm) / (rs + 1e-8)
    daily["effective_n"] = daily["weight_sum"] ** 2 / (daily["weight_sq_sum"] + 1e-8)
    log_eff = np.log1p(daily["effective_n"])
    z = lambda s: (s - s.mean()) / (s.std() + 1e-8)
    daily["heat"] = z(daily["sent_strength_w"]) * z(log_eff)
    return daily.reset_index()[["date", "sent_norm_w", "sent_strength_w", "sent_std",
                                "neg_z", "effective_n", "heat"]]


# --------------------------------------------------------- 파이프라인
def attach_egarch(df_raw: pd.DataFrame) -> pd.DataFrame:
    cache = pd.read_csv(EGARCH_CACHE)
    assert len(cache) == len(df_raw), f"egarch cache {len(cache)} != df_raw {len(df_raw)}"
    d = df_raw.copy()
    d["egarch_vol"] = cache["egarch_vol"].values
    d["vol_shock"] = d["egarch_vol"].pct_change().clip(-5, 5)
    d["vol_regime_high"] = (d["egarch_vol"] > d["egarch_vol"].rolling(60).quantile(0.7)).astype(float)
    d["vol_ma5"] = d["egarch_vol"].rolling(5).mean()
    d["vol_trend"] = (d["egarch_vol"] > d["vol_ma5"]).astype(float)
    return d


def compute_rsi(s, w=14):
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / w, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / w, adjust=False).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def build_features(df: pd.DataFrame, sent_ma_window: int = 10) -> pd.DataFrame:
    df = df.copy()
    sc = [c for c in df.columns if any(k in c for k in ["sent_", "neg_z", "effective", "heat"])]
    df[sc] = df[sc].ffill()
    df["neg_z_inv"] = -df["neg_z"]
    df["sent_std_inv"] = -df["sent_std"]
    df["sent_energy"] = df["sent_strength_w"] * df["sent_norm_w"]
    df["sent_norm_ma5"] = df["sent_norm_w"].rolling(5).mean()
    df["sent_norm_diff"] = df["sent_norm_w"].diff()
    df["neg_z_ma5"] = df["neg_z"].rolling(5).mean()
    df["sent_composite"] = df["sent_norm_w"] * 0.4 + df["neg_z_inv"] * 0.3 + df["sent_energy"] * 0.3
    df["sent_composite_ma10"] = df["sent_composite"].rolling(sent_ma_window).mean()  # <-- 실험 대상
    df["sent_composite_diff"] = df["sent_composite"].diff(5)
    price = df["kospi_close"]
    for wq in (20, 60, 120):
        df[f"ma{wq}"] = price.rolling(wq).mean()
    df["ma_ratio_5_20"] = price.rolling(5).mean() / (df["ma20"] + 1e-9) - 1
    df["ma_ratio_20_60"] = df["ma20"] / (df["ma60"] + 1e-9) - 1
    df["ma_ratio_60_120"] = df["ma60"] / (df["ma120"] + 1e-9) - 1
    df["rsi14"] = compute_rsi(price, 14)
    df["mom5"] = price.pct_change(5); df["mom20"] = price.pct_change(20); df["mom60"] = price.pct_change(60)
    df["above_ma20"] = (price > df["ma20"]).astype(float)
    df["above_ma60"] = (price > df["ma60"]).astype(float)
    df["above_ma120"] = (price > df["ma120"]).astype(float)
    df["trend_strength"] = df["above_ma20"] + df["above_ma60"] + df["above_ma120"]
    df["vol_ratio"] = df["egarch_vol"] / (df["egarch_vol"].rolling(60).mean() + 1e-9)
    for i in range(2, 8):
        df[f"sub_index{i}_lag1"] = df[f"sub_index{i}"].shift(1)
    df["egarch_vol_lag1"] = df["egarch_vol"].shift(1)
    df["egarch_vol_ma5_lag1"] = df["vol_ma5"].shift(1)
    df["vol_shock_lag1"] = df["vol_shock"].shift(1)
    df["vol_regime_lag1"] = df["vol_regime_high"].shift(1)
    df["neg_z_vol_ratio"] = (df["neg_z_inv"] / (df["egarch_vol_lag1"] + 1e-6)).clip(-10, 10)
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["target_reg"] = df["log_return_t+1"]
    df["target_5d"] = df["log_return_t+1"].rolling(5).sum().shift(-4)
    feat_cols = [c for c in df.columns if c not in ["date", "target_reg", "target_5d"]]
    df[feat_cols] = df[feat_cols].shift(1)
    return df.dropna().reset_index(drop=True)


def walkforward_kfgi(df: pd.DataFrame, feats: list[str], dir_vec: np.ndarray,
                     min_obs: int, alpha: float = 1.0) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    raw = np.full(n, np.nan)
    bounds = [(0, None) if d >= 0 else (None, 0) for d in dir_vec]

    def obj(b, x, y):
        r = y - x @ b
        return float(r @ r + alpha * b @ b)

    def grad(b, x, y):
        r = y - x @ b
        return -2 * x.T @ r + 2 * alpha * b

    Xall = df[feats].values
    yall = df["target_5d"].values
    for t in range(min_obs, n):
        x_tr, y_tr = Xall[:t], yall[:t]
        m = ~np.isnan(x_tr).any(axis=1) & ~np.isnan(y_tr)
        x_tr, y_tr = x_tr[m], y_tr[m]
        if len(x_tr) < 30:
            continue
        try:
            sca = StandardScaler()
            xs = np.clip(sca.fit_transform(x_tr), -3, 3)
            res = minimize(obj, np.zeros(len(feats)), args=(xs, y_tr), jac=grad,
                           method="L-BFGS-B", bounds=bounds, options={"maxiter": 300, "ftol": 1e-10})
            w = res.x / (np.sum(np.abs(res.x)) + 1e-12)
            xt = Xall[t:t + 1]
            if np.isnan(xt).any():
                continue
            raw[t] = float((np.clip(sca.transform(xt), -3, 3) @ w).ravel()[0])
        except Exception:
            continue
    valid = raw[~np.isnan(raw)]
    p1, p99 = np.percentile(valid, 1), np.percentile(valid, 99)
    df["K_FGI"] = 100 * (np.clip(raw, p1, p99) - p1) / (p99 - p1 + 1e-12)
    return df.dropna(subset=["K_FGI"]).reset_index(drop=True)


def classify_regime(row):
    if row["K_FGI"] > 65 and row["vol_regime_high"] == 0 and row["trend_strength"] >= 2:
        return "bull"
    if row["K_FGI"] < 25 or row["vol_shock"] > CFG["vol_shock_thr"]:
        return "crisis"
    return "normal"


def kfgi_mult(k, momentum=True):
    m = 0.5 + (k / 100.0) * 1.1 if momentum else 1.6 - (k / 100.0) * 1.1
    return float(np.clip(m, 0.5, 1.6))


def compute_positions(df: pd.DataFrame, momentum=True) -> pd.DataFrame:
    d = df.copy()
    d["tw"] = d["trend_strength"].apply(lambda x: TREND_MAP.get(int(x), 0.0))
    d["vt_pos"] = np.sqrt(TARGET_DAILY_VOL / (d["egarch_vol"] + 1e-9)).clip(CFG["min_leverage"], CFG["max_leverage"])
    d["kfgi_mult"] = d["K_FGI"].apply(lambda k: kfgi_mult(k, momentum))
    rm = d["regime"].map({"bull": 1.2, "normal": 1.0, "crisis": 0.7})
    d["weight"] = (d["tw"] * d["vt_pos"] * d["kfgi_mult"] * rm).clip(0.0, CFG["max_leverage"] * 1.5)
    d.loc[d["vol_shock"] > CFG["vol_shock_thr"], "weight"] = 0.0
    d["cumret_3d"] = d["target_reg"].rolling(3).sum().shift(1).fillna(0)
    d.loc[d["cumret_3d"] < CFG["tail_loss_thr"], "weight"] = 0.0
    d["weight_lag"] = d["weight"].shift(1).fillna(0)
    d["turnover"] = d["weight_lag"].diff().abs().fillna(d["weight_lag"].iloc[0])
    d["strat_ret"] = d["weight_lag"] * d["target_reg"] - d["turnover"] * CFG["fee"]
    return d


def metrics(ret: pd.Series, mkt: pd.Series, label: str) -> dict:
    ann = ret.mean() * 252
    vol = ret.std() * np.sqrt(252) + 1e-9
    cum = np.exp(ret.cumsum())
    mdd = float((cum / cum.cummax() - 1).min())
    q = np.quantile(ret, 0.05)
    cvar = float(ret[ret <= q].mean())
    down = mkt < 0
    dvol = float(ret[down].std() * np.sqrt(252))
    defense = float(ret[down].mean() / (mkt[down].mean() + 1e-9))
    return dict(label=label, ann_ret=float(ann), sharpe=float(ann / vol), mdd=mdd,
                calmar=float(ann / (abs(mdd) + 1e-9)), cvar5=cvar, downside_vol=dvol,
                downday_defense=defense, total_return=float(np.exp(ret.sum()) - 1))


def run_pipeline(df_raw: pd.DataFrame, sent_ma_window: int = 10) -> pd.DataFrame:
    d = attach_egarch(df_raw)
    d = build_features(d, sent_ma_window=sent_ma_window)
    feats = [f for f in KFGI_FEATS_BASE if f in d.columns]
    dir_vec = np.array([DIRECTION.get(f, 1) for f in feats])
    d = walkforward_kfgi(d, feats, dir_vec, CFG["kfgi_min_obs"])
    d["regime"] = d.apply(classify_regime, axis=1)
    return compute_positions(d, momentum=True)


# --------------------------------------------------------- main
def main():
    kfg = pd.read_csv(KFG_FINAL, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    mkt_full = kfg[["date", "log_return_t+1"]].rename(columns={"log_return_t+1": "mkt"})
    comments = load_comments()
    print(f"[data] KFG_final {len(kfg)} rows, comments {len(comments)}")

    sent_base_cols = ["sent_norm_w", "sent_strength_w", "sent_std", "neg_z", "effective_n", "heat"]
    results = []
    ts = {}

    def swap_sent(gamma):
        ds = daily_sentiment(comments, gamma)
        m = kfg.drop(columns=sent_base_cols).merge(ds, on="date", how="left")
        m[sent_base_cols] = m[sent_base_cols].ffill()
        return m

    # ---- 실험 1: gamma
    for g in [1.0, 2.0, 3.0]:
        print(f"\n[gamma={g}] 파이프라인 재실행 ...")
        res = run_pipeline(swap_sent(g), sent_ma_window=10)
        mk = mkt_full.set_index("date")["mkt"].reindex(res["date"]).values
        mm = metrics(res["strat_ret"].reset_index(drop=True),
                     pd.Series(mk), f"gamma={g:g}")
        mm["experiment"] = "gamma"; mm["param"] = g
        results.append(mm); ts[f"gamma={g:g}"] = res[["date", "strat_ret", "K_FGI", "weight"]].copy()
        print("   ", {k: round(v, 4) for k, v in mm.items() if isinstance(v, float)})

    # ---- 실험 2: 감성 이동평균 window (gamma=2 고정)
    base2 = swap_sent(2.0)
    for wwin in [5, 10, 20]:
        print(f"\n[sent_MA={wwin}] 파이프라인 재실행 ...")
        res = run_pipeline(base2, sent_ma_window=wwin)
        mk = mkt_full.set_index("date")["mkt"].reindex(res["date"]).values
        mm = metrics(res["strat_ret"].reset_index(drop=True), pd.Series(mk), f"sent_MA={wwin}")
        mm["experiment"] = "sent_ma"; mm["param"] = wwin
        results.append(mm); ts[f"sent_MA={wwin}"] = res[["date", "strat_ret"]].copy()
        print("   ", {k: round(v, 4) for k, v in mm.items() if isinstance(v, float)})

    out = pd.DataFrame(results)
    out.to_csv(BASE / "downstream_sensitivity.csv", index=False)
    print("\n=== 요약 ===")
    print(out[["experiment", "param", "sharpe", "mdd", "cvar5", "downside_vol",
               "downday_defense", "calmar", "total_return"]].to_string(index=False))

    # 그림
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    gg = out[out.experiment == "gamma"]
    ax[0].plot(gg.param, gg.sharpe, "o-", label="Sharpe")
    ax[0].plot(gg.param, gg.calmar, "s-", label="Calmar")
    ax[0].axvline(2, color="crimson", ls="--", lw=1); ax[0].legend()
    ax[0].set(title="γ → 전략 Sharpe / Calmar", xlabel="γ")
    ax[1].plot(gg.param, gg.mdd, "o-", label="MDD")
    ax[1].plot(gg.param, gg.cvar5, "s-", label="CVaR 5%")
    ax[1].plot(gg.param, gg.downside_vol, "^-", label="Downside vol")
    ax[1].axvline(2, color="crimson", ls="--", lw=1); ax[1].legend()
    ax[1].set(title="γ → 하방위험 지표", xlabel="γ")
    for name, t in ts.items():
        if name.startswith("gamma"):
            ax[2].plot(pd.to_datetime(t["date"]), np.exp(t["strat_ret"].cumsum()), label=name, lw=1)
    ax[2].legend(); ax[2].set(title="γ별 누적수익 곡선")
    fig.savefig(FIG / "H_gamma_strategy.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    mm = out[out.experiment == "sent_ma"]
    ax[0].plot(mm.param, mm.sharpe, "o-", label="Sharpe")
    ax[0].plot(mm.param, mm.calmar, "s-", label="Calmar")
    ax[0].axvline(10, color="crimson", ls="--", lw=1); ax[0].legend()
    ax[0].set(title="감성 MA window → Sharpe / Calmar", xlabel="window(days)")
    ax[1].plot(mm.param, mm.mdd, "o-", label="MDD")
    ax[1].plot(mm.param, mm.cvar5, "s-", label="CVaR 5%")
    ax[1].plot(mm.param, mm.downday_defense, "^-", label="하락일 방어율")
    ax[1].axvline(10, color="crimson", ls="--", lw=1); ax[1].legend()
    ax[1].set(title="감성 MA window → 하방위험", xlabel="window(days)")
    fig.savefig(FIG / "I_sentma_strategy.png"); plt.close(fig)

    (BASE / "downstream_sensitivity.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n저장: {BASE/'downstream_sensitivity.csv'}")


if __name__ == "__main__":
    main()
