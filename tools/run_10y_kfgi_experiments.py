from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


warnings.filterwarnings("ignore")

CFG = dict(
    egarch_min_obs=60,
    kfgi_min_obs=60,
    n_splits=5,
    fee=0.00015,
    target_vol_ann=0.15,
    max_leverage=2.0,
    min_leverage=0.1,
    vol_shock_thr=2.0,
    tail_loss_thr=-0.07,
    seed=42,
)
TARGET_DAILY_VOL = CFG["target_vol_ann"] / np.sqrt(252)
TREND_MAP = {0: 0.0, 1: 0.35, 2: 0.70, 3: 1.0}


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def find_child(parent: Path, normalized_name: str) -> Path:
    for child in parent.iterdir():
        if nfc(child.name) == normalized_name:
            return child
    raise FileNotFoundError(f"{normalized_name} not found under {parent}")


def resolve_paths() -> tuple[Path, Path, Path, Path]:
    cwd = Path.cwd().resolve()
    project_root = cwd.parent if nfc(cwd.name) == "kfgi_최종" else cwd
    if nfc(project_root.name) == "kfgi_최종":
        project_root = project_root.parent
    paper_dir = find_child(project_root, "논문용")
    package_dir = find_child(paper_dir, "10y_kfgi_paper_package")
    final_dataset_dir = find_child(package_dir, "01_final_dataset")
    data_path = find_child(final_dataset_dir, "KFG_final_10y.csv")
    out_dir = package_dir / "06_10y_experiments"
    return project_root, package_dir, data_path, out_dir


def setup_matplotlib() -> None:
    if sys.platform.startswith("win"):
        plt.rcParams["font.family"] = "Malgun Gothic"
    else:
        plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False


def fit_egarch_walkforward(log_returns: pd.Series, min_obs: int, cache_path: Path) -> pd.Series:
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        return pd.Series(cached["egarch_vol"].values, index=log_returns.index, name="egarch_vol")

    n = len(log_returns)
    sigma = np.full(n, np.nan)
    for t in range(min_obs, n):
        train = log_returns.iloc[:t].dropna()
        if not HAS_ARCH:
            sigma[t] = train.iloc[-20:].std() if len(train) >= 20 else np.nan
        else:
            try:
                am = arch_model(train * 100, vol="EGARCH", p=1, q=1, dist="skewt", rescale=False)
                res = am.fit(disp="off", show_warning=False)
                fc = res.forecast(horizon=1, reindex=False)
                sigma[t] = float(np.sqrt(fc.variance.values[-1, 0])) / 100
            except Exception:
                sigma[t] = train.iloc[-20:].std() if len(train) >= 20 else np.nan
        if t % 100 == 0:
            print(f"    EGARCH {t}/{n}")

    out = pd.DataFrame({"egarch_vol": sigma})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return pd.Series(sigma, index=log_returns.index, name="egarch_vol")


def add_egarch_features(df_raw: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    log_ret = np.log(df_raw["kospi_close"] / df_raw["kospi_close"].shift(1)).fillna(0)
    cache_path = out_dir / "cache" / "egarch_vol_10y.csv"
    df_raw["egarch_vol"] = fit_egarch_walkforward(log_ret, CFG["egarch_min_obs"], cache_path)
    df_raw["vol_shock"] = df_raw["egarch_vol"].pct_change().clip(-5, 5)
    df_raw["vol_regime_high"] = (
        df_raw["egarch_vol"] > df_raw["egarch_vol"].rolling(60).quantile(0.7)
    ).astype(float)
    df_raw["vol_ma5"] = df_raw["egarch_vol"].rolling(5).mean()
    df_raw["vol_trend"] = (df_raw["egarch_vol"] > df_raw["vol_ma5"]).astype(float)
    print(f"    EGARCH valid rows={df_raw['egarch_vol'].notna().sum():,}")
    return df_raw


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sent_cols = [c for c in df.columns if any(k in c for k in ["sent_", "neg_z", "effective", "heat"])]
    df[sent_cols] = df[sent_cols].ffill()

    df["neg_z_inv"] = -df["neg_z"]
    df["sent_std_inv"] = -df["sent_std"]
    df["sent_energy"] = df["sent_strength_w"] * df["sent_norm_w"]
    df["sent_norm_ma5"] = df["sent_norm_w"].rolling(5).mean()
    df["sent_norm_diff"] = df["sent_norm_w"].diff()
    df["neg_z_ma5"] = df["neg_z"].rolling(5).mean()
    df["sent_composite"] = (
        df["sent_norm_w"] * 0.4 + df["neg_z_inv"] * 0.3 + df["sent_energy"] * 0.3
    )
    df["sent_composite_ma10"] = df["sent_composite"].rolling(10).mean()
    df["sent_composite_diff"] = df["sent_composite"].diff(5)

    price = df["kospi_close"]
    df["ma20"] = price.rolling(20).mean()
    df["ma60"] = price.rolling(60).mean()
    df["ma120"] = price.rolling(120).mean()
    df["ma_ratio_5_20"] = price.rolling(5).mean() / (df["ma20"] + 1e-9) - 1
    df["ma_ratio_20_60"] = df["ma20"] / (df["ma60"] + 1e-9) - 1
    df["ma_ratio_60_120"] = df["ma60"] / (df["ma120"] + 1e-9) - 1
    df["rsi14"] = compute_rsi(price, 14)
    df["mom5"] = price.pct_change(5)
    df["mom20"] = price.pct_change(20)
    df["mom60"] = price.pct_change(60)
    df["above_ma20"] = (price > df["ma20"]).astype(float)
    df["above_ma60"] = (price > df["ma60"]).astype(float)
    df["above_ma120"] = (price > df["ma120"]).astype(float)
    df["trend_strength"] = df["above_ma20"] + df["above_ma60"] + df["above_ma120"]
    df["vol_ratio"] = df["egarch_vol"] / (df["egarch_vol"].rolling(60).mean() + 1e-9)

    # sub_index1 is intentionally excluded from modeling features.
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

    before = len(df)
    feature_cols = [c for c in df.columns if c not in ["date", "target_reg", "target_5d"]]
    df[feature_cols] = df[feature_cols].shift(1)
    df = df.dropna().reset_index(drop=True)
    print(f"    feature rows {before:,}->{len(df):,}, start={df['date'].min().date()}")
    return df


KFGI_FEATS_BASE = (
    [f"sub_index{i}" for i in range(2, 8)]
    + [
        "sent_norm_w", "sent_energy", "sent_std_inv",
        "neg_z_inv", "sent_composite", "sent_composite_ma10",
        "egarch_vol", "vol_regime_high", "vol_ratio",
    ]
)
DIRECTION = {
    **{f"sub_index{i}": 1 for i in range(2, 8)},
    "sent_norm_w": 1,
    "sent_energy": 1,
    "sent_std_inv": 1,
    "neg_z_inv": 1,
    "sent_composite": 1,
    "sent_composite_ma10": 1,
    "egarch_vol": -1,
    "vol_regime_high": -1,
    "vol_ratio": -1,
}


def get_kfgi_feats(df: pd.DataFrame, exclude_sentiment: bool = False) -> tuple[list[str], np.ndarray]:
    feats = [f for f in KFGI_FEATS_BASE if f in df.columns]
    if exclude_sentiment:
        feats = [f for f in feats if not any(k in f for k in ["sent_", "neg_z", "composite"])]
    dir_vec = np.array([DIRECTION.get(f, 1) for f in feats])
    return feats, dir_vec


def create_walkforward_kfgi(
    df: pd.DataFrame,
    feats: list[str],
    dir_vec: np.ndarray,
    min_obs: int,
    cache_path: Path | None = None,
    alpha: float = 1.0,
) -> pd.DataFrame:
    if cache_path and cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        return df.merge(cached[["date", "K_FGI"]], on="date", how="left").dropna(subset=["K_FGI"]).reset_index(drop=True)

    df = df.copy()
    n = len(df)
    raw_scores = np.full(n, np.nan)
    bounds = [(0, None) if d >= 0 else (None, 0) for d in dir_vec]

    def objective(beta, x, y):
        residuals = y - x @ beta
        return float(np.sum(residuals**2) + alpha * np.sum(beta**2))

    def gradient(beta, x, y):
        residuals = y - x @ beta
        return -2 * x.T @ residuals + 2 * alpha * beta

    for t in range(min_obs, n):
        x_train = df[feats].iloc[:t].values
        y_train = df["target_5d"].iloc[:t].values
        mask = ~np.isnan(x_train).any(axis=1) & ~np.isnan(y_train)
        x_train, y_train = x_train[mask], y_train[mask]
        if len(x_train) < 30:
            continue
        try:
            scaler = StandardScaler()
            x_scaled = np.clip(scaler.fit_transform(x_train), -3, 3)
            res = minimize(
                objective,
                np.zeros(len(feats)),
                args=(x_scaled, y_train),
                jac=gradient,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-10},
            )
            coef = res.x
            weights = coef / (np.sum(np.abs(coef)) + 1e-12)
            x_t = df[feats].iloc[t:t + 1].values
            if np.isnan(x_t).any():
                continue
            raw_scores[t] = float(np.clip(scaler.transform(x_t), -3, 3) @ weights)
        except Exception:
            continue
        if t % 250 == 0:
            print(f"    K-FGI {t}/{n}")

    valid = raw_scores[~np.isnan(raw_scores)]
    if len(valid) == 0:
        raise ValueError("No valid K-FGI estimates")
    p1, p99 = np.percentile(valid, 1), np.percentile(valid, 99)
    df["K_FGI"] = 100 * (np.clip(raw_scores, p1, p99) - p1) / (p99 - p1 + 1e-12)
    out = df.dropna(subset=["K_FGI"]).reset_index(drop=True)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out[["date", "K_FGI"]].to_csv(cache_path, index=False)
    print(f"    K-FGI rows {len(df):,}->{len(out):,}, start={out['date'].min().date()}")
    return out


def perf(returns: pd.Series, label: str, ref: pd.Series | None = None) -> dict:
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252) + 1e-9
    sharpe = ann_ret / ann_vol
    cum = np.exp(returns.cumsum())
    mdd = (cum / cum.cummax() - 1).min()
    calmar = ann_ret / (abs(mdd) + 1e-9)
    total = np.exp(returns.sum()) - 1
    defense = np.nan
    if ref is not None:
        ref_aligned = ref.reindex(returns.index)
        down = ref_aligned < 0
        defense = returns[down].mean() / (ref_aligned[down].mean() + 1e-9)
    return {
        "label": label,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
        "total_return": total,
        "downside_defense": defense,
    }


def classify_regime(row: pd.Series) -> str:
    if row["K_FGI"] > 65 and row["vol_regime_high"] == 0 and row["trend_strength"] >= 2:
        return "bull"
    if row["K_FGI"] < 25 or row["vol_shock"] > CFG["vol_shock_thr"]:
        return "crisis"
    return "normal"


def kfgi_scale_mult(kfgi_val: float, is_momentum: bool = True) -> float:
    mult = 0.5 + (kfgi_val / 100.0) * 1.1 if is_momentum else 1.6 - (kfgi_val / 100.0) * 1.1
    return float(np.clip(mult, 0.5, 1.6))


def compute_positions(df: pd.DataFrame, kfgi_momentum: bool = True, fee: float | None = None) -> pd.DataFrame:
    fee = CFG["fee"] if fee is None else fee
    d = df.copy()
    d["tw"] = d["trend_strength"].apply(lambda x: TREND_MAP.get(int(x), 0.0))
    raw_vt = TARGET_DAILY_VOL / (d["egarch_vol"] + 1e-9)
    d["vt_pos"] = np.sqrt(raw_vt).clip(CFG["min_leverage"], CFG["max_leverage"])
    d["kfgi_mult"] = d["K_FGI"].apply(lambda k: kfgi_scale_mult(k, is_momentum=kfgi_momentum))
    regime_mult = d["regime"].map({"bull": 1.2, "normal": 1.0, "crisis": 0.7})
    d["weight"] = (d["tw"] * d["vt_pos"] * d["kfgi_mult"] * regime_mult).clip(0.0, CFG["max_leverage"] * 1.5)
    d.loc[d["vol_shock"] > CFG["vol_shock_thr"], "weight"] = 0.0
    d["cumret_3d"] = d["target_reg"].rolling(3).sum().shift(1).fillna(0)
    d.loc[d["cumret_3d"] < CFG["tail_loss_thr"], "weight"] = 0.0
    d["weight_lag"] = d["weight"].shift(1).fillna(0)
    d["turnover"] = d["weight_lag"].diff().abs().fillna(d["weight_lag"].iloc[0])
    d["cost"] = d["turnover"] * fee
    d["strat_ret"] = d["weight_lag"] * d["target_reg"] - d["cost"]
    return d


def bench_pure_trend(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["weight"] = d["trend_strength"].apply(lambda x: TREND_MAP.get(int(x), 0.0))
    d["weight_lag"] = d["weight"].shift(1).fillna(0)
    d["turnover"] = d["weight_lag"].diff().abs().fillna(d["weight_lag"].iloc[0])
    d["cost"] = d["turnover"] * CFG["fee"]
    d["strat_ret"] = d["weight_lag"] * d["target_reg"] - d["cost"]
    return d


def bench_trend_vol(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    tw = d["trend_strength"].apply(lambda x: TREND_MAP.get(int(x), 0.0))
    vt = np.sqrt(TARGET_DAILY_VOL / (d["egarch_vol"] + 1e-9)).clip(CFG["min_leverage"], CFG["max_leverage"])
    d["weight"] = (tw * vt).clip(0, CFG["max_leverage"])
    d.loc[d["vol_shock"] > CFG["vol_shock_thr"], "weight"] = 0.0
    d["weight_lag"] = d["weight"].shift(1).fillna(0)
    d["turnover"] = d["weight_lag"].diff().abs().fillna(d["weight_lag"].iloc[0])
    d["cost"] = d["turnover"] * CFG["fee"]
    d["strat_ret"] = d["weight_lag"] * d["target_reg"] - d["cost"]
    return d


LGBM_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.02,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.7,
    bagging_freq=5,
    lambda_l1=0.1,
    lambda_l2=0.1,
    min_child_samples=20,
    verbosity=-1,
    seed=CFG["seed"],
)


def run_lgbm(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, lgb.Booster, pd.DataFrame]:
    splitter = TimeSeriesSplit(n_splits=CFG["n_splits"])
    x, y = df[features].values, df["target_5d"].values
    rows = []
    preds = []
    last_model = None
    for fold, (tr, te) in enumerate(splitter.split(x), start=1):
        model = lgb.train(LGBM_PARAMS, lgb.Dataset(x[tr], label=y[tr]), num_boost_round=500)
        pred = model.predict(x[te])
        rmse = float(np.sqrt(mean_squared_error(y[te], pred)))
        rows.append({"fold": fold, "rmse": rmse, "start": df["date"].iloc[te[0]], "end": df["date"].iloc[te[-1]]})
        preds.append(pd.DataFrame({"date": df["date"].iloc[te].values, "pred_5d": pred, "target_5d": y[te]}))
        last_model = model
    return pd.concat(preds, ignore_index=True), last_model, pd.DataFrame(rows)


def calc_dd(ret: pd.Series) -> pd.Series:
    cum = np.exp(ret.cumsum())
    return (cum / cum.cummax() - 1) * 100


def plot_main(df, main_df, trend_df, trendvol_df, df_ns, main_ns, rho_5d, kdir, path: Path) -> None:
    dates = df["date"]
    shock_mask = df["vol_shock"] > CFG["vol_shock_thr"]
    ref = df["target_reg"]
    fig = plt.figure(figsize=(17, 13))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    ax1.plot(dates, np.exp(ref.cumsum()), label="KOSPI200 B&H", color="#9CA3AF", lw=1.5)
    ax1.plot(dates, np.exp(trend_df["strat_ret"].cumsum()), label="Trend", color="#16A34A", lw=1.5, ls="-.")
    ax1.plot(dates, np.exp(trendvol_df["strat_ret"].cumsum()), label="Trend+EGARCH vol", color="#F59E0B", lw=1.5, ls="--")
    ax1.plot(dates, np.exp(main_df["strat_ret"].cumsum()), label=f"Main + K-FGI ({kdir})", color="#2563EB", lw=2.5)
    ax1.plot(df_ns["date"], np.exp(main_ns["strat_ret"].cumsum()), label="Main without sentiment", color="#7C3AED", lw=1.2, ls=":")
    ax1.fill_between(dates, 0.5, 4, where=shock_mask, color="#FCA5A5", alpha=0.2, label="vol_shock")
    ax1.set_title(f"10Y K-FGI x EGARCH cumulative return (rho 5d={rho_5d:.3f})")
    ax1.legend(fontsize=8.5)
    ax1.grid(alpha=0.2)

    for ret, label, color, alpha in [
        (ref, "KOSPI200", "#9CA3AF", 0.40),
        (trend_df["strat_ret"], "Trend", "#16A34A", 0.35),
        (trendvol_df["strat_ret"], "Trend+EGARCH", "#F59E0B", 0.35),
        (main_df["strat_ret"], "Main", "#2563EB", 0.50),
    ]:
        ax2.fill_between(dates, calc_dd(ret), color=color, alpha=alpha, label=label)
    ax2.set_title("Drawdown (%)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    raw_vt = TARGET_DAILY_VOL / (df["egarch_vol"] + 1e-9)
    vt_soft = np.sqrt(raw_vt).clip(CFG["min_leverage"], CFG["max_leverage"])
    ax3.fill_between(dates, df["egarch_vol"] * 100, alpha=0.5, color="#F59E0B", label="EGARCH sigma (%)")
    ax3r = ax3.twinx()
    ax3r.plot(dates, vt_soft, color="#2563EB", lw=1.2, label="vol target position")
    ax3.set_title("EGARCH sigma vs vol targeting")
    ax3.grid(alpha=0.2)

    kv = df["K_FGI"].values
    ax4.plot(dates, kv, color="#7C3AED", lw=1.0)
    ax4.axhline(25, color="#2563EB", linestyle="--", lw=0.8, label="Fear")
    ax4.axhline(75, color="#DC2626", linestyle="--", lw=0.8, label="Greed")
    ax4.fill_between(dates, 0, 100, where=(kv < 25), color="#BFDBFE", alpha=0.45)
    ax4.fill_between(dates, 0, 100, where=(kv > 75), color="#FCA5A5", alpha=0.35)
    ax4.set_ylim(0, 100)
    ax4.set_title("K-FGI")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.2)

    ax5.plot(dates, main_df["weight"], color="#2563EB", lw=0.9, label="Main")
    ax5.plot(dates, trend_df["weight"], color="#16A34A", lw=0.9, alpha=0.7, label="Trend", ls="-.")
    ax5.plot(df_ns["date"], main_ns["weight"], color="#7C3AED", lw=0.9, alpha=0.7, label="Without sentiment", ls=":")
    ax5.axhline(1.0, color="#6B7280", linestyle=":", lw=0.8)
    ax5.set_title("Position weight")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.2)

    fig.suptitle("K-FGI x EGARCH 10-year experiment", fontsize=13, fontweight="bold", y=1.01)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(model: lgb.Booster, features: list[str], out_dir: Path) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    top15 = imp.head(15)
    colors = [
        "#DC2626" if any(k in f for k in ["sent_", "neg_z", "composite"])
        else "#F59E0B" if any(k in f for k in ["egarch", "vol_", "ma_ratio", "rsi", "mom", "above", "trend"])
        else "#2563EB"
        for f in top15["feature"]
    ]
    ax.barh(top15["feature"][::-1], top15["importance"][::-1], color=colors[::-1])
    ax.legend(handles=[
        Patch(color="#DC2626", label="Sentiment"),
        Patch(color="#F59E0B", label="Trend/volatility"),
        Patch(color="#2563EB", label="Other"),
    ], fontsize=9)
    ax.grid(alpha=0.2, axis="x")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "feature_importance_10y.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return imp


def run_stat_tests(main_df: pd.DataFrame, df: pd.DataFrame, df_ns: pd.DataFrame) -> pd.DataFrame:
    common_idx = main_df.index.intersection(df.index)
    excess = (main_df.loc[common_idx, "strat_ret"] - df.loc[common_idx, "target_reg"]).dropna()
    t_a, p_a = stats.ttest_1samp(excess, 0)
    down_mask = df.loc[common_idx, "target_reg"] < 0
    excess_down = excess[down_mask[down_mask].index.intersection(excess.index)].dropna()
    t_b, p_b = stats.ttest_1samp(excess_down, 0)
    q25, q75 = main_df["K_FGI"].quantile([0.25, 0.75])
    greed = main_df.loc[main_df["K_FGI"] >= q75, "target_reg"].dropna()
    fear = main_df.loc[main_df["K_FGI"] <= q25, "target_reg"].dropna()
    t_c, p_c = stats.ttest_ind(greed, fear, equal_var=False)
    common_ns = main_df.index.intersection(df_ns.index)
    valid = (
        main_df.loc[common_ns, "K_FGI"].notna()
        & df_ns.loc[common_ns, "K_FGI"].notna()
        & main_df.loc[common_ns, "target_5d"].notna()
    )
    kw = main_df.loc[common_ns, "K_FGI"][valid].values
    kwo = df_ns.loc[common_ns, "K_FGI"][valid].values
    r5 = main_df.loc[common_ns, "target_5d"][valid].values
    rng = np.random.default_rng(42)
    boot_diffs = []
    for _ in range(1000):
        idx = rng.choice(len(kw), len(kw), replace=True)
        boot_diffs.append(spearmanr(kw[idx], r5[idx])[0] - spearmanr(kwo[idx], r5[idx])[0])
    return pd.DataFrame([
        {"test": "overall_excess_ttest", "stat": t_a, "p_value": p_a, "n": len(excess)},
        {"test": "downside_excess_ttest", "stat": t_b, "p_value": p_b, "n": len(excess_down)},
        {"test": "kfgi_extreme_return_ttest", "stat": t_c, "p_value": p_c, "n": len(greed) + len(fear)},
        {"test": "sentiment_bootstrap_delta_rho", "stat": float(np.mean(boot_diffs)), "p_value": float(np.mean(np.array(boot_diffs) <= 0)), "n": len(kw)},
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-egarch", action="store_true")
    parser.add_argument("--force-kfgi", action="store_true")
    args = parser.parse_args()

    setup_matplotlib()
    _, package_dir, data_path, out_dir = resolve_paths()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "cache").mkdir(exist_ok=True)

    if args.force_egarch:
        cache = out_dir / "cache" / "egarch_vol_10y.csv"
        if cache.exists():
            cache.unlink()
    if args.force_kfgi:
        for cache in [out_dir / "cache" / "kfgi_with_sentiment.csv", out_dir / "cache" / "kfgi_without_sentiment.csv"]:
            if cache.exists():
                cache.unlink()

    print("=" * 70)
    print("K-FGI x EGARCH 10Y experiment")
    print(f"input={data_path}")
    print(f"output={out_dir}")
    print(f"arch_available={HAS_ARCH}")
    print("=" * 70)

    df_raw = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df_raw = add_egarch_features(df_raw, out_dir)
    df = build_features(df_raw)

    feats, dir_vec = get_kfgi_feats(df, exclude_sentiment=False)
    print(f"    K-FGI feats with sentiment={feats}")
    df = create_walkforward_kfgi(df, feats, dir_vec, CFG["kfgi_min_obs"], out_dir / "cache" / "kfgi_with_sentiment.csv")
    rho_1d, p_1d = spearmanr(df["K_FGI"], df["target_reg"])
    rho_5d, p_5d = spearmanr(df["K_FGI"], df["target_5d"])
    kfgi_momentum = rho_5d > 0
    kdir = "momentum" if kfgi_momentum else "contrarian"
    print(f"    rho_1d={rho_1d:.4f}, p={p_1d:.4f}; rho_5d={rho_5d:.4f}, p={p_5d:.4f}; direction={kdir}")

    df["regime"] = df.apply(classify_regime, axis=1)
    main_df = compute_positions(df, kfgi_momentum)
    trend_df = bench_pure_trend(df)
    trendvol_df = bench_trend_vol(df)

    feats_ns, dir_vec_ns = get_kfgi_feats(df, exclude_sentiment=True)
    print(f"    K-FGI feats without sentiment={feats_ns}")
    df_ns = create_walkforward_kfgi(df.drop(columns=["K_FGI"]), feats_ns, dir_vec_ns, CFG["kfgi_min_obs"], out_dir / "cache" / "kfgi_without_sentiment.csv")
    df_ns["regime"] = df_ns.apply(classify_regime, axis=1)
    main_ns = compute_positions(df_ns, kfgi_momentum)

    performance = pd.DataFrame([
        perf(df["target_reg"], "buy_hold_kospi200"),
        perf(trend_df["strat_ret"], "trend_only", df["target_reg"]),
        perf(trendvol_df["strat_ret"], "trend_egarch_vol", df["target_reg"]),
        perf(main_df["strat_ret"], "main_kfgi_sentiment", df["target_reg"]),
        perf(main_ns["strat_ret"], "main_kfgi_no_sentiment", df_ns["target_reg"]),
    ])
    performance.to_csv(out_dir / "tables" / "performance_summary.csv", index=False)

    features = (
        [f"sub_index{i}_lag1" for i in range(2, 8)]
        + [
            "sent_norm_w", "sent_energy", "sent_std_inv", "neg_z_inv",
            "sent_norm_ma5", "neg_z_ma5", "sent_norm_diff",
            "sent_composite", "sent_composite_ma10", "sent_composite_diff",
            "egarch_vol_lag1", "egarch_vol_ma5_lag1",
            "vol_shock_lag1", "vol_regime_lag1", "neg_z_vol_ratio",
            "ma_ratio_5_20", "ma_ratio_20_60", "ma_ratio_60_120",
            "rsi14", "mom5", "mom20", "mom60",
            "above_ma20", "above_ma60", "above_ma120",
            "trend_strength", "vol_ratio", "K_FGI", "dayofweek", "month",
        ]
    )
    features = [f for f in features if f in df.columns]
    pred_df, lgbm_model, lgbm_cv = run_lgbm(df, features)
    pred_df.to_csv(out_dir / "tables" / "lgbm_oos_predictions.csv", index=False)
    lgbm_cv.to_csv(out_dir / "tables" / "lgbm_cv_summary.csv", index=False)
    imp = plot_feature_importance(lgbm_model, features, out_dir)
    imp.to_csv(out_dir / "tables" / "feature_importance.csv", index=False)

    split_date = pd.Timestamp("2025-01-01")
    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()
    oos = pd.DataFrame([
        perf(compute_positions(train_df, kfgi_momentum)["strat_ret"], "train_pre_2025", train_df["target_reg"]),
        perf(compute_positions(test_df, kfgi_momentum)["strat_ret"], "test_2025", test_df["target_reg"]),
    ])
    oos.to_csv(out_dir / "tables" / "oos_2025_performance.csv", index=False)

    yearly_rows = []
    for year, sub in df.groupby(df["date"].dt.year):
        if len(sub) >= 50:
            yearly_rows.append({"year": year, **perf(compute_positions(sub, kfgi_momentum)["strat_ret"], f"{year}", sub["target_reg"])})
    pd.DataFrame(yearly_rows).to_csv(out_dir / "tables" / "yearly_performance.csv", index=False)

    regime_rows = []
    main_df_for_regime = main_df.copy()
    for regime, sub in main_df_for_regime.groupby("regime"):
        if len(sub) >= 30:
            regime_rows.append({"regime": regime, **perf(sub["strat_ret"], regime, sub["target_reg"])})
    pd.DataFrame(regime_rows).to_csv(out_dir / "tables" / "regime_performance.csv", index=False)

    fee_rows = []
    for fee in [0.00015, 0.00030, 0.00045]:
        res = compute_positions(test_df, kfgi_momentum, fee=fee)
        fee_rows.append({"fee": fee, **perf(res["strat_ret"], f"fee_{fee}", test_df["target_reg"])})
    pd.DataFrame(fee_rows).to_csv(out_dir / "tables" / "fee_stress_test.csv", index=False)

    stat_tests = run_stat_tests(main_df, df, df_ns)
    stat_tests.to_csv(out_dir / "tables" / "statistical_tests.csv", index=False)

    sensitivity_rows = []
    for kmin, kmax in [(0.3, 1.4), (0.5, 1.6), (0.5, 2.0), (0.7, 1.4), (0.7, 1.6)]:
        for bull_r, norm_r, crisis_r in [(1.1, 1.0, 0.8), (1.2, 1.0, 0.7), (1.3, 1.0, 0.6), (1.2, 1.0, 0.5)]:
            d = df.copy()
            d["tw"] = d["trend_strength"].apply(lambda x: TREND_MAP.get(int(x), 0.0))
            d["vt_pos"] = np.sqrt(TARGET_DAILY_VOL / (d["egarch_vol"] + 1e-9)).clip(CFG["min_leverage"], CFG["max_leverage"])
            if kfgi_momentum:
                d["kfgi_mult"] = (kmin + (d["K_FGI"] / 100.0) * (kmax - kmin)).clip(kmin, kmax)
            else:
                d["kfgi_mult"] = (kmax - (d["K_FGI"] / 100.0) * (kmax - kmin)).clip(kmin, kmax)
            rm = d["regime"].map({"bull": bull_r, "normal": norm_r, "crisis": crisis_r})
            d["weight"] = (d["tw"] * d["vt_pos"] * d["kfgi_mult"] * rm).clip(0, CFG["max_leverage"] * 1.5)
            d.loc[d["vol_shock"] > CFG["vol_shock_thr"], "weight"] = 0
            d["cumret_3d"] = d["target_reg"].rolling(3).sum().shift(1).fillna(0)
            d.loc[d["cumret_3d"] < CFG["tail_loss_thr"], "weight"] = 0
            d["weight_lag"] = d["weight"].shift(1).fillna(0)
            d["turnover"] = d["weight_lag"].diff().abs().fillna(d["weight_lag"].iloc[0])
            d["cost"] = d["turnover"] * CFG["fee"]
            d["strat_ret"] = d["weight_lag"] * d["target_reg"] - d["cost"]
            sensitivity_rows.append({
                "kfgi_range": f"{kmin}-{kmax}",
                "regime_mult": f"{bull_r}/{norm_r}/{crisis_r}",
                "adopted": kmin == 0.5 and kmax == 1.6 and bull_r == 1.2 and crisis_r == 0.7,
                **perf(d["strat_ret"], "sensitivity", d["target_reg"]),
            })
    pd.DataFrame(sensitivity_rows).sort_values("sharpe", ascending=False).to_csv(out_dir / "tables" / "sensitivity_analysis.csv", index=False)

    result_df = main_df.copy()
    result_cols = [
        "date", "K_FGI", "regime", "egarch_vol", "vol_shock", "vol_regime_high",
        "trend_strength", "weight", "weight_lag", "turnover", "cost", "strat_ret",
        "target_reg", "target_5d", "kospi_close",
    ]
    result_df[result_cols].to_csv(out_dir / "kfgi_10y_timeseries.csv", index=False)
    strategy = pd.DataFrame({
        "date": df["date"],
        "buy_hold": df["target_reg"],
        "trend_only": trend_df["strat_ret"],
        "trend_egarch_vol": trendvol_df["strat_ret"],
        "main_kfgi_sentiment": main_df["strat_ret"],
    })
    strategy.to_csv(out_dir / "strategy_returns_10y.csv", index=False)

    plot_main(df, main_df, trend_df, trendvol_df, df_ns, main_ns, rho_5d, kdir, out_dir / "figures" / "kfgi_egarch_10y_result.png")

    metadata = {
        "input": str(data_path),
        "output": str(out_dir),
        "arch_available": HAS_ARCH,
        "rows_raw": len(df_raw),
        "rows_features": len(df),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "kfgi_features": feats,
        "subindex1_policy": "kept in source CSV, excluded from K-FGI/modeling features",
        "rho_1d": float(rho_1d),
        "rho_1d_p": float(p_1d),
        "rho_5d": float(rho_5d),
        "rho_5d_p": float(p_5d),
        "kfgi_direction": kdir,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    print("\nDone.")
    print(performance.to_string(index=False))
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
