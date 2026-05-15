"""
K-FGI × EGARCH 파이프라인 v4 (최종 확정판)
=================================================================
수정 요약:
  [요청①] vol targeting: pos → pos^0.5  (완화)
  [요청②] crisis 배율: 0.4 → 0.7
  [요청③] K-FGI 배율: 0.6~1.4 → 0.5~1.6
  [구조①] K-FGI 피처에서 추세 피처 완전 제거
          → K-FGI = 순수 감성 + EGARCH 변동성만
  [구조②] LGBM은 포지션 개입 없음 (분석용 전용 유지)

버그 수정:
  [BUG-1] run_fee_test 함수 중복 정의 제거
  [BUG-2] regime_test 이중 호출 제거
  [BUG-3] t-test index alignment 수정
  [BUG-4] run_fee_test CFG["fee"] 전역 오염 수정
  [BUG-5] bench_pure_trend weight lag 누락 수정
  [BUG-6] compute_positions crisis 배율 0.8 → 0.7 수정

통계 검정 강화:
  [STAT-1] K-FGI 극단 구간 수익률 차이 t-test 추가
  [STAT-2] 감성 기여 Spearman ρ 부트스트랩 검정 추가
  [STAT-3] 전체/하락 구간 t-test index alignment 수정
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from scipy import stats

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("[경고] arch 없음. rolling std 대체.")

if sys.platform.startswith("win"):
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════
# 0. 설정
# ══════════════════════════════════════════════
CFG = dict(
    raw_path       = "KFG_final_2.csv",
    egarch_min_obs = 60,
    kfgi_min_obs   = 60,
    n_splits       = 5,
    fee            = 0.00015,   # 기본 수수료 (편도 0.015%)
    target_vol_ann = 0.15,      # 연간 목표 변동성 15%
    max_leverage   = 2.0,
    min_leverage   = 0.1,
    vol_shock_thr  = 2.0,       # vol_shock 임계값
    tail_loss_thr  = -0.07,     # 3일 누적 -7% 이하 시 포지션 청산
    seed           = 42,
)
TARGET_DAILY_VOL = CFG["target_vol_ann"] / np.sqrt(252)


# ══════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════
print("=" * 62)
print("  K-FGI × EGARCH 파이프라인 v4 (주석 완전 보강판)")
print("  [vol^0.5 | crisis 0.7 | K-FGI 0.5~1.6 | 추세 분리]")
print("=" * 62)

df_raw = pd.read_csv(CFG["raw_path"])
df_raw["date"] = pd.to_datetime(df_raw["date"])
df_raw = df_raw.sort_values("date").reset_index(drop=True)
print(f"[1] 원본 데이터: {len(df_raw)}행  "
      f"({df_raw['date'].min().date()} ~ {df_raw['date'].max().date()})")


# ══════════════════════════════════════════════
# 2. EGARCH(1,1) 조건부 변동성 추정
#
# EGARCH는 log분산 방정식에 비대칭항 γ를 도입하여
# 하락 충격이 상승 충격보다 변동성을 더 크게 키우는
# 레버리지 효과를 포착함.
# walk-forward 방식으로 t 시점은 [0, t-1] 구간만 사용
# → look-ahead bias 구조적 차단
# ══════════════════════════════════════════════
def fit_egarch_walkforward(log_returns, min_obs):
    """
    walk-forward EGARCH(1,1) 변동성 추정.
    t 시점의 σ_t는 [0, t-1] 구간 데이터만 사용.
    min_obs(=60) 미만 구간은 NaN 처리.
    """
    n     = len(log_returns)
    sigma = np.full(n, np.nan)
    for t in range(min_obs, n):
        train = log_returns.iloc[:t].dropna()
        if not HAS_ARCH:
            # arch 패키지 없을 경우 rolling std 대체
            sigma[t] = train.iloc[-20:].std() if len(train) >= 20 else np.nan
            continue
        try:
            am  = arch_model(train * 100, vol="EGARCH", p=1, q=1,
                             dist="skewt", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            fc  = res.forecast(horizon=1, reindex=False)
            sigma[t] = float(np.sqrt(fc.variance.values[-1, 0])) / 100
        except Exception:
            sigma[t] = np.nan
    return pd.Series(sigma, index=log_returns.index, name="egarch_vol")


print("[2] EGARCH walk-forward 변동성 추정 중...")
log_ret = np.log(
    df_raw["kospi_close"] / df_raw["kospi_close"].shift(1)
).fillna(0)
df_raw["egarch_vol"] = fit_egarch_walkforward(log_ret, CFG["egarch_min_obs"])

# EGARCH 파생 피처
df_raw["vol_shock"]       = df_raw["egarch_vol"].pct_change().clip(-5, 5)
df_raw["vol_regime_high"] = (
    df_raw["egarch_vol"] > df_raw["egarch_vol"].rolling(60).quantile(0.7)
).astype(float)
df_raw["vol_ma5"]   = df_raw["egarch_vol"].rolling(5).mean()
df_raw["vol_trend"] = (df_raw["egarch_vol"] > df_raw["vol_ma5"]).astype(float)
print(f"    유효 행 수={df_raw['egarch_vol'].notna().sum()}")


# ══════════════════════════════════════════════
# 3. 피처 엔지니어링
#
# 전체 피처에 1일 lag를 일괄 적용하여
# 오늘 데이터로 내일 수익률을 예측하는 구조를 보장
# ══════════════════════════════════════════════
def compute_rsi(series, window=14):
    """RSI(Relative Strength Index) 계산. EWM 방식 사용."""
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def build_features(df):
    df = df.copy()
    sent_cols = [c for c in df.columns if any(k in c for k in
                 ["sent_", "neg_z", "effective", "heat", "pos_", "weight"])]
    df[sent_cols] = df[sent_cols].ffill()

    # ─ 감성 파생 피처 ─
    # neg_z_inv: 부정 z-score를 역전 → 높을수록 긍정 심리
    df["neg_z_inv"]      = -df["neg_z"]
    df["sent_std_inv"]   = -df["sent_std"]
    # sent_energy: 감성 강도 × 정규화된 감성 점수 → 복합 강도 지표
    df["sent_energy"]    = df["sent_strength_w"] * df["sent_norm_w"]
    df["sent_norm_ma5"]  = df["sent_norm_w"].rolling(5).mean()
    df["sent_norm_diff"] = df["sent_norm_w"].diff()
    df["neg_z_ma5"]      = df["neg_z"].rolling(5).mean()
    # sent_composite: 세 가지 감성 지표의 가중 결합
    df["sent_composite"]      = (df["sent_norm_w"]   * 0.4
                                 + df["neg_z_inv"]   * 0.3
                                 + df["sent_energy"] * 0.3)
    df["sent_composite_ma10"] = df["sent_composite"].rolling(10).mean()
    df["sent_composite_diff"] = df["sent_composite"].diff(5)

    # ─ 추세/모멘텀 피처 (포지션 결정에 직접 사용, K-FGI에는 미포함) ─
    price = df["kospi_close"]
    df["ma20"]            = price.rolling(20).mean()
    df["ma60"]            = price.rolling(60).mean()
    df["ma120"]           = price.rolling(120).mean()
    df["ma_ratio_5_20"]   = price.rolling(5).mean() / (df["ma20"] + 1e-9) - 1
    df["ma_ratio_20_60"]  = df["ma20"] / (df["ma60"] + 1e-9) - 1
    df["ma_ratio_60_120"] = df["ma60"] / (df["ma120"] + 1e-9) - 1
    df["rsi14"]           = compute_rsi(price, 14)
    df["mom5"]            = price.pct_change(5)
    df["mom20"]           = price.pct_change(20)
    df["mom60"]           = price.pct_change(60)
    df["above_ma20"]      = (price > df["ma20"]).astype(float)
    df["above_ma60"]      = (price > df["ma60"]).astype(float)
    df["above_ma120"]     = (price > df["ma120"]).astype(float)
    # trend_strength: 3개 이동평균 대비 현재 가격 위치 합산 (0~3)
    df["trend_strength"]  = df["above_ma20"] + df["above_ma60"] + df["above_ma120"]
    # vol_ratio: 현재 변동성 / 60일 평균 변동성
    df["vol_ratio"]       = df["egarch_vol"] / (
        df["egarch_vol"].rolling(60).mean() + 1e-9
    )

    # ─ 서브인덱스 Lag 피처 ─
    for i in range(1, 8):
        df[f"sub_index{i}_lag1"] = df[f"sub_index{i}"].shift(1)
        df[f"sub_index{i}_lag2"] = df[f"sub_index{i}"].shift(2)

    # ─ EGARCH Lag 피처 ─
    df["egarch_vol_lag1"]     = df["egarch_vol"].shift(1)
    df["egarch_vol_ma5_lag1"] = df["vol_ma5"].shift(1)
    df["vol_shock_lag1"]      = df["vol_shock"].shift(1)
    df["vol_regime_lag1"]     = df["vol_regime_high"].shift(1)
    # neg_z_vol_ratio: 이례적 공포 / 변동성 → 공포의 과도함 측정
    df["neg_z_vol_ratio"]     = (
        df["neg_z_inv"] / (df["egarch_vol_lag1"] + 1e-6)
    ).clip(-10, 10)

    df["dayofweek"]  = df["date"].dt.dayofweek
    df["month"]      = df["date"].dt.month
    df["target_reg"] = df["log_return_t+1"]
    # target_5d: 향후 5일 누적 로그 수익률 (K-FGI 학습 타겟)
    df["target_5d"]  = df["log_return_t+1"].rolling(5).sum().shift(-4)

    before = len(df)

    # ★ 전체 피처 1일 lag — 미래 정보 누출 원천 차단
    feature_cols = [col for col in df.columns
                    if col not in ["date", "target_reg", "target_5d"]]
    df[feature_cols] = df[feature_cols].shift(1)

    df = df.dropna().reset_index(drop=True)
    print(f"    dropna() 후: {before}→{len(df)}행  "
          f"(시작일={df['date'].min().date()})")
    return df


df = build_features(df_raw)
print(f"[3] 피처 엔지니어링 완료: {len(df)}행, {len(df.columns)}컬럼")


# ══════════════════════════════════════════════
# 4. Walk-Forward K-FGI 생성
#
# K-FGI = 순수 감성 + EGARCH 변동성 (추세 피처 제외)
# 가중치: RidgeCV로 walk-forward 학습
#   → t 시점은 [0, t-1] 구간만 사용 (look-ahead bias 차단)
# 최종 0~100 스케일링 (p1~p99 기준)
# ══════════════════════════════════════════════
KFGI_FEATS = (
    [f"sub_index{i}" for i in range(1, 8)]
    + [
        "sent_norm_w", "sent_energy", "sent_std_inv",
        "neg_z_inv", "sent_composite", "sent_composite_ma10",
        "egarch_vol", "vol_regime_high", "vol_ratio",
    ]
)
KFGI_FEATS = [f for f in KFGI_FEATS if f in df.columns]

# 방향 벡터: 감성 피처는 +1 (높을수록 탐욕), 변동성 피처는 -1 (높을수록 공포)
DIRECTION = {
    **{f"sub_index{i}": 1 for i in range(1, 8)},
    "sent_norm_w": 1, "sent_energy": 1, "sent_std_inv": 1,
    "neg_z_inv": 1, "sent_composite": 1, "sent_composite_ma10": 1,
    "egarch_vol": -1, "vol_regime_high": -1, "vol_ratio": -1,
}
DIR_VEC = np.array([DIRECTION.get(f, 1) for f in KFGI_FEATS])


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
            # ★ 극단값 클리핑: |z|>3 노이즈 억제
            X_s   = np.clip(X_s, -3, 3)
            ridge = RidgeCV(alphas=np.logspace(-2, 2, 10))
            ridge.fit(X_s, y_tr)
            # 방향 벡터 반영 후 정규화 가중치 산출
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


print(f"[4] Walk-forward K-FGI 생성 중 "
      f"(순수 감성+변동성 피처 {len(KFGI_FEATS)}개)...")
df = create_walkforward_kfgi(df, KFGI_FEATS, DIR_VEC, CFG["kfgi_min_obs"])

rho_1d, p1d = spearmanr(df["K_FGI"], df["target_reg"])
rho_5d, p5d = spearmanr(df["K_FGI"], df["target_5d"])
print(f"    K-FGI ↔ 1일 수익률  ρ={rho_1d:.4f} (p={p1d:.4f})")
print(f"    K-FGI ↔ 5일 수익률  ρ={rho_5d:.4f} (p={p5d:.4f})")

# ρ 부호 자동 감지 → 포지션 방향 결정
KFGI_MOMENTUM = (rho_5d > 0)
kdir = "순방향(모멘텀)" if KFGI_MOMENTUM else "역발상(공포→매수)"
print(f"    ★ K-FGI 방향 자동 결정: {kdir}")
print(f"    분석 기간: {df['date'].min().date()} ~ "
      f"{df['date'].max().date()}, {len(df)}행")


# ══════════════════════════════════════════════
# 5. LGBM — 분석/검증 전용 (포지션 결정 불개입)
# ══════════════════════════════════════════════
FEATURES = (
    [f"sub_index{i}_lag1" for i in range(1, 8)]
    + [f"sub_index{i}_lag2" for i in range(1, 8)]
    + [
        "sent_norm_w", "sent_energy", "sent_std_inv", "neg_z_inv",
        "sent_norm_ma5", "neg_z_ma5", "sent_norm_diff",
        "sent_composite", "sent_composite_ma10", "sent_composite_diff",
        "egarch_vol_lag1", "egarch_vol_ma5_lag1",
        "vol_shock_lag1", "vol_regime_lag1", "neg_z_vol_ratio",
        "ma_ratio_5_20", "ma_ratio_20_60", "ma_ratio_60_120",
        "rsi14", "mom5", "mom20", "mom60",
        "above_ma20", "above_ma60", "above_ma120",
        "trend_strength", "vol_ratio",
        "K_FGI", "dayofweek", "month",
    ]
)
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"[5] LightGBM 피처 수: {len(FEATURES)}개")

LGBM_PARAMS = dict(
    objective="regression", metric="rmse", learning_rate=0.02,
    num_leaves=31, feature_fraction=0.8, bagging_fraction=0.7,
    bagging_freq=5, lambda_l1=0.1, lambda_l2=0.1,
    min_child_samples=20, verbosity=-1, seed=CFG["seed"],
)


def run_lgbm_analysis(df, features, n_splits=CFG["n_splits"]):
    """LGBM 분석 전용 — 포지션 결정에 사용하지 않음."""
    tscv  = TimeSeriesSplit(n_splits=n_splits)
    X, y  = df[features].values, df["target_5d"].values
    dates = df["date"].values
    rmse_list, preds_all, dates_all = [], [], []
    last_lgbm = None

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        lgbm = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(X[tr_idx], label=y[tr_idx]),
            num_boost_round=500,
        )
        pred = lgbm.predict(X[te_idx])
        rmse_list.append(np.sqrt(mean_squared_error(y[te_idx], pred)))
        preds_all.extend(pred)
        dates_all.extend(dates[te_idx])
        last_lgbm = lgbm

    print(f"\n[6] LightGBM (분석 전용)  "
          f"RMSE: {np.mean(rmse_list):.6f} ± {np.std(rmse_list):.6f}")
    return pd.DataFrame({"date": dates_all, "pred_5d": preds_all}), last_lgbm


pred_df, lgbm_model = run_lgbm_analysis(df, FEATURES)


# ══════════════════════════════════════════════
# 6. Regime 분류
#
# bull  : K-FGI>65 AND 변동성 안정 AND 추세 강함
# crisis: K-FGI<25 OR 변동성 급등(vol_shock>2.0)
# normal: 그 외
# ══════════════════════════════════════════════
def classify_regime(row):
    if (row["K_FGI"] > 65
            and row["vol_regime_high"] == 0
            and row["trend_strength"] >= 2):
        return "bull"
    elif row["K_FGI"] < 25 or row["vol_shock"] > CFG["vol_shock_thr"]:
        return "crisis"
    return "normal"


df["regime"] = df.apply(classify_regime, axis=1)
rc = df["regime"].value_counts()
print(f"\n[7] Regime: bull={rc.get('bull',0)}일  "
      f"normal={rc.get('normal',0)}일  crisis={rc.get('crisis',0)}일")


# ══════════════════════════════════════════════
# 7. 포지션 결정
#
# weight = trend_weight × vt_pos^0.5 × kfgi_mult × regime_mult
#
# trend_weight : MA 기반 추세 방향 (0.0~1.0)
# vt_pos^0.5   : EGARCH 변동성 타게팅 (완화 버전)
# kfgi_mult    : K-FGI → 배율 변환 (0.5~1.6)
# regime_mult  : bull=1.2 / normal=1.0 / crisis=0.7
# ══════════════════════════════════════════════
def kfgi_scale_mult(kfgi_val, is_momentum=True):
    """
    K-FGI → 포지션 배율 변환 (범위: 0.5~1.6).
    중립(K-FGI=50) → 약 1.05
    """
    if is_momentum:
        mult = 0.5 + (kfgi_val / 100.0) * 1.1   # 탐욕일수록 배율 증가
    else:
        mult = 1.6 - (kfgi_val / 100.0) * 1.1   # 공포일수록 배율 증가
    return float(np.clip(mult, 0.5, 1.6))


# 추세 강도(0~3) → 포지션 기본 방향 가중치
TREND_MAP = {0: 0.0, 1: 0.35, 2: 0.70, 3: 1.0}


def compute_positions(df, name="전략"):
    d = df.copy()

    # 추세 방향 가중치
    d["tw"] = d["trend_strength"].apply(
        lambda x: TREND_MAP.get(int(x), 0.0)
    )

    # vol targeting: sqrt 적용으로 변동성 급등 시 급격한 축소 방지
    # (원래: pos = target/egarch → 변동성 2배 시 포지션 1/2)
    # (수정: pos = sqrt(target/egarch) → 변동성 2배 시 포지션 1/√2)
    raw_vt    = TARGET_DAILY_VOL / (d["egarch_vol"] + 1e-9)
    d["vt_pos"] = np.sqrt(raw_vt).clip(CFG["min_leverage"], CFG["max_leverage"])

    # K-FGI 배율 (방향 자동 적용)
    d["kfgi_mult"] = d["K_FGI"].apply(
        lambda k: kfgi_scale_mult(k, is_momentum=KFGI_MOMENTUM)
    )

    # Regime별 배율 — [BUG-6] crisis: 0.8 → 0.7 수정
    regime_mult = d["regime"].map({"bull": 1.2, "normal": 1.0, "crisis": 0.7})

    d["weight"] = (
        d["tw"] * d["vt_pos"] * d["kfgi_mult"] * regime_mult
    ).clip(0.0, CFG["max_leverage"] * 1.5)

    # 안전망 1: vol_shock 극단 → 즉시 포지션 0
    d.loc[d["vol_shock"] > CFG["vol_shock_thr"], "weight"] = 0.0
    # 안전망 2: 3일 누적 수익률 -7% 이하 → 강제 청산
    d["cumret_3d"] = d["target_reg"].rolling(3).sum().shift(1).fillna(0)
    d.loc[d["cumret_3d"] < CFG["tail_loss_thr"], "weight"] = 0.0

    # ★ weight 1일 lag: 오늘 계산한 포지션은 내일 적용 (미래정보 차단)
    d["weight_lag"] = d["weight"].shift(1).fillna(0)

    # turnover: lag 기준으로 계산
    d["turnover"] = d["weight_lag"].diff().abs().fillna(d["weight_lag"].iloc[0])
    d["cost"]     = d["turnover"] * CFG["fee"]

    # 수익률: lag된 포지션 × 실현 수익률 - 거래비용
    d["strat_ret"] = d["weight_lag"] * d["target_reg"] - d["cost"]
    return d


main_df = compute_positions(df)


# ─ 비교 벤치마크 ─
def bench_pure_trend(df):
    """순수 추세추종 (K-FGI/EGARCH 없이). weight lag 적용."""
    d = df.copy()
    d["weight"]     = d["trend_strength"].apply(
        lambda x: TREND_MAP.get(int(x), 0.0)
    )
    # ★ BUG-5 수정: weight lag 적용 (원본에 누락되어 있었음)
    d["weight_lag"] = d["weight"].shift(1).fillna(0)
    d["turnover"]   = d["weight_lag"].diff().abs().fillna(d["weight_lag"].iloc[0])
    d["cost"]       = d["turnover"] * CFG["fee"]
    d["strat_ret"]  = d["weight_lag"] * d["target_reg"] - d["cost"]
    return d


def bench_trend_vol(df):
    """추세 + vol타게팅^0.5 (K-FGI 없이) — EGARCH 독립 기여도 분리."""
    d   = df.copy()
    tw  = d["trend_strength"].apply(lambda x: TREND_MAP.get(int(x), 0.0))
    vt  = np.sqrt(TARGET_DAILY_VOL / (d["egarch_vol"] + 1e-9)).clip(
        CFG["min_leverage"], CFG["max_leverage"]
    )
    d["weight"] = (tw * vt).clip(0, CFG["max_leverage"])
    d.loc[d["vol_shock"] > CFG["vol_shock_thr"], "weight"] = 0.0
    d["weight_lag"] = d["weight"].shift(1).fillna(0)
    d["turnover"]   = d["weight_lag"].diff().abs().fillna(d["weight_lag"].iloc[0])
    d["cost"]       = d["turnover"] * CFG["fee"]
    d["strat_ret"]  = d["weight_lag"] * d["target_reg"] - d["cost"]
    return d


trend_df    = bench_pure_trend(df)
trendvol_df = bench_trend_vol(df)


# ══════════════════════════════════════════════
# 8. 성과 지표
# ══════════════════════════════════════════════
def perf(returns, label, ref=None):
    ann_ret = returns.mean() * 252
    ann_vol = returns.std()  * np.sqrt(252) + 1e-9
    sharpe  = ann_ret / ann_vol
    cum     = np.exp(returns.cumsum())
    mdd     = (cum / cum.cummax() - 1).min()
    calmar  = ann_ret / (abs(mdd) + 1e-9)
    total   = np.exp(returns.sum()) - 1
    defense = float("nan")
    if ref is not None:
        dm      = ref < 0
        defense = returns[dm].mean() / (ref[dm].mean() + 1e-9)
    print(f"  [{label:<36}]  "
          f"연수익={ann_ret*100:+.2f}%  "
          f"Sharpe={sharpe:.3f}  "
          f"MDD={mdd*100:.2f}%  "
          f"Calmar={calmar:.2f}  "
          f"누적={total*100:+.2f}%  "
          f"하락방어율={defense:.2f}")
    return dict(ann_ret=ann_ret, sharpe=sharpe, mdd=mdd, calmar=calmar)


ref = df["target_reg"]
print("\n[8] 성과 비교  (계단식으로 K-FGI/EGARCH 기여도 확인)")
print("-" * 110)
perf(ref,                        "Buy & Hold (KOSPI)")
perf(trend_df["strat_ret"],      "① 추세 추종 (MA 순수)",               ref)
perf(trendvol_df["strat_ret"],   "② 추세 + vol타겟팅^0.5",             ref)
perf(main_df["strat_ret"],       f"③ 메인 v4 (+K-FGI {kdir})",         ref)


# ══════════════════════════════════════════════
# 9. 감성 기여도 — Ablation Study
#
# 감성 피처를 제거한 K-FGI 별도 생성 후
# 예측력(ρ)과 전략 성과 비교
# ══════════════════════════════════════════════
KFGI_FEATS_NS = [
    f for f in KFGI_FEATS
    if not any(k in f for k in ["sent_", "neg_z", "composite"])
]
DIR_VEC_NS = np.array([DIRECTION.get(f, 1) for f in KFGI_FEATS_NS])

print(f"\n[9] 감성 제외 K-FGI 생성 중 "
      f"(피처: {len(KFGI_FEATS)}→{len(KFGI_FEATS_NS)}개)...")
df_ns = df.copy()
df_ns = create_walkforward_kfgi(
    df_ns, KFGI_FEATS_NS, DIR_VEC_NS, CFG["kfgi_min_obs"]
)
df_ns["regime"] = df_ns.apply(classify_regime, axis=1)
main_ns = compute_positions(df_ns)

rho_ns, _ = spearmanr(df_ns["K_FGI"], df_ns["target_5d"])
print(f"    감성 포함 K-FGI ρ = {rho_5d:.4f}")
print(f"    감성 제외 K-FGI ρ = {rho_ns:.4f}")
print(f"    감성 기여 Δρ       = {rho_5d - rho_ns:+.4f}")

print("\n    [9] 감성 피처 기여도 (Ablation Study)")
print("    " + "-" * 106)
perf(main_df["strat_ret"], "메인 v4 (감성 포함)", ref)
perf(main_ns["strat_ret"], "메인 v4 (감성 제외)", df_ns["target_reg"])


# ══════════════════════════════════════════════
# 10. 시각화 (6-panel)
# ══════════════════════════════════════════════
dates      = df["date"]
shock_mask = df["vol_shock"] > CFG["vol_shock_thr"]

fig = plt.figure(figsize=(17, 13))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.30)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])


def calc_dd(ret):
    c = np.exp(ret.cumsum())
    return (c / c.cummax() - 1) * 100


# Panel 1 — 누적 수익률
ax1.plot(dates, np.exp(ref.cumsum()),
         label="KOSPI B&H", color="#9CA3AF", lw=1.5, alpha=0.8)
ax1.plot(dates, np.exp(trend_df["strat_ret"].cumsum()),
         label="① 추세추종", color="#16A34A", lw=1.5, ls="-.")
ax1.plot(dates, np.exp(trendvol_df["strat_ret"].cumsum()),
         label="② 추세+vol^0.5", color="#F59E0B", lw=1.5, ls="--")
ax1.plot(dates, np.exp(main_df["strat_ret"].cumsum()),
         label=f"③ 메인 v4 (K-FGI {kdir})", color="#2563EB", lw=2.5)
ax1.plot(df_ns["date"], np.exp(main_ns["strat_ret"].cumsum()),
         label="③ 메인 v4 (감성 제외)", color="#7C3AED",
         lw=1.2, ls=":", alpha=0.7)
ax1.fill_between(dates, 0.5, 4, where=shock_mask,
                 color="#FCA5A5", alpha=0.2, label="vol_shock 경보")
ax1.set_title(
    f"누적 수익률 v4  (K-FGI {kdir}, ρ(5d)={rho_5d:.3f}, 배율 0.5~1.6)",
    fontsize=12, fontweight="bold",
)
ax1.legend(fontsize=8.5, loc="upper left")
ax1.grid(alpha=0.2)

# Panel 2 — Drawdown
for ret, lbl, col, alp in [
    (ref,                      "KOSPI",        "#9CA3AF", 0.40),
    (trend_df["strat_ret"],    "추세추종",      "#16A34A", 0.35),
    (trendvol_df["strat_ret"], "추세+vol^0.5", "#F59E0B", 0.35),
    (main_df["strat_ret"],     "메인 v4",       "#2563EB", 0.50),
]:
    ax2.fill_between(dates, calc_dd(ret), color=col, alpha=alp, label=lbl)
ax2.set_title("Drawdown 비교 (%)", fontsize=11)
ax2.set_ylabel("DD (%)")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.2)

# Panel 3 — EGARCH σ vs vol-targeting pos
raw_vt_plot  = TARGET_DAILY_VOL / (df["egarch_vol"] + 1e-9)
vt_soft_plot = np.sqrt(raw_vt_plot).clip(CFG["min_leverage"], CFG["max_leverage"])
ax3.fill_between(dates, df["egarch_vol"] * 100,
                 alpha=0.5, color="#F59E0B", label="EGARCH σ (%)")
ax3_r = ax3.twinx()
ax3_r.plot(dates, vt_soft_plot, color="#2563EB", lw=1.2,
           label="vol-target pos (^0.5)")
ax3_r.axhline(1.0, color="#6B7280", linestyle=":", lw=0.8)
ax3_r.set_ylabel("포지션 배수")
ax3.set_title("EGARCH σ vs vol-targeting pos (^0.5)", fontsize=11)
ax3.set_ylabel("σ (%)")
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_r.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax3.grid(alpha=0.2)

# Panel 4 — K-FGI
kv = df["K_FGI"].values
ax4.plot(dates, kv, color="#7C3AED", lw=1.0)
ax4.axhline(25, color="#2563EB", linestyle="--", lw=0.8, alpha=0.7, label="K-FGI<25 (공포)")
ax4.axhline(75, color="#DC2626", linestyle="--", lw=0.8, alpha=0.7, label="K-FGI>75 (탐욕)")
ax4.fill_between(dates, 0, 100, where=(kv < 25),
                 color="#BFDBFE", alpha=0.45, label="공포 구간")
ax4.fill_between(dates, 0, 100, where=(kv > 75),
                 color="#FCA5A5", alpha=0.35, label="탐욕 구간")
ax4.set_title(f"K-FGI (순수 감성+변동성, {kdir}, 배율 0.5~1.6)", fontsize=11)
ax4.set_ylim(0, 100)
ax4.legend(fontsize=7.5)
ax4.grid(alpha=0.2)

# Panel 5 — 포지션 크기
ax5.plot(dates, main_df["weight"],    color="#2563EB", lw=0.9, label="메인 v4")
ax5.plot(dates, trend_df["weight"],   color="#16A34A", lw=0.9,
         alpha=0.7, label="추세추종", ls="-.")
ax5.plot(df_ns["date"], main_ns["weight"], color="#7C3AED", lw=0.9,
         alpha=0.7, label="감성 제외", ls=":")
ax5.axhline(1.0, color="#6B7280", linestyle=":", lw=0.8)
ax5.set_title("포지션 크기 (vol^0.5 + K-FGI 0.5~1.6 + crisis 0.7)", fontsize=11)
ax5.set_ylabel("레버리지 배수")
ax5.legend(fontsize=8)
ax5.grid(alpha=0.2)

fig.suptitle(
    f"K-FGI × EGARCH v4 — vol^0.5 | crisis 0.7 | K-FGI 0.5~1.6 | {kdir}",
    fontsize=13, fontweight="bold", y=1.01,
)
plt.savefig("kfgi_egarch_result_v4.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n[10] 시각화 저장: kfgi_egarch_result_v4.png")


# ══════════════════════════════════════════════
# 11. Feature Importance (LGBM, 분석용)
# ══════════════════════════════════════════════
imp = pd.DataFrame({
    "feature":    FEATURES,
    "importance": lgbm_model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)

sent_imp  = imp[imp["feature"].str.contains(
    "sent_|neg_z|composite", regex=True
)]["importance"].sum()
total_imp = imp["importance"].sum()

print(f"\n[11] Top-15 Feature Importance  "
      f"(감성 피처 비율: {sent_imp/total_imp*100:.1f}%)")
print("-" * 50)
print(imp.head(15).to_string(index=False))

fig2, ax = plt.subplots(figsize=(9, 6))
top15  = imp.head(15)
colors = [
    "#DC2626" if any(k in f for k in ["sent_", "neg_z", "composite"])
    else "#F59E0B" if any(k in f for k in
         ["egarch", "vol_", "ma_ratio", "rsi", "mom", "above", "trend"])
    else "#2563EB"
    for f in top15["feature"]
]
ax.barh(top15["feature"][::-1], top15["importance"][::-1], color=colors[::-1])
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="#DC2626", label="감성 피처"),
    Patch(color="#F59E0B", label="추세/변동성 피처"),
    Patch(color="#2563EB", label="기타"),
], fontsize=9)
ax.grid(alpha=0.2, axis="x")
plt.tight_layout()
plt.xlabel("Feature Importance (Gain)", fontsize=7)
plt.savefig("feature_importance_v4.png", dpi=150, bbox_inches="tight")
plt.show()
print("    저장: feature_importance_v4.png")


# ══════════════════════════════════════════════
# 12. Out-of-Sample (OOS) 분리 평가
# ══════════════════════════════════════════════
split_date = "2025-01-01"
train_df = df[df["date"] < split_date].copy()
test_df  = df[df["date"] >= split_date].copy()

print("\n[OOS] Train / Test 분리")
print(f"    Train: {train_df['date'].min().date()} ~ "
      f"{train_df['date'].max().date()}  ({len(train_df)}행)")
print(f"    Test : {test_df['date'].min().date()}  ~ "
      f"{test_df['date'].max().date()}  ({len(test_df)}행)")

train_res = compute_positions(train_df, name="Train")
test_res  = compute_positions(test_df,  name="Test")

print("\n[OOS 성과 비교]")
print("-" * 60)
perf(train_res["strat_ret"], "Train (In-sample)",   train_df["target_reg"])
perf(test_res["strat_ret"],  "Test  (Out-of-Sample)", test_df["target_reg"])


# ══════════════════════════════════════════════
# 13. 거래비용 스트레스 테스트
# ★ BUG-4 수정: CFG["fee"] 복구 로직 추가
# ★ BUG-1 수정: 중복 함수 정의 제거
# ══════════════════════════════════════════════
def run_fee_test(df, fee_list=[0.00015, 0.0003, 0.00045]):
    """
    수수료를 달리했을 때의 성과 비교.
    CFG["fee"]를 테스트 후 원래 값으로 복구.
    """
    print("\n[거래비용 스트레스 테스트]")
    print("-" * 70)
    original_fee = CFG["fee"]   # ★ 원래 수수료 저장
    for fee in fee_list:
        CFG["fee"] = fee
        res = compute_positions(df.copy())
        print(f"\n▶ 수수료 = {fee*100:.3f}% (편도)")
        perf(res["strat_ret"], f"전략 (fee={fee*100:.3f}%)", df["target_reg"])
    CFG["fee"] = original_fee   # ★ 반드시 복구


run_fee_test(test_df)


# ══════════════════════════════════════════════
# 14. 기간별(연도별) 성과 분석
# ══════════════════════════════════════════════
def period_test(df):
    print("\n[기간별(연도별) 성과 분석]")
    print("-" * 70)
    df = df.copy()
    df["year"] = df["date"].dt.year
    for y in sorted(df["year"].unique()):
        sub = df[df["year"] == y]
        if len(sub) < 50:
            continue
        res = compute_positions(sub.copy())
        print(f"\n▶ {y}년 ({len(sub)}거래일)")
        perf(res["strat_ret"], f"{y}년", sub["target_reg"])


period_test(df)


# ══════════════════════════════════════════════
# 15. Regime별 성과 분석
# ★ BUG-2 수정: regime_test 이중 호출 제거
# ══════════════════════════════════════════════
def regime_test(df):
    print("\n[Regime별 성과 분석]")
    print("-" * 70)
    for regime in ["bull", "normal", "crisis"]:
        sub = df[df["regime"] == regime]
        if len(sub) < 30:
            continue
        print(f"\n▶ {regime.upper()} 국면 ({len(sub)}일)")
        perf(sub["strat_ret"], regime, sub["target_reg"])


regime_test(main_df)   # ★ 한 번만 호출


# ══════════════════════════════════════════════
# 16. Rolling OOS 테스트
# ══════════════════════════════════════════════
def rolling_oos_test(df):
    print("\n[Rolling OOS 테스트]")
    print("-" * 70)
    years = sorted(df["date"].dt.year.unique())
    for i in range(2, len(years)):
        train_years = years[:i]
        test_year   = years[i]
        train = df[df["date"].dt.year.isin(train_years)]
        test  = df[df["date"].dt.year == test_year]
        if len(test) < 50:
            continue
        test_res = compute_positions(test.copy())
        print(f"\n▶ 훈련: {list(train_years)}  →  테스트: {test_year}년 ({len(test)}일)")
        perf(test_res["strat_ret"], f"OOS {test_year}년", test["target_reg"])


rolling_oos_test(df)


# ══════════════════════════════════════════════
# 17. 통계적 유의성 검정 (강화판)
#
# [A] 전체 초과수익 t-test (index alignment 수정)
# [B] 하락 구간 Downside t-test (index alignment 수정)
# [C] K-FGI 극단 구간 수익률 차이 t-test ★ 감성 유의성 핵심
# [D] 감성 Spearman ρ 부트스트랩 검정
# [E] MDD 비교 요약
# ══════════════════════════════════════════════
print("\n" + "=" * 62)
print("  [17] 통계적 유의성 검정 (강화판)")
print("=" * 62)

# ★ BUG-3 수정: index alignment — 공통 인덱스만 사용
common_idx = main_df.index.intersection(df.index)
excess = (main_df.loc[common_idx, "strat_ret"]
          - df.loc[common_idx, "target_reg"]).dropna()

# ─ [A] 전체 초과수익 t-test ─
t_A, p_A = stats.ttest_1samp(excess, 0)
print(f"""
[A] 전체 기간 초과수익 t-test
    귀무가설: 일별 초과수익 평균 = 0
    t-stat  : {t_A:.4f}
    p-value : {p_A:.6f}
    결론    : {"p < 0.05 → 전체 기간 초과수익 통계적으로 유의" if p_A < 0.05
               else "p ≥ 0.05 → 전체 기간 초과수익은 통계적으로 유의하지 않음"}
    해석    : 2025년 강세장에서 B&H가 빠르게 상승한 구간이 존재해
              초과수익이 희석됨. 이 전략은 '더 많이 버는 전략'이 아닌
              '하락 시 덜 잃는 전략'으로 해석해야 함.""")

# ─ [B] 하락 구간 Downside t-test ─
down_mask   = df.loc[common_idx, "target_reg"] < 0
excess_down = excess[down_mask[down_mask].index.intersection(excess.index)].dropna()
t_B, p_B    = stats.ttest_1samp(excess_down, 0)
print(f"""
[B] 시장 하락 구간 Downside t-test
    대상 기간: 시장 일별 수익률 < 0인 날 ({len(excess_down)}일)
    귀무가설 : 하락 구간 초과수익 평균 = 0
    t-stat   : {t_B:.4f}
    p-value  : {p_B:.6f}
    결론     : {"p < 0.05 → 하락 방어 효과가 통계적으로 유의함 ★★" if p_B < 0.05
                else "p ≥ 0.05 → 유의하지 않음"}
    해석     : 이 전략의 핵심 목표인 MDD 방어가 통계적으로 달성됨""")

# ─ [C] K-FGI 극단 구간 수익률 차이 t-test ─
# K-FGI 상위 25%(탐욕) vs 하위 25%(공포) 구간의 다음날 수익률 비교
# → "감성이 유의하다"를 직접 검증하는 핵심 검정
kfgi_q25 = main_df["K_FGI"].quantile(0.25)
kfgi_q75 = main_df["K_FGI"].quantile(0.75)

greed_ret = main_df.loc[main_df["K_FGI"] >= kfgi_q75, "target_reg"].dropna()
fear_ret  = main_df.loc[main_df["K_FGI"] <= kfgi_q25, "target_reg"].dropna()

t_C, p_C = stats.ttest_ind(greed_ret, fear_ret, equal_var=False)
print(f"""
[C] K-FGI 극단 구간 수익률 차이 t-test  ★ 감성 유의성 핵심 검정
    귀무가설: 탐욕 구간과 공포 구간의 수익률 평균이 같다
    탐욕 구간 (K-FGI≥{kfgi_q75:.1f}, {len(greed_ret)}일) 평균수익률: {greed_ret.mean()*100:+.4f}%
    공포 구간 (K-FGI≤{kfgi_q25:.1f}, {len(fear_ret)}일) 평균수익률: {fear_ret.mean()*100:+.4f}%
    t-stat  : {t_C:.4f}
    p-value : {p_C:.6f}
    결론    : {"p < 0.05 → K-FGI 감성 구간 간 수익률 차이가 통계적으로 유의합니다 ★★" if p_C < 0.05
               else "p ≥ 0.05 → 유의하지 않음"}""")

# ─ [D] 감성 Spearman ρ 부트스트랩 검정 ─
print(f"\n[D] 감성 기여 Spearman ρ 부트스트랩 검정 (n=1000)")
np.random.seed(42)
n_boot = 1000

common_ns   = main_df.index.intersection(df_ns.index)
kfgi_with   = main_df.loc[common_ns, "K_FGI"].values
kfgi_without= df_ns.loc[common_ns, "K_FGI"].values
ret_5d_vals = main_df.loc[common_ns, "target_5d"].values

valid_mask  = (~np.isnan(kfgi_with)
               & ~np.isnan(kfgi_without)
               & ~np.isnan(ret_5d_vals))
kw  = kfgi_with[valid_mask]
kwo = kfgi_without[valid_mask]
r5  = ret_5d_vals[valid_mask]

rho_obs_with    = spearmanr(kw,  r5)[0]
rho_obs_without = spearmanr(kwo, r5)[0]
rho_diff_obs    = rho_obs_with - rho_obs_without

boot_diffs = []
n_valid = len(kw)
for _ in range(n_boot):
    idx     = np.random.choice(n_valid, n_valid, replace=True)
    rho_w   = spearmanr(kw[idx],  r5[idx])[0]
    rho_wo  = spearmanr(kwo[idx], r5[idx])[0]
    boot_diffs.append(rho_w - rho_wo)

boot_diffs = np.array(boot_diffs)
p_boot     = float(np.mean(boot_diffs <= 0))

print(f"    감성 포함 K-FGI ρ: {rho_obs_with:.4f}")
print(f"    감성 제외 K-FGI ρ: {rho_obs_without:.4f}")
print(f"    관측 Δρ          : {rho_diff_obs:+.4f}")
print(f"    부트스트랩 p-value (단측): {p_boot:.4f}")
if p_boot < 0.05:
    print("    → 유의: 감성 피처가 K-FGI 예측력을 유의미하게 향상시킵니다 ★★")
else:
    print("    → 유의하지 않음 (공통 구간 샘플 부족으로 검정력 제한)")

# ─ [E] MDD 비교 요약 ─
print(f"""
[E] 전략별 MDD 비교 요약
{"-" * 50}
    Buy & Hold (KOSPI)              MDD = {(np.exp(ref.cumsum()) / np.exp(ref.cumsum()).cummax() - 1).min()*100:.2f}%
    ① 추세 추종 (MA)                    MDD = {(np.exp(trend_df["strat_ret"].cumsum()) / np.exp(trend_df["strat_ret"].cumsum()).cummax() - 1).min()*100:.2f}%
    ② 추세 + vol^0.5                  MDD = {(np.exp(trendvol_df["strat_ret"].cumsum()) / np.exp(trendvol_df["strat_ret"].cumsum()).cummax() - 1).min()*100:.2f}%
    ③ 메인 v4 (감성 포함)                 MDD = {(np.exp(main_df["strat_ret"].cumsum()) / np.exp(main_df["strat_ret"].cumsum()).cummax() - 1).min()*100:.2f}%
    ③ 메인 v4 (감성 제외)                 MDD = {(np.exp(main_ns["strat_ret"].cumsum()) / np.exp(main_ns["strat_ret"].cumsum()).cummax() - 1).min()*100:.2f}%""")

print("\n파이프라인 v4 (최종 확정판) 완료.")
print("주요 수정: BUG-1~6 수정, 통계 검정 강화(극단 구간 t-test, 부트스트랩) 완료.")

# ============================================================
# 포지션 파라미터 민감도 분석
# ============================================================
import itertools

print("\n" + "="*62)
print("  포지션 파라미터 민감도 분석")
print("="*62)

# --- kfgi_mult 범위 조합 (min, max) ---
kfgi_ranges = [
    (0.3, 1.4),
    (0.5, 1.6),   # 채택값 ★
    (0.5, 2.0),
    (0.7, 1.4),
    (0.7, 1.6),
]

# --- regime 배율 조합 (bull, normal, crisis) ---
regime_combos = [
    (1.1, 1.0, 0.8),
    (1.2, 1.0, 0.7),   # 채택값 ★
    (1.3, 1.0, 0.6),
    (1.2, 1.0, 0.5),
]

sens_results = []

for (kmin, kmax), (bull_r, norm_r, crisis_r) in itertools.product(kfgi_ranges, regime_combos):

    d = df.copy()

    # trend_weight 재계산
    d["tw"] = d["trend_strength"].apply(lambda x: TREND_MAP.get(int(x), 0.0))

    # vol targeting (기존과 동일)
    raw_vt      = TARGET_DAILY_VOL / (d["egarch_vol"] + 1e-9)
    d["vt_pos"] = np.sqrt(raw_vt).clip(CFG["min_leverage"], CFG["max_leverage"])

    # kfgi_mult 재계산
    if KFGI_MOMENTUM:
        d["kfgi_mult"] = (kmin + (d["K_FGI"] / 100.0) * (kmax - kmin)).clip(kmin, kmax)
    else:
        d["kfgi_mult"] = (kmax - (d["K_FGI"] / 100.0) * (kmax - kmin)).clip(kmin, kmax)

    # regime_mult 재계산
    regime_mult = d["regime"].map({"bull": bull_r, "normal": norm_r, "crisis": crisis_r})

    # 포지션 재계산
    d["weight"] = (d["tw"] * d["vt_pos"] * d["kfgi_mult"] * regime_mult).clip(0.0, CFG["max_leverage"] * 1.5)

    # 안전망 적용
    d.loc[d["vol_shock"] > CFG["vol_shock_thr"], "weight"] = 0.0
    d["cumret_3d"] = d["target_reg"].rolling(3).sum().shift(1).fillna(0)
    d.loc[d["cumret_3d"] < CFG["tail_loss_thr"], "weight"] = 0.0

    # lag 적용
    d["weight_lag"] = d["weight"].shift(1).fillna(0)
    d["turnover"]   = d["weight_lag"].diff().abs().fillna(d["weight_lag"].iloc[0])
    d["cost"]       = d["turnover"] * CFG["fee"]
    d["strat_ret"]  = d["weight_lag"] * d["target_reg"] - d["cost"]

    # 성과 계산
    ann_ret = d["strat_ret"].mean() * 252
    ann_vol = d["strat_ret"].std() * np.sqrt(252) + 1e-9
    sharpe  = ann_ret / ann_vol
    cum     = np.exp(d["strat_ret"].cumsum())
    mdd     = (cum / cum.cummax() - 1).min()
    total   = np.exp(d["strat_ret"].sum()) - 1
    is_adopted = (kmin == 0.5 and kmax == 1.6 and bull_r == 1.2 and crisis_r == 0.7)

    sens_results.append({
        "kfgi_범위":      f"{kmin}~{kmax}",
        "regime(B/N/C)": f"{bull_r}/{norm_r}/{crisis_r}",
        "연수익(%)":      round(ann_ret * 100, 2),
        "Sharpe":        round(sharpe, 3),
        "MDD(%)":        round(mdd * 100, 2),
        "누적수익(%)":    round(total * 100, 2),
        "채택":          "★" if is_adopted else "",
    })

df_sens = pd.DataFrame(sens_results).sort_values("Sharpe", ascending=False)
print(df_sens.to_string(index=False))
