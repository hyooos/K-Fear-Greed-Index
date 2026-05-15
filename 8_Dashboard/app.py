"""
K-FGI Strategy Dashboard
CSV(KFG_final_2.csv)를 직접 읽어 EGARCH walk-forward 변동성을 계산한 뒤
스나이퍼/불도저 전략 성과를 시각화한다.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

# ═══════════════════════════════════════════════
# 1. 페이지 설정
# ═══════════════════════════════════════════════
st.set_page_config(page_title="K-FGI Strategy", layout="wide", page_icon="📈")
st.markdown("""
<style>
    .main .block-container h1 {
        font-size:3rem; font-weight:800; color:#00adb5; margin-bottom:0;
    }
    .main .block-container > div > p {
        font-size:1.2rem; font-weight:500; color:#aaaaaa;
    }
    h3 {
        font-weight:700 !important; color:#eeeeee !important;
        border-bottom:2px solid #00adb5; padding-bottom:10px; margin-top:30px !important;
    }
    [data-testid="stMetricValue"] {
        font-size:2rem !important; color:#00adb5 !important; font-weight:700 !important;
    }
    hr { margin:2em 0; border-top:1px solid #444 !important; }
</style>
""", unsafe_allow_html=True)

st.title("⚡Panic Buy, Party Sell")
st.markdown(": 감정을 지배하는 투자 전략")
st.divider()

# ═══════════════════════════════════════════════
# 2. 설정값 (modeling_v4_fixed.py CFG와 동일)
# ═══════════════════════════════════════════════
CFG = dict(
    target_vol_ann=0.15,
    max_leverage=2.0,
    min_leverage=0.1,
    vol_shock_thr=2.0,
    tail_loss_thr=-0.07,
    fee=0.00015,
    egarch_min_obs=60,
    kfgi_min_obs=60,
)
TARGET_DAILY_VOL = CFG["target_vol_ann"] / np.sqrt(252)

KFGI_FEATS_ORDERED = (
    [f"sub_index{i}" for i in range(1, 8)]
    + ["sent_norm_w", "sent_energy", "sent_std_inv",
       "neg_z_inv", "sent_composite", "sent_composite_ma10",
       "egarch_vol", "vol_regime_high", "vol_ratio"]
)
DIRECTION = {
    **{f"sub_index{i}": 1 for i in range(1, 8)},
    "sent_norm_w": 1, "sent_energy": 1, "sent_std_inv": 1,
    "neg_z_inv": 1, "sent_composite": 1, "sent_composite_ma10": 1,
    "egarch_vol": -1, "vol_regime_high": -1, "vol_ratio": -1,
}
TREND_MAP = {0: 0.00, 1: 0.35, 2: 0.70, 3: 1.00}

# CSV 경로 (dashboard.py와 같은 폴더에 위치)
CSV_PATH = os.path.join(os.path.dirname(__file__), "KFG_final_2.csv")


# ═══════════════════════════════════════════════
# 3. EGARCH walk-forward (modeling_v4_fixed.py와 동일)
# ═══════════════════════════════════════════════
def fit_egarch_walkforward(log_returns: pd.Series, min_obs: int) -> pd.Series:
    """
    t 시점은 [0, t-1] 구간 데이터만 사용 → look-ahead bias 차단.
    arch 패키지 없으면 rolling std(20일)로 대체.
    """
    n = len(log_returns)
    sigma = np.full(n, np.nan)
    for t in range(min_obs, n):
        train = log_returns.iloc[:t].dropna()
        if not HAS_ARCH:
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


# ═══════════════════════════════════════════════
# 4. 유틸 함수
# ═══════════════════════════════════════════════
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def compute_metrics(ret: pd.Series, periods: int = 252) -> dict:
    ret = ret.dropna()
    if len(ret) == 0:
        return {}
    ann_ret  = (1 + ret).prod() ** (periods / len(ret)) - 1
    ann_vol  = ret.std() * np.sqrt(periods)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0.0
    neg_ret  = ret[ret < 0]
    downside = neg_ret.std() * np.sqrt(periods) if len(neg_ret) > 0 else 1e-9
    sortino  = ann_ret / downside
    cum      = (1 + ret).cumprod()
    mdd      = (cum / cum.cummax() - 1).min()
    calmar   = ann_ret / abs(mdd) if mdd != 0 else 0.0
    win_rate = (ret > 0).mean()
    max_consec = cur = 0
    for r in ret:
        cur = cur + 1 if r < 0 else 0
        max_consec = max(max_consec, cur)
    return dict(
        ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
        sortino=sortino, calmar=calmar, mdd=mdd,
        win_rate=win_rate, max_consec_loss=max_consec,
    )


def make_monthly_heatmap(ret: pd.Series, dates: pd.Series) -> go.Figure:
    monthly = (
        pd.DataFrame({"ret": ret.values, "date": pd.to_datetime(dates.values)})
        .set_index("date")["ret"]
        .resample("ME")
        .apply(lambda x: (1 + x).prod() - 1)
    )
    pivot = (
        monthly.reset_index()
        .assign(year=lambda d: d["date"].dt.year,
                month=lambda d: d["date"].dt.month)
        .pivot(index="year", columns="month", values="ret")
    )
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    z    = pivot.values * 100
    text = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=[month_names[m-1] for m in pivot.columns],
        y=[str(y) for y in pivot.index],
        text=text, texttemplate="%{text}",
        colorscale=[
            [0.0, "#d32f2f"], [0.35, "#ff8a65"],
            [0.5, "#37474f"],
            [0.65, "#4db6ac"], [1.0, "#00695c"],
        ],
        zmid=0, colorbar=dict(title="수익률(%)"), hoverongaps=False,
    ))
    fig.update_layout(
        title="<b>월별 수익률 히트맵</b>",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eeeeee"), xaxis=dict(side="top"),
        height=max(220, 60*len(pivot)+120),
        margin=dict(l=60, r=20, t=80, b=20),
    )
    return fig


# ═══════════════════════════════════════════════
# 5. 데이터 로드 + 전체 파이프라인 (캐시)
#    modeling_v4_fixed.py 파이프라인 그대로 재현
# ═══════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_and_compute() -> pd.DataFrame:
    # ── 원본 CSV 로드 ──
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ── EGARCH walk-forward ──
    log_ret = np.log(
        df["kospi_close"] / df["kospi_close"].shift(1)
    ).fillna(0)
    df["egarch_vol"] = fit_egarch_walkforward(log_ret, CFG["egarch_min_obs"])

    # EGARCH 파생 피처
    df["vol_shock"]       = df["egarch_vol"].pct_change().clip(-5, 5)
    df["vol_regime_high"] = (
        df["egarch_vol"] > df["egarch_vol"].rolling(60).quantile(0.7)
    ).astype(float)
    df["vol_ma5"]   = df["egarch_vol"].rolling(5).mean()
    df["vol_ratio"] = df["egarch_vol"] / (df["egarch_vol"].rolling(60).mean() + 1e-9)

    # ── 감성 파생 피처 ──
    df["neg_z_inv"]           = -df["neg_z"]
    df["sent_std_inv"]        = -df["sent_std"]
    df["sent_energy"]         = df["sent_strength_w"] * df["sent_norm_w"]
    df["sent_norm_ma5"]       = df["sent_norm_w"].rolling(5).mean()
    df["sent_norm_diff"]      = df["sent_norm_w"].diff()
    df["neg_z_ma5"]           = df["neg_z"].rolling(5).mean()
    df["sent_composite"]      = (df["sent_norm_w"]*0.4
                                 + df["neg_z_inv"]*0.3
                                 + df["sent_energy"]*0.3)
    df["sent_composite_ma10"] = df["sent_composite"].rolling(10).mean()
    df["sent_composite_diff"] = df["sent_composite"].diff(5)

    # ── 추세/모멘텀 피처 ──
    price = df["kospi_close"]
    df["ma20"]           = price.rolling(20).mean()
    df["ma60"]           = price.rolling(60).mean()
    df["ma120"]          = price.rolling(120).mean()
    df["ma_ratio_5_20"]  = price.rolling(5).mean() / (df["ma20"] + 1e-9) - 1
    df["ma_ratio_20_60"] = df["ma20"] / (df["ma60"] + 1e-9) - 1
    df["rsi14"]          = compute_rsi(price, 14)
    df["mom5"]           = price.pct_change(5)
    df["mom20"]          = price.pct_change(20)
    df["above_ma20"]     = (price > df["ma20"]).astype(float)
    df["above_ma60"]     = (price > df["ma60"]).astype(float)
    df["above_ma120"]    = (price > df["ma120"]).astype(float)
    df["trend_strength"] = df["above_ma20"] + df["above_ma60"] + df["above_ma120"]

    df["target_reg"] = df["log_return_t+1"]
    df["target_5d"]  = df["log_return_t+1"].rolling(5).sum().shift(-4)

    # ── 전체 피처 1일 lag (미래 정보 차단) ──
    lag_cols = [c for c in df.columns if c not in ["date", "target_reg", "target_5d"]]
    df[lag_cols] = df[lag_cols].shift(1)
    df = df.dropna(subset=["target_reg", "egarch_vol"]).reset_index(drop=True)

    # ── Walk-forward K-FGI ──
    kfgi_feats = [f for f in KFGI_FEATS_ORDERED if f in df.columns]
    dir_vec    = np.array([DIRECTION.get(f, 1) for f in kfgi_feats])
    min_obs    = CFG["kfgi_min_obs"]
    n          = len(df)
    raw_scores = np.full(n, np.nan)

    for t in range(min_obs, n):
        X_tr = df[kfgi_feats].iloc[:t].values
        y_tr = df["target_5d"].iloc[:t].values
        mask = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr)
        X_tr, y_tr = X_tr[mask], y_tr[mask]
        if len(X_tr) < 30:
            continue
        try:
            sc    = StandardScaler()
            X_s   = np.clip(sc.fit_transform(X_tr), -3, 3)
            ridge = RidgeCV(alphas=np.logspace(-2, 2, 10))
            ridge.fit(X_s, y_tr)
            coef  = ridge.coef_ * dir_vec
            w     = coef / (np.sum(np.abs(coef)) + 1e-12)
            x_t   = df[kfgi_feats].iloc[t:t+1].values
            if np.isnan(x_t).any():
                continue
            raw_scores[t] = float(np.clip(sc.transform(x_t), -3, 3) @ w)
        except Exception:
            continue

    valid       = raw_scores[~np.isnan(raw_scores)]
    p1, p99     = np.percentile(valid, 1), np.percentile(valid, 99)
    df["K_FGI"] = 100 * (np.clip(raw_scores, p1, p99) - p1) / (p99 - p1 + 1e-12)
    df          = df.dropna(subset=["K_FGI"]).reset_index(drop=True)

    # K-FGI 방향 자동 결정
    from scipy.stats import spearmanr
    rho_5d, _ = spearmanr(df["K_FGI"], df["target_5d"].fillna(0))
    df.attrs["kfgi_momentum"] = bool(rho_5d > 0)

    # ── Regime 분류 ──
    def classify_regime(row):
        if (row["K_FGI"] > 65
                and row["vol_regime_high"] == 0
                and row["trend_strength"] >= 2):
            return "bull"
        elif row["K_FGI"] < 25 or row["vol_shock"] > CFG["vol_shock_thr"]:
            return "crisis"
        return "normal"

    df["regime"] = df.apply(classify_regime, axis=1)

    # ── 공통 파생 ──
    df["tw"]        = df["trend_strength"].map(TREND_MAP).fillna(0.0)
    raw_vt          = TARGET_DAILY_VOL / (df["egarch_vol"] + 1e-9)
    df["vt_pos"]    = np.sqrt(raw_vt).clip(CFG["min_leverage"], CFG["max_leverage"])
    regime_mult     = df["regime"].map({"bull": 1.2, "normal": 1.0, "crisis": 0.7})
    df["cumret_3d"] = df["target_reg"].rolling(3).sum().shift(1).fillna(0)
    df["market_trend"] = (df["kospi_close"] > df["ma20"]).astype(int)

    # ── 스나이퍼 포지션 (모멘텀: 탐욕 → 비중 증가) ──
    is_mom = df.attrs["kfgi_momentum"]
    if is_mom:
        df["kfgi_mult_sniper"] = (0.5 + (df["K_FGI"] / 100.0) * 1.1).clip(0.5, 1.6)
    else:
        df["kfgi_mult_sniper"] = (1.6 - (df["K_FGI"] / 100.0) * 1.1).clip(0.5, 1.6)

    df["weight_sniper"] = (
        df["tw"] * df["vt_pos"] * df["kfgi_mult_sniper"] * regime_mult
    ).clip(0.0, CFG["max_leverage"] * 1.5)
    df.loc[df["vol_shock"]  > CFG["vol_shock_thr"],  "weight_sniper"] = 0.0
    df.loc[df["cumret_3d"]  < CFG["tail_loss_thr"],  "weight_sniper"] = 0.0
    df["weight_sniper_lag"] = df["weight_sniper"].shift(1).fillna(0)
    df["turnover_sniper"]   = (df["weight_sniper_lag"].diff().abs()
                               .fillna(df["weight_sniper_lag"].iloc[0]))
    df["sniper_ret"] = (df["weight_sniper_lag"] * df["target_reg"]
                        - df["turnover_sniper"] * CFG["fee"])

    # ── 불도저 포지션 (역발상: 공포 → 비중 증가) ──
    df["kfgi_mult_bull"] = (1.6 - (df["K_FGI"] / 100.0) * 1.1).clip(0.5, 1.6)
    df["weight_bull"] = (
        df["tw"] * df["vt_pos"] * df["kfgi_mult_bull"] * regime_mult
    ).clip(0.0, CFG["max_leverage"] * 1.5)
    df.loc[df["vol_shock"]  > CFG["vol_shock_thr"],  "weight_bull"] = 0.0
    df.loc[df["cumret_3d"]  < CFG["tail_loss_thr"],  "weight_bull"] = 0.0
    df["weight_bull_lag"] = df["weight_bull"].shift(1).fillna(0)
    df["turnover_bull"]   = (df["weight_bull_lag"].diff().abs()
                             .fillna(df["weight_bull_lag"].iloc[0]))
    df["bull_ret"] = (df["weight_bull_lag"] * df["target_reg"]
                      - df["turnover_bull"] * CFG["fee"])

    df["bnh_ret"] = df["target_reg"]
    return df


# ═══════════════════════════════════════════════
# 6. 사이드바 — 전략 설정 (파일 업로더 없음)
# ═══════════════════════════════════════════════
st.sidebar.title("K-Fear & Greed Index")
st.sidebar.markdown("---")

if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

with st.sidebar.form(key="setup_form"):
    strategy_choice = st.radio(
        "1️⃣ 투자 전략 선택",
        ("스나이퍼 (MDD 방어형)", "불도저 (공격/위험감수형)"),
    )
    position_choice = st.radio("2️⃣ 현재 포지션", ("매수 포지션", "매도 포지션"))
    capital_options = {
        "100만 원":   1_000_000,
        "500만 원":   5_000_000,
        "1,000만 원": 10_000_000,
        "5,000만 원": 50_000_000,
        "1억 원":     100_000_000,
        "5억 원":     500_000_000,
    }
    selected_capital_str = st.selectbox(
        "3️⃣ 초기 투자 금액", list(capital_options.keys()), index=2,
    )
    initial_capital = capital_options[selected_capital_str]
    submit_button = st.form_submit_button("분석 시작", use_container_width=True)

if submit_button:
    st.session_state.analysis_started = True

# ═══════════════════════════════════════════════
# 7. 데이터 로드 (캐시 — walk-forward 포함)
# ═══════════════════════════════════════════════
if not os.path.exists(CSV_PATH):
    st.error(f"CSV 파일을 찾을 수 없습니다: `{CSV_PATH}`\n\n"
             "dashboard.py와 같은 폴더에 `KFG_final_2.csv`를 두세요.")
    st.stop()

with st.status(
    "K-FGI walk-forward 계산 중... (EGARCH + RidgeCV, 첫 실행 시 수분 소요)",
    expanded=False,
) as _status:
    st.write("📐 EGARCH walk-forward 변동성 추정 중...")
    st.write("📐 Walk-forward K-FGI 인덱스 생성 중...")
    df = load_and_compute()
    _status.update(
        label=(f"✅ 완료 ({len(df):,}행, "
               f"{df['date'].min().strftime('%Y.%m')} ~ "
               f"{df['date'].max().strftime('%Y.%m')})"),
        state="complete", expanded=False,
    )

KFGI_MOMENTUM = df.attrs.get("kfgi_momentum", True)
kdir = "순방향(모멘텀)" if KFGI_MOMENTUM else "역발상(공포→매수)"

# 사이드바 요약
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**📊 데이터 요약**  \n"
    f"기간: `{df['date'].min().strftime('%Y.%m')}` ~ `{df['date'].max().strftime('%Y.%m')}`  \n"
    f"거래일: `{len(df):,}일`  \n"
    f"K-FGI 방향: `{kdir}`  \n"
    f"K-FGI 현재값: `{df['K_FGI'].iloc[-1]:.1f}`"
)

# 날짜 슬라이더
with st.sidebar:
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    start_date, end_date = st.slider(
        "4️⃣ 백테스팅 기간",
        min_value=min_date, max_value=max_date,
        value=(min_date, max_date),
    )

# ═══════════════════════════════════════════════
# 8. 분석 시작 전 랜딩 화면
# ═══════════════════════════════════════════════
if not st.session_state.analysis_started:
    st.markdown("""
    <div style='text-align:center;color:#aaaaaa;margin-top:60px;'>
        <h3>시장의 심리를 읽고, 데이터로 증명합니다.</h3>
        <p>K-FGI(한국형 공포탐욕지수)를 활용한 맞춤형 퀀트 리포트를 생성해보세요.</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════
# 9. 필터링 + 공통 변수
# ═══════════════════════════════════════════════
mask       = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
fdf        = df.loc[mask].copy().reset_index(drop=True)
is_sniper  = "스나이퍼" in strategy_choice
strat_col  = "sniper_ret"       if is_sniper else "bull_ret"
weight_col = "weight_sniper_lag" if is_sniper else "weight_bull_lag"
strat_name = strategy_choice.split(" ")[0]
other_name = "불도저"            if is_sniper else "스나이퍼"
other_col  = "bull_ret"         if is_sniper else "sniper_ret"

fdf["cum_strat"]   = (1 + fdf[strat_col]).cumprod()
fdf["cum_bnh"]     = (1 + fdf["bnh_ret"]).cumprod()
fdf["strat_asset"] = initial_capital * fdf["cum_strat"]
fdf["bnh_asset"]   = initial_capital * fdf["cum_bnh"]
fdf["dd_strat"]    = fdf["cum_strat"] / fdf["cum_strat"].cummax() - 1
fdf["dd_bnh"]      = fdf["cum_bnh"]   / fdf["cum_bnh"].cummax()   - 1

COMMON = dict(
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#eeeeee"),
    xaxis=dict(showgrid=True, gridcolor="#333", zeroline=False, linecolor="#555"),
    yaxis=dict(showgrid=True, gridcolor="#333", zeroline=False, linecolor="#555"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#444", borderwidth=1),
    margin=dict(l=60, r=40, t=60, b=40),
)

# ═══════════════════════════════════════════════
# 섹션 A: 시장 진단 & 액션 플랜
# ═══════════════════════════════════════════════
st.subheader(f"💡 맞춤형 시장 진단 & 액션 플랜 ({position_choice.split(' ')[1]})")

latest      = fdf.iloc[-1]
kfgi_now    = latest["K_FGI"]
trend_now   = "상승 추세 📈" if latest["market_trend"] == 1 else "하락 추세 📉"
regime_now  = latest["regime"]
weight_now  = latest[weight_col]
prev_weight = fdf.iloc[-2][weight_col] if len(fdf) > 1 else weight_now

col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=kfgi_now,
        title={"text": "<b>K-FGI</b> (공포탐욕지수)", "font": {"color":"#eeeeee","size":20}},
        number={"font": {"color":"#eeeeee","size":40}},
        gauge={
            "axis": {"range":[0,100], "tickcolor":"#eeeeee"},
            "bar":  {"color":"#eeeeee","thickness":0.15},
            "bgcolor":"#222831","bordercolor":"#444",
            "steps":[
                {"range":[0, 25],  "color":"#00adb5"},
                {"range":[25,45],  "color":"#00565b"},
                {"range":[45,55],  "color":"#444"},
                {"range":[55,65],  "color":"#8a3011"},
                {"range":[65,100], "color":"#ff5722"},
            ],
            "threshold":{"line":{"color":"white","width":4},
                         "thickness":0.8,"value":kfgi_now},
        }
    ))
    fig_gauge.update_layout(
        height=280, margin=dict(l=20,r=20,t=50,b=20),
        paper_bgcolor="rgba(0,0,0,0)", font={"color":"#eeeeee"},
    )
    st.plotly_chart(fig_gauge, use_container_width=True, key="gauge")

with col2:
    st.markdown("### 포지션 맞춤 가이드")
    if kfgi_now < 25:
        state_text, color = "🥶 극단적 공포 (Extreme Fear)", "#00adb5"
        buy_g  = "적극 매수 기회! 시장의 패닉을 역이용하세요."
        sell_g = "투매(패닉셀) 절대 금지! 바닥에서 파는 격입니다."
    elif kfgi_now < 45:
        state_text, color = "😨 공포 (Fear)", "#007a80"
        buy_g  = "저가 분할 매수 접근. 좋은 가격 구간입니다."
        sell_g = "손절 자제. 기술적 반등을 인내하세요."
    elif kfgi_now < 55:
        state_text, color = "😐 중립 (Neutral)", "#aaaaaa"
        buy_g  = "시장 방향 확인 후 유연한 진입."
        sell_g = "추세 이탈 전까지 홀딩."
    elif kfgi_now < 65:
        state_text, color = "😏 탐욕 (Greed)", "#c74318"
        buy_g  = "신규 진입 비중 축소. 상승 여력 제한적."
        sell_g = "분할 익절 시작, 현금 비중 확대."
    else:
        state_text, color = "🔥 극단적 탐욕 (Extreme Greed)", "#ff5722"
        buy_g  = "추격 매수(FOMO) 절대 금지! 시장 과열 징후."
        sell_g = "적극 익절! 이익 실현 및 현금화."

    st.info(f"**현재 상태:** {state_text}\n\n**추세:** {trend_now}")
    guide = buy_g if "매수" in position_choice else sell_g
    label = "매수 타이밍 진단" if "매수" in position_choice else "매도/홀딩 진단"
    st.markdown(f"""
    <div style="background:#2b303a;padding:15px;border-left:5px solid {color};border-radius:5px;margin-top:10px;">
        <h4 style="margin-top:0;color:{color};">{label}</h4>
        <p style="font-size:1.1em;color:#eeeeee;margin-bottom:0;">👉 <b>{guide}</b></p>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("### 시스템 권장 비중")
    regime_cfg = {
        "bull":   ("🐂 Bull 국면 (×1.2)", "#00adb5"),
        "normal": ("📊 Normal 국면 (×1.0)", "#aaaaaa"),
        "crisis": ("⚠️ Crisis 국면 (×0.7)", "#ff5722"),
    }
    rlabel, rcolor = regime_cfg.get(regime_now, ("📊 Normal", "#aaaaaa"))
    st.markdown(f"""
    <div style="text-align:center;padding:8px;background:#393e46;border-radius:5px;margin-bottom:10px;">
        <span style="color:{rcolor};font-weight:bold;font-size:1.05em;">{rlabel}</span>
    </div>""", unsafe_allow_html=True)

    st.metric(
        label=f"{strat_name} 전략 목표 비중",
        value=f"{weight_now:.2f}x",
        delta=f"{weight_now - prev_weight:+.2f}x 변동",
    )

    vol_shock_on = latest["vol_shock"] > CFG["vol_shock_thr"]
    tail_loss_on = latest["cumret_3d"] < CFG["tail_loss_thr"]
    if vol_shock_on:
        st.warning("⚡ vol_shock 안전망 발동 중 (변동성 급등)")
    elif tail_loss_on:
        st.warning("🛑 tail_loss 안전망 발동 중 (연속 손실)")

    st.caption(
        "※ 1x=100% 투자, >1x=레버리지 / 최대 2.0x · 최소 0.1x\n"
        f"K-FGI 방향: {kdir}\n"
        "weight = trend × vt^0.5 × kfgi_mult × regime_mult"
    )

st.divider()

# ═══════════════════════════════════════════════
# 섹션 B: 퀀트 성과 지표
# ═══════════════════════════════════════════════
st.subheader(f"📊 {strat_name} 전략 백테스팅 성과")

m_strat = compute_metrics(fdf[strat_col])
m_other = compute_metrics(fdf[other_col])
m_bnh   = compute_metrics(fdf["bnh_ret"])

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "최종 자산 평가액",
    f"{int(fdf['strat_asset'].iloc[-1]):,} 원",
    f"연환산 {m_strat['ann_ret']*100:+.1f}%",
)
k2.metric(
    "샤프 비율",
    f"{m_strat['sharpe']:.3f}",
    f"코스피 {m_bnh['sharpe']:.3f} 대비",
)
k3.metric(
    "전략 MDD",
    f"{m_strat['mdd']*100:.1f}%",
    f"코스피 {m_bnh['mdd']*100:.1f}% 대비",
    delta_color="inverse",
)
k4.metric(
    "승률",
    f"{m_strat['win_rate']*100:.1f}%",
    f"코스피 {m_bnh['win_rate']*100:.1f}% 대비",
)

st.markdown("<br>", unsafe_allow_html=True)

compare_df = pd.DataFrame({
    "지표": ["연환산 수익률","연환산 변동성","샤프 비율",
              "소르티노 비율","칼마 비율","최대 낙폭(MDD)",
              "승률","최대 연속 손실"],
    strat_name: [
        f"{m_strat['ann_ret']*100:+.2f}%", f"{m_strat['ann_vol']*100:.2f}%",
        f"{m_strat['sharpe']:.3f}",         f"{m_strat['sortino']:.3f}",
        f"{m_strat['calmar']:.3f}",         f"{m_strat['mdd']*100:.2f}%",
        f"{m_strat['win_rate']*100:.1f}%",  f"{m_strat['max_consec_loss']}일",
    ],
    other_name: [
        f"{m_other['ann_ret']*100:+.2f}%", f"{m_other['ann_vol']*100:.2f}%",
        f"{m_other['sharpe']:.3f}",         f"{m_other['sortino']:.3f}",
        f"{m_other['calmar']:.3f}",         f"{m_other['mdd']*100:.2f}%",
        f"{m_other['win_rate']*100:.1f}%",  f"{m_other['max_consec_loss']}일",
    ],
    "코스피 B&H": [
        f"{m_bnh['ann_ret']*100:+.2f}%", f"{m_bnh['ann_vol']*100:.2f}%",
        f"{m_bnh['sharpe']:.3f}",          f"{m_bnh['sortino']:.3f}",
        f"{m_bnh['calmar']:.3f}",          f"{m_bnh['mdd']*100:.2f}%",
        f"{m_bnh['win_rate']*100:.1f}%",   f"{m_bnh['max_consec_loss']}일",
    ],
}).set_index("지표")

st.dataframe(compare_df, use_container_width=True)
st.divider()

# ═══════════════════════════════════════════════
# 섹션 C: 누적 자산 곡선 + 드로우다운
# ═══════════════════════════════════════════════
fig_eq = go.Figure()
peak_asset = initial_capital * fdf["cum_strat"].cummax()
fig_eq.add_trace(go.Scatter(
    x=fdf["date"], y=peak_asset, mode="lines",
    line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig_eq.add_trace(go.Scatter(
    x=fdf["date"], y=fdf["strat_asset"], mode="lines",
    name=f"<b>{strat_name} 전략</b>",
    line=dict(color="#00adb5", width=2.5),
    fill="tonexty", fillcolor="rgba(255,87,34,0.1)",
))
fig_eq.add_trace(go.Scatter(
    x=fdf["date"], y=fdf["bnh_asset"], mode="lines",
    name="KOSPI Buy&Hold", line=dict(color="#777", width=1.5, dash="dot"),
))
fig_eq.update_layout(
    title="<b>누적 자산 성장 추이</b> (음영=드로우다운 구간)",
    yaxis_title="자산 평가액 (원)", height=420, **COMMON,
)
st.plotly_chart(fig_eq, use_container_width=True, key="equity")

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=fdf["date"], y=fdf["dd_strat"]*100, mode="lines",
    name=f"{strat_name} 전략", line=dict(color="#00adb5", width=1.8),
    fill="tozeroy", fillcolor="rgba(0,173,181,0.12)",
))
fig_dd.add_trace(go.Scatter(
    x=fdf["date"], y=fdf["dd_bnh"]*100, mode="lines",
    name="KOSPI B&H", line=dict(color="#ff5722", width=1.2, dash="dot"),
))
dd_layout = COMMON.copy()
dd_layout.update(dict(
    title="<b>드로우다운 (Drawdown)</b>",
    yaxis=dict(title="드로우다운 (%)", ticksuffix="%", **COMMON["yaxis"]),
    height=280,
))
fig_dd.update_layout(**dd_layout)
st.plotly_chart(fig_dd, use_container_width=True, key="drawdown")

# ═══════════════════════════════════════════════
# 섹션 D: 롤링 60일 샤프 비율
# ═══════════════════════════════════════════════
win = 60
_rs = fdf[strat_col].rolling(win)
_rb = fdf["bnh_ret"].rolling(win)
fdf["rolling_sharpe_strat"] = (
    _rs.mean() / _rs.std().replace(0, np.nan) * np.sqrt(252)
).replace([np.inf, -np.inf], np.nan)
fdf["rolling_sharpe_bnh"] = (
    _rb.mean() / _rb.std().replace(0, np.nan) * np.sqrt(252)
).replace([np.inf, -np.inf], np.nan)

fig_sharpe = go.Figure()
fig_sharpe.add_hline(y=0, line_dash="solid", line_color="#444")
fig_sharpe.add_hline(
    y=1, line_dash="dot", line_color="#00adb5",
    annotation_text="Sharpe = 1", annotation_font_color="#00adb5",
)
fig_sharpe.add_trace(go.Scatter(
    x=fdf["date"], y=fdf["rolling_sharpe_strat"], mode="lines",
    name=f"{strat_name} (60일)", line=dict(color="#00adb5", width=2),
))
fig_sharpe.add_trace(go.Scatter(
    x=fdf["date"], y=fdf["rolling_sharpe_bnh"], mode="lines",
    name="KOSPI B&H (60일)", line=dict(color="#777", width=1.5, dash="dot"),
))
sh_layout = COMMON.copy()
sh_layout.update(dict(
    title="<b>롤링 60일 샤프 비율 (연환산)</b>",
    yaxis=dict(title="샤프 비율", **COMMON["yaxis"]),
    height=300,
))
fig_sharpe.update_layout(**sh_layout)
st.plotly_chart(fig_sharpe, use_container_width=True, key="rolling_sharpe")
st.divider()

# ═══════════════════════════════════════════════
# 섹션 E: K-FGI + KOSPI (이중축)
# ═══════════════════════════════════════════════
fig_kfgi = go.Figure()
fig_kfgi.add_trace(go.Scatter(
    x=fdf["date"], y=fdf["kospi_close"],
    name="KOSPI", line=dict(color="#aaa", width=1.5),
))
fig_kfgi.add_trace(go.Scatter(
    x=fdf["date"], y=fdf["K_FGI"], name="<b>K-FGI</b>", yaxis="y2",
    line=dict(color="#ff5722", width=2), opacity=0.85,
))
fig_kfgi.add_hrect(
    y0=0, y1=25, yref="y2", fillcolor="#00adb5", opacity=0.15, line_width=0,
    annotation_text="극단 공포 (<25)", annotation_font_color="#00adb5",
)
fig_kfgi.add_hrect(
    y0=65, y1=100, yref="y2", fillcolor="#ff5722", opacity=0.15, line_width=0,
    annotation_text="극단 탐욕 (>65)", annotation_font_color="#ff5722",
    annotation_position="top left",
)
kl = COMMON.copy()
kl.update(dict(
    title="<b>KOSPI 흐름과 K-FGI 심리 국면</b> (임계값 25 / 65)",
    yaxis=dict(title="KOSPI", **COMMON["yaxis"]),
    yaxis2=dict(
        title="K-FGI (0~100)", overlaying="y", side="right",
        range=[0,100], showgrid=False, tickfont=dict(color="#ff5722"),
    ),
    height=420,
))
fig_kfgi.update_layout(**kl)
st.plotly_chart(fig_kfgi, use_container_width=True, key="kfgi")

# ═══════════════════════════════════════════════
# 섹션 F: 포지션 비중 + EGARCH 변동성
# ═══════════════════════════════════════════════
fig_pos = go.Figure()
fig_pos.add_trace(go.Scatter(
    x=fdf["date"], y=fdf[weight_col], name="포지션 비중 (lag)",
    mode="lines", line=dict(color="#00adb5", width=1.5),
    fill="tozeroy", fillcolor="rgba(0,173,181,0.12)",
))
fig_pos.add_trace(go.Scatter(
    x=fdf["date"], y=fdf["egarch_vol"]*100, name="EGARCH σ (%)", yaxis="y2",
    line=dict(color="#F59E0B", width=1.2), opacity=0.7,
))
fig_pos.add_hline(y=1.0, line_dash="dash", line_color="#aaa",
                   annotation_text="기준 1.0x")
pl = COMMON.copy()
pl.update(dict(
    title="<b>포지션 비중 vs EGARCH 변동성</b> (vol^0.5 타겟팅)",
    yaxis=dict(title="포지션 (x)", **COMMON["yaxis"]),
    yaxis2=dict(
        title="EGARCH σ (%)", overlaying="y", side="right",
        showgrid=False, tickfont=dict(color="#F59E0B"),
    ),
    height=320,
))
fig_pos.update_layout(**pl)
st.plotly_chart(fig_pos, use_container_width=True, key="position")

# ═══════════════════════════════════════════════
# 섹션 G: 월별 수익률 히트맵
# ═══════════════════════════════════════════════
fig_heatmap = make_monthly_heatmap(fdf[strat_col], fdf["date"])
st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap")
st.divider()

# ═══════════════════════════════════════════════
# 섹션 H: 시장 국면 분포 + 규칙표
# ═══════════════════════════════════════════════
rc         = fdf["regime"].value_counts()
color_map  = {"bull":"#00adb5","normal":"#aaaaaa","crisis":"#ff5722"}
fig_pie    = go.Figure(go.Pie(
    labels=rc.index.tolist(), values=rc.values.tolist(),
    marker_colors=[color_map.get(r,"#aaaaaa") for r in rc.index],
    hole=0.45, textinfo="label+percent",
))
fig_pie.update_layout(
    title="<b>시장 국면 분포</b>",
    paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#eeeeee"),
    height=300, showlegend=False,
)
c_left, c_right = st.columns([1, 2])
with c_left:
    st.plotly_chart(fig_pie, use_container_width=True, key="pie")
with c_right:
    st.markdown("### 국면별 포지션 규칙")
    st.markdown(f"""
| 국면 | 조건 | Regime 배율 |
|------|------|-------------|
| 🐂 Bull | K-FGI > 65 + 변동성 안정 + trend ≥ 2 | × 1.2 |
| 📊 Normal | 그 외 | × 1.0 |
| ⚠️ Crisis | K-FGI < 25 또는 vol_shock > 2.0 | × 0.7 |

**안전망 (즉시 포지션 0)**
- `vol_shock_guard` : 전일 대비 변동성 > 2.0배
- `tail_loss_guard` : 3일 누적 수익 < −7%

**포지션 산출식 ({strat_name})**
`weight = trend_weight × vol_target^0.5 × kfgi_mult × regime_mult`
`kfgi_mult (스나이퍼) = 0.5 + (K_FGI/100) × 1.1  →  [0.5, 1.6]`
`kfgi_mult (불도저)   = 1.6 − (K_FGI/100) × 1.1  →  [0.5, 1.6]`
K-FGI 방향: **{kdir}**
    """)

st.divider()

# ═══════════════════════════════════════════════
# 섹션 I: 최근 30일 시그널 로그
# ═══════════════════════════════════════════════
st.subheader("🔍 최근 30일 시그널 로그")

_sig = fdf.tail(30)[[
    "date", "K_FGI", "regime", weight_col,
    strat_col, "vol_shock", "cumret_3d", "trend_strength",
]].copy()
_sig.columns = ["날짜","K-FGI","국면","포지션(x)","일간수익률(%)","vol_shock","3일누적(%)","추세강도"]
_sig["날짜"]          = _sig["날짜"].dt.strftime("%Y-%m-%d")
_sig["K-FGI"]         = _sig["K-FGI"].round(1)
_sig["포지션(x)"]     = _sig["포지션(x)"].round(3)
_sig["일간수익률(%)"] = (_sig["일간수익률(%)"] * 100).round(3)
_sig["vol_shock"]     = _sig["vol_shock"].round(3)
_sig["3일누적(%)"]    = (_sig["3일누적(%)"] * 100).round(2)
_sig["추세강도"]      = _sig["추세강도"].astype(int)
_sig["국면"]          = _sig["국면"].map(
    {"bull":"🐂 bull","normal":"📊 normal","crisis":"⚠️ crisis"}
)
st.dataframe(_sig.set_index("날짜"), use_container_width=True, height=400)

# ═══════════════════════════════════════════════
# 섹션 J: 백테스팅 결과 CSV 다운로드
# ═══════════════════════════════════════════════
st.divider()
st.subheader("💾 백테스팅 결과 다운로드")

export_cols = {
    "date":        "날짜",
    "K_FGI":       "K-FGI",
    "regime":      "국면",
    weight_col:    "포지션(x)",
    strat_col:     f"{strat_name}_수익률",
    "bnh_ret":     "코스피_수익률",
    "cum_strat":   f"{strat_name}_누적수익",
    "cum_bnh":     "코스피_누적수익",
    "strat_asset": f"{strat_name}_자산평가액(원)",
    "bnh_asset":   "코스피_자산평가액(원)",
    "dd_strat":    f"{strat_name}_드로우다운",
    "vol_shock":   "vol_shock",
    "cumret_3d":   "3일누적수익",
    "egarch_vol":  "EGARCH_변동성",
    "trend_strength": "추세강도",
}
export_df = fdf[list(export_cols.keys())].rename(columns=export_cols).copy()
st.download_button(
    label=f"📥  {strat_name} 전략 결과 CSV 다운로드",
    data=export_df.to_csv(index=False, encoding="utf-8-sig"),
    file_name=f"kfgi_{strat_name}_{start_date}_{end_date}.csv",
    mime="text/csv",
    use_container_width=True,
)
