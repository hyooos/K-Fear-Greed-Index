import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from arch import arch_model

    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


st.set_page_config(page_title="K-FGI Dashboard", layout="wide", page_icon="📊")

st.markdown(
    """
<style>
    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"],
    [data-testid="stMain"], section[data-testid="stSidebar"],
    .stMainBlockContainer, .main .block-container {
        background-color: #0e1117 !important;
        color: #eeeeee !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        background-color: #161b22 !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, div, .stMarkdown {
        color: #eeeeee !important;
    }
    [data-testid="stMetricValue"] {
        color: #00adb5 !important;
        font-weight: 700 !important;
    }
    .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
        background-color: #00adb5 !important;
        color: #0e1117 !important;
        border: none !important;
        font-weight: 700 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

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
TREND_MAP = {0: 0.00, 1: 0.35, 2: 0.70, 3: 1.00}
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KFG_final_2.csv")

COMMON_LAYOUT = dict(
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#eeeeee"),
    xaxis=dict(showgrid=True, gridcolor="#2a2a2a", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#2a2a2a", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#444", borderwidth=1),
    margin=dict(l=55, r=40, t=55, b=40),
)

KFGI_FEATS = [
    *[f"sub_index{i}" for i in range(1, 8)],
    "sent_norm_w",
    "sent_energy",
    "sent_std_inv",
    "neg_z_inv",
    "sent_composite",
    "sent_composite_ma10",
    "egarch_vol",
    "vol_regime_high",
    "vol_ratio",
]
DIRECTION = {
    **{f"sub_index{i}": 1 for i in range(1, 8)},
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


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def fit_egarch_walkforward(log_returns: pd.Series, min_obs: int, retrain_every: int = 21, progress_bar=None) -> pd.Series:
    n = len(log_returns)
    sigma = np.full(n, np.nan)
    retrain_steps = list(range(min_obs, n, retrain_every))
    total = max(len(retrain_steps), 1)

    for step_i, t0 in enumerate(retrain_steps):
        t_end = min(t0 + retrain_every, n)
        train = log_returns.iloc[:t0].dropna()
        if progress_bar is not None:
            progress_bar.progress(
                min(int(step_i / total * 100), 99),
                text=f"EGARCH walk-forward {step_i + 1}/{total} ({t0}/{n})",
            )

        if not HAS_ARCH:
            sigma[t0:t_end] = train.iloc[-20:].std() if len(train) >= 20 else np.nan
            continue

        try:
            am = arch_model(train * 100, vol="EGARCH", p=1, q=1, dist="skewt", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=t_end - t0, reindex=False)
            sigma[t0:t_end] = np.sqrt(fc.variance.values[-1]) / 100
        except Exception:
            sigma[t0:t_end] = train.iloc[-20:].std() if len(train) >= 20 else np.nan

    if progress_bar is not None:
        progress_bar.progress(100, text="EGARCH 완료")
    return pd.Series(sigma, index=log_returns.index, name="egarch_vol")


def safe_scale(raw_scores: np.ndarray) -> np.ndarray:
    valid = raw_scores[~np.isnan(raw_scores)]
    out = np.full(len(raw_scores), np.nan)
    if len(valid) == 0:
        return np.full(len(raw_scores), 50.0)
    if len(valid) == 1 or np.allclose(valid, valid[0]):
        out[~np.isnan(raw_scores)] = 50.0
        return out
    p1, p99 = np.percentile(valid, [1, 99])
    denom = p99 - p1
    if abs(denom) < 1e-12:
        out[~np.isnan(raw_scores)] = 50.0
        return out
    return 100 * (np.clip(raw_scores, p1, p99) - p1) / denom


def perf_metrics(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if len(ret) == 0:
        return {}
    ann_ret = ret.mean() * 252
    ann_vol = ret.std() * np.sqrt(252) + 1e-9
    cum = np.exp(ret.cumsum())
    mdd = (cum / cum.cummax() - 1).min()
    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol,
        "mdd": mdd,
        "calmar": ann_ret / (abs(mdd) + 1e-9),
        "total": float(np.exp(ret.sum()) - 1),
    }


def make_position(d: pd.DataFrame, use_kfgi: bool = True, use_vol: bool = True) -> pd.Series:
    d = d.copy()
    tw = d["trend_strength"].map(TREND_MAP).fillna(0.0)

    if use_vol:
        raw_vt = TARGET_DAILY_VOL / (d["egarch_vol"] + 1e-9)
        vt = np.sqrt(raw_vt).clip(CFG["min_leverage"], CFG["max_leverage"])
    else:
        vt = pd.Series(1.0, index=d.index)

    if use_kfgi and "K_FGI" in d.columns:
        kfgi_mult = (0.5 + (d["K_FGI"] / 100.0) * 1.1).clip(0.5, 1.6)
        regime_mult = d["regime"].map({"bull": 1.2, "normal": 1.0, "crisis": 0.7}).fillna(1.0)
    else:
        kfgi_mult = pd.Series(1.0, index=d.index)
        regime_mult = pd.Series(1.0, index=d.index)

    w = (tw * vt * kfgi_mult * regime_mult).clip(0.0, CFG["max_leverage"] * 1.5)
    if use_vol:
        w[d["vol_shock"] > CFG["vol_shock_thr"]] = 0.0
        cumret3 = d["target_reg"].rolling(3).sum().shift(1).fillna(0)
        w[cumret3 < CFG["tail_loss_thr"]] = 0.0

    w_lag = w.shift(1).fillna(0)
    turnover = w_lag.diff().abs().fillna(w_lag.iloc[0])
    return w_lag * d["target_reg"] - turnover * CFG["fee"]


@st.cache_data(show_spinner=False)
def build_pipeline() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    log_ret = np.log(df["kospi_close"] / df["kospi_close"].shift(1)).fillna(0)
    df["egarch_vol"] = fit_egarch_walkforward(log_ret, CFG["egarch_min_obs"])
    df["vol_shock"] = df["egarch_vol"].pct_change().clip(-5, 5)
    df["vol_regime_high"] = (df["egarch_vol"] > df["egarch_vol"].rolling(60).quantile(0.7)).astype(float)
    df["vol_ratio"] = df["egarch_vol"] / (df["egarch_vol"].rolling(60).mean() + 1e-9)

    df["neg_z_inv"] = -df["neg_z"]
    df["sent_std_inv"] = -df["sent_std"]
    df["sent_energy"] = df["sent_strength_w"] * df["sent_norm_w"]
    df["sent_composite"] = df["sent_norm_w"] * 0.4 + df["neg_z_inv"] * 0.3 + df["sent_energy"] * 0.3
    df["sent_composite_ma10"] = df["sent_composite"].rolling(10).mean()

    price = df["kospi_close"]
    df["ma20"] = price.rolling(20).mean()
    df["ma60"] = price.rolling(60).mean()
    df["ma120"] = price.rolling(120).mean()
    df["above_ma20"] = (price > df["ma20"]).astype(float)
    df["above_ma60"] = (price > df["ma60"]).astype(float)
    df["above_ma120"] = (price > df["ma120"]).astype(float)
    df["trend_strength"] = df["above_ma20"] + df["above_ma60"] + df["above_ma120"]
    df["market_trend"] = (price > df["ma20"]).astype(int)
    df["rsi14"] = compute_rsi(price, 14)

    df["target_reg"] = df["log_return_t+1"]
    df["target_5d"] = df["log_return_t+1"].rolling(5).sum().shift(-4)

    lag_cols = [c for c in df.columns if c not in ["date", "target_reg", "target_5d"]]
    df[lag_cols] = df[lag_cols].shift(1)
    df = df.dropna(subset=["target_reg", "egarch_vol"]).reset_index(drop=True)

    feats = [f for f in KFGI_FEATS if f in df.columns]
    dir_vec = np.array([DIRECTION.get(f, 1) for f in feats])
    raw_scores = np.full(len(df), np.nan)

    for t in range(CFG["kfgi_min_obs"], len(df)):
        X_tr = df[feats].iloc[:t].values
        y_tr = df["target_5d"].iloc[:t].values
        mask = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr)
        X_tr, y_tr = X_tr[mask], y_tr[mask]
        if len(X_tr) < 30:
            continue
        try:
            sc = StandardScaler()
            X_s = np.clip(sc.fit_transform(X_tr), -3, 3)
            ridge = RidgeCV(alphas=np.logspace(-2, 2, 10))
            ridge.fit(X_s, y_tr)
            coef = ridge.coef_ * dir_vec
            w = coef / (np.sum(np.abs(coef)) + 1e-12)
            x_t = df[feats].iloc[t:t + 1].values
            if np.isnan(x_t).any():
                continue
            raw_scores[t] = float(np.clip(sc.transform(x_t), -3, 3) @ w)
        except Exception:
            continue

    df["K_FGI"] = safe_scale(raw_scores)
    df = df.dropna(subset=["K_FGI"]).reset_index(drop=True)

    rho_5d = spearmanr(df["K_FGI"], df["target_5d"].fillna(0)).statistic
    df.attrs["rho_5d"] = 0.0 if pd.isna(rho_5d) else float(rho_5d)
    df.attrs["kfgi_momentum"] = bool((0.0 if pd.isna(rho_5d) else rho_5d) > 0)

    def classify_regime(row):
        if row["K_FGI"] > 65 and row["vol_regime_high"] == 0 and row["trend_strength"] >= 2:
            return "bull"
        if row["K_FGI"] < 25 or row["vol_shock"] > CFG["vol_shock_thr"]:
            return "crisis"
        return "normal"

    df["regime"] = df.apply(classify_regime, axis=1)
    df["ret_bnh"] = df["target_reg"]
    df["ret_trend"] = make_position(df, use_kfgi=False, use_vol=False)
    df["ret_trend_vol"] = make_position(df, use_kfgi=False, use_vol=True)
    df["ret_main"] = make_position(df, use_kfgi=True, use_vol=True)

    df["weight_main"] = (
        df["trend_strength"].map(TREND_MAP).fillna(0)
        * np.sqrt(TARGET_DAILY_VOL / (df["egarch_vol"] + 1e-9)).clip(CFG["min_leverage"], CFG["max_leverage"])
        * (0.5 + (df["K_FGI"] / 100.0) * 1.1).clip(0.5, 1.6)
        * df["regime"].map({"bull": 1.2, "normal": 1.0, "crisis": 0.7}).fillna(1.0)
    ).clip(0.0, CFG["max_leverage"] * 1.5).shift(1).fillna(0)

    df["cumret_3d"] = df["target_reg"].rolling(3).sum().shift(1).fillna(0)
    df.loc[df["vol_shock"] > CFG["vol_shock_thr"], "weight_main"] = 0.0
    df.loc[df["cumret_3d"] < CFG["tail_loss_thr"], "weight_main"] = 0.0
    return df


if "started" not in st.session_state:
    st.session_state.started = False
if "initial_capital" not in st.session_state:
    st.session_state.initial_capital = 10_000_000


def start_analysis():
    st.session_state.started = True


st.sidebar.title("K-FGI 대시보드")
st.sidebar.markdown("감성 및 변동성 결합 지표를 활용한 리스크 관리 전략")
st.sidebar.divider()

capital_map = {
    "100만 원": 1_000_000,
    "500만 원": 5_000_000,
    "1,000만 원": 10_000_000,
    "5,000만 원": 50_000_000,
    "1억 원": 100_000_000,
    "5억 원": 500_000_000,
}

with st.sidebar.form("cfg_form"):
    labels = list(capital_map.keys())
    current_label = next((k for k, v in capital_map.items() if v == st.session_state.initial_capital), "1,000만 원")
    cap_label = st.selectbox("초기 투자금액", labels, index=labels.index(current_label))
    submitted = st.form_submit_button("분석 시작", on_click=start_analysis, width="stretch")

if submitted:
    st.session_state.initial_capital = capital_map[cap_label]
    st.rerun()

initial_capital = st.session_state.initial_capital

if not os.path.exists(CSV_PATH):
    st.error(f"`KFG_final_2.csv` 파일을 찾을 수 없습니다: `{CSV_PATH}`")
    st.stop()

if not st.session_state.started:
    st.title("📊 K-FGI × EGARCH 전략 대시보드")
    st.markdown(
        """
**감성 및 변동성 결합 지표(K-FGI)를 활용한 주식시장 리스크 관리 전략 연구**

왼쪽 사이드바에서 투자금액을 설정하고 **분석 시작** 버튼을 눌러주세요.

---
| 전략 | 설명 |
|------|------|
| Buy & Hold | KOSPI 단순 보유 |
| ① 추세 추종 | MA 이동평균 기반 순수 추세 |
| ② 추세 + EGARCH | 추세 + 변동성 타겟팅(^0.5) |
| ③ K-FGI 전략 | 추세 + EGARCH + K-FGI + 시장국면 |
"""
    )
    st.stop()

with st.spinner("EGARCH walk-forward + K-FGI 파이프라인을 계산 중입니다. 잠시만 기다려주세요."):
    df = build_pipeline()

rho_5d = df.attrs.get("rho_5d", 0.0)
kdir_str = "순방향(모멘텀)" if df.attrs.get("kfgi_momentum", True) else "역발상(공포→매수)"

with st.sidebar:
    st.divider()
    min_d, max_d = df["date"].min().date(), df["date"].max().date()
    st.markdown(
        f"**데이터:** `{df['date'].min().strftime('%Y.%m')}` ~ `{df['date'].max().strftime('%Y.%m')}`  \n"
        f"K-FGI 현재: **{df['K_FGI'].iloc[-1]:.1f}**  \n"
        f"K-FGI 방향: **{kdir_str}**  \n"
        f"ρ(5d) = **{rho_5d:.4f}**"
    )
    start_d, end_d = st.slider("백테스팅 기간", min_value=min_d, max_value=max_d, value=(min_d, max_d))

mask = (df["date"].dt.date >= start_d) & (df["date"].dt.date <= end_d)
fdf = df.loc[mask].copy().reset_index(drop=True)
if fdf.empty:
    st.warning("선택한 기간에 데이터가 없습니다.")
    st.stop()

for col in ["ret_bnh", "ret_trend", "ret_trend_vol", "ret_main"]:
    fdf[f"cum_{col}"] = fdf[col].cumsum().apply(np.exp)

fdf["strat_asset"] = initial_capital * fdf["cum_ret_main"]
fdf["bnh_asset"] = initial_capital * fdf["cum_ret_bnh"]
fdf["dd_main"] = fdf["cum_ret_main"] / fdf["cum_ret_main"].cummax() - 1

st.title("📊 K-FGI × EGARCH 전략 대시보드")
st.caption(f"분석 기간: {start_d} ~ {end_d}")
if not HAS_ARCH:
    st.warning("`arch` 패키지가 없어 EGARCH 대신 rolling std fallback이 사용됩니다.")

latest = fdf.iloc[-1]
m_main = perf_metrics(fdf["ret_main"])
m_bnh = perf_metrics(fdf["ret_bnh"])

st.subheader("🧭 Fear & Greed Gauge")

kfgi_now = float(latest["K_FGI"])
if kfgi_now < 25:
    state_lbl, state_desc, state_color = "Extreme Fear", "패닉 구간입니다. 심리가 크게 위축돼 있습니다.", "#1976d2"
elif kfgi_now < 45:
    state_lbl, state_desc, state_color = "Fear", "보수적 심리가 우세한 구간입니다.", "#1565c0"
elif kfgi_now < 55:
    state_lbl, state_desc, state_color = "Neutral", "심리가 중립에 가까운 균형 구간입니다.", "#78909c"
elif kfgi_now < 75:
    state_lbl, state_desc, state_color = "Greed", "위험 선호가 강해지는 구간입니다.", "#e64a19"
else:
    state_lbl, state_desc, state_color = "Extreme Greed", "과열 신호가 강한 구간입니다.", "#c62828"

g1, g2 = st.columns([1.15, 0.85])

with g1:
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=kfgi_now,
            title={"text": "<b>K-FGI Fear & Greed</b>", "font": {"color": "#eeeeee", "size": 20}},
            number={"font": {"color": "#eeeeee", "size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#aaa"},
                "bar": {"color": "#f5f5f5", "thickness": 0.18},
                "bgcolor": "#1a1a2e",
                "bordercolor": "#444",
                "steps": [
                    {"range": [0, 25], "color": "#1565c0"},
                    {"range": [25, 45], "color": "#1976d2"},
                    {"range": [45, 55], "color": "#37474f"},
                    {"range": [55, 75], "color": "#bf360c"},
                    {"range": [75, 100], "color": "#d32f2f"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.8,
                    "value": kfgi_now,
                },
            },
        )
    )
    fig_gauge.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eeeeee"),
    )
    st.plotly_chart(fig_gauge, width="stretch")

with g2:
    st.markdown("#### 현재 심리 상태")
    st.markdown(
        f"""
<div style="background:#1e2a3a;padding:16px;border-left:4px solid {state_color};
            border-radius:8px;margin-bottom:14px;">
  <div style="font-size:1.1rem;font-weight:700;color:{state_color};">{state_lbl}</div>
  <div style="color:#dddddd;margin-top:6px;">{state_desc}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.metric("권장 포지션", f"{latest['weight_main']:.2f}x", latest["regime"])
    st.metric("K-FGI 방향", kdir_str)
    st.metric("ρ(5d)", f"{rho_5d:.4f}")

c1, c2 = st.columns(2)
c1.metric("연환산 수익률", f"{m_main['ann_ret'] * 100:+.2f}%", f"B&H {m_bnh['ann_ret'] * 100:+.2f}%")
c2.metric("MDD", f"{m_main['mdd'] * 100:.2f}%", f"B&H {m_bnh['mdd'] * 100:.2f}%")

fig_eq = go.Figure()
for col, lbl, clr, dash in [
    ("cum_ret_bnh", "B&H", "#9CA3AF", "dot"),
    ("cum_ret_trend", "추세", "#16A34A", "dashdot"),
    ("cum_ret_trend_vol", "추세+EGARCH", "#F59E0B", "dash"),
    ("cum_ret_main", "K-FGI 전략", "#2563EB", "solid"),
]:
    fig_eq.add_trace(go.Scatter(x=fdf["date"], y=fdf[col], name=lbl, line=dict(color=clr, width=2, dash=dash)))
fig_eq.update_layout(title="<b>누적 수익률</b>", yaxis_title="누적 배수", height=380, **COMMON_LAYOUT)
st.plotly_chart(fig_eq, width="stretch")

col1, col2 = st.columns(2)

with col1:
    fig_kfgi = go.Figure()
    fig_kfgi.add_trace(go.Scatter(x=fdf["date"], y=fdf["kospi_close"], name="KOSPI", line=dict(color="#aaaaaa", width=1.5)))
    fig_kfgi.add_trace(go.Scatter(x=fdf["date"], y=fdf["K_FGI"], name="K-FGI", yaxis="y2", line=dict(color="#ff5722", width=2)))
    kfgi_layout = {k: v for k, v in COMMON_LAYOUT.items() if k != "yaxis"}
    fig_kfgi.update_layout(
        **kfgi_layout,
        title="<b>KOSPI vs K-FGI</b>",
        yaxis={"title": "KOSPI", "showgrid": True, "gridcolor": "#2a2a2a"},
        yaxis2={"title": "K-FGI", "overlaying": "y", "side": "right", "range": [0, 100], "showgrid": False},
        height=360,
    )
    st.plotly_chart(fig_kfgi, width="stretch")

with col2:
    fig_pos = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08)
    fig_pos.add_trace(go.Scatter(x=fdf["date"], y=fdf["egarch_vol"] * 100, name="EGARCH σ (%)", line=dict(color="#F59E0B", width=1.5)), row=1, col=1)
    fig_pos.add_trace(go.Scatter(x=fdf["date"], y=fdf["weight_main"], name="포지션", line=dict(color="#2563EB", width=1.5)), row=2, col=1)
    fig_pos.update_layout(
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eeeeee"),
        hovermode="x unified",
        title="<b>변동성 vs 포지션</b>",
        margin=dict(l=55, r=30, t=45, b=40),
    )
    fig_pos.update_yaxes(title_text="σ (%)", row=1, col=1, showgrid=True, gridcolor="#2a2a2a")
    fig_pos.update_yaxes(title_text="포지션", row=2, col=1, showgrid=True, gridcolor="#2a2a2a")
    st.plotly_chart(fig_pos, width="stretch")

st.subheader("📅 월별 수익률 히트맵 (K-FGI 전략)")
monthly = (
    pd.DataFrame({"r": fdf["ret_main"].values, "d": pd.to_datetime(fdf["date"].values)})
    .set_index("d")["r"]
    .resample("ME")
    .apply(lambda x: np.exp(x.sum()) - 1)
)
pivot = (
    monthly.reset_index()
    .assign(year=lambda d: d["d"].dt.year, month=lambda d: d["d"].dt.month)
    .pivot(index="year", columns="month", values="r")
)
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
heat_values = pivot.values * 100
heat_text = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in heat_values]

fig_hm = go.Figure(
    go.Heatmap(
        z=heat_values,
        x=[month_names[c - 1] for c in pivot.columns],
        y=[str(y) for y in pivot.index],
        text=heat_text,
        texttemplate="%{text}",
        colorscale=[
            [0.0, "#1565c0"],
            [0.35, "#4fc3f7"],
            [0.5, "#37474f"],
            [0.65, "#ff8a65"],
            [1.0, "#c62828"],
        ],
        zmid=0,
        colorbar=dict(title="수익률(%)"),
    )
)
fig_hm.update_layout(
    title="<b>월별 수익률 히트맵</b>",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#eeeeee"),
    xaxis=dict(side="top"),
    height=max(240, 60 * len(pivot) + 100),
    margin=dict(l=60, r=20, t=80, b=20),
)
st.plotly_chart(fig_hm, width="stretch")

st.subheader("최근 30일 시그널 로그")
sig = fdf.tail(30)[["date", "K_FGI", "regime", "weight_main", "ret_main", "vol_shock", "trend_strength", "egarch_vol"]].copy()
sig["date"] = sig["date"].dt.strftime("%Y-%m-%d")
sig["K_FGI"] = sig["K_FGI"].round(1)
sig["weight_main"] = sig["weight_main"].round(3)
sig["ret_main"] = (sig["ret_main"] * 100).round(3)
sig["vol_shock"] = sig["vol_shock"].round(3)
sig["egarch_vol"] = (sig["egarch_vol"] * 100).round(3)
st.dataframe(sig.rename(columns={
    "date": "날짜",
    "K_FGI": "K-FGI",
    "regime": "국면",
    "weight_main": "포지션(×)",
    "ret_main": "일간 수익률(%)",
    "vol_shock": "vol_shock",
    "trend_strength": "추세강도",
    "egarch_vol": "EGARCH σ",
}).set_index("날짜"), width="stretch", height=360)

exp_df = fdf[["date", "K_FGI", "regime", "weight_main", "ret_bnh", "ret_trend", "ret_trend_vol", "ret_main", "egarch_vol", "vol_shock", "trend_strength"]].copy()
st.download_button(
    label="📥 전략 결과 CSV 다운로드",
    data=exp_df.to_csv(index=False, encoding="utf-8-sig"),
    file_name=f"kfgi_result_{start_d}_{end_d}.csv",
    mime="text/csv",
    width="stretch",
)
