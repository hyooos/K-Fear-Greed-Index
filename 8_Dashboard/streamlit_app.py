from pathlib import Path
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="K-Fear&Greed Index",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).resolve().parents[1]
OUTPUTS = BASE / "paper_outputs"
DATA = OUTPUTS / "data"
TABLES = OUTPUTS / "tables"

STRATEGIES = {
    "main_kfgi_sentiment": "K-FGI 전략",
    "buy_hold": "KOSPI200 보유",
    "trend_only": "Trend",
    "trend_egarch_vol": "Trend + EGARCH",
}

@st.cache_data
def load_data():
    kfgi = pd.read_csv(DATA / "kfgi_10y_timeseries.csv", parse_dates=["date"])
    ret = pd.read_csv(DATA / "strategy_returns_10y.csv", parse_dates=["date"])
    perf = pd.read_csv(TABLES / "performance_summary.csv")
    return kfgi, ret, perf


def pct(value):
    return f"{value * 100:.1f}%"


def krw(value):
    if value >= 100_000_000:
        return f"{value / 100_000_000:.2f}억원"
    if value >= 10_000:
        return f"{value / 10_000:.0f}만원"
    return f"{value:,.0f}원"


def parse_money(text):
    cleaned = str(text).replace(",", "").replace(" ", "")
    if not cleaned.isdigit():
        return None
    return int(cleaned)


def format_money_input(key):
    value = parse_money(st.session_state.get(key, ""))
    if value is not None:
        st.session_state[key] = f"{value:,}"


def money_stepper(label, key, default, step, min_value, max_value):
    st.markdown(f"<div class='money-label'>{label}</div>", unsafe_allow_html=True)
    if key not in st.session_state:
        st.session_state[key] = f"{default:,}"

    minus_col, input_col, plus_col = st.columns([0.18, 0.64, 0.18], gap="small")
    current = parse_money(st.session_state[key])
    current = default if current is None else current

    with minus_col:
        if st.button("-", key=f"{key}_minus", use_container_width=True):
            st.session_state[key] = f"{max(min_value, current - step):,}"
    with plus_col:
        if st.button("+", key=f"{key}_plus", use_container_width=True):
            st.session_state[key] = f"{min(max_value, current + step):,}"
    with input_col:
        value_text = st.text_input(
            label,
            key=key,
            label_visibility="collapsed",
            on_change=format_money_input,
            args=(key,),
        )

    return value_text


def simulate_account(ret_view, strategy, initial_capital, monthly_contribution):
    returns = ret_view[strategy].fillna(0)
    dates = ret_view["date"]
    balance = float(initial_capital)
    contributed = float(initial_capital)
    values = []
    contributions = []
    last_month = None

    for date, log_return in zip(dates, returns):
        month = date.to_period("M")
        if last_month is not None and month != last_month and monthly_contribution:
            balance += monthly_contribution
            contributed += monthly_contribution
        balance *= float(np.exp(log_return))
        values.append(balance)
        contributions.append(contributed)
        last_month = month

    nav = pd.Series(values, index=ret_view.index)
    contributed_series = pd.Series(contributions, index=ret_view.index)
    return nav, contributed_series


def calc_stats(ret_view, strategy, initial_capital, monthly_contribution):
    returns = ret_view[strategy].dropna()
    nav, contributed_series = simulate_account(ret_view, strategy, initial_capital, monthly_contribution)
    contributed = contributed_series.iloc[-1] if not contributed_series.empty else initial_capital
    if returns.empty:
        return {
            "nav": nav,
            "contributed": contributed,
            "final_value": initial_capital,
            "profit": 0.0,
            "total_return": 0.0,
            "ann_ret": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "mdd": 0.0,
        }
    running_max = nav.cummax()
    drawdown = nav / running_max - 1
    years = max(len(returns) / 252, 1 / 252)
    time_weighted_return = np.exp(returns.sum()) - 1
    ann_ret = (1 + time_weighted_return) ** (1 / years) - 1
    ann_vol = returns.std(ddof=0) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol else 0.0
    return {
        "nav": nav,
        "contributed": contributed,
        "final_value": nav.iloc[-1],
        "profit": nav.iloc[-1] - contributed,
        "total_return": nav.iloc[-1] / contributed - 1,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "mdd": drawdown.min(),
    }


def fgi_state(value):
    if value < 25:
        return "극단적 공포"
    if value < 45:
        return "공포"
    if value < 55:
        return "중립"
    if value < 75:
        return "탐욕"
    return "극단적 탐욕"


def fgi_state_detail(value):
    if value < 25:
        return "위험 회피 심리가 강한 구간"
    if value < 45:
        return "방어적 포지션을 우선하는 구간"
    if value < 55:
        return "방향성이 뚜렷하지 않은 중립 구간"
    if value < 75:
        return "위험 선호가 강화되는 구간"
    return "과열 가능성을 점검할 구간"


def gauge_color(value):
    if value < 25:
        return "#1684fc"
    if value < 45:
        return "#21a8d8"
    if value < 65:
        return "#2fb47c"
    if value < 75:
        return "#ff9f1c"
    return "#ff3b30"


def state_chip_class(value):
    if value < 25:
        return "chip extreme-fear"
    if value < 45:
        return "chip fear"
    if value < 55:
        return "chip neutral"
    if value < 75:
        return "chip greed"
    return "chip extreme-greed"


def nearest_row_before(df, target_date):
    rows = df[df["date"] <= target_date]
    if rows.empty:
        return df.iloc[0]
    return rows.iloc[-1]


def comparison_rows(kfgi_view):
    latest_date = kfgi_view["date"].max()
    points = [
        ("전일", nearest_row_before(kfgi_view.iloc[:-1], latest_date) if len(kfgi_view) > 1 else kfgi_view.iloc[-1]),
        ("1주 전", nearest_row_before(kfgi_view, latest_date - pd.Timedelta(days=7))),
        ("1개월 전", nearest_row_before(kfgi_view, latest_date - pd.DateOffset(months=1))),
        ("1년 전", nearest_row_before(kfgi_view, latest_date - pd.DateOffset(years=1))),
    ]
    return points


def gauge_point(cx, cy, radius, value):
    angle = math.radians(180 - (value / 100) * 180)
    return cx + radius * math.cos(angle), cy - radius * math.sin(angle)


def arc_line_path(cx, cy, radius, start_value, end_value):
    x1, y1 = gauge_point(cx, cy, radius, start_value)
    x2, y2 = gauge_point(cx, cy, radius, end_value)
    large_arc = 1 if end_value - start_value > 50 else 0
    return f"M {x1:.2f} {y1:.2f} A {radius} {radius} 0 {large_arc} 1 {x2:.2f} {y2:.2f}"


def needle_polygon(cx, cy, tip_radius, value):
    angle = math.radians(180 - (value / 100) * 180)
    ux, uy = math.cos(angle), -math.sin(angle)
    px, py = -uy, ux
    tip_x, tip_y = cx + tip_radius * ux, cy + tip_radius * uy
    base_x, base_y = cx - 4 * ux, cy - 4 * uy
    left_x, left_y = base_x + 8 * px, base_y + 8 * py
    right_x, right_y = base_x - 8 * px, base_y - 8 * py
    return f"{tip_x:.2f},{tip_y:.2f} {left_x:.2f},{left_y:.2f} {right_x:.2f},{right_y:.2f}"


def make_gauge_svg(value):
    cx, cy = 360, 258
    radius = 220
    needle_color = "#111827"
    pointer = needle_polygon(cx, cy, 178, value)
    segments = [
        (0, 25, "#d9ebff"),
        (25, 45, "#d7f3fb"),
        (45, 55, "#e7f8ef"),
        (55, 75, "#fff0cf"),
        (75, 100, "#ffe0de"),
    ]
    paths = "\n".join(
        f'<path d="{arc_line_path(cx, cy, radius, start, end)}" fill="none" stroke="{color}" stroke-width="42" stroke-linecap="butt"/>'
        for start, end, color in segments
    )
    ticks = ""
    for tick in [0, 25, 50, 75, 100]:
        tx, ty = gauge_point(cx, cy, 168, tick)
        ticks += f'<text x="{tx:.1f}" y="{ty + 7:.1f}" text-anchor="middle" class="gauge-tick">{tick}</text>'
    return f"""
    <div class="needle-gauge">
      <svg viewBox="0 0 720 340" role="img" aria-label="K-FGI {value:.0f}, {fgi_state(value)}">
        <path d="{arc_line_path(cx, cy, radius, 0, 100)}" fill="none" stroke="rgba(255,255,255,.72)" stroke-width="52"/>
        {paths}
        {ticks}
        <circle cx="{cx}" cy="{cy}" r="96" fill="rgba(255,255,255,.70)"/>
        <polygon points="{pointer}" fill="{needle_color}" class="gauge-needle"/>
        <circle cx="{cx}" cy="{cy}" r="22" fill="{needle_color}"/>
        <circle cx="{cx}" cy="{cy}" r="6" fill="white"/>
        <text x="{cx}" y="326" text-anchor="middle" class="gauge-score">{value:.0f}</text>
      </svg>
      <div class="gauge-legend">
        <span><i class="legend-extreme-fear"></i>극단적 공포</span>
        <span><i class="legend-fear"></i>공포</span>
        <span><i class="legend-neutral"></i>중립</span>
        <span><i class="legend-greed"></i>탐욕</span>
        <span><i class="legend-extreme-greed"></i>극단적 탐욕</span>
      </div>
    </div>
    """


def make_fgi_chart(kfgi_view):
    fig = go.Figure()
    y_main = kfgi_view["K_FGI_MA"] if "K_FGI_MA" in kfgi_view else kfgi_view["K_FGI"]
    fig.add_trace(
        go.Scatter(
            x=kfgi_view["date"],
            y=kfgi_view["K_FGI"],
            mode="lines",
            name="K-FGI",
            line={"width": 1.3, "color": "rgba(15, 23, 42, .32)"},
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>K-FGI %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=kfgi_view["date"],
            y=y_main,
            mode="lines",
            name="5거래일 평균",
            line={"width": 2.6, "color": "#263a5f"},
            customdata=np.stack(
                [
                    kfgi_view["K_FGI"].map(fgi_state),
                    kfgi_view["weight"],
                    kfgi_view["egarch_vol"] * 100,
                    kfgi_view["kospi_close"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "5거래일 평균 %{y:.1f}<br>"
                "FGI 상태 %{customdata[0]}<br>"
                "투자비중 %{customdata[1]:.2f}x<br>"
                "EGARCH 변동성 %{customdata[2]:.2f}%<br>"
                "KOSPI200 %{customdata[3]:,.2f}<extra></extra>"
            ),
        )
    )
    fig.add_hrect(y0=0, y1=25, fillcolor="#d9ebff", opacity=0.55, line_width=0)
    fig.add_hrect(y0=25, y1=45, fillcolor="#d8f4fb", opacity=0.45, line_width=0)
    fig.add_hrect(y0=45, y1=65, fillcolor="#ddf6ea", opacity=0.40, line_width=0)
    fig.add_hrect(y0=65, y1=80, fillcolor="#fff0cf", opacity=0.50, line_width=0)
    fig.add_hrect(y0=80, y1=100, fillcolor="#ffe0de", opacity=0.60, line_width=0)
    fig.add_hline(y=75, line_dash="dot", line_color="#ff3b30", annotation_text="Extreme Greed")
    fig.add_hline(y=55, line_dash="dot", line_color="#ff9f1c", annotation_text="Greed")
    fig.add_hline(y=45, line_dash="dot", line_color="#2fb47c", annotation_text="Neutral")
    fig.add_hline(y=25, line_dash="dot", line_color="#1684fc", annotation_text="Fear")
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_range=[0, 100],
        yaxis_title="K-FGI",
        xaxis_title=None,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.88)",
        legend=dict(orientation="h", y=1.06, x=0),
    )
    return fig


def make_nav_chart(ret_view, initial_capital, monthly_contribution):
    fig = go.Figure()
    colors = {
        "buy_hold": "#8b95a5",
        "trend_only": "#21a8d8",
        "trend_egarch_vol": "#ff9f1c",
        "main_kfgi_sentiment": "#ff3b30",
    }
    for col, name in STRATEGIES.items():
        nav, contributed_series = simulate_account(ret_view, col, initial_capital, monthly_contribution)
        fig.add_trace(
            go.Scatter(
                x=ret_view["date"],
                y=nav,
                mode="lines",
                name=name,
                line={"width": 3 if col == "main_kfgi_sentiment" else 2, "color": colors[col]},
                customdata=contributed_series,
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    "%{fullData.name}: %{y:,.0f}원<br>"
                    "그날까지 투입원금: %{customdata:,.0f}원<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        height=410,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="평가금액",
        xaxis_title=None,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.9)",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(190, 225, 255, .95), transparent 28%),
            radial-gradient(circle at 86% 4%, rgba(223, 214, 255, .92), transparent 30%),
            radial-gradient(circle at 72% 78%, rgba(190, 247, 235, .62), transparent 28%),
            linear-gradient(135deg, #f8fbff 0%, #dff8ff 44%, #f7f3ff 100%);
    }
    [data-testid="stHeader"] { background: rgba(255,255,255,0); }
    .block-container { padding-top: 1.45rem; max-width: 1440px; }
    .hero {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 24px;
        margin-bottom: 10px;
    }
    .hero h1 {
        margin: 0;
        font-size: clamp(2.05rem, 4vw, 2.55rem);
        letter-spacing: 0;
        color: #0f172a;
        word-break: keep-all;
        overflow-wrap: normal;
    }
    .hero p { margin: 6px 0 0; color: #475569; font-size: 1rem; }
    .pill {
        display: inline-flex;
        align-items: center;
        border: 1px solid rgba(71, 85, 105, .18);
        border-radius: 999px;
        padding: 8px 13px;
        background: rgba(255,255,255,.72);
        color: #263a5f;
        font-weight: 700;
        white-space: nowrap;
        box-shadow: 0 10px 24px rgba(31, 41, 55, .06);
    }
    div[data-testid="stMetric"], .status-card, .panel, .compare-card, .indicator-card {
        background: rgba(255,255,255,.82);
        border: 1px solid rgba(148, 163, 184, .28);
        box-shadow: 0 18px 42px rgba(31, 41, 55, .09);
        backdrop-filter: blur(12px);
        border-radius: 8px;
        padding: 18px 18px;
    }
    div[data-testid="stMetric"] label { color: #334155 !important; font-weight: 750; }
    div[data-testid="stMetricValue"] { color: #0f172a; }
    .status-card h3 { margin: 0 0 8px; font-size: 1.08rem; color: #0f172a; }
    .status-card .big { font-size: 2.15rem; font-weight: 850; color: #111827; line-height: 1.05; }
    .status-card .muted { color: #64748b; margin-top: 8px; font-size: .94rem; }
    .panel-title {
        font-size: 1.23rem;
        font-weight: 850;
        color: #0f172a;
        margin: 0 0 12px;
    }
    .gauge-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.1rem;
        font-weight: 850;
        color: #111827;
        margin: 4px 0 0;
    }
    .gauge-title::before {
        content: "";
        width: 7px;
        height: 28px;
        background: linear-gradient(180deg, #1684fc, #ff3b30);
        display: inline-block;
    }
    .needle-gauge {
        width: 100%;
        min-height: 285px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .needle-gauge svg {
        width: 100%;
        max-width: 560px;
        height: auto;
        filter: drop-shadow(0 18px 32px rgba(15, 23, 42, .10));
    }
    .gauge-tick {
        fill: #7a7f87;
        font-size: 19px;
        font-weight: 650;
    }
    .gauge-score {
        fill: #111827;
        font-size: 58px;
        font-weight: 900;
    }
    .gauge-needle {
        filter: drop-shadow(0 5px 5px rgba(15, 23, 42, .22));
    }
    .gauge-legend {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 9px 14px;
        margin-top: -8px;
        color: #475569;
        font-size: .9rem;
        font-weight: 700;
    }
    .gauge-legend span {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
    }
    .gauge-legend i {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        display: inline-block;
    }
    .legend-extreme-fear { background: #d9ebff; border: 1px solid #1684fc; }
    .legend-fear { background: #d7f3fb; border: 1px solid #21a8d8; }
    .legend-neutral { background: #e7f8ef; border: 1px solid #2fb47c; }
    .legend-greed { background: #fff0cf; border: 1px solid #ff9f1c; }
    .legend-extreme-greed { background: #ffe0de; border: 1px solid #ff3b30; }
    .compare-card { margin-top: 18px; }
    .compare-card h3 {
        margin: 0 0 18px;
        font-size: 1.22rem;
        color: #111827;
    }
    .compare-row {
        display: grid;
        grid-template-columns: 92px 92px 1fr 54px;
        align-items: center;
        gap: 12px;
        margin: 21px 0;
        color: #111827;
    }
    .compare-row .when { color: #737373; font-weight: 650; }
    .compare-row .state { font-weight: 850; }
    .compare-row .rule { border-top: 2px dashed rgba(148, 163, 184, .42); }
    .chip {
        justify-self: end;
        min-width: 48px;
        height: 48px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 850;
        border: 2px solid #a1a1aa;
        background: #f4f4f5;
        color: #27272a;
    }
    .chip.extreme-fear { background: #d9ebff; border-color: #1684fc; color: #0f4c9a; }
    .chip.fear { background: #d7f3fb; border-color: #21a8d8; color: #075985; }
    .chip.neutral { background: #e7f8ef; border-color: #2fb47c; color: #166534; }
    .chip.greed { background: #fff0cf; border-color: #ff9f1c; color: #92400e; }
    .chip.extreme-greed { background: #ffe0de; border-color: #ff3b30; color: #991b1b; }
    .indicator-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin: 12px 0 22px;
    }
    .indicator-card h3 {
        margin: 0 0 8px;
        font-size: 1rem;
        color: #111827;
    }
    .indicator-card .value {
        font-size: 1.7rem;
        font-weight: 850;
        color: #111827;
        margin-bottom: 6px;
    }
    .indicator-card .desc {
        color: #64748b;
        line-height: 1.45;
        font-size: .94rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(226,232,240,.86);
        border-radius: 999px;
        padding: 10px 18px;
        color: #334155;
        font-weight: 800;
    }
    .stTabs [aria-selected="true"] {
        background: #263a5f !important;
        color: white !important;
    }
    .money-label {
        color: #334155;
        font-size: .92rem;
        font-weight: 700;
        margin: 12px 0 6px;
    }
    section[data-testid="stSidebar"] .stButton button {
        border-radius: 8px;
        min-height: 40px;
        font-size: 1.05rem;
        font-weight: 850;
        border: 1px solid rgba(148, 163, 184, .45);
        background: rgba(255,255,255,.82);
    }
    @media (max-width: 900px) {
        .indicator-grid { grid-template-columns: 1fr; }
        .compare-row { grid-template-columns: 72px 84px 1fr 48px; gap: 8px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

kfgi, ret, perf = load_data()

min_date = kfgi["date"].min().date()
max_date = kfgi["date"].max().date()

st.markdown(
    f"""
    <div class="hero">
      <div>
        <h1>K-Fear&Greed Index</h1>
        <p>시장 분위기가 공포인지 탐욕인지 보고, 내 투자금이 어떻게 움직였을지 쉽게 확인합니다.</p>
      </div>
      <div class="pill">{min_date} ~ {max_date}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("투자 시뮬레이션")
    selected_range = st.date_input(
        "기간",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    initial_capital_text = money_stepper(
        "초기 투자금",
        key="initial_capital_input",
        default=100_000_000,
        step=10_000_000,
        min_value=1_000_000,
        max_value=10_000_000_000,
    )
    monthly_contribution_text = money_stepper(
        "매월 추가 투자금",
        key="monthly_contribution_input",
        default=0,
        step=1_000_000,
        min_value=0,
        max_value=100_000_000,
    )
    selected_strategy = st.selectbox(
        "주 전략",
        options=list(STRATEGIES.keys()),
        format_func=lambda value: STRATEGIES[value],
    )

if len(selected_range) != 2:
    st.warning("확인할 시작일과 종료일을 모두 선택해 주세요.")
    st.stop()

initial_capital = parse_money(initial_capital_text)
monthly_contribution = parse_money(monthly_contribution_text)

if initial_capital is None or monthly_contribution is None:
    st.warning("투자금은 숫자와 쉼표만 입력해 주세요. 예: 100,000,000")
    st.stop()

if not 1_000_000 <= initial_capital <= 10_000_000_000:
    st.warning("초기 투자금은 1,000,000원부터 10,000,000,000원까지 입력할 수 있습니다.")
    st.stop()

if not 0 <= monthly_contribution <= 100_000_000:
    st.warning("매월 추가 투자금은 0원부터 100,000,000원까지 입력할 수 있습니다.")
    st.stop()

start_date, end_date = selected_range

mask = (kfgi["date"].dt.date >= start_date) & (kfgi["date"].dt.date <= end_date)
kfgi_view = kfgi.loc[mask].copy()
ret_view = ret.loc[(ret["date"].dt.date >= start_date) & (ret["date"].dt.date <= end_date)].copy()

if kfgi_view.empty or ret_view.empty:
    st.error("선택한 기간에 데이터가 없습니다.")
    st.stop()

kfgi_view["K_FGI_MA"] = kfgi_view["K_FGI"].rolling(5, min_periods=1).mean()
latest = kfgi_view.iloc[-1]
previous = kfgi_view.iloc[-2] if len(kfgi_view) > 1 else latest
latest_state = fgi_state(float(latest["K_FGI"]))
selected_stats = calc_stats(ret_view, selected_strategy, initial_capital, monthly_contribution)
bench_stats = calc_stats(ret_view, "buy_hold", initial_capital, monthly_contribution)
delta_vs_bh = selected_stats["final_value"] - bench_stats["final_value"]
comparison_html = ""
for label, row in comparison_rows(kfgi_view):
    score = float(row["K_FGI"])
    comparison_html += f"""
      <div class="compare-row">
        <div class="when">{label}</div>
        <div class="state">{fgi_state(score)}</div>
        <div class="rule"></div>
        <div class="{state_chip_class(score)}">{score:.0f}</div>
      </div>
    """

gauge_col, compare_col = st.columns([1.05, 0.95], gap="large")

with gauge_col:
    st.markdown('<div class="gauge-title">K-FGI 현재 상태</div>', unsafe_allow_html=True)
    st.markdown(make_gauge_svg(float(latest["K_FGI"])), unsafe_allow_html=True)
    st.caption(f"마지막 업데이트 기준일: {latest['date'].date()}")

with compare_col:
    st.markdown(
        f"""
        <div class="compare-card">
          <h3>과거와 비교</h3>
          <div class="compare-row">
            <div class="when">현재</div>
            <div class="state">{latest_state}</div>
            <div class="rule"></div>
            <div class="{state_chip_class(float(latest["K_FGI"]))}">{latest["K_FGI"]:.0f}</div>
          </div>
          {comparison_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="indicator-grid">
      <div class="indicator-card">
        <h3>시장 분위기</h3>
        <div class="value">{latest_state}</div>
        <div class="desc">K-FGI {latest["K_FGI"]:.1f} · {fgi_state_detail(float(latest["K_FGI"]))}</div>
      </div>
      <div class="indicator-card">
        <h3>추천 투자 정도</h3>
        <div class="value">{latest["weight"]:.2f}x</div>
        <div class="desc">1.00x는 전액 투자, 0.50x는 절반만 투자한다는 뜻입니다.</div>
      </div>
      <div class="indicator-card">
        <h3>내 예상 평가금액</h3>
        <div class="value">{krw(selected_stats["final_value"])}</div>
        <div class="desc">투입 {krw(selected_stats["contributed"])} · 손익 {selected_stats["profit"]:+,.0f}원</div>
      </div>
      <div class="indicator-card">
        <h3>단순 보유와 비교</h3>
        <div class="value">{delta_vs_bh:+,.0f}원</div>
        <div class="desc">KOSPI200 현재 {latest["kospi_close"]:,.2f} · 당일 {latest["target_reg"] * 100:+.2f}%</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["한눈에 보기", "K-FGI 흐름", "투자 결과"])

with tabs[0]:
    left, right = st.columns([1.35, 1], gap="medium")
    with left:
        st.markdown('<div class="panel-title">전략별 누적 평가금액</div>', unsafe_allow_html=True)
        st.plotly_chart(make_nav_chart(ret_view, initial_capital, monthly_contribution), use_container_width=True, key="overview_nav")
    with right:
        st.markdown('<div class="panel-title">최근 20거래일 상태</div>', unsafe_allow_html=True)
        recent = kfgi_view.tail(20)[["date", "K_FGI", "weight", "target_reg"]].copy()
        recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
        recent["state"] = recent["K_FGI"].map(fgi_state)
        recent["target_reg"] = recent["target_reg"].map(lambda value: f"{value * 100:+.2f}%")
        st.dataframe(
            recent.rename(
                columns={
                    "date": "날짜",
                    "K_FGI": "K-FGI",
                    "state": "FGI 상태",
                    "weight": "비중",
                    "target_reg": "수익률",
                }
            )[["날짜", "K-FGI", "FGI 상태", "비중", "수익률"]],
            use_container_width=True,
            hide_index=True,
        )

with tabs[1]:
    st.markdown('<div class="panel-title">Fear & Greed 흐름</div>', unsafe_allow_html=True)
    fig = make_fgi_chart(kfgi_view)
    st.plotly_chart(fig, use_container_width=True, key="kfgi_timeseries")

with tabs[2]:
    st.markdown('<div class="panel-title">투자 결과 요약</div>', unsafe_allow_html=True)
    summary_rows = []
    for key, name in STRATEGIES.items():
        stats = calc_stats(ret_view, key, initial_capital, monthly_contribution)
        summary_rows.append(
            {
                "전략": name,
                "투입원금": stats["contributed"],
                "최종금액": stats["final_value"],
                "손익": stats["profit"],
                "총수익률": stats["total_return"],
                "연환산수익률": stats["ann_ret"],
                "연변동성": stats["ann_vol"],
                "위험 대비 점수": stats["sharpe"],
                "최대 하락폭": stats["mdd"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    st.dataframe(
        summary.style.format(
            {
                "투입원금": "{:,.0f}원",
                "최종금액": "{:,.0f}원",
                "손익": "{:,.0f}원",
                "총수익률": "{:.1%}",
                "연환산수익률": "{:.1%}",
                "연변동성": "{:.1%}",
                "위험 대비 점수": "{:.2f}점",
                "최대 하락폭": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.plotly_chart(make_nav_chart(ret_view, initial_capital, monthly_contribution), use_container_width=True, key="backtest_nav")
