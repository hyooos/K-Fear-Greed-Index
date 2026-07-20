from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="K-FGI 10Y Dashboard", layout="wide")

BASE = Path(__file__).resolve().parents[1]
OUTPUTS = BASE / "paper_outputs"
DATA = OUTPUTS / "data"
TABLES = OUTPUTS / "tables"


@st.cache_data
def load_data():
    kfgi = pd.read_csv(DATA / "kfgi_10y_timeseries.csv", parse_dates=["date"])
    ret = pd.read_csv(DATA / "strategy_returns_10y.csv", parse_dates=["date"])
    perf = pd.read_csv(TABLES / "performance_summary.csv")
    imp = pd.read_csv(TABLES / "feature_importance.csv")
    sens = pd.read_csv(TABLES / "sensitivity_analysis.csv")
    yearly = pd.read_csv(TABLES / "yearly_performance.csv")
    stat = pd.read_csv(TABLES / "statistical_tests.csv")
    return kfgi, ret, perf, imp, sens, yearly, stat


kfgi, ret, perf, imp, sens, yearly, stat = load_data()

st.title("K-FGI x EGARCH 10-Year Dashboard")
st.caption("sub_index1은 원자료에는 보존하되, 모멘텀 중복 방지를 위해 K-FGI 및 모델 피처에서는 제외.")

main = perf[perf["label"] == "main_kfgi_sentiment"].iloc[0]
bench = perf[perf["label"] == "buy_hold_kospi200"].iloc[0]
cols = st.columns(5)
cols[0].metric("Experiment period", f"{kfgi['date'].min().date()} ~ {kfgi['date'].max().date()}")
cols[1].metric("Main total return", f"{main['total_return']*100:.1f}%")
cols[2].metric("Main Sharpe", f"{main['sharpe']:.3f}")
cols[3].metric("Main MDD", f"{main['mdd']*100:.1f}%", delta=f"{(main['mdd']-bench['mdd'])*100:.1f}%p vs B&H")
cols[4].metric("Buy&Hold total", f"{bench['total_return']*100:.1f}%")

tabs = st.tabs(["Performance", "K-FGI", "Weights", "Feature Importance", "Validation", "Data"])

with tabs[0]:
    st.subheader("전략 성과")
    fig = go.Figure()
    labels = {
        "buy_hold": "KOSPI200 B&H",
        "trend_only": "Trend",
        "trend_egarch_vol": "Trend + EGARCH",
        "main_kfgi_sentiment": "K-FGI Strategy",
    }
    for col, name in labels.items():
        fig.add_trace(go.Scatter(x=ret["date"], y=np.exp(ret[col].cumsum()), mode="lines", name=name))
    fig.update_layout(height=460, yaxis_title="Cumulative return (base=1.0)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(perf, use_container_width=True)

with tabs[1]:
    st.subheader("K-FGI 시계열")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kfgi["date"], y=kfgi["K_FGI"].rolling(5).mean(), mode="lines", name="K-FGI 5D MA"))
    fig.add_hline(y=65, line_dash="dot", line_color="red", annotation_text="Greed 65")
    fig.add_hline(y=25, line_dash="dot", line_color="blue", annotation_text="Fear 25")
    fig.update_layout(height=440, yaxis_range=[0, 100], yaxis_title="K-FGI")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(kfgi[["date", "K_FGI", "regime", "egarch_vol", "weight", "strat_ret"]], use_container_width=True)

with tabs[2]:
    st.subheader("가중치 민감도")
    heat = sens.pivot_table(index="kfgi_range", columns="regime_mult", values="sharpe", aggfunc="mean")
    fig = px.imshow(heat, text_auto=".3f", aspect="auto", color_continuous_scale="YlGnBu")
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sens.sort_values("sharpe", ascending=False), use_container_width=True)

with tabs[3]:
    st.subheader("LightGBM Feature Importance")
    top = imp.head(20).copy()
    fig = px.bar(top.iloc[::-1], x="importance", y="feature", orientation="h", height=560)
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("검증 결과")
    left, right = st.columns(2)
    with left:
        st.write("연도별 성과")
        st.plotly_chart(px.bar(yearly, x="year", y="total_return", color="total_return", height=360), use_container_width=True)
    with right:
        st.write("통계 검정")
        stat_plot = stat.copy()
        stat_plot["minus_log10_p"] = -np.log10(stat_plot["p_value"].clip(lower=1e-300))
        st.plotly_chart(px.bar(stat_plot, x="test", y="minus_log10_p", height=360), use_container_width=True)
    st.dataframe(stat, use_container_width=True)

with tabs[5]:
    st.subheader("파일 위치")
    st.code(str(BASE))
    st.write("주요 입력 및 산출물은 패키지 내부 `data`, `06_10y_experiments`, `figure 모음`에 저장되어 있습니다.")
