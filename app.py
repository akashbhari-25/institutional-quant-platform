import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Institutional Quant Platform",
    layout="wide"
)

st.title("Institutional Quant Research Platform")

st.markdown("""
Multi-factor alpha engine with:
- Quant factor scoring
- Long/short portfolio
- Regime detection
- ML alpha prediction
- Monte Carlo simulation
- Tail risk analysis
""")

st.sidebar.header("Controls")

selected_metric = st.sidebar.selectbox(
    "Select Factor",
    [
        "CAGR",
        "Momentum_12M",
        "Sortino",
        "Annualized_Volatility"
    ]
)

st.subheader("Factor Table")

st.dataframe(factor_table)

st.subheader("Composite Alpha Scores")

fig = px.bar(
    factor_scores.reset_index(),
    x="Ticker",
    y="Composite_Alpha_Score",
    color="Composite_Alpha_Score",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Strategy vs Benchmark")

fig2 = px.line(
    x=strategy_cumulative.index,
    y=strategy_cumulative.values,
    labels={"x":"Date","y":"Portfolio Value"},
    title="Long/Short Strategy"
)

st.plotly_chart(fig2, use_container_width=True)

st.subheader("Feature Importance")

fig3 = px.bar(
    feature_importance,
    x="Factor",
    y="Importance",
    template="plotly_dark",
    color="Importance"
)

st.plotly_chart(fig3, use_container_width=True)