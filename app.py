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
import yfinance as yf
import pandas as pd
import numpy as np

tickers = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "LT.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "ITC.NS"
]

prices = yf.download(
    tickers,
    start="2018-01-01",
    auto_adjust=True
)["Close"]

returns = prices.pct_change().dropna()

summary_stats = pd.DataFrame({
    "CAGR": (
        (prices.iloc[-1] / prices.iloc[0]) **
        (252 / len(prices)) - 1
    ),
    "Volatility": (
        returns.std() * np.sqrt(252)
    ),
    "Sharpe": (
        (returns.mean() / returns.std()) * np.sqrt(252)
    )
})

momentum_6m = prices.pct_change(126).iloc[-1]

momentum_12m = prices.pct_change(252).iloc[-1]

momentum_df = pd.DataFrame({
    "Momentum_6M": momentum_6m,
    "Momentum_12M": momentum_12m
})

rolling_max = prices.rolling(252).max()

drawdown = (
    prices / rolling_max
) - 1

max_drawdown = drawdown.min()

drawdown_df = pd.DataFrame({
    "Max_Drawdown": max_drawdown
})

downside_returns = returns.copy()

downside_returns[
    downside_returns > 0
] = 0

sortino_ratio = (
    returns.mean() /
    downside_returns.std()
) * np.sqrt(252)

risk_metrics_df = pd.DataFrame({
    "Sortino": sortino_ratio
})

factor_table = pd.concat([
    summary_stats,
    momentum_df,
    drawdown_df,
    risk_metrics_df
], axis=1)

factor_table = factor_table.round(4)
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
