import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(
    page_title="Institutional Quant Platform",
    layout="wide"
)

st.title("Institutional Quant Platform")
st.sidebar.header("Quant Controls")

selected_tickers = st.sidebar.multiselect(
    "Select Stocks",
    [
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
    ],

    default=[
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS"
    ]
)

start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("2018-01-01")
)

strategy_type = st.sidebar.selectbox(
    "Strategy Type",
    [
        "Momentum",
        "Low Volatility",
        "Quality",
        "Multi Factor"
    ]
)

momentum_weight = st.sidebar.slider(
    "Momentum Weight",
    0.0,
    1.0,
    0.30
)

volatility_weight = st.sidebar.slider(
    "Volatility Weight",
    0.0,
    1.0,
    0.20
)

quality_weight = st.sidebar.slider(
    "Quality Weight",
    0.0,
    1.0,
    0.20
)

risk_weight = st.sidebar.slider(
    "Risk Control Weight",
    0.0,
    1.0,
    0.30
)
st.markdown("""
### Features
- Quant factor scoring
- Long/short portfolio
- Regime detection
- ML alpha prediction
- Monte Carlo simulation
- Tail risk analysis
""")

# =========================
# DATA DOWNLOAD
# =========================

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

# =========================
# FACTOR ENGINE
# =========================

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

drawdown = (prices / rolling_max) - 1

max_drawdown = drawdown.min()

drawdown_df = pd.DataFrame({
    "Max_Drawdown": max_drawdown
})

downside_returns = returns.copy()

downside_returns[downside_returns > 0] = 0

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

# =========================
# DASHBOARD
# =========================

st.header("Factor Table")

st.dataframe(
    factor_table,
    use_container_width=True
)
