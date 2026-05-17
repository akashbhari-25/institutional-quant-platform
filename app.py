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

tickers = selected_tickers
    

prices = yf.download(
    tickers,
  start=start_date,
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
# =========================
# FACTOR SCORE ENGINE
# =========================

factor_scores = pd.DataFrame(index=factor_table.index)

factor_scores["CAGR_Score"] = (
    factor_table["CAGR"].rank(pct=True)
)

factor_scores["Momentum_Score"] = (
    factor_table["Momentum_12M"].rank(pct=True)
)

factor_scores["Volatility_Score"] = (
    1 - factor_table["Volatility"].rank(pct=True)
)

factor_scores["Drawdown_Score"] = (
    1 - factor_table["Max_Drawdown"].rank(pct=True)
)

factor_scores["Sortino_Score"] = (
    factor_table["Sortino"].rank(pct=True)
)

factor_scores["Composite_Score"] = (
    factor_scores["CAGR_Score"] * momentum_weight +
    factor_scores["Momentum_Score"] * momentum_weight +
    factor_scores["Volatility_Score"] * volatility_weight +
    factor_scores["Drawdown_Score"] * risk_weight +
    factor_scores["Sortino_Score"] * quality_weight
)

factor_scores = factor_scores.sort_values(
    "Composite_Score",
    ascending=False
)

# =========================
# PORTFOLIO CONSTRUCTION
# =========================

top_stocks = factor_scores.head(3).index

portfolio_returns = (
    returns[top_stocks]
    .mean(axis=1)
)

portfolio_cumulative = (
    1 + portfolio_returns
).cumprod()

benchmark_returns = (
    returns.mean(axis=1)
)

benchmark_cumulative = (
    1 + benchmark_returns
).cumprod()

# =========================
# PERFORMANCE CHART
# =========================

st.header("Portfolio Performance")

chart_data = pd.DataFrame({
    "Quant Portfolio": portfolio_cumulative,
    "Benchmark": benchmark_cumulative
})

st.line_chart(chart_data)
# =========================
# PERFORMANCE METRICS
# =========================

portfolio_cagr = (
    (portfolio_cumulative.iloc[-1]) **
    (252 / len(portfolio_returns))
) - 1

portfolio_volatility = (
    portfolio_returns.std() *
    np.sqrt(252)
)

portfolio_sharpe = (
    portfolio_returns.mean() /
    portfolio_returns.std()
) * np.sqrt(252)

portfolio_drawdown = (
    portfolio_cumulative /
    portfolio_cumulative.cummax() - 1
).min()

# =========================
# KPI DASHBOARD
# =========================

st.header("Portfolio Analytics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "CAGR",
    f"{portfolio_cagr:.2%}"
)

col2.metric(
    "Volatility",
    f"{portfolio_volatility:.2%}"
)

col3.metric(
    "Sharpe Ratio",
    f"{portfolio_sharpe:.2f}"
)

col4.metric(
    "Max Drawdown",
    f"{portfolio_drawdown:.2%}"
)
# =========================
# TOP STOCKS
# =========================

st.header("Top Ranked Stocks")

st.dataframe(
    factor_scores[["Composite_Score"]]
    .head(5),
    use_container_width=True
)
# =========================
# MONTE CARLO SIMULATION
# =========================

simulation_days = 252

simulation_runs = 200

portfolio_mean = portfolio_returns.mean()

portfolio_std = portfolio_returns.std()

mc_paths = pd.DataFrame()

for i in range(simulation_runs):

    simulated_returns = np.random.normal(
        portfolio_mean,
        portfolio_std,
        simulation_days
    )

    simulated_path = (
        1 + simulated_returns
    ).cumprod()

    mc_paths[i] = simulated_path
