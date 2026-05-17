import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import plotly.graph_objects as go

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

# =========================
# KPI DASHBOARD
# =========================

portfolio_return = (
    returns.mean().mean() * 252
)

portfolio_volatility = (
    returns.std().mean() * np.sqrt(252)
)

portfolio_sharpe = (
    portfolio_return /
    portfolio_volatility
)

portfolio_drawdown = (
    drawdown.min().mean()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Expected Return",
    f"{portfolio_return:.2%}"
)

col2.metric(
    "Portfolio Volatility",
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
# =========================
# MONTE CARLO VISUALIZATION
# =========================

st.header("Monte Carlo Portfolio Simulation")

fig_mc = px.line(
    mc_paths,
    template="plotly_dark"
)

fig_mc.update_layout(
    height=700,
    xaxis_title="Trading Days",
    yaxis_title="Portfolio Value"
)

st.plotly_chart(
    fig_mc,
    use_container_width=True
)
# =========================
# VALUE AT RISK
# =========================

var_95 = np.percentile(
    portfolio_returns,
    5
)

cvar_95 = portfolio_returns[
    portfolio_returns <= var_95
].mean()

st.header("Tail Risk Metrics")

col1, col2 = st.columns(2)

col1.metric(
    "95% VaR",
    f"{var_95:.2%}"
)
# =========================
# ROLLING SHARPE ANALYSIS
# =========================

portfolio_returns = returns.mean(axis=1)

rolling_sharpe = (
    portfolio_returns.rolling(126).mean()
    /
    portfolio_returns.rolling(126).std()
) * np.sqrt(252)

rolling_sharpe = rolling_sharpe.dropna()

st.header("Rolling Sharpe Ratio")

fig_sharpe = px.line(
    rolling_sharpe,
    template="plotly_dark",
    title="6-Month Rolling Sharpe Ratio"
)

fig_sharpe.update_layout(
    xaxis_title="Date",
    yaxis_title="Sharpe Ratio"
)

st.plotly_chart(
    fig_sharpe,
    use_container_width=True
)
col2.metric(
    "95% CVaR",
    f"{cvar_95:.2%}"
)

# =========================
# ROLLING DRAWDOWN ANALYTICS
# =========================

portfolio_cumulative = (
    1 + portfolio_returns
).cumprod()

rolling_peak = portfolio_cumulative.cummax()

drawdown_series = (
    portfolio_cumulative / rolling_peak
) - 1

st.header("Portfolio Drawdown Analysis")

fig_drawdown = px.area(
    drawdown_series,
    template="plotly_dark",
    title="Portfolio Drawdown Over Time"
)

fig_drawdown.update_layout(
    xaxis_title="Date",
    yaxis_title="Drawdown"
)

st.plotly_chart(
    fig_drawdown,
    use_container_width=True
)
# =========================
# PORTFOLIO ALLOCATION
# =========================

factor_score = (
    factor_table["Sharpe"]
    +
    factor_table["Momentum_12M"]
    +
    factor_table["Sortino"]
)

factor_score = factor_score.clip(lower=0)

portfolio_weights = (
    factor_score /
    factor_score.sum()
)

allocation_df = pd.DataFrame({
    "Stock": portfolio_weights.index,
    "Weight": portfolio_weights.values
})

st.header("Portfolio Allocation")

fig_allocation = px.pie(
    allocation_df,
    names="Stock",
    values="Weight",
    hole=0.4,
    template="plotly_dark",
    title="Portfolio Weight Distribution"
)

st.plotly_chart(
    fig_allocation,
    use_container_width=True
)
# =========================
# EFFICIENT FRONTIER
# =========================

st.header("Efficient Frontier Optimization")

mean_returns = returns.mean() * 252

cov_matrix = returns.cov() * 252

num_portfolios = 3000

results = np.zeros((3, num_portfolios))

weights_record = []

for i in range(num_portfolios):

    weights = np.random.random(len(selected_tickers))

    weights /= np.sum(weights)

    portfolio_return = np.sum(
        mean_returns * weights
    )

    portfolio_volatility = np.sqrt(
        np.dot(
            weights.T,
            np.dot(cov_matrix, weights)
        )
    )

    sharpe_ratio = (
        portfolio_return /
        portfolio_volatility
    )

    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = sharpe_ratio

    weights_record.append(weights)

efficient_df = pd.DataFrame({
    "Return": results[0],
    "Volatility": results[1],
    "Sharpe": results[2]
})

fig_frontier = px.scatter(
    efficient_df,
    x="Volatility",
    y="Return",
    color="Sharpe",
    template="plotly_dark",
    title="Efficient Frontier Simulation"
)

st.plotly_chart(
    fig_frontier,
    use_container_width=True
)
