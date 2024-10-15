import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objs as go
import streamlit as st

# Step 1: Fetch historical stock prices
def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    return returns

# Step 2: Portfolio performance calculations
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (returns - 0.01) / std_dev  # Assume risk-free rate = 0.01
    return returns, std_dev, sharpe_ratio

# Step 3: Compute Efficient Frontier
def efficient_frontier(mean_returns, cov_matrix, num_portfolios=100):
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns, std_dev, sharpe_ratio = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = std_dev
        results[1, i] = returns
        results[2, i] = sharpe_ratio  # Sharpe Ratio
        weights_record.append(weights)

    return results, weights_record

# Step 4: Plot Efficient Frontier and Custom Portfolio
def plot_efficient_frontier(results, weights_record, custom_weights, custom_return, custom_std_dev, tickers):
    hover_texts = []
    for i, weights in enumerate(weights_record):
        portfolio_info = [f"{ticker}: {weight:.2%}" for ticker, weight in zip(tickers, weights)]
        hover_text = (
            f"Portfolio Weights:<br>{'<br>'.join(portfolio_info)}<br>"
            f"Return: {results[1, i]:.2%}<br>"
            f"Volatility: {results[0, i]:.2%}<br>"
            f"Sharpe Ratio: {results[2, i]:.2f}"
        )
        hover_texts.append(hover_text)

    trace1 = go.Scatter(
        x=results[0, :], y=results[1, :],
        mode='markers',
        marker=dict(color=results[2, :], size=5, showscale=True),
        name='Efficient Frontier',
        text=hover_texts,
        hoverinfo='text'
    )

    custom_portfolio_info = [f"{ticker}: {weight:.2%}" for ticker, weight in zip(tickers, custom_weights)]
    custom_hover_text = (
        f"Custom Portfolio:<br>{'<br>'.join(custom_portfolio_info)}<br>"
        f"Return: {custom_return:.2%}<br>"
        f"Volatility: {custom_std_dev:.2%}<br>"
        f"Sharpe Ratio: {(custom_return - 0.01) / custom_std_dev:.2f}"
    )

    trace2 = go.Scatter(
        x=[custom_std_dev], y=[custom_return],
        mode='markers',
        marker=dict(color='red', size=12, symbol='star'),
        name='Custom Portfolio',
        text=[custom_hover_text],
        hoverinfo='text'
    )

    layout = go.Layout(
        title='Efficient Frontier with Custom Portfolio',
        xaxis=dict(title='Volatility (Std Dev)'),
        yaxis=dict(title='Return'),
        showlegend=True
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return fig

# Other code remains unchanged...

# Step 5: Streamlit app layout
st.title("Portfolio Optimization with Efficient Frontier")

# Allow user to upload CSV file
uploaded_file = st.file_uploader("Upload CSV File (TICKER, WEIGHT)", type=["csv"])

# Initialize tickers and weights
tickers = []
weights = []

# Handle file upload
if uploaded_file is not None:
    try:
        # Read CSV into DataFrame
        df = pd.read_csv(uploaded_file, header=None)

        # Validate CSV format
        if df.shape[0] == 1:
            tickers = df.iloc[0].tolist()
        elif df.shape[0] == 2:
            tickers = df.iloc[0].tolist()
            weights = df.iloc[1].tolist()
            if np.sum(weights) != 1:
                st.warning("Weights from CSV do not sum to 1. Normalizing them.")
                weights = weights / np.sum(weights)
        else:
            st.error("CSV must have one or two rows (Tickers and Weights).")

    except Exception as e:
        st.error(f"Error reading the CSV file: {str(e)}")

# Fallback to manual ticker input if no file is uploaded
if not tickers:
    ticker_input = st.text_input("Enter Tickers (Comma Separated)", value="AAPL,MSFT,GOOGL,AMZN,TSLA")
    tickers = [ticker.strip() for ticker in ticker_input.split(',')]

# Fetch historical data
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))

# Option to choose input method for weights
input_method = st.radio("Choose Input Method for Portfolio Weights", ('Slider', 'Number Input'))

st.subheader("Input Portfolio Weights")

if not weights:
    # Default weights are evenly distributed
    weights = [1 / len(tickers)] * len(tickers)

# Adjust weights dynamically based on input method
for i, ticker in enumerate(tickers):
    if input_method == 'Slider':
        weight = st.slider(f"Weight for {ticker}:", min_value=0.0, max_value=1.0, value=weights[i], step=0.01)
        weights[i] = weight
    else:  # Number Input
        weight = st.number_input(f"Weight for {ticker}:", min_value=0.0, max_value=1.0, value=weights[i], step=0.01)
        weights[i] = weight

st.write(f"Sum of weights: {sum(weights):.2f}")

# Normalize weights to ensure they sum to 1
weights = np.array(weights)
if np.sum(weights) != 1:
    weights = weights / np.sum(weights)

# Slider to set the number of portfolios
num_portfolios = st.slider("Select Number of Portfolios for Efficient Frontier", min_value=100, max_value=10000, value=10000, step=100)

# Dropdown to select the market index for comparison
market_options = {
    "S&P 500": "^GSPC",
    "DJIA": "^DJI",
    "Russell 1000": "^RUI",
    "NASDAQ": "^IXIC"
}

selected_market = st.selectbox("Select Market Index to Compare", list(market_options.keys()))
if st.button("Compare to Market"):
    selected_weights = weights

    # Fetch historical data for the selected portfolio
    selected_portfolio_returns = get_data(tickers, start_date, end_date)
    weighted_portfolio_returns = (selected_portfolio_returns * selected_weights).sum(axis=1)

    # Convert portfolio returns index to timezone-naive
    weighted_portfolio_returns.index = weighted_portfolio_returns.index.tz_localize(None)

    # Fetch historical data for the selected market index
    market_symbol = market_options[selected_market]
    market_data = yf.download(market_symbol, start=start_date, end=end_date)['Adj Close']
    market_returns = market_data.pct_change().dropna()

    # Convert market returns index to timezone-naive
    market_returns.index = market_returns.index.tz_localize(None)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Portfolio': weighted_portfolio_returns,
        selected_market: market_returns
    })

    # Drop rows with NaN values that may arise due to date mismatches
    comparison_df = comparison_df.dropna()

    # Plot comparison
    comparison_fig = go.Figure()
    comparison_fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['Portfolio'], mode='lines', name='Selected Portfolio'))
    comparison_fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df[selected_market], mode='lines', name=selected_market))

    comparison_fig.update_layout(
        title=f'Historical Returns: Selected Portfolio vs {selected_market}',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        legend_title='Legend',
        template='plotly_white'
    )

    # Display comparison plot
    st.plotly_chart(comparison_fig)

# Button to update portfolio
if st.button("Update Portfolio"):
    # Fetch historical data
    returns = get_data(tickers, start_date, end_date)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Compute portfolio performance for custom weights
    custom_return, custom_std_dev, custom_sharpe = portfolio_performance(weights, mean_returns, cov_matrix)

    # Compute efficient frontier using the slider value
    results, weights_record = efficient_frontier(mean_returns, cov_matrix, num_portfolios)

    # Plot efficient frontier and custom portfolio
    fig = plot_efficient_frontier(results, weights_record, weights, custom_return, custom_std_dev, tickers)

    # Display the plot
    st.plotly_chart(fig)

    # Show additional metrics
    st.subheader("Portfolio Performance Metrics")
    st.write(f"Expected Return: {custom_return:.2%}")
    st.write(f"Expected Volatility: {custom_std_dev:.2%}")
    st.write(f"Sharpe Ratio: {(custom_return - 0.01) / custom_std_dev:.2f}")
