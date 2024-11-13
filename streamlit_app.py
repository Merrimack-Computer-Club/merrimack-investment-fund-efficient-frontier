import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objs as go
import streamlit as st
import io

# Set the page layout as wide.
st.set_page_config(page_title="Merrimack Computer Club Portfolio Analysis Tool", layout="wide", page_icon="assets/merrimack-computer-club.png")

# Step 7: Streamlit app layout
# Use Markdown to create custom header with image on the right
st.title("Portfolio Optimization with Efficient Frontier and Historic Return Analyis")

mk = '''
This portfolio optimization and analysis tool helps users visualize and evaluate investment portfolios using the **efficient frontier**. It employs **mean-variance optimization** to calculate the optimal portfolio allocation for selected assets, such as stocks, mutual funds, ETFs, or other asset classes, and plots the trade-off between risk and return.

### Key Features:

1. **Efficient Frontier Calculation**:
   - Computes the efficient frontier, representing the optimal portfolios offering the highest expected annual return for a given level of risk, based on historical asset returns.

2. **Monte Carlo Simulation**:
   - Uses **Monte Carlo simulation** to resample portfolio inputs and generate a range of possible outcomes, improving diversification and providing more robust performance insights.

3. **Interactive User Interface**:
   - An intuitive web-based interface to:
     - Upload portfolio data or input tickers.
     - Adjust risk-free rates, asset allocations, and parameters.
     - Visualize performance metrics like expected return, volatility, and Sharpe ratio.

4. **Portfolio Customization**:
   - Allows users to specify asset allocations and constraints (e.g., ensuring weights sum to 1). Custom portfolios can be plotted on the efficient frontier for comparison.

5. **Market Comparison**:
   - Compares portfolio performance against major market indices (e.g., S&P 500, DJIA, NASDAQ), showing cumulative returns and daily rates of return.

6. **Visualization**:
   - Provides interactive charts for:
     - The efficient frontier with portfolio points and custom portfolio.
     - A correlation matrix of asset returns.
     - Historical rate of return and cumulative return for both the portfolio and market indices.

7. **Error Handling**:
   - Includes error handling for smooth operation, with feedback for issues like missing data or invalid inputs.

This tool empowers users to optimize portfolios, manage risk, and make informed decisions about asset allocation and performance.
'''
st.markdown(mk)

# Step 1: Fetch historical stock prices with error handling
def get_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        returns = data.pct_change().dropna()
        return returns
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Step 2: Portfolio performance calculations with adjustable risk-free rate
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    # Annualize returns and volatility
    annualized_return = np.sum(mean_returns * weights) * 252  # Annualizing daily returns
    annualized_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualizing volatility
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std_dev
    return annualized_return, annualized_std_dev, sharpe_ratio

# Step 3: Compute Efficient Frontier
def efficient_frontier(mean_returns, cov_matrix, num_portfolios=100, risk_free_rate=0.01):
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns, std_dev, sharpe_ratio = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
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
        f"Sharpe Ratio: {(custom_return - risk_free_rate) / custom_std_dev:.2f}"
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

# Step 5: Portfolio Summary Table
def portfolio_summary_table(results, weights_record, tickers):
    max_sharpe_idx = np.argmax(results[2])
    min_var_idx = np.argmin(results[0])

    summary_data = {
        'Portfolio': ['Maximum Sharpe Ratio', 'Minimum Variance'],
        'Expected Annual Return': [results[1, max_sharpe_idx], results[1, min_var_idx]],
        'Expected Volatility': [results[0, max_sharpe_idx], results[0, min_var_idx]],
        'Sharpe Ratio': [results[2, max_sharpe_idx], results[2, min_var_idx]],
        'Weights': [weights_record[max_sharpe_idx], weights_record[min_var_idx]]
    }
    summary_df = pd.DataFrame(summary_data)

    summary_df['Weights'] = summary_df['Weights'].apply(lambda w: ', '.join([f"{tickers[i]}: {w[i]:.2%}" for i in range(len(w))]))

    st.subheader("Efficient Frontier Portfolio Summary")
    st.write(summary_df)

# Step 6: Correlation Matrix Display
def display_correlation_matrix(returns, tickers):
    st.subheader("Correlation Matrix")
    correlation_matrix = returns.corr()
    st.write(correlation_matrix)

# CSV upload
uploaded_file = st.file_uploader("Upload CSV File (TICKER, WEIGHT)", type=["csv"])

tickers = []
weights = []

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=None)
        if df.shape[0] == 1:
            tickers = df.iloc[0].tolist()
        elif df.shape[0] == 2:
            tickers = df.iloc[0].tolist()
            weights = df.iloc[1].apply(pd.to_numeric, errors='coerce')
            weights = weights.tolist()
            print(tickers)
            print(weights)
            if np.sum(weights) != 1:
                st.warning("Weights do not sum to 1; normalizing weights.")
                weights = weights / np.sum(weights)
        else:
            st.error("CSV must have one or two rows (Tickers and Weights).")
    except:
        st.error("Error parsing given CSV file.")
        
if not tickers:
    ticker_input = st.text_input("Enter Tickers (Comma Separated)", value="AAPL,MSFT,GOOGL,AMZN,TSLA")
    tickers = [ticker.strip() for ticker in ticker_input.split(',')]

start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
# Function to get the last business day
def get_last_business_day():
    today = pd.to_datetime("today").normalize()  # Current date
    # If today is a weekend (Saturday or Sunday), move to the previous Friday
    if today.weekday() == 5:  # Saturday
        last_business_day = today - pd.Timedelta(days=1)
    elif today.weekday() == 6:  # Sunday
        last_business_day = today - pd.Timedelta(days=2)
    else:  # Weekdays
        last_business_day = today
    return last_business_day

# Set the end date to the last business day
end_date = st.date_input("End Date", value=get_last_business_day())

# Find the current risk free rate
def get_risk_free_rate():
    ticker = "^IRX"  # Symbol for 3-month US Treasury Bill yield
    data = yf.Ticker(ticker)
    # Get the most recent closing value of the yield
    rate = data.history(period="1d")['Close'].iloc[-1] / 100  # Convert to decimal (e.g., 4.5% -> 0.045)
    return rate

# Get the current risk-free rate
risk_free_rate = get_risk_free_rate()
st.text(f'The current risk-free rate for a  3-month US Treasury Bill yield of {risk_free_rate:.4f}%')
risk_free_rate = st.number_input("Set Risk-Free Rate (Annual)", min_value=0.0, max_value=1.0, value=risk_free_rate, step=0.0001, format="%.4f", help="Risk-free rate for calculating Sharpe ratio")

input_method = st.radio("Choose Input Method for Portfolio Weights", ('Slider', 'Number Input'))
st.subheader("Input Portfolio Weights")

if len(weights) <= 0:
    weights = [1 / len(tickers)] * len(tickers)

for i, ticker in enumerate(tickers):
    if input_method == 'Slider':
        weight = st.slider(f"Weight for {ticker}:", min_value=0.0, max_value=1.0, value=weights[i], format="%0.4f", step=0.0001)
        weights[i] = weight
    else:
        weight = st.number_input(f"Weight for {ticker}:", min_value=0.0, max_value=1.0, value=weights[i], format="%0.4f", step=0.0001)
        weights[i] = weight

weights = np.array(weights)
if np.sum(weights) != 1:
    st.warning("Weights do not sum to 1; normalizing weights.")
    weights = weights / np.sum(weights)

num_portfolios = st.slider("Select Number of Portfolios for Efficient Frontier", min_value=100, max_value=10000, value=5000, step=100)
# Check if tickers or weights are empty
if len(tickers) <= 0:
    st.warning("No tickers or weights provided. Please input tickers and weights.")
else:
# Create a DataFrame with two rows: one for tickers and one for weights
    df = pd.DataFrame([tickers, weights])

    # Convert DataFrame to CSV
    csv = df.to_csv(index=False, header=False)

    # Convert the CSV string into a BytesIO object
    csv_file = io.StringIO(csv)

    # Add download button for CSV file
    st.download_button(
        label="Download CSV",
        data=csv_file.getvalue(),
        file_name="portfolio.csv",
        mime="text/csv"
    )

if st.button("Deploy Efficient Frontier"):
    returns = get_data(tickers, start_date, end_date)
    if returns is not None:
        display_correlation_matrix(returns, tickers)
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        custom_return, custom_std_dev, custom_sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        results, weights_record = efficient_frontier(mean_returns, cov_matrix, num_portfolios, risk_free_rate)
        fig = plot_efficient_frontier(results, weights_record, weights, custom_return, custom_std_dev, tickers)
        st.plotly_chart(fig)
        st.subheader("Portfolio Performance Metrics")
        st.write(f"Expected Annual Return: {custom_return:.2%}")
        st.write(f"Expected Volatility: {custom_std_dev:.2%}")
        st.write(f"Sharpe Ratio: {(custom_return - risk_free_rate) / custom_std_dev:.2f}")
        portfolio_summary_table(results, weights_record, tickers)

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

    # Plot historical rate of return as cumulative percentage return
    rate_of_return_fig = go.Figure()

    # Calculate cumulative return as percentage
    portfolio_cumulative_return = (1 + comparison_df['Portfolio']).cumprod() - 1
    market_cumulative_return = (1 + comparison_df[selected_market]).cumprod() - 1

    # Add traces for the cumulative return
    rate_of_return_fig.add_trace(go.Scatter(x=comparison_df.index, y=portfolio_cumulative_return * 100, mode='lines', name='Selected Portfolio'))
    rate_of_return_fig.add_trace(go.Scatter(x=comparison_df.index, y=market_cumulative_return * 100, mode='lines', name=selected_market))

    # Update layout with appropriate titles and labels
    rate_of_return_fig.update_layout(
        title=f'Historical Rate of Return (Cumulative Percentage): Selected Portfolio vs {selected_market}',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        legend_title='Legend',
        template='plotly_white'
    )

    # Display historical rate of return plot
    st.plotly_chart(rate_of_return_fig)
