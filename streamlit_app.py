import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import streamlit as st
import io

# Author tag
AUTHOR_URL = "https://www.linkedin.com/in/alex-elguezabal/"
WEBSITE_URL = "https://merrimackcomputer.club/"

# Set the page configuration and layout as wide.
st.set_page_config(page_title="Merrimack Computer Club Portfolio Analysis Tool", layout="wide", page_icon="assets/merrimack-computer-club.png")

# Page Information Ref.
mcc_image = f"""
<a href="{WEBSITE_URL}" target="_blank">
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSixZF48ZnadAhRjcI-4_g8pCN4nDtLYabSNg&s" alt="Merrimack Computer Club Website" style="width:100; text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);">
</a>
"""

# Render the clickable image
st.markdown(mcc_image, unsafe_allow_html=True)

# Use Markdown to create custom header with image on the right
st.title("Merrimack College Student Managed Fund - Portfolio Analysis Tool")

# Add logo
mk = '''
This tool helps users visualize and optimize investment portfolios using the **efficient frontier**. It calculates the optimal portfolio allocation for various assets and plots the trade-off between risk and return.

## Key Features

- **Efficient Frontier**: Identify optimal portfolios for the best risk-return balance.
- **Monte Carlo Simulation**: Enhance diversification with robust outcome projections.
- **Portfolio Customization**: Adjust allocations, set constraints, and analyze key metrics like return, volatility, and Sharpe ratio.
- **Market Comparison**: Benchmark performance against indices such as S&P 500 and NASDAQ.
- **Backtesting**: Review historical performance for selected and optimized portfolios.
- **Alpha & Beta Analysis**: Evaluate excess returns and market risk using CAPM metrics.
- **Interactive Tools**: Seamlessly upload/download CSVs, fine-tune weights, and integrate dynamic risk-free rates.
- **Visualization**: Explore interactive charts for efficient frontiers, asset correlations, and cumulative returns.
- **Error Handling**: Ensure smooth operations with clear feedback and issue resolution.

This tool empowers users to optimize portfolios, manage risk, and make informed decisions about asset allocation and performance.
'''
st.markdown(mk)

# Dropdown to select the market index for comparison
market_options = {
    "S&P 500": "^GSPC",
    "DJIA": "^DJI",
    "Russell 1000": "^RUI",
    "NASDAQ": "^IXIC"
}

# Step 1: Fetch historical stock prices with error handling
def get_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
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
        'Portfolio': ['Maximum Sharpe Ratio (Optimized Portfolio)', 'Minimum Variance'],
        'Expected Annual Return': [results[1, max_sharpe_idx], results[1, min_var_idx]],
        'Expected Volatility': [results[0, max_sharpe_idx], results[0, min_var_idx]],
        'Sharpe Ratio': [results[2, max_sharpe_idx], results[2, min_var_idx]],
        'Weights': [weights_record[max_sharpe_idx], weights_record[min_var_idx]]
    }
    summary_df = pd.DataFrame(summary_data)

    summary_df['Weights'] = summary_df['Weights'].apply(lambda w: ', '.join([f"{tickers[i]}: {w[i]:.2%}" for i in range(len(w))]))

    st.subheader("Portfolio Summary: Efficient Frontier")

    #####################################################
    #
    #   Sharpe Ratio Dataframe
    #
    #####################################################

    # Create a DataFrame to store portfolio statistics
    summary_sharpe_df = pd.DataFrame({
        'Expected Annual Return': results[1],
        'Expected Volatility': results[0],
        'Sharpe Ratio': results[2],
        'Weights': weights_record
    })

    # Sort by Sharpe Ratio in descending order
    summary_sharpe_df = summary_sharpe_df.sort_values(by='Sharpe Ratio', ascending=False).reset_index(drop=True)

    # Convert weights to readable format
    summary_sharpe_df['Weights'] = summary_sharpe_df['Weights'].apply(
        lambda w: ', '.join([f"{tickers[i]}: {w[i]:.2%}" for i in range(len(w))])
    )

    summary, sharp_ratio = st.tabs(["🗃 Summary", "📈 Sharp Ratio Data"])
    
    summary.subheader("Summary of Efficient Frontier")
    summary.write(summary_df)    
    sharp_ratio.subheader("Portfolios sorted by Sharpe Ratio")
    sharp_ratio.dataframe(summary_sharpe_df.style.format({"Expected Annual Return": "{:.2%}", 
                                          "Expected Volatility": "{:.2%}", 
                                          "Sharpe Ratio": "{:.4f}"}))

# Step 6: Correlation Matrix Display
def display_correlation_matrix(returns, tickers):
    correlation_matrix = returns.corr()
    st.write(correlation_matrix)

# Step 7: Portfolio & Optimized Portfolio Backtest
def portfolio_backtest(results, weights_record, selected_weights, tickers, start_date, end_date):
    max_sharpe_idx = np.argmax(results[2])
    # Fetch historical data for the selected portfolio
    selected_portfolio_returns = get_data(tickers, start_date, end_date)
    weighted_portfolio_returns = (selected_portfolio_returns * selected_weights).sum(axis=1)
    optimized_portfolio_weights = weights_record[max_sharpe_idx]
    weighted_optimized_portfolio_returns = (selected_portfolio_returns * optimized_portfolio_weights).sum(axis=1)

    # Convert portfolio returns index to timezone-naive
    weighted_portfolio_returns.index = weighted_portfolio_returns.index.tz_localize(None)
    weighted_optimized_portfolio_returns.index = weighted_optimized_portfolio_returns.index.tz_localize(None)

    # Fetch historical data for the selected market index
    market_symbol = market_options[selected_market]
    market_data = yf.download(market_symbol, start=start_date, end=end_date, auto_adjust=True)['Close']
    market_returns = market_data.pct_change().dropna().squeeze()

    # Convert market returns index to timezone-naive
    market_returns.index = market_returns.index.tz_localize(None)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Optimized Portfolio': weighted_optimized_portfolio_returns,
        'Selected Portfolio': weighted_portfolio_returns,
        selected_market: market_returns
    })

    # Drop rows with NaN values that may arise due to date mismatches
    comparison_df = comparison_df.dropna()

    # Plot comparison
    comparison_fig = go.Figure()
    comparison_fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['Optimized Portfolio'], mode='lines', name='Optimized Portfolio'))
    comparison_fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['Selected Portfolio'], mode='lines', name='Selected Portfolio'))
    comparison_fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df[selected_market], mode='lines', name=selected_market))

    comparison_fig.update_layout(
        title=f'Historical Returns: Optimized Portfolio vs Selected Portfolio vs {selected_market}',
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
    optimized_portfolio_cumulative_return = (1 + comparison_df['Optimized Portfolio']).cumprod() - 1
    selected_portfolio_cumulative_return = (1 + comparison_df['Selected Portfolio']).cumprod() - 1
    market_cumulative_return = (1 + comparison_df[selected_market]).cumprod() - 1

    # Add traces for the cumulative return
    rate_of_return_fig.add_trace(go.Scatter(x=comparison_df.index, y=optimized_portfolio_cumulative_return * 100, mode='lines', name='Optimized Portfolio'))
    rate_of_return_fig.add_trace(go.Scatter(x=comparison_df.index, y=selected_portfolio_cumulative_return * 100, mode='lines', name='Selected Portfolio'))
    rate_of_return_fig.add_trace(go.Scatter(x=comparison_df.index, y=market_cumulative_return * 100, mode='lines', name=selected_market))

    # Update layout with appropriate titles and labels
    rate_of_return_fig.update_layout(
        title=f'Historical Rate of Return (Cumulative Percentage): Optimized Portfolio vs Selected Portfolio vs {selected_market}',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        legend_title='Legend',
        template='plotly_white'
    )

    # Display historical rate of return plot
    st.plotly_chart(rate_of_return_fig)

# Step 8: Calculate the R alpha using CAPM forumula, beta using slope of regression through volitilty.
def calculate_alpha_beta(portfolio_returns, market_returns, risk_free_rate):
    """
    Calculate alpha and beta for a portfolio using the CAPM model.
    Alpha is the excess return relative to the market, adjusted for risk-free rate.
    Beta measures the systematic risk of the portfolio.
    """
    # Ensure both series are tz-naive
    portfolio_returns.index = portfolio_returns.index.tz_localize(None)
    market_returns.index = market_returns.index.tz_localize(None)

    # Ensure both series have the same length and drop NaN values
    combined_df = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
    portfolio_returns = combined_df.iloc[:, 0]
    market_returns = combined_df.iloc[:, 1]

    # Perform linear regression to calculate beta (slope) and alpha (intercept)
    X = market_returns.values.reshape(-1, 1)  # Independent variable (market returns)
    y = portfolio_returns.values              # Dependent variable (portfolio returns)
    X = np.hstack([np.ones_like(X), X])       # Add column for intercept
    coef = np.linalg.lstsq(X, y, rcond=None)[0]  # Solve for regression coefficients
    intercept, beta = coef[0], coef[1]        # Intercept and slope from regression

    # Calculate average returns
    R = portfolio_returns.sum()         # Portfolio return
    Rm = market_returns.sum()          # Market return

    # Calculate alpha using the CAPM formula
    alpha = (R - risk_free_rate) - beta * (Rm - risk_free_rate)

    return alpha, beta

# Step 9: display the alpha and beta in a readable table
def display_alpha_beta(results, weights_record, weights, tickers, start_date, end_date, risk_free_rate):
    """
    Calculate and display alpha and beta for the selected and optimized portfolios.
    """
    # Fetch portfolio returns and market returns
    returns = get_data(tickers, start_date, end_date)
    if returns is not None:
        selected_portfolio_returns = (returns * weights).sum(axis=1)
        max_sharpe_idx = np.argmax(results[2])
        optimized_weights = weights_record[max_sharpe_idx]
        optimized_portfolio_returns = (returns * optimized_weights).sum(axis=1)

        # Fetch market data
        market_symbol = market_options[selected_market]
        market_data = yf.download(market_symbol, start=start_date, end=end_date, auto_adjust=True)['Close']
        market_returns = market_data.pct_change().dropna()

        # Calculate alpha and beta for both portfolios
        selected_alpha, selected_beta = calculate_alpha_beta(selected_portfolio_returns, market_returns, risk_free_rate)
        optimized_alpha, optimized_beta = calculate_alpha_beta(optimized_portfolio_returns, market_returns, risk_free_rate)

        # Create a DataFrame to display the results
        data = {
            "Portfolio": ["Selected Portfolio", "Optimized Portfolio"],
            "Alpha": [selected_alpha, optimized_alpha],
            "Beta": [selected_beta, optimized_beta]
        }
        summary_df = pd.DataFrame(data)

        # Display the table
        st.write(summary_df)

# Step 10: Add graph for tickers most likely to be big performers in the next year
def display_top_performers(mean_returns, tickers):
    # Example DataFrame where the tickers are the index and returns are the values
    data = {
        "Mean Returns": mean_returns
    }

    # Create DataFrame
    df = pd.DataFrame(data, index=tickers)

    # Sort the DataFrame by 'Mean Returns' column in descending order
    df_sorted = df.sort_values(by="Mean Returns", ascending=False)

    # Extract top tickers and their associated returns as arrays
    top_tickers = df_sorted.index.to_numpy()  # Extract tickers
    top_returns = df_sorted["Mean Returns"].to_numpy()  # Extract returns

    # Custom color scale from dark gold to bright gold
    color_scale = [
        [0, 'rgb(204, 153, 0)'],  # Darker gold
        [1, 'rgb(255, 215, 0)']   # Bright gold
    ]

    # Plot the bar chart with the custom color scale
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_tickers[:10],  # Show top 10 performers
        y=top_returns[:10],
        name='Top Performers',
        marker=dict(
            color=top_returns[:10],  # Apply color based on the returns
            colorscale=color_scale,  # Use the custom color scale
            showscale=True           # Show the color scale bar
        )
    ))

    fig.update_layout(
        title='Top 10 Most Likely Big Performers in the Next Year',
        xaxis_title='Ticker',
        yaxis_title='Expected Return',
        template='plotly_white'
    )

    st.plotly_chart(fig)

##################################################################################
#                                                                                #
# Application UI                                                                 #
#                                                                                #                                               
##################################################################################

tickers = []
weights = []
sub_portfolio_percentage = 0.25

with st.sidebar:
        st.header("Portfolio Selection")
        # CSV upload
        uploaded_file = st.file_uploader("Upload CSV File (TICKER, WEIGHT)", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, header=None)
                if df.shape[0] == 1:
                    tickers = df.iloc[0].tolist()
                elif df.shape[0] == 2:
                    tickers = df.iloc[0].tolist()
                    weights = df.iloc[1].apply(pd.to_numeric, errors='coerce')
                    weights = weights.tolist()
                    if np.sum(weights) != 1:
                        st.warning("Weights do not sum to 1; normalizing weights.")
                        weights = weights / np.sum(weights)
                else:
                    st.error("CSV must have one or two rows (Tickers and Weights).")
            except:
                st.error("Error parsing given CSV file.")
                
        # Set default tickers if none exist.
        if not tickers:
            tickers = ["AAPL","MSFT","GOOGL","AMZN","TSLA"]

        ticker_input = st.text_input("Enter Tickers (Comma Separated)", value=','.join(tickers))
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
            ticker = "^TNX"  # Symbol for 10 year US Treasury Bill yield
            data = yf.Ticker(ticker)
            # Get the most recent closing value of the yield
            rate = data.history(period="1d")
            # If there is an error getting the Risk free rate, give 0.0
            if(len(rate) == 0):
                return 0.0
            rate = rate['Close'].iloc[-1] / 100  # Convert to decimal (e.g., 4.5% -> 0.045)
            return rate

        # Set the selected comparison market index
        selected_market = st.selectbox("Select Market Index to Compare", list(market_options.keys()))

        # Get the current risk-free rate
        risk_free_rate = get_risk_free_rate()
        st.text(f'The current risk-free rate for a  10-year US Treasury Bill yield of {risk_free_rate:.4f}%')
        risk_free_rate = st.number_input("Set Risk-Free Rate (Annual)", min_value=0.0, max_value=1.0, value=risk_free_rate, step=0.0001, format="%.4f", help="Risk-free rate for calculating Sharpe ratio.")

        input_method = st.radio("Choose Input Method for Portfolio Weights", ('Slider', 'Number Input'))
        st.subheader("Input Portfolio Weights")

        if len(weights) <= 0:
            weights = [1 / len(tickers)] * len(tickers)

        elif len(weights) < len(tickers):  # If the ticker-weight matrix is not square
            st.warning("A new ticker has been added, please review wights as they may have been adjusted.")
            # Extend the weights array by adding 0 for the new tickers
            weights = np.concatenate([weights, np.zeros(len(tickers) - len(weights))])

        for i, ticker in enumerate(tickers):
            name = yf.Ticker(ticker).get_info()['shortName']
            if input_method == 'Slider':
                weight = st.slider(f"Weight for {ticker} ({name}):", min_value=0.0, max_value=1.0, value=weights[i], format="%0.4f", step=0.0001)
                weights[i] = weight
            else:
                weight = st.number_input(f"Weight for {ticker} ({name}):", min_value=0.0, max_value=1.0, value=weights[i], format="%0.4f", step=0.0001)
                weights[i] = weight

        weights = np.array(weights)
        if np.sum(weights) != 1:
            st.warning("Weights do not sum to 1; normalizing weights.")
            weights = weights / np.sum(weights)

        sub_portfolio_percentage = st.number_input("Set Sub-portfolio Weighting", min_value=0.0, max_value=1.0, value=sub_portfolio_percentage, step=0.0001, format="%.4f", help="Sub-portfolio weighting.")

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
                label="Download Portfolio CSV",
                data=csv_file.getvalue(),
                file_name="portfolio.csv",
                mime="text/csv"
            )

num_portfolios=5000
num_portfolios_input_method = st.radio("Select Number of Portfolios for Efficient Frontier", ('Slider', 'Number Input'))
if num_portfolios_input_method=='Slider': 
    num_portfolios = st.slider("Select Number of Portfolios for Efficient Frontier", min_value=100, max_value=10000, value=5000, step=100)
else:
    num_portfolios = st.number_input("Number of Portfolios", min_value=100, value=num_portfolios)

if st.button("Deploy Efficient Frontier"):
    returns = get_data(tickers, start_date, end_date)
    if returns is not None:
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        custom_return, custom_std_dev, custom_sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        results, weights_record = efficient_frontier(mean_returns, cov_matrix, num_portfolios, risk_free_rate)
        fig = plot_efficient_frontier(results, weights_record, weights, custom_return, custom_std_dev, tickers)
        st.plotly_chart(fig)
        st.subheader("Portfolio Performance Metrics")
        st.write(f"Expected Annual Return: {custom_return:.3%}")
        st.write(f"Expected Volatility: {custom_std_dev:.3%}")
        st.write(f"Sharpe Ratio: {(custom_return - risk_free_rate) / custom_std_dev:.3f}")
        portfolio_summary_table(results, weights_record, tickers)
        st.subheader("Optimized & Select Portfolio Backtest")
        portfolio_backtest(results, weights_record, weights, tickers, start_date, end_date)
        st.subheader("Optimized & Selected Portfolio Alpha,Beta")
        display_alpha_beta(results, weights_record, weights, tickers, start_date, end_date, risk_free_rate)
        st.subheader("Correlation Matrix")
        display_correlation_matrix(returns, tickers)
        st.subheader("Predictive Strongest Performers (Mean-Return Analysis)")
        display_top_performers(mean_returns, tickers)
        st.subheader("Subportfolio Header")
        max_sharpe_idx = np.argmax(results[2])
        optimized_portfolio_weights = weights_record[max_sharpe_idx]
        df = pd.DataFrame([tickers, optimized_portfolio_weights])
        # Scale the optimized portfolio weights
        scaled_weights = optimized_portfolio_weights * sub_portfolio_percentage

        # Create a DataFrame to display the stock tickers and their scaled weights
        scaled_weights_df = pd.DataFrame({
            "Ticker": tickers,
            "Weight (%)": scaled_weights * 100
        })

        # Display the table in the app
        st.table(scaled_weights_df)

        # Option to download the scaled weights as CSV
        csv_scaled = scaled_weights_df.to_csv(index=False)
        st.download_button(
            label="Download Sub-portfolio CSV",
            data=csv_scaled,
            file_name="sub-portfolio.csv",
            mime="text/csv"
        )

        # Download CSV for Optimized Portfolio
        # Create a DataFrame with two rows: one for tickers and one for weights
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False, header=False)
        # Convert the CSV string into a BytesIO object
        csv_file = io.StringIO(csv)
        # Add download button for CSV file
        st.download_button(
            label="Download Optimized Portfolio CSV",
            data=csv_file.getvalue(),
            file_name="optimized-portfolio.csv",
            mime="text/csv"
        )

      


            

# Page Information Ref. -- footer
linkedin_image = f"""
<div style="
    position: fixed;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;
    text-align: center;
    padding: 10px;
    font-size: small;
    ">
    © 2024 Merrimack Computer Club | <a href="https://github.com/Merrimack-Computer-Club" target="_blank">Github</a> Authors Alexander Elguezabal <a href="{AUTHOR_URL}" target="_blank">LinkedIn</a>

</div>
"""

# Render the clickable image
st.markdown(linkedin_image, unsafe_allow_html=True)