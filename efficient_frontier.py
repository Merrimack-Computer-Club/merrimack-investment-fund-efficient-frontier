from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

import portfolio_analysis as pa


AUTHOR_URL = "https://www.linkedin.com/in/alex-elguezabal/"
WEBSITE_URL = "https://merrimackcomputer.club/"
CHART_LABEL_FONT = dict(size=19, weight=700)

st.set_page_config(
    page_title="Merrimack Computer Club Portfolio Analysis Tool",
    layout="wide", page_icon="assets/merrimack-computer-club.png",
)
st.sidebar.caption(
    "Appearance: use the top-right menu > Settings > Theme to choose light or dark."
)

mcc_image = f"""
<a href="{WEBSITE_URL}" target="_blank">
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSixZF48ZnadAhRjcI-4_g8pCN4nDtLYabSNg&s" alt="Merrimack Computer Club Website" style="width:100; text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);">
</a>
"""
st.markdown(mcc_image, unsafe_allow_html=True)
st.title("Merrimack College Student Managed Fund - Portfolio Analysis Tool")
st.markdown("""
This tool helps users visualize and optimize investment portfolios using the
**efficient frontier**: the lowest-risk portfolios for a range of estimated returns.

## Key Features

- **Efficient Frontier**: Optimize fully invested, long-only allocations.
- **Random Portfolio Samples**: Compare possible allocations, not forecasts of future prices.
- **Portfolio Customization**: Adjust weights and compare annual return, volatility, and Sharpe ratio.
- **Market Comparison**: Compare historical daily and cumulative returns with a market index.
- **Alpha & Beta Analysis**: Compare annualized excess return and sensitivity to the market.
- **Fama-French Analysis**: Estimate market, size, and value factor exposures.
- **Interactive Tools**: Upload/download portfolio CSVs and set an annual risk-free rate.
- **Visualization**: Explore the frontier, asset correlations, and historical mean returns.

Estimates describe the selected historical sample; they do not guarantee future performance.
""")

market_options = {
    "S&P 500": "^GSPC", "DJIA": "^DJI",
    "Russell 1000": "^RUI", "NASDAQ": "^IXIC",
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers, start_date, end_date):
    return pa.download_prices(tickers, start_date, end_date)


@st.cache_data(ttl=3600, show_spinner=False)
def load_risk_free_rate():
    return pa.get_risk_free_rate()


@st.cache_data(ttl=3600, show_spinner=False)
def load_asset_name(ticker):
    return pa.get_asset_name(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fama_french_factors():
    return pa.download_fama_french_factors()


def sharpe_text(value):
    return f"{value:.3f}" if value is not None else "Undefined (zero volatility)"


def weights_text(weights, separator=", "):
    return separator.join(f"{ticker}: {weight:.2%}" for ticker, weight in weights.items())


def portfolio_hover(portfolio):
    return (
        f"{weights_text(portfolio.weights, '<br>')}<br>"
        f"Annual Return: {portfolio.annual_return:.2%}<br>"
        f"Annual Volatility: {portfolio.volatility:.2%}<br>"
        f"Sharpe Ratio: {sharpe_text(portfolio.sharpe)}"
    )


def portfolio_table(portfolios, names=None):
    frame = pd.DataFrame([{
        "Expected Annual Return": portfolio.annual_return,
        "Expected Volatility": portfolio.volatility,
        "Sharpe Ratio": portfolio.sharpe,
        "Weights": weights_text(portfolio.weights),
    } for portfolio in portfolios])
    if names is not None:
        frame.insert(0, "Portfolio", names)
    return frame


def styled_metrics(frame):
    return frame.style.format({
        "Expected Annual Return": "{:.2%}",
        "Expected Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.4f}",
    }, na_rep="Undefined")


def chart_figure(data=None):
    fig = go.Figure(data, layout=dict(template="streamlit"))
    # Native Plotly axis gray needs large, bold text to meet contrast in light mode.
    fig.update_layout(
        font=CHART_LABEL_FONT,
        xaxis=dict(
            tickfont=CHART_LABEL_FONT, nticks=4,
            title=dict(font=CHART_LABEL_FONT, standoff=12),
        ),
        yaxis=dict(
            tickfont=CHART_LABEL_FONT, nticks=4,
            title=dict(font=CHART_LABEL_FONT, standoff=12),
        ),
        hoverlabel=dict(font=CHART_LABEL_FONT),
        legend=dict(orientation="h", x=0, xanchor="left", y=-0.25, yanchor="top"),
        margin=dict(t=75, b=155),
        height=520,
    )
    return fig


def plot_efficient_frontier(optimization, samples, custom):
    colors = [portfolio.sharpe if portfolio.sharpe is not None else np.nan for portfolio in samples]
    has_sharpe = any(portfolio.sharpe is not None for portfolio in samples)
    fig = chart_figure()
    palette = fig.layout.template.layout.colorway
    fig.add_trace(go.Scatter(
        x=[portfolio.volatility for portfolio in samples],
        y=[portfolio.annual_return for portfolio in samples],
        mode="markers", name="Random portfolios (not optimized)",
        marker=dict(
            color=colors if has_sharpe else palette[4],
            colorscale=fig.layout.template.layout.colorscale.sequential,
            size=5, showscale=has_sharpe,
            colorbar=dict(
                title=dict(text="Sharpe", font=CHART_LABEL_FONT),
                tickfont=CHART_LABEL_FONT, nticks=4, thickness=12, outlinewidth=0,
            ),
        ),
        text=[portfolio_hover(portfolio) for portfolio in samples], hoverinfo="text",
    ))
    fig.add_trace(go.Scatter(
        x=[portfolio.volatility for portfolio in optimization.frontier],
        y=[portfolio.annual_return for portfolio in optimization.frontier],
        mode="lines+markers", name="Efficient frontier",
        line=dict(color=palette[0], width=3),
        text=[portfolio_hover(portfolio) for portfolio in optimization.frontier],
        hoverinfo="text",
    ))
    highlights = [("Custom Portfolio", custom, palette[2], "star")]
    if optimization.maximum_sharpe is not None:
        highlights.append(("Maximum Sharpe Portfolio", optimization.maximum_sharpe, palette[1], "diamond"))
    for name, portfolio, color, symbol in highlights:
        fig.add_trace(go.Scatter(
            x=[portfolio.volatility], y=[portfolio.annual_return], name=name,
            mode="markers", marker=dict(color=color, size=12, symbol=symbol),
            text=[portfolio_hover(portfolio)], hoverinfo="text",
        ))
    fig.update_layout(
        title="Efficient Frontier with Custom Portfolio",
        xaxis=dict(title=dict(text="Annual Volatility (Std Dev)"), tickformat=".0%"),
        yaxis=dict(title=dict(text="Expected Annual Return"), tickformat=".0%"),
        showlegend=True,
    )
    return fig


def plot_returns(frame, title, axis_title):
    fig = chart_figure()
    palette = fig.layout.template.layout.colorway
    styles = {"Optimized Portfolio": (1, "solid"), "Selected Portfolio": (2, "dash")}
    for name in frame:
        color_index, dash = styles.get(name, (3, "dot"))
        fig.add_trace(go.Scatter(
            x=frame.index, y=frame[name], mode="lines", name=name,
            line=dict(color=palette[color_index], dash=dash),
        ))
    fig.update_layout(
        title=title, xaxis_title_text="Date",
        yaxis=dict(title=dict(text=axis_title), tickformat=".1%"),
    )
    st.plotly_chart(fig, width="stretch", theme="streamlit")


def display_historical_comparison(prices, selected_weights, optimized_weights, market, start, end, rate):
    st.subheader("Historical Portfolio & Market Comparison")
    st.caption(
        "In-sample (hindsight) comparison: optimized weights were chosen using the estimation history "
        "above, not a separate training period. Both portfolios rebalance to their fixed weights each trading day. "
        "Trading costs and taxes are excluded. Stocks use adjusted closes; the selected price index "
        "does not include reinvested dividends. No currency conversion is performed."
    )
    try:
        benchmark = load_prices([market_options[market]], start, end)
        comparison, cumulative = pa.historical_comparison(
            prices, benchmark, selected_weights, optimized_weights, market
        )
    except pa.PortfolioError as error:
        st.error(f"Historical comparison unavailable: {error}")
        return
    st.caption(
        f"Common comparison period: {cumulative.index[0]:%Y-%m-%d} to "
        f"{cumulative.index[-1]:%Y-%m-%d} ({len(comparison)} daily returns). "
        "All cumulative returns start at zero on the same closing date."
    )
    plot_returns(comparison, f"Daily Returns vs {market}", "Daily Return")
    plot_returns(cumulative, f"Cumulative Returns vs {market}", "Cumulative Return")
    st.subheader("Portfolio Alpha & Beta")
    try:
        rows = []
        for name in comparison.columns[:-1]:
            alpha, beta = pa.calculate_alpha_beta(comparison[name], comparison[market], rate)
            rows.append({"Portfolio": name, "Annualized Alpha": alpha, "Beta": beta})
    except pa.PortfolioError as error:
        st.warning(str(error))
    else:
        st.caption(
            "Alpha is the annualized daily excess-return intercept; beta measures sensitivity "
            "to the market. The annual risk-free input is divided by 252 for daily calculations."
        )
        st.dataframe(
            pd.DataFrame(rows).style.format({"Annualized Alpha": "{:.2%}", "Beta": "{:.3f}"}),
            hide_index=True, width="stretch",
        )


def factor_exposure_summary(name, regression):
    if regression.market_beta > 1.1:
        market = "above-market sensitivity"
    elif regression.market_beta < 0.9:
        market = "below-market sensitivity"
    else:
        market = "roughly market-level sensitivity"
    if regression.smb_beta > 0.1:
        size = "a small-cap tilt"
    elif regression.smb_beta < -0.1:
        size = "a large-cap tilt"
    else:
        size = "little size tilt"
    if regression.hml_beta > 0.1:
        value = "a value tilt"
    elif regression.hml_beta < -0.1:
        value = "a growth tilt"
    else:
        value = "little value/growth tilt"
    return f"**{name}:** {market}, {size}, and {value} over this sample."


def display_fama_french_analysis(returns, selected_weights, optimized_weights):
    st.subheader("Fama-French Three-Factor Analysis")
    st.markdown(
        "The regression estimates `portfolio excess return = alpha + "
        "market beta * (Mkt-RF) + size beta * SMB + value beta * HML`. "
        "SMB is the historical return of small companies minus big companies; "
        "HML is high book-to-market (value) minus low book-to-market (growth)."
    )
    try:
        factors = load_fama_french_factors()
        portfolios = [("Selected Portfolio", selected_weights)]
        if optimized_weights is not None:
            portfolios.insert(0, ("Optimized Portfolio", optimized_weights))
        results = [
            (name, pa.fama_french_regression(pa.weighted_returns(returns, weights), factors))
            for name, weights in portfolios
        ]
    except pa.PortfolioError as error:
        st.warning(f"Fama-French analysis unavailable: {error}")
        return

    rows = [{
        "Portfolio": name,
        "Annualized Alpha": result.annual_alpha,
        "Market Beta": result.market_beta,
        "SMB Beta": result.smb_beta,
        "HML Beta": result.hml_beta,
        "R-squared": result.r_squared,
        "Observations": result.observations,
    } for name, result in results]
    st.dataframe(
        pd.DataFrame(rows).style.format({
            "Annualized Alpha": "{:.2%}",
            "Market Beta": "{:.3f}",
            "SMB Beta": "{:.3f}",
            "HML Beta": "{:.3f}",
            "R-squared": "{:.1%}",
        }, na_rep="Undefined"),
        hide_index=True, width="stretch",
    )
    for name, result in results:
        st.markdown(factor_exposure_summary(name, result))
    first_result = results[0][1]
    st.caption(
        f"Factor overlap: {first_result.start_date:%Y-%m-%d} to "
        f"{first_result.end_date:%Y-%m-%d}. Alpha is the daily intercept multiplied by 252; "
        "R-squared is the share of variation explained in-sample. The official US daily "
        "factor data supplies its own one-month T-bill risk-free return, so this regression "
        "does not use the sidebar's 10-year Treasury input. Exposures are historical "
        "co-movements, not forecasts or proof of manager skill. They are most applicable "
        "to US equities and may be misleading for international assets or mixed currencies. "
        "[Source: Kenneth French Data Library]"
        "(https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)."
    )


def display_results(tickers, weights, start, end, rate, sample_count, market, sub_weight):
    rate = pa.validate_risk_free_rate(rate)
    with st.spinner("Loading prices and solving portfolio allocations..."):
        prices = load_prices(tickers, start, end)
        returns = pa.daily_returns(prices)
        mean_returns, covariance = returns.mean(), returns.cov()
        custom = pa.portfolio_performance(weights, mean_returns, covariance, rate)
        optimization = pa.efficient_frontier(mean_returns, covariance, rate)
        samples = pa.sample_portfolios(mean_returns, covariance, rate, sample_count)
    if prices.attrs.get("trimmed_rows", 0):
        st.warning(
            "The available histories have different start or end dates. "
            "Calculations use only their shared history, shown below."
        )
    if np.linalg.matrix_rank(covariance.to_numpy()) < len(tickers):
        st.warning(
            "Some return histories are redundant or too short to estimate independent risks. "
            "Near-zero estimated risk may be misleading; use a longer history or remove redundant assets."
        )
    st.caption(
        f"Estimation period: {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}; "
        f"{len(returns)} daily returns. Annual estimates use 252 trading days. "
        "Expected return is an annualized arithmetic mean, not compound growth or a forecast."
    )
    st.plotly_chart(
        plot_efficient_frontier(optimization, samples, custom),
        width="stretch", theme="streamlit",
    )
    st.subheader("Portfolio Performance Metrics")
    st.write(f"Expected Annual Return: {custom.annual_return:.3%}")
    st.write(f"Expected Volatility: {custom.volatility:.3%}")
    st.write(f"Sharpe Ratio: {sharpe_text(custom.sharpe)}")

    optimized = optimization.maximum_sharpe
    summary = [optimization.minimum_variance]
    names = ["Minimum Variance"]
    if optimized is None:
        st.warning(
            "A finite maximum Sharpe ratio is unavailable because a near-zero-volatility "
            "portfolio is feasible. No maximum-Sharpe allocation or export is selected."
        )
    else:
        summary.insert(0, optimized)
        names.insert(0, "Maximum Sharpe Ratio (Optimized Portfolio)")
    if (mean_returns * pa.TRADING_DAYS <= rate).all():
        st.warning(
            "All estimated asset returns are at or below the risk-free rate. "
            "Maximizing a negative Sharpe ratio can favor higher risk; it is not a recommendation."
        )
    st.subheader("Portfolio Summary: Efficient Frontier")
    summary_tab, samples_tab = st.tabs(["Summary", "Random Portfolio Data"])
    summary_tab.dataframe(styled_metrics(portfolio_table(summary, names)), hide_index=True)
    samples_tab.caption("Random allocations ranked by Sharpe ratio, not optimized frontier portfolios.")
    samples_tab.dataframe(
        styled_metrics(portfolio_table(samples).sort_values("Sharpe Ratio", ascending=False)),
        hide_index=True,
    )

    optimized_weights = optimized.weights if optimized is not None else None
    display_historical_comparison(prices, weights, optimized_weights, market, start, end, rate)
    display_fama_french_analysis(returns, weights, optimized_weights)
    st.subheader("Correlation Matrix")
    st.dataframe(returns.corr())
    st.caption("Correlation is undefined (blank) for an asset with zero volatility.")
    st.subheader("Historical Mean Returns")
    historical_means = (mean_returns * pa.TRADING_DAYS).sort_values(ascending=False).head(10)
    fig = chart_figure(go.Bar(x=historical_means.index, y=historical_means.values))
    fig.update_layout(
        title="Top Historical Mean Returns<br>(Not a Forecast)", xaxis_title_text="Ticker",
        xaxis=dict(tickmode="linear", dtick=1),
        yaxis=dict(title=dict(text="Annualized Arithmetic Mean Return"), tickformat=".1%"),
        margin=dict(b=100), height=450,
    )
    st.plotly_chart(fig, width="stretch", theme="streamlit")
    if optimized is not None:
        st.subheader("Optimized Sub-portfolio Allocation")
        st.caption(
            f"These allocations sum to {sub_weight:.1%} of a larger portfolio. "
            "The rest of that larger portfolio is not modeled."
        )
        scaled = pd.DataFrame({
            "Ticker": optimized.weights.index,
            "Weight (%)": optimized.weights.to_numpy() * sub_weight * 100,
        })
        st.table(scaled)
        st.download_button(
            "Download Sub-portfolio CSV", data=scaled.to_csv(index=False),
            file_name="sub-portfolio.csv", mime="text/csv",
        )
        st.download_button(
            "Download Optimized Portfolio CSV", data=portfolio_csv(optimized.weights),
            file_name="optimized-portfolio.csv", mime="text/csv",
        )


def portfolio_csv(weights):
    return pd.DataFrame([weights.index.tolist(), weights.tolist()]).to_csv(index=False, header=False)


with st.sidebar:
    st.header("Portfolio Selection")
    if "portfolio_tickers" not in st.session_state:
        st.session_state["portfolio_tickers"] = "AAPL,MSFT,GOOGL,AMZN,TSLA"
    uploaded_file = st.file_uploader("Upload Portfolio CSV", type=["csv"])
    st.caption("CSV format: first row tickers, optional second row non-negative weights.")
    if uploaded_file is not None:
        content = uploaded_file.getvalue()
        if content != st.session_state.get("portfolio_upload"):
            try:
                uploaded_tickers, uploaded_weights = pa.read_portfolio_csv(BytesIO(content))
            except pa.PortfolioError as error:
                st.error(str(error))
                st.stop()
            st.session_state["portfolio_tickers"] = ",".join(uploaded_tickers)
            st.session_state["portfolio_weights"] = uploaded_weights.to_dict()
            for ticker, weight in uploaded_weights.items():
                for mode in ("Slider", "Number Input"):
                    st.session_state[f"{mode}_weight_{ticker}"] = float(weight)
            st.session_state["portfolio_upload"] = content
        st.caption("Uploaded weights are matched by ticker and normalized to 100%.")
    else:
        st.session_state.pop("portfolio_upload", None)
    ticker_input = st.text_input("Enter Tickers (Comma Separated)", key="portfolio_tickers")
    try:
        tickers = pa.validate_tickers(ticker_input)
    except pa.PortfolioError as error:
        st.error(str(error))
        st.stop()

    start_date = st.date_input("Start Date", value=pd.Timestamp("2020-01-01"))
    today = pd.Timestamp.today().normalize()
    last_business_day = today - pd.Timedelta(days=max(today.weekday() - 4, 0))
    end_date = st.date_input("End Date", value=last_business_day)
    st.caption("Start Date is inclusive; End Date is exclusive.")
    try:
        pa.validate_dates(start_date, end_date)
    except pa.PortfolioError as error:
        st.error(str(error))
        st.stop()

    selected_market = st.selectbox("Select Market Index to Compare", list(market_options))
    try:
        current_rate = load_risk_free_rate()
    except pa.PortfolioError as error:
        current_rate = None
        st.warning(f"{error} Enter the annual risk-free rate manually before deploying.")
    else:
        st.text(f"Latest 10-year U.S. Treasury yield: {current_rate:.2%}")
    risk_free_rate = st.number_input(
        "Set Risk-Free Rate (Annual)", min_value=0.0, max_value=1.0,
        value=current_rate, step=0.0001, format="%.4f",
        help="Annual decimal rate: 0.04 means 4%. The Treasury yield is a proxy, not a guarantee.",
    )

    input_method = st.radio("Choose Input Method for Portfolio Weights", ("Slider", "Number Input"))
    st.subheader("Input Portfolio Weights")
    previous_weights = st.session_state.get("portfolio_weights", {})
    raw_weights = []
    for ticker in tickers:
        try:
            name = load_asset_name(ticker)
        except pa.PortfolioError as error:
            st.warning(str(error))
            name = ticker
        key = f"{input_method}_weight_{ticker}"
        if key not in st.session_state:
            default = 1 / len(tickers) if not previous_weights else 0.0
            st.session_state[key] = float(previous_weights.get(ticker, default))
        widget = st.slider if input_method == "Slider" else st.number_input
        raw_weights.append(widget(
            f"Weight for {ticker} ({name}):", min_value=0.0, max_value=1.0,
            format="%.4f", step=0.0001, key=key,
        ))
    st.session_state["portfolio_weights"] = dict(zip(tickers, raw_weights))
    try:
        weights = pa.normalize_weights(raw_weights, tickers)
    except pa.PortfolioError as error:
        st.error(str(error))
        st.stop()
    if not np.isclose(sum(raw_weights), 1, rtol=0, atol=1e-8):
        st.warning(
            f"Weights total {sum(raw_weights):.2%}. Calculations and CSV downloads use "
            "the same proportions, normalized to 100%."
        )
    sub_portfolio_percentage = st.number_input(
        "Set Sub-portfolio Weighting", min_value=0.0, max_value=1.0,
        value=0.25, step=0.0001, format="%.4f",
        help="Fraction of a larger portfolio assigned to the optimized allocation.",
    )
    st.download_button(
        "Download Portfolio CSV", data=portfolio_csv(weights),
        file_name="portfolio.csv", mime="text/csv",
    )

sample_method = st.radio("Random Portfolio Sample Input", ("Slider", "Number Input"))
sample_widget = st.slider if sample_method == "Slider" else st.number_input
num_portfolios = sample_widget(
    "Number of Random Portfolio Samples", min_value=100, max_value=10000, value=5000, step=100,
)
st.caption(
    "Random samples are visual context only. The efficient frontier and optimal allocations "
    "are solved separately and do not depend on the sample count."
)
if st.button("Deploy Efficient Frontier"):
    try:
        display_results(
            tickers, weights, start_date, end_date, risk_free_rate, int(num_portfolios),
            selected_market, sub_portfolio_percentage,
        )
    except pa.PortfolioError as error:
        st.error(str(error))

st.markdown(
    "© 2024 Merrimack Computer Club | "
    "[Github](https://github.com/Merrimack-Computer-Club) "
    f"Authors Alexander Elguezabal [LinkedIn]({AUTHOR_URL})"
)
