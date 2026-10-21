import streamlit as st
from yfinance.exceptions import YFRateLimitError

from analyst_data import (
    INDEX_SOURCES,
    IndexDataError,
    get_index_tickers,
    get_stock_data,
)


st.set_page_config(page_title="Global Analyst Reversion Tool", layout="wide")
st.title("🌐 Global Analyst Mean Reversion Tool")
st.markdown("""
Welcome to the **Global Analyst Reversion Tool**!
This app compares each stock in a market index with its mean analyst price target.
An index itself does not have a company analyst target, and some stocks have no
analyst coverage.

👉 Select a market from the sidebar to get started.
""")

index_option = st.sidebar.selectbox(
    "Choose a Market Index", options=["", *INDEX_SOURCES]
)


@st.cache_data(ttl=86400, show_spinner=False)
def load_constituents(index_name):
    return get_index_tickers(index_name)


@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(constituents):
    progress = st.progress(0, text="Loading stock prices and analyst targets...")
    try:
        return get_stock_data(
            constituents,
            progress=lambda done, total: progress.progress(
                done / total, text=f"Loaded {done} of {total} stocks..."
            ),
        )
    finally:
        progress.empty()


if index_option:
    if st.sidebar.button("Refresh data"):
        load_constituents.clear()
        load_stock_data.clear()

    st.caption(f"Constituent source: {INDEX_SOURCES[index_option]}")
    if index_option == "Nikkei 225 (Japan)":
        st.caption(
            "The Nikkei constituent source is dated January 2024; "
            "it may not reflect later index changes."
        )
    try:
        with st.spinner(f"Loading {index_option} data. Large indexes can take several minutes..."):
            index_df = load_constituents(index_option)
            stock_df, issues = load_stock_data(index_df)
    except IndexDataError as error:
        st.error(str(error))
        st.info("The constituent provider may be unavailable. Use Refresh data to retry.")
    except YFRateLimitError:
        st.error("Yahoo Finance is limiting requests. Wait a few minutes, then refresh the data.")
    else:
        if not issues.empty:
            st.warning(
                f"Data requests failed for {issues['Ticker'].nunique()} stocks. "
                "Available prices are retained when only the analyst target fails."
            )
            with st.expander("Data request details"):
                st.dataframe(issues, use_container_width=True, hide_index=True)
        if stock_df.empty:
            st.warning("No valid stock prices were retrieved. Try refreshing the data later.")
        else:
            covered = int(stock_df["Mean Analyst Target"].notna().sum())
            st.markdown(f"### 📈 {index_option} Stock Summary")
            st.caption(
                f"Prices loaded for {len(stock_df)} of {len(index_df)} constituents; "
                f"analyst targets available for {covered}. "
                "Prices and targets use each stock's quoted currency/unit, not a common currency. "
                "Current Price is the latest closing price. Changes exclude dividends. "
                "Blank values mean the required history or analyst target is unavailable."
            )
            if covered == 0:
                st.info("Yahoo Finance returned no analyst targets for these stocks.")
            st.dataframe(
                stock_df.sort_values(
                    by=["Upside to Target", "Ticker"],
                    ascending=[False, True],
                    na_position="last",
                ).reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
