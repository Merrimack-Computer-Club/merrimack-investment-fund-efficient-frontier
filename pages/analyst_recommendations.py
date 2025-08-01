import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta, timezone
from io import StringIO

# --- Page Config ---
st.set_page_config(page_title="Global Analyst Reversion Tool", layout="wide")

# --- Always-Visible Description ---
st.title("🌐 Global Analyst Mean Reversion Tool")
st.markdown("""
Welcome to the **Global Analyst Reversion Tool**!  
This app helps identify potential mean reversion opportunities by comparing a stock’s current price to the mean analyst target, along with recent performance data.

👉 Select a market from the sidebar to get started.
""")

# --- Sidebar Selection ---
index_option = st.sidebar.selectbox(
    "Choose a Market Index",
    options=["", "S&P 500 (USA)", "NASDAQ-100 (USA)", "Dow Jones (USA)", "FTSE 100 (UK)", "Nikkei 225 (Japan)", "SSE Composite (China)"]
)

# --- Ticker Loaders ---
@st.cache_data(show_spinner=False)
def get_index_tickers(index_name):
    try:
        if index_name == "S&P 500 (USA)":
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'id': 'constituents'})
            df = pd.read_html(StringIO(str(table)))[0]
            df = df[['Symbol', 'Security']].rename(columns={'Symbol': 'Ticker', 'Security': 'Company'})

        elif index_name == "FTSE 100 (UK)":
            url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find_all('table')[6]  # Fourth table is correct
            df = pd.read_html(StringIO(str(table)))[0]
            df = df[['Company', 'Ticker']].rename(columns={'Company': 'Company', 'Ticker': 'Ticker'})

        elif index_name == "NASDAQ-100 (USA)":
            url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
            soup = BeautifulSoup(requests.get(url).text, 'html.parser')
            table = soup.find('table', {'id': 'constituents'})
            df = pd.read_html(StringIO(str(table)))[0]
            df = df[['Ticker', 'Company']]

        elif index_name == "Dow Jones (USA)":
            url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
            soup = BeautifulSoup(requests.get(url).text, 'html.parser')
            table = soup.find('table', {'id': 'constituents'})
            df = pd.read_html(StringIO(str(table)))[0]
            df = df[['Symbol', 'Company']].rename(columns={'Symbol': 'Ticker'})

        elif index_name == "Nikkei 225 (Japan)":
            # Using list from TopForeignStocks.com with validated tickers :contentReference[oaicite:1]{index=1}
            url = "https://topforeignstocks.com/indices/the-components-of-the-nikkei-225-index/"
            soup = BeautifulSoup(requests.get(url).text, 'html.parser')
            table = soup.find('table')
            df = pd.read_html(StringIO(str(table)))[0]
            df = df[['Code', 'Company']].rename(columns={'Code': 'Ticker'})
            df['Ticker'] = df['Ticker'].astype(str).str.zfill(4) + '.T'

        elif index_name == "SSE Composite (China)":
            # Use table from Investing.com or TradingView for Shanghai Composite :contentReference[oaicite:2]{index=2}
            url = "https://www.investing.com/indices/shanghai-composite-components"
            soup = BeautifulSoup(requests.get(url).text, 'html.parser')
            table = soup.find('table')
            df = pd.read_html(StringIO(str(table)))[0]
            df = df.rename(columns={'Name': 'Company'})
            df['Ticker'] = df['Name'].apply(lambda x: x.replace(' ', '')) + '.SS'
            df = df[['Ticker', 'Company']]

        else:
            return pd.DataFrame()

        return df[['Ticker', 'Company']]

    except Exception as e:
        st.error(f"❌ Failed to load {index_name} tickers: {e}")
        return pd.DataFrame()

# --- Stock Data Fetcher ---
@st.cache_data(show_spinner=True)
def get_stock_data(tickers):
    results = []
    end_date = datetime.now(timezone.utc)
    thirty_days_ago = end_date - timedelta(days=30)
    ytd_start = datetime(end_date.year, 1, 1, tzinfo=timezone.utc)

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")

            if hist.empty or len(hist) < 2:
                continue

            current_price = hist['Close'].iloc[-1]
            last_close = hist['Close'].iloc[-2]

            price_30d_ago = hist[hist.index >= thirty_days_ago]['Close'].iloc[0]
            price_ytd = hist[hist.index >= ytd_start]['Close'].iloc[0]
            price_year_ago = hist['Close'].iloc[0]

            mean_target = stock.info.get('targetMeanPrice', np.nan)

            results.append({
                "Ticker": ticker,
                "Company": stock.info.get("shortName", ""),
                "Current Price": current_price,
                "Last Close": last_close,
                "30D Change": ((current_price - price_30d_ago) / price_30d_ago) * 100,
                "YTD Change": ((current_price - price_ytd) / price_ytd) * 100,
                "1Y Change": ((current_price - price_year_ago) / price_year_ago) * 100,
                "Mean Analyst Target": mean_target,
                "Upside to Target": ((mean_target - current_price) / current_price) * 100
            })

        except Exception as e:
            print(f"Error for {ticker}: {e}")
            continue

    df = pd.DataFrame(results)

    return df

# --- Load and Display Table ---
if index_option:
    with st.spinner(f"Loading {index_option} data..."):
        index_df = get_index_tickers(index_option)
        if not index_df.empty:
            stock_df = get_stock_data(index_df['Ticker'].tolist())
            if not stock_df.empty:
                st.markdown(f"### 📈 {index_option} Stock Summary")
                st.dataframe(
                    stock_df.sort_values(by="Upside to Target", ascending=False).reset_index(drop=True),
                    use_container_width=True
                )
            else:
                st.warning("No valid stock data retrieved.")
        else:
            st.warning("No tickers found for selected index.")
