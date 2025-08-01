import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="S&P 500 Analyst Reversion Tool", layout="wide")
st.title("📊 S&P 500 Analyst Mean Reversion Tool")

@st.cache_data(show_spinner=False)
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(str(table))[0]
    return df[['Symbol', 'Security']].rename(columns={"Symbol": "Ticker", "Security": "Company"})

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
                "Upside to Target": ((mean_target - current_price) / current_price) * 100 if pd.notna(mean_target) else np.nan
            })

        except Exception as e:
            print(f"Error for {ticker}: {e}")
            continue

    df = pd.DataFrame(results)

    # Format numeric columns
    for col in ["Current Price", "Last Close", "Mean Analyst Target"]:
        df[col] = df[col].map(lambda x: f"${x:.2f}" if pd.notna(x) else "—")

    for col in ["30D Change", "YTD Change", "1Y Change", "Upside to Target"]:
        df[col] = df[col].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")

    return df

sp500_df = get_sp500_tickers()
tickers = sp500_df['Ticker'].tolist()

with st.spinner("Fetching stock data (this may take a minute)..."):
    stock_df = get_stock_data(tickers)

if not stock_df.empty:
    st.markdown("### 📈 Stock Summary Table")
    st.dataframe(
        stock_df.sort_values(by="Upside to Target", ascending=False).reset_index(drop=True),
        use_container_width=True
    )
else:
    st.error("No stock data could be retrieved.")
