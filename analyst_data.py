"""Constituent lists and price comparisons for the analyst recommendations page."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO, StringIO
import re

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from curl_cffi.requests.exceptions import RequestException as YahooRequestError
from xlrd.biffh import XLRDError
from yfinance.exceptions import YFException, YFRateLimitError


INDEX_SOURCES = {
    "S&P 500 (USA)": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "NASDAQ-100 (USA)": "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
    "Russell 1000": "https://en.wikipedia.org/wiki/List_of_Russell_1000_companies",
    "Dow Jones (USA)": (
        "https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies"
    ),
    "FTSE 100 (UK)": "https://en.wikipedia.org/wiki/FTSE_100_Index",
    "Nikkei 225 (Japan)": (
        "https://topforeignstocks.com/indices/the-components-of-the-nikkei-225-index/"
    ),
    "SSE Composite (China)": (
        "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/"
        "file/autofile/cons/000001cons.xls"
    ),
}
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; MerrimackInvestmentFund/1.0; "
        "+https://github.com/Merrimack-Computer-Club/"
        "merrimack-investment-fund-efficient-frontier)"
    )
}
STOCK_COLUMNS = [
    "Ticker", "Company", "Current Price", "Last Close", "30D Change", "YTD Change",
    "1Y Change", "Mean Analyst Target", "Upside to Target",
]


class IndexDataError(ValueError):
    """The constituent provider could not supply a usable list."""


def normalize_ticker(value, index_name):
    ticker = re.sub(r"\[[^\]]*\]", "", str(value)).strip().upper()
    if index_name == "FTSE 100 (UK)":
        ticker = ticker.removesuffix(".L").rstrip(".").replace(".", "-") + ".L"
        pattern = r"[A-Z0-9]+(?:-[A-Z0-9]+)*\.L"
    elif index_name in ("Nikkei 225 (Japan)", "SSE Composite (China)"):
        suffix, width = (".T", 4) if index_name == "Nikkei 225 (Japan)" else (".SS", 6)
        ticker = ticker.removesuffix(suffix)
        ticker = re.sub(r"^(\d+)\.0$", r"\1", ticker).zfill(width) + suffix
        pattern = r"[0-9][0-9A-Z]{3}\.T" if suffix == ".T" else r"\d{6}\.SS"
    else:
        ticker = ticker.replace(".", "-")
        pattern = r"[A-Z0-9]+(?:-[A-Z0-9]+)*"
    if not re.fullmatch(pattern, ticker):
        raise IndexDataError(f"Invalid ticker {value!r} in the {index_name} source.")
    return ticker


def parse_constituents(tables, index_name):
    matches = []
    for table in tables:
        if isinstance(table.columns, pd.MultiIndex):
            continue
        table = table.copy()
        table.columns = [
            re.sub(r"\[[^\]]*\]", "", str(column)).strip()
            for column in table.columns
        ]
        ticker_column = next(
            (c for c in table.columns if c in ("Symbol", "Ticker", "Code")
             or c.endswith("Constituent Code")),
            None,
        )
        company_column = next(
            (c for c in table.columns if c in ("Security", "Company", "Company Name")
             or c.endswith("Constituent Name(Eng)")),
            None,
        )
        if ticker_column is not None and company_column is not None:
            matches.append(
                table[[ticker_column, company_column]].set_axis(
                    ["Ticker", "Company"], axis=1
                )
            )
    if not matches:
        raise IndexDataError(f"No constituent table found for {index_name}.")

    result = pd.concat(matches, ignore_index=True).dropna(how="all")
    if result.empty or result.isna().any().any():
        raise IndexDataError(f"The {index_name} constituent list is empty or incomplete.")
    result["Ticker"] = result["Ticker"].map(lambda value: normalize_ticker(value, index_name))
    result["Company"] = result["Company"].astype(str).str.strip()
    if result["Company"].eq("").any():
        raise IndexDataError(f"The {index_name} constituent list has missing company names.")
    return result.drop_duplicates("Ticker").reset_index(drop=True)


def get_index_tickers(index_name):
    if index_name not in INDEX_SOURCES:
        raise IndexDataError(f"Unsupported market index: {index_name}")
    try:
        response = requests.get(
            INDEX_SOURCES[index_name], headers=REQUEST_HEADERS, timeout=30
        )
        response.raise_for_status()
        if index_name == "SSE Composite (China)":
            tables = [pd.read_excel(BytesIO(response.content), engine="xlrd", dtype=str)]
        else:
            tables = pd.read_html(StringIO(response.text))
        return parse_constituents(tables, index_name)
    except (requests.RequestException, ValueError, XLRDError) as error:
        raise IndexDataError(f"Could not load {index_name}: {error}") from error


def positive_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) and number > 0 else np.nan


def summarize_stock(ticker, company, history, mean_target, as_of=None):
    if history.empty or "Close" not in history:
        raise ValueError("No closing-price history was returned.")
    prices = pd.to_numeric(history["Close"], errors="coerce")
    # Keep the exchange's trading dates, rather than shifting them to UTC dates.
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices = prices[np.isfinite(prices) & (prices > 0)].sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    today = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    today = today.tz_localize(None).normalize()
    prices = prices[prices.index <= today]
    if len(prices) < 2:
        raise ValueError("Fewer than two valid closing prices were returned.")

    current_price = float(prices.iloc[-1])

    def change_since(cutoff):
        baseline = prices[prices.index <= cutoff]
        if baseline.empty:
            return np.nan
        return (current_price / baseline.iloc[-1] - 1) * 100

    mean_target = positive_number(mean_target)
    return {
        "Ticker": ticker,
        "Company": company,
        "Current Price": current_price,
        "Last Close": float(prices.iloc[-2]),
        "30D Change": change_since(today - pd.Timedelta(days=30)),
        "YTD Change": change_since(today.replace(month=1, day=1) - pd.Timedelta(days=1)),
        "1Y Change": change_since(today - pd.DateOffset(years=1)),
        "Mean Analyst Target": mean_target,
        "Upside to Target": (mean_target / current_price - 1) * 100,
    }


def get_stock_row(ticker, company, as_of=None):
    stock = yf.Ticker(ticker)
    try:
        history = stock.history(
            period="2y", auto_adjust=False, raise_errors=True, timeout=20
        )
        row = summarize_stock(ticker, company, history, None, as_of)
    except YFRateLimitError:
        raise
    except (YFException, YahooRequestError, requests.RequestException, ValueError) as error:
        return None, [{"Ticker": ticker, "Issue": f"Price history unavailable: {error}"}]

    issues = []
    try:
        targets = stock.get_analyst_price_targets()
        if targets is not None and not isinstance(targets, dict):
            raise ValueError("Unexpected analyst-target response.")
        target = positive_number((targets or {}).get("mean"))
        row["Mean Analyst Target"] = target
        row["Upside to Target"] = (target / row["Current Price"] - 1) * 100
    except YFRateLimitError:
        raise
    except (YFException, YahooRequestError, requests.RequestException, ValueError) as error:
        issues.append({"Ticker": ticker, "Issue": f"Analyst target unavailable: {error}"})
    return row, issues


def get_stock_data(constituents, progress=None):
    results, issues = [], []
    executor = ThreadPoolExecutor(max_workers=4)
    futures = {
        executor.submit(get_stock_row, row.Ticker, row.Company): row.Ticker
        for row in constituents.drop_duplicates("Ticker").itertuples(index=False)
    }
    try:
        for count, future in enumerate(as_completed(futures), start=1):
            row, stock_issues = future.result()
            if row is not None:
                results.append(row)
            issues.extend(stock_issues)
            if progress is not None:
                progress(count, len(futures))
    finally:
        # In particular, stop queued requests immediately when Yahoo rate-limits us.
        executor.shutdown(wait=True, cancel_futures=True)
    return (
        pd.DataFrame(results, columns=STOCK_COLUMNS),
        pd.DataFrame(issues, columns=["Ticker", "Issue"]),
    )
