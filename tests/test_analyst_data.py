from io import StringIO
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests
from curl_cffi.requests.exceptions import RequestException as YahooRequestError
from yfinance.exceptions import YFRateLimitError

from analyst_data import (
    INDEX_SOURCES,
    STOCK_COLUMNS,
    IndexDataError,
    get_index_tickers,
    get_stock_data,
    get_stock_row,
    normalize_ticker,
    parse_constituents,
    summarize_stock,
)


def history(values=(80, 100, 110, 120), tz=None):
    return pd.DataFrame(
        {"Close": values},
        index=pd.DatetimeIndex(
            ["2024-06-28", "2024-12-31", "2025-05-30", "2025-06-30"], tz=tz
        ),
    )


class ConstituentTests(unittest.TestCase):
    def test_yahoo_symbol_formats(self):
        cases = [
            ("BRK.B", "S&P 500 (USA)", "BRK-B"),
            ("BF.B[1]", "Russell 1000", "BF-B"),
            (" aapl ", "NASDAQ-100 (USA)", "AAPL"),
            ("MMM", "Dow Jones (USA)", "MMM"),
            ("AZN", "FTSE 100 (UK)", "AZN.L"),
            ("AZN.L", "FTSE 100 (UK)", "AZN.L"),
            ("BT.A", "FTSE 100 (UK)", "BT-A.L"),
            ("RR.", "FTSE 100 (UK)", "RR.L"),
            ("7203", "Nikkei 225 (Japan)", "7203.T"),
            ("7203.T", "Nikkei 225 (Japan)", "7203.T"),
            (7203.0, "Nikkei 225 (Japan)", "7203.T"),
            ("285A", "Nikkei 225 (Japan)", "285A.T"),
            (600000, "SSE Composite (China)", "600000.SS"),
            ("600000.SS", "SSE Composite (China)", "600000.SS"),
        ]
        for value, index_name, expected in cases:
            with self.subTest(value=value, index_name=index_name):
                self.assertEqual(normalize_ticker(value, index_name), expected)

    def test_company_names_are_not_shanghai_tickers(self):
        with self.assertRaises(IndexDataError):
            normalize_ticker("Pudong Development Bank", "SSE Composite (China)")

    def test_every_index_schema(self):
        cases = [
            ("S&P 500 (USA)", "Symbol", "Security", "BRK.B", "BRK-B"),
            ("NASDAQ-100 (USA)", "Ticker", "Company", "AAPL", "AAPL"),
            ("Russell 1000", "Symbol", "Company", "AAPL", "AAPL"),
            ("Dow Jones (USA)", "Symbol", "Company", "MMM", "MMM"),
            ("FTSE 100 (UK)", "Ticker", "Company", "AZN", "AZN.L"),
            ("Nikkei 225 (Japan)", "Code", "Company Name", "7203.T", "7203.T"),
            (
                "SSE Composite (China)", "\u6210\u4efd\u5238\u4ee3\u7801Constituent Code",
                "\u6210\u4efd\u5238\u82f1\u6587\u540d\u79f0Constituent Name(Eng)",
                "600000", "600000.SS",
            ),
        ]
        for name, ticker_col, company_col, ticker, expected in cases:
            with self.subTest(index=name):
                frame = pd.DataFrame({ticker_col: [ticker], company_col: ["Example"]})
                result = parse_constituents([frame], name)
                self.assertEqual(
                    result.to_dict("records"), [{"Ticker": expected, "Company": "Example"}]
                )

    def test_html_table_order_and_ids_do_not_matter(self):
        html = """
        <table id="constituents"><tr><th>Year</th><th>Return</th></tr>
        <tr><td>2024</td><td>5</td></tr></table>
        <table><tr><th>Company</th><th>Ticker[1]</th></tr>
        <tr><td>Example</td><td>AZN</td></tr>
        <tr><td>Example</td><td>AZN</td></tr></table>
        """
        result = parse_constituents(pd.read_html(StringIO(html)), "FTSE 100 (UK)")
        self.assertEqual(result["Ticker"].tolist(), ["AZN.L"])

    def test_collects_all_matching_tables(self):
        tables = [
            pd.DataFrame({"Code": ["7203.T"], "Company Name": ["Toyota"]}),
            pd.DataFrame({"Code": ["6758.T"], "Company Name": ["Sony"]}),
        ]
        result = parse_constituents(tables, "Nikkei 225 (Japan)")
        self.assertEqual(result["Ticker"].tolist(), ["7203.T", "6758.T"])

    def test_missing_or_incomplete_table_fails_explicitly(self):
        for frame in [
            pd.DataFrame({"Name": ["Example"]}),
            pd.DataFrame(columns=["Symbol", "Company"]),
            pd.DataFrame({"Symbol": [None], "Company": ["Example"]}),
            pd.DataFrame({"Symbol": ["AAPL"], "Company": [" "]}),
        ]:
            with self.subTest(columns=list(frame.columns)):
                with self.assertRaises(IndexDataError):
                    parse_constituents([frame], "Russell 1000")

    @patch("analyst_data.requests.get")
    def test_loads_relocated_wikipedia_pages_with_timeout(self, get):
        get.return_value.text = (
            "<table><tr><th>Symbol</th><th>Company</th></tr>"
            "<tr><td>AAPL</td><td>Apple</td></tr></table>"
        )
        for name, page in [
            ("NASDAQ-100 (USA)", "List_of_NASDAQ-100_companies"),
            ("Russell 1000", "List_of_Russell_1000_companies"),
            ("Dow Jones (USA)", "List_of_Dow_Jones_Industrial_Average_companies"),
        ]:
            with self.subTest(index=name):
                get_index_tickers(name)
                self.assertTrue(get.call_args.args[0].endswith(page))
                self.assertEqual(get.call_args.kwargs["timeout"], 30)
                get.return_value.raise_for_status.assert_called()

    @patch("analyst_data.pd.read_excel")
    @patch("analyst_data.requests.get")
    def test_sse_uses_official_workbook_not_company_names(self, get, read_excel):
        get.return_value.content = b"workbook"
        read_excel.return_value = pd.DataFrame({
            "Constituent Code": ["600000"],
            "Constituent Name(Eng)": ["Pudong Bank"],
        })
        result = get_index_tickers("SSE Composite (China)")
        self.assertEqual(result["Ticker"].tolist(), ["600000.SS"])
        self.assertEqual(read_excel.call_args.kwargs, {"engine": "xlrd", "dtype": str})
        self.assertIn("000001cons.xls", get.call_args.args[0])

    @patch("analyst_data.requests.get")
    def test_http_and_parse_errors_are_not_empty_successes(self, get):
        get.return_value.raise_for_status.side_effect = requests.HTTPError("503")
        with self.assertRaisesRegex(IndexDataError, "503"):
            get_index_tickers("Russell 1000")
        get.return_value.raise_for_status.side_effect = None
        get.return_value.text = "<html>Unavailable</html>"
        with self.assertRaises(IndexDataError):
            get_index_tickers("Russell 1000")
        get.side_effect = requests.Timeout("timed out")
        with self.assertRaisesRegex(IndexDataError, "timed out"):
            get_index_tickers("Russell 1000")

    @patch("analyst_data.requests.get")
    def test_unknown_index_does_not_make_a_request(self, get):
        with self.assertRaises(IndexDataError):
            get_index_tickers("Unknown")
        get.assert_not_called()


class StockTests(unittest.TestCase):
    def test_returns_use_prior_closes_across_timezones(self):
        for tz in [None, "America/New_York", "Europe/London", "Asia/Tokyo"]:
            with self.subTest(tz=tz):
                row = summarize_stock("TEST", "Example", history(tz=tz), 150, "2025-06-30")
                self.assertEqual(row["Current Price"], 120)
                self.assertEqual(row["Last Close"], 110)
                self.assertAlmostEqual(row["30D Change"], (120 / 110 - 1) * 100)
                self.assertAlmostEqual(row["YTD Change"], 20)
                self.assertAlmostEqual(row["1Y Change"], 50)
                self.assertAlmostEqual(row["Upside to Target"], 25)

    def test_new_listing_keeps_prices_and_leaves_missing_periods_blank(self):
        row = summarize_stock("NEW", "New listing", history().iloc[-2:], 150, "2025-06-30")
        self.assertEqual(row["Current Price"], 120)
        self.assertTrue(np.isnan(row["YTD Change"]))
        self.assertTrue(np.isnan(row["1Y Change"]))
        recent = pd.DataFrame(
            {"Close": [110, 120]}, index=pd.to_datetime(["2025-06-27", "2025-06-30"])
        )
        row = summarize_stock("NEW", "New listing", recent, 150, "2025-06-30")
        self.assertTrue(np.isnan(row["30D Change"]))
        with self.assertRaises(ValueError):
            summarize_stock("NEW", "New listing", history().iloc[-2:], 150, "2025-06-01")

    def test_insufficient_prices_fail_explicitly(self):
        for frame in [pd.DataFrame(), history().iloc[-1:], history([0, None, -1, np.inf])]:
            with self.subTest(frame=frame):
                with self.assertRaises(ValueError):
                    summarize_stock("TEST", "Example", frame, 150, "2025-06-30")

    def test_invalid_targets_do_not_drop_prices_or_create_infinity(self):
        for target in [None, np.nan, np.inf, 0, -1, "unavailable"]:
            with self.subTest(target=target):
                row = summarize_stock("TEST", "Example", history(), target, "2025-06-30")
                self.assertEqual(row["Current Price"], 120)
                self.assertTrue(np.isnan(row["Mean Analyst Target"]))
                self.assertTrue(np.isnan(row["Upside to Target"]))

    def test_new_year_before_first_trading_day_does_not_crash(self):
        row = summarize_stock("TEST", "Example", history(), 150, "2025-01-01")
        self.assertEqual(row["Current Price"], 100)
        self.assertEqual(row["YTD Change"], 0)

    def test_invalid_and_unsorted_prices_are_handled(self):
        frame = history([80, 100, np.nan, 120]).iloc[::-1]
        row = summarize_stock("TEST", "Example", frame, 150, "2025-06-30")
        self.assertEqual(row["Current Price"], 120)
        self.assertEqual(row["Last Close"], 100)

    @patch("analyst_data.yf.Ticker")
    def test_quote_requests_use_unadjusted_close_and_target_endpoint(self, ticker):
        ticker.return_value.history.return_value = history()
        ticker.return_value.get_analyst_price_targets.return_value = {"mean": 150}
        row, issues = get_stock_row("TEST", "Example", "2025-06-30")
        self.assertEqual(issues, [])
        self.assertAlmostEqual(row["Upside to Target"], 25)
        ticker.return_value.history.assert_called_once_with(
            period="2y", auto_adjust=False, raise_errors=True, timeout=20
        )
        ticker.return_value.get_analyst_price_targets.assert_called_once()

    @patch("analyst_data.yf.Ticker")
    def test_target_failure_preserves_price_row(self, ticker):
        ticker.return_value.history.return_value = history()
        ticker.return_value.get_analyst_price_targets.side_effect = YahooRequestError("offline")
        row, issues = get_stock_row("TEST", "Example", "2025-06-30")
        self.assertEqual(row["Current Price"], 120)
        self.assertTrue(np.isnan(row["Mean Analyst Target"]))
        self.assertIn("offline", issues[0]["Issue"])

    @patch("analyst_data.yf.Ticker")
    def test_missing_coverage_preserves_price_row(self, ticker):
        ticker.return_value.history.return_value = history()
        for targets in [None, {}, {"mean": None}]:
            with self.subTest(targets=targets):
                ticker.return_value.get_analyst_price_targets.return_value = targets
                row, issues = get_stock_row("TEST", "Example", "2025-06-30")
                self.assertEqual(row["Current Price"], 120)
                self.assertTrue(np.isnan(row["Mean Analyst Target"]))
                self.assertEqual(issues, [])

    @patch("analyst_data.yf.Ticker")
    def test_history_failure_is_reported(self, ticker):
        ticker.return_value.history.side_effect = YahooRequestError("offline")
        row, issues = get_stock_row("TEST", "Example")
        self.assertIsNone(row)
        self.assertIn("Price history unavailable", issues[0]["Issue"])
        ticker.return_value.get_analyst_price_targets.assert_not_called()

    @patch("analyst_data.yf.Ticker")
    def test_rate_limits_propagate_instead_of_dropping_stocks(self, ticker):
        ticker.return_value.history.side_effect = YFRateLimitError()
        with self.assertRaises(YFRateLimitError):
            get_stock_row("TEST", "Example")
        ticker.return_value.history.side_effect = None
        ticker.return_value.history.return_value = history()
        ticker.return_value.get_analyst_price_targets.side_effect = YFRateLimitError()
        with self.assertRaises(YFRateLimitError):
            get_stock_row("TEST", "Example", "2025-06-30")

    @patch("analyst_data.get_stock_row")
    def test_batch_retains_successes_reports_failures_and_updates_progress(self, get_row):
        row = summarize_stock("OK", "Example", history(), 150, "2025-06-30")
        get_row.side_effect = lambda ticker, company: (
            (row, []) if ticker == "OK" else (None, [{"Ticker": ticker, "Issue": "offline"}])
        )
        progress = Mock()
        result, issues = get_stock_data(
            pd.DataFrame({"Ticker": ["OK", "BAD", "OK"], "Company": ["A", "B", "A"]}),
            progress=progress,
        )
        self.assertEqual(result["Ticker"].tolist(), ["OK"])
        self.assertEqual(issues["Ticker"].tolist(), ["BAD"])
        self.assertEqual(get_row.call_count, 2)
        progress.assert_called_with(2, 2)

    def test_empty_batch_has_stable_columns(self):
        result, issues = get_stock_data(pd.DataFrame(columns=["Ticker", "Company"]))
        self.assertEqual(list(result.columns), STOCK_COLUMNS)
        self.assertEqual(list(issues.columns), ["Ticker", "Issue"])


if __name__ == "__main__":
    unittest.main()
