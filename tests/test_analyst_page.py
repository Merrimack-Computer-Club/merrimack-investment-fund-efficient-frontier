from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest
from yfinance.exceptions import YFRateLimitError

from analyst_data import INDEX_SOURCES, STOCK_COLUMNS, IndexDataError


PAGE = Path(__file__).resolve().parents[1].joinpath("pages", "analyst_recommendations.py")


class AnalystPageTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        self.constituents = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT"], "Company": ["Apple", "Microsoft"],
        })
        self.stocks = pd.DataFrame([
            ["AAPL", "Apple", 100, 99, 1, 2, 3, 120, 20],
            ["MSFT", "Microsoft", 200, 198, 1, 2, 3, np.nan, np.nan],
        ], columns=STOCK_COLUMNS)
        self.issues = pd.DataFrame(columns=["Ticker", "Issue"])

    def tearDown(self):
        st.cache_data.clear()

    def select_index(self):
        app = AppTest.from_file(str(PAGE)).run()
        self.assertEqual(app.sidebar.selectbox[0].options, ["", *INDEX_SOURCES])
        app.sidebar.selectbox[0].select("Dow Jones (USA)").run()
        self.assertEqual(len(app.exception), 0)
        return app

    @patch("analyst_data.get_stock_data")
    @patch("analyst_data.get_index_tickers")
    def test_displays_coverage_and_refresh_reloads(self, load_index, load_stocks):
        load_index.return_value = self.constituents
        load_stocks.return_value = (self.stocks, self.issues)
        app = self.select_index()
        self.assertEqual(app.dataframe[0].value["Ticker"].tolist(), ["AAPL", "MSFT"])
        self.assertIn("analyst targets available for 1", app.caption[-1].value)
        app.run()
        self.assertEqual(load_stocks.call_count, 1)
        app.sidebar.button[0].click().run()
        self.assertEqual(load_index.call_count, 2)
        self.assertEqual(load_stocks.call_count, 2)
        self.assertEqual(len(app.exception), 0)

    @patch("analyst_data.get_stock_data")
    @patch("analyst_data.get_index_tickers")
    def test_constituent_errors_are_visible_and_not_cached(self, load_index, load_stocks):
        load_index.side_effect = IndexDataError("Provider unavailable")
        app = self.select_index()
        self.assertIn("Provider unavailable", app.error[0].value)
        app.run()
        self.assertEqual(load_index.call_count, 2)
        load_stocks.assert_not_called()

    @patch("analyst_data.get_stock_data")
    @patch("analyst_data.get_index_tickers")
    def test_rate_limit_is_visible_and_not_cached(self, load_index, load_stocks):
        load_index.return_value = self.constituents
        load_stocks.side_effect = YFRateLimitError()
        app = self.select_index()
        self.assertIn("limiting requests", app.error[0].value)
        app.run()
        self.assertEqual(load_stocks.call_count, 2)

    @patch("analyst_data.get_stock_data")
    @patch("analyst_data.get_index_tickers")
    def test_partial_failures_show_details_without_hiding_prices(self, load_index, load_stocks):
        load_index.return_value = self.constituents
        load_stocks.return_value = (
            self.stocks,
            pd.DataFrame([{"Ticker": "MSFT", "Issue": "Analyst target unavailable"}]),
        )
        app = self.select_index()
        self.assertIn("failed for 1 stocks", app.warning[0].value)
        self.assertEqual(len(app.dataframe), 2)
        self.assertEqual(len(app.dataframe[1].value), 2)

    @patch("analyst_data.get_stock_data")
    @patch("analyst_data.get_index_tickers")
    def test_missing_coverage_is_explained(self, load_index, load_stocks):
        load_index.return_value = self.constituents
        self.stocks["Mean Analyst Target"] = np.nan
        self.stocks["Upside to Target"] = np.nan
        load_stocks.return_value = (self.stocks, self.issues)
        app = self.select_index()
        self.assertIn("no analyst targets", app.info[0].value)
        self.assertEqual(len(app.dataframe[0].value), 2)


if __name__ == "__main__":
    unittest.main()
