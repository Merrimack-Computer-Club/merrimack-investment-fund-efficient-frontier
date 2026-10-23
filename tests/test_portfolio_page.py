from io import BytesIO, StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

import portfolio_analysis as pa


PAGE = Path(__file__).resolve().parents[1].joinpath("efficient_frontier.py")


class PortfolioPageTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        self.tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
        dates = pd.bdate_range("2024-01-01", periods=41)
        rng = np.random.default_rng(123)
        returns = rng.normal(0.0007, 0.01, (40, len(self.tickers)))
        self.prices = pd.DataFrame(
            np.vstack([np.ones(len(self.tickers)), np.cumprod(1 + returns, axis=0)]) * 100,
            index=dates, columns=self.tickers,
        )
        self.benchmark = pd.DataFrame(
            {"^GSPC": np.r_[1, np.cumprod(1 + rng.normal(0.0005, 0.008, 40))] * 100},
            index=dates,
        )
        self.download = self.start_patch("portfolio_analysis.download_prices", side_effect=self.price_response)
        self.rate = self.start_patch("portfolio_analysis.get_risk_free_rate", return_value=0.045)
        self.names = self.start_patch("portfolio_analysis.get_asset_name", side_effect=lambda ticker: ticker)

    def tearDown(self):
        st.cache_data.clear()

    def start_patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def price_response(self, tickers, start, end):
        if tickers == ["^GSPC"]:
            return self.benchmark.copy()
        return self.prices.loc[:, sorted(tickers)].copy()

    def app(self):
        app = AppTest.from_file(str(PAGE), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def deploy(self, app, count=100):
        next(slider for slider in app.slider if slider.label == "Number of Random Portfolio Samples").set_value(count)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        return app

    def text(self, elements):
        return "\n".join(element.value for element in elements)

    def charts(self, app):
        return [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]

    def set_weight(self, app, ticker, value):
        next(widget for widget in app.sidebar.slider if widget.key == f"Slider_weight_{ticker}").set_value(value)

    def test_real_frontier_and_optimized_summary_do_not_depend_on_sample_count(self):
        app = self.deploy(self.app())
        self.assertEqual(len(app.error), 0)
        self.assertEqual(self.download.call_count, 2)
        charts = self.charts(app)
        self.assertEqual(charts[0]["data"][0]["name"], "Random portfolios (not optimized)")
        self.assertEqual(len(charts[0]["data"][0]["x"]), 100)
        self.assertEqual(charts[0]["data"][1]["name"], "Efficient frontier")
        self.assertEqual(len(charts[0]["data"][1]["x"]), 50)
        summary = app.dataframe[0].value.copy()
        self.deploy(app, count=200)
        pd.testing.assert_frame_equal(app.dataframe[0].value, summary)
        self.assertEqual(self.download.call_count, 2)
        self.assertEqual(len(self.charts(app)[0]["data"][0]["x"]), 200)

    def test_all_charts_use_native_theme_and_consistent_non_color_series_cues(self):
        app = self.deploy(self.app())
        charts = self.charts(app)
        self.assertEqual(len(charts), 4)
        for element, chart in zip(app.get("plotly_chart"), charts):
            self.assertEqual(element.proto.theme, "streamlit")
            for layout in (chart["layout"], chart["layout"]["template"]["layout"]):
                self.assertNotIn("paper_bgcolor", layout)
                self.assertNotIn("plot_bgcolor", layout)
                self.assertNotIn("color", layout.get("font", {}))
            for axis in ("xaxis", "yaxis"):
                for font in (chart["layout"][axis]["tickfont"], chart["layout"][axis]["title"]["font"]):
                    self.assertGreaterEqual(font["size"], 19)
                    self.assertEqual(font["weight"], 700)
                    self.assertNotIn("color", font)
            self.assertEqual(chart["layout"]["legend"]["orientation"], "h")

        template = charts[0]["layout"]["template"]["layout"]
        palette = template["colorway"]
        samples, frontier, custom, optimized = charts[0]["data"]
        self.assertEqual(samples["marker"]["colorscale"], template["colorscale"]["sequential"])
        self.assertEqual(frontier["line"]["color"], palette[0])
        self.assertEqual(custom["marker"]["color"], palette[2])
        self.assertEqual(optimized["marker"]["color"], palette[1])
        self.assertEqual(custom["marker"]["symbol"], "star")
        self.assertEqual(optimized["marker"]["symbol"], "diamond")
        for chart in charts[1:3]:
            traces = {trace["name"]: trace for trace in chart["data"]}
            self.assertEqual(traces["Optimized Portfolio"]["line"], {"color": palette[1], "dash": "solid"})
            self.assertEqual(traces["Selected Portfolio"]["line"], {"color": palette[2], "dash": "dash"})
            self.assertEqual(traces["S&P 500"]["line"], {"color": palette[3], "dash": "dot"})
        self.assertEqual(charts[3]["layout"]["xaxis"]["tickmode"], "linear")
        self.assertEqual(charts[3]["layout"]["xaxis"]["dtick"], 1)

    def test_theme_help_and_attribution_use_native_text(self):
        app = self.app()
        self.assertIn("Settings > Theme", self.text(app.sidebar.caption))
        self.assertIn("Authors Alexander Elguezabal", self.text(app.markdown))
        self.assertNotIn("position: fixed", self.text(app.markdown))

    def test_reordered_tickers_keep_their_own_weights_and_returns(self):
        app = self.app()
        app.sidebar.text_input[0].set_value("MSFT,AAPL").run()
        self.set_weight(app, "MSFT", 1.0)
        self.set_weight(app, "AAPL", 0.0)
        self.deploy(app)
        expected = pa.daily_returns(self.prices[["MSFT"]])["MSFT"].mean() * 252
        self.assertIn(f"Expected Annual Return: {expected:.3%}", self.text(app.markdown))
        custom_trace = self.charts(app)[0]["data"][2]
        self.assertIn("MSFT: 100.00%", custom_trace["text"][0])
        self.assertIn("AAPL: 0.00%", custom_trace["text"][0])

    def test_add_remove_reorder_and_weight_input_mode_preserve_ticker_associations(self):
        app = self.app()
        self.set_weight(app, "AAPL", 0.7)
        self.set_weight(app, "MSFT", 0.3)
        app.run()
        app.sidebar.text_input[0].set_value("MSFT,AAPL,NVDA").run()
        values = {widget.key: widget.value for widget in app.sidebar.slider}
        self.assertAlmostEqual(values["Slider_weight_MSFT"], 0.3)
        self.assertAlmostEqual(values["Slider_weight_AAPL"], 0.7)
        self.assertEqual(values["Slider_weight_NVDA"], 0)
        app.sidebar.radio[0].set_value("Number Input").run()
        values = {widget.key: widget.value for widget in app.sidebar.number_input}
        self.assertAlmostEqual(values["Number Input_weight_MSFT"], 0.3)
        self.assertAlmostEqual(values["Number Input_weight_AAPL"], 0.7)
        self.assertEqual(values["Number Input_weight_NVDA"], 0)
        app.sidebar.text_input[0].set_value("AAPL,MSFT").run()
        app.sidebar.radio[0].set_value("Slider").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.sidebar.slider), 2)
        values = {widget.key: widget.value for widget in app.sidebar.slider}
        self.assertAlmostEqual(values["Slider_weight_MSFT"], 0.3)
        self.assertAlmostEqual(values["Slider_weight_AAPL"], 0.7)

    def test_invalid_tickers_and_zero_weights_show_actionable_errors(self):
        for value in ("", "AAPL,", "AAPL,aapl"):
            with self.subTest(value=value):
                app = self.app()
                app.sidebar.text_input[0].set_value(value).run()
                self.assertEqual(len(app.exception), 0)
                self.assertGreater(len(app.error), 0)
        app = self.app()
        for widget in app.sidebar.slider:
            widget.set_value(0.0)
        app.run()
        self.assertIn("greater than zero", self.text(app.error))
        self.assertEqual(len(app.exception), 0)
        self.download.assert_not_called()

    def test_invalid_dates_stop_before_analysis(self):
        app = self.app()
        app.sidebar.date_input[1].set_value(pd.Timestamp("2019-01-01").date()).run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Start Date must be before", self.text(app.error))
        self.download.assert_not_called()

    def test_unavailable_rate_requires_manual_input_and_is_not_silently_zero(self):
        self.rate.side_effect = pa.PortfolioError("No current Treasury yield")
        app = self.app()
        self.assertIn("manually", self.text(app.warning))
        rate_input = next(widget for widget in app.sidebar.number_input if widget.label == "Set Risk-Free Rate (Annual)")
        self.assertIsNone(rate_input.value)
        self.deploy(app)
        self.assertIn("Enter an annual risk-free rate", self.text(app.error))
        self.download.assert_not_called()
        rate_input = next(widget for widget in app.sidebar.number_input if widget.label == "Set Risk-Free Rate (Annual)")
        rate_input.set_value(0.04).run()
        self.deploy(app)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(self.charts(app)), 4)

    def test_treasury_percentage_and_metadata_failures_are_displayed_correctly(self):
        self.names.side_effect = pa.PortfolioError("Company name unavailable")
        app = self.app()
        self.assertIn("4.50%", self.text(app.text))
        self.assertIn("Company name unavailable", self.text(app.warning))
        self.deploy(app)
        self.assertEqual(len(app.error), 0)

    def test_failed_price_requests_can_be_retried(self):
        self.download.side_effect = pa.PortfolioError("Missing closing prices for AAPL")
        app = self.deploy(self.app())
        self.assertIn("Missing closing prices", self.text(app.error))
        self.assertEqual(len(self.charts(app)), 0)
        self.download.side_effect = self.price_response
        self.deploy(app)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(self.charts(app)), 4)

    def test_single_asset_and_negative_sharpe_are_supported(self):
        app = self.app()
        app.sidebar.text_input[0].set_value("AAPL").run()
        rate = next(widget for widget in app.sidebar.number_input if widget.label == "Set Risk-Free Rate (Annual)")
        rate.set_value(1.0)
        self.deploy(app)
        self.assertEqual(len(app.error), 0)
        self.assertIn("Maximizing a negative Sharpe ratio", self.text(app.warning))
        self.assertEqual(len(self.charts(app)[0]["data"][1]["x"]), 1)
        self.assertIn("AAPL: 100.00%", app.dataframe[0].value.iloc[0]["Weights"])

    def test_numeric_sample_count_is_bounded_like_the_slider(self):
        app = self.app()
        next(radio for radio in app.radio if radio.label == "Random Portfolio Sample Input").set_value("Number Input").run()
        sample_count = next(widget for widget in app.number_input if widget.label == "Number of Random Portfolio Samples")
        self.assertEqual(sample_count.proto.min, 100)
        self.assertEqual(sample_count.proto.max, 10000)
        sample_count.set_value(100)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(self.charts(app)[0]["data"][0]["x"]), 100)

    def test_unavailable_benchmark_does_not_hide_valid_portfolio_results(self):
        def response(tickers, start, end):
            if tickers == ["^GSPC"]:
                raise pa.PortfolioError("Benchmark not available")
            return self.price_response(tickers, start, end)

        self.download.side_effect = response
        app = self.deploy(self.app())
        self.assertIn("Historical comparison unavailable", self.text(app.error))
        self.assertIn("Expected Annual Return", self.text(app.markdown))
        self.assertEqual(len(self.charts(app)), 2)

    def test_backtest_labels_compounding_and_annual_alpha_are_consistent(self):
        app = self.deploy(self.app())
        self.assertIn("In-sample (hindsight)", self.text(app.caption))
        self.assertIn("rebalance", self.text(app.caption))
        charts = self.charts(app)
        self.assertEqual(charts[1]["layout"]["yaxis"]["title"]["text"], "Daily Return")
        self.assertEqual(charts[2]["layout"]["yaxis"]["title"]["text"], "Cumulative Return")
        for daily, cumulative in zip(charts[1]["data"], charts[2]["data"]):
            self.assertEqual(cumulative["y"][0], 0)
            np.testing.assert_allclose(cumulative["y"][1:], np.cumprod(1 + np.array(daily["y"])) - 1)
        self.assertIn("Annualized Alpha", app.dataframe[2].value.columns)
        self.assertIn("Not a Forecast", charts[3]["layout"]["title"]["text"])

    def test_constant_prices_show_undefined_sharpe_without_optimized_exports(self):
        self.prices[:] = 100.0
        with patch("streamlit.download_button", wraps=st.download_button) as download:
            app = self.deploy(self.app())
        self.assertIn("finite maximum Sharpe ratio is unavailable", self.text(app.warning))
        self.assertIn("Undefined (zero volatility)", self.text(app.markdown))
        self.assertIn("redundant or too short", self.text(app.warning))
        self.assertEqual(len(app.error), 0)
        labels = [call.args[0] for call in download.call_args_list]
        self.assertNotIn("Download Optimized Portfolio CSV", labels)
        self.assertNotIn("Download Sub-portfolio CSV", labels)
        self.assertNotIn("Optimized Portfolio", [trace["name"] for trace in self.charts(app)[1]["data"]])
        chart = self.charts(app)[0]
        self.assertEqual(
            chart["data"][0]["marker"]["color"],
            chart["layout"]["template"]["layout"]["colorway"][4],
        )

    def test_zero_volatility_benchmark_reports_undefined_alpha_beta(self):
        self.benchmark[:] = 100.0
        app = self.deploy(self.app())
        self.assertIn("benchmark has zero volatility", self.text(app.warning))
        self.assertEqual(len(app.error), 0)

    def test_exports_use_the_same_normalized_and_optimized_allocations(self):
        app = self.app()
        app.sidebar.text_input[0].set_value("MSFT,AAPL").run()
        self.set_weight(app, "MSFT", 0.6)
        self.set_weight(app, "AAPL", 0.2)
        with patch("streamlit.download_button", wraps=st.download_button) as download:
            self.deploy(app)
        self.assertIn("normalized to 100%", self.text(app.warning))
        exports = {call.args[0]: call.kwargs["data"] for call in download.call_args_list}
        _, selected = pa.read_portfolio_csv(StringIO(exports["Download Portfolio CSV"]))
        self.assertEqual(selected.index.tolist(), ["MSFT", "AAPL"])
        np.testing.assert_allclose(selected, [0.75, 0.25])
        _, optimized = pa.read_portfolio_csv(StringIO(exports["Download Optimized Portfolio CSV"]))
        scaled = pd.read_csv(StringIO(exports["Download Sub-portfolio CSV"])).set_index("Ticker")["Weight (%)"]
        np.testing.assert_allclose(scaled.reindex(optimized.index), optimized * 0.25 * 100, atol=1e-10)
        self.assertAlmostEqual(scaled.sum(), 25)
        summary_weights = app.dataframe[0].value.iloc[0]["Weights"]
        for ticker, weight in optimized.items():
            self.assertIn(f"{ticker}: {weight:.2%}", summary_weights)

    def test_uploaded_weights_follow_ticker_edits_and_reject_bad_csv(self):
        with patch("streamlit.file_uploader", return_value=BytesIO(b"MSFT,AAPL\n60,40\n")):
            app = self.app()
            values = {widget.key: widget.value for widget in app.sidebar.slider}
            self.assertAlmostEqual(values["Slider_weight_MSFT"], 0.6)
            self.assertAlmostEqual(values["Slider_weight_AAPL"], 0.4)
            app.sidebar.text_input[0].set_value("AAPL,MSFT,NVDA").run()
            values = {widget.key: widget.value for widget in app.sidebar.slider}
            self.assertAlmostEqual(values["Slider_weight_MSFT"], 0.6)
            self.assertAlmostEqual(values["Slider_weight_AAPL"], 0.4)
            self.assertEqual(values["Slider_weight_NVDA"], 0)
        with patch("streamlit.file_uploader", return_value=BytesIO(b"MSFT,AAPL\nbad,1\n")):
            invalid = self.app()
            self.assertIn("finite, non-negative number", self.text(invalid.error))
            self.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
