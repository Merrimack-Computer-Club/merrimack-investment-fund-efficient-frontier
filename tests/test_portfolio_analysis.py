from io import BytesIO, StringIO
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult
from yfinance.exceptions import YFRateLimitError

import portfolio_analysis as pa


def factor_archive(rows):
    content = (
        "Created from test data.\n\n"
        ",Mkt-RF,SMB,HML,RF\n"
        f"{rows}\n\nCopyright test"
    )
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("F-F_Research_Data_Factors_daily.csv", content)
    return buffer.getvalue()


class PortfolioInputTests(unittest.TestCase):
    def test_tickers_and_labelled_weights(self):
        tickers = pa.validate_tickers(" msft, aapl ")
        weights = pa.normalize_weights([60, 40], tickers)
        self.assertEqual(tickers, ["MSFT", "AAPL"])
        np.testing.assert_allclose(weights, [0.6, 0.4])
        reordered = pa.normalize_weights(weights, ["AAPL", "MSFT"])
        np.testing.assert_allclose(reordered, [0.4, 0.6])

    def test_invalid_tickers_are_not_silently_dropped(self):
        for values in ("", "AAPL,", "AAPL,aapl", "AA PL", [np.nan], []):
            with self.subTest(values=values), self.assertRaises(pa.PortfolioError):
                pa.validate_tickers(values)

    def test_invalid_weights_are_rejected_before_normalizing(self):
        for values in ([0, 0], [-1, 2], [np.nan, 1], [np.inf, 1], ["bad", 1], [1], [[1, 2]]):
            with self.subTest(values=values), self.assertRaises(pa.PortfolioError):
                pa.normalize_weights(values, ["A", "B"])
        with self.assertRaises(pa.PortfolioError):
            pa.normalize_weights(pd.Series([1], index=["C"]), ["A"])

    def test_csv_roundtrip_and_optional_weights(self):
        tickers, weights = pa.read_portfolio_csv(StringIO("msft,aapl\n60,40\n"))
        self.assertEqual(tickers, ["MSFT", "AAPL"])
        np.testing.assert_allclose(weights, [0.6, 0.4])
        csv = pd.DataFrame([tickers, weights.tolist()]).to_csv(index=False, header=False)
        tickers_again, weights_again = pa.read_portfolio_csv(StringIO(csv))
        self.assertEqual(tickers_again, tickers)
        pd.testing.assert_series_equal(weights, weights_again)
        _, equal = pa.read_portfolio_csv(StringIO("AAPL,MSFT\n"))
        np.testing.assert_allclose(equal, [0.5, 0.5])

    def test_csv_rejects_partial_zero_and_non_numeric_allocations(self):
        for csv in ("", "A,B\n0,0", "A,B\n1,", "A,B\n1,bad", "A,A\n1,2", "A,B\n1,2\n3,4"):
            with self.subTest(csv=csv), self.assertRaises(pa.PortfolioError):
                pa.read_portfolio_csv(StringIO(csv))

    @patch("portfolio_analysis.yf.download")
    def test_inverted_or_empty_dates_do_not_download(self, download):
        for start, end in (("2024-02-01", "2024-01-01"), ("2024-01-01", "2024-01-01"), (pd.NaT, pd.NaT)):
            with self.subTest(start=start), self.assertRaises(pa.PortfolioError):
                pa.download_prices(["A"], start, end)
        download.assert_not_called()

    def test_risk_free_rate_validation(self):
        for value in (None, "bad", np.nan, np.inf, -0.01, 1.1):
            with self.subTest(value=value), self.assertRaises(pa.PortfolioError):
                pa.validate_risk_free_rate(value)
        self.assertEqual(pa.validate_risk_free_rate(0), 0)


class PriceHistoryTests(unittest.TestCase):
    def setUp(self):
        self.prices = pd.DataFrame({
            "AAPL": [100, 110, 121, 133.1],
            "MSFT": [100, 120, 144, 172.8],
        }, index=pd.bdate_range("2024-01-01", periods=4))

    @patch("portfolio_analysis.yf.download")
    def test_yahoo_column_order_never_reassigns_weights(self, download):
        download.return_value = pd.concat({"Close": self.prices}, axis=1)
        prices = pa.download_prices(["MSFT", "AAPL"], "2024-01-01", "2024-02-01")
        self.assertEqual(prices.columns.tolist(), ["MSFT", "AAPL"])
        returns = pa.daily_returns(prices)
        weights = pa.normalize_weights([1, 0], ["MSFT", "AAPL"])
        np.testing.assert_allclose(pa.weighted_returns(returns, weights), 0.2)
        portfolio = pa.portfolio_performance(weights, returns.mean(), returns.cov(), 0.02)
        self.assertAlmostEqual(portfolio.annual_return, 0.2 * 252)
        self.assertTrue(download.call_args.kwargs["auto_adjust"])

    @patch("portfolio_analysis.yf.download")
    def test_single_and_ticker_first_yahoo_columns(self, download):
        for data in (
            self.prices[["AAPL"]].rename(columns={"AAPL": "Close"}),
            pd.concat({"Close": self.prices[["AAPL"]]}, axis=1),
            pd.concat({"Close": self.prices[["AAPL"]]}, axis=1).swaplevel(axis=1),
        ):
            with self.subTest(columns=data.columns):
                download.return_value = data
                result = pa.download_prices(["AAPL"], "2024-01-01", "2024-02-01")
                np.testing.assert_allclose(result["AAPL"], self.prices["AAPL"])

    @patch("portfolio_analysis.yf.download")
    def test_missing_tickers_and_failed_downloads_are_explicit(self, download):
        responses = (
            pd.DataFrame(), pd.concat({"Close": self.prices[["AAPL"]]}, axis=1),
            pd.concat({"Open": self.prices}, axis=1),
        )
        for data in responses:
            with self.subTest(columns=data.columns), self.assertRaises(pa.PortfolioError):
                download.return_value = data
                pa.download_prices(["AAPL", "MSFT"], "2024-01-01", "2024-02-01")
        download.side_effect = YFRateLimitError()
        with self.assertRaisesRegex(pa.PortfolioError, "Could not download"):
            pa.download_prices(["AAPL"], "2024-01-01", "2024-02-01")

    def test_internal_gaps_do_not_create_zero_or_multiday_daily_returns(self):
        prices = self.prices.copy()
        prices.iloc[1, 0] = np.nan
        with self.assertRaisesRegex(pa.PortfolioError, "Missing closing prices.*AAPL"):
            pa.daily_returns(prices)

    def test_shared_boundary_history_is_reported(self):
        prices = self.prices.copy()
        prices.iloc[0, 1] = np.nan
        shared = pa.clean_prices(prices)
        self.assertEqual(shared.attrs["trimmed_rows"], 1)
        self.assertEqual(shared.index[0], prices.index[1])
        self.assertEqual(len(pa.daily_returns(shared)), 2)

    def test_invalid_and_insufficient_prices_fail(self):
        for value in (0, -1, np.inf):
            prices = self.prices.copy()
            prices.iloc[1, 0] = value
            with self.subTest(value=value), self.assertRaises(pa.PortfolioError):
                pa.clean_prices(prices)
        for prices in (
            self.prices.iloc[:2], self.prices.assign(AAPL=np.nan),
            pd.concat([self.prices, self.prices.iloc[:1]]),
        ):
            with self.subTest(prices=prices), self.assertRaises(pa.PortfolioError):
                pa.clean_prices(prices)

    @patch("portfolio_analysis.yf.Ticker")
    def test_treasury_quote_is_converted_once_and_missing_is_not_zero(self, ticker):
        ticker.return_value.history.return_value = pd.DataFrame({"Close": [4.5]})
        self.assertEqual(pa.get_risk_free_rate(), 0.045)
        ticker.return_value.history.return_value = pd.DataFrame()
        with self.assertRaisesRegex(pa.PortfolioError, "No current"):
            pa.get_risk_free_rate()
        ticker.return_value.history.return_value = pd.DataFrame({"Close": [np.nan]})
        with self.assertRaises(pa.PortfolioError):
            pa.get_risk_free_rate()


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.mean = pd.Series([0.1, 0.2], index=["A", "B"]) / 252
        self.covariance = pd.DataFrame(
            np.diag([0.04, 0.09]) / 252, index=self.mean.index, columns=self.mean.index
        )

    def test_matches_analytical_minimum_variance_and_maximum_sharpe(self):
        result = pa.efficient_frontier(self.mean, self.covariance, 0.02)
        minimum, maximum = result.minimum_variance, result.maximum_sharpe
        np.testing.assert_allclose(minimum.weights, [9 / 13, 4 / 13], atol=1e-7)
        self.assertAlmostEqual(minimum.annual_return, 0.130769230769, places=8)
        self.assertAlmostEqual(minimum.volatility, 0.166410058868, places=8)
        self.assertIsNotNone(maximum)
        np.testing.assert_allclose(maximum.weights, [0.5, 0.5], atol=1e-7)
        self.assertAlmostEqual(maximum.sharpe, 0.721110255093, places=8)

    def test_every_frontier_point_is_feasible_and_on_the_upper_branch(self):
        result = pa.efficient_frontier(self.mean, self.covariance, 0.02, points=31)
        self.assertEqual(len(result.frontier), 31)
        for portfolio in result.frontier:
            self.assertAlmostEqual(portfolio.weights.sum(), 1, places=7)
            self.assertTrue((portfolio.weights >= -1e-8).all())
            weight_b = (portfolio.annual_return - 0.1) / 0.1
            np.testing.assert_allclose(portfolio.weights, [1 - weight_b, weight_b], atol=1e-7)
            variance = 0.04 * (1 - weight_b) ** 2 + 0.09 * weight_b ** 2
            self.assertAlmostEqual(portfolio.volatility ** 2, variance, places=8)
        self.assertTrue(np.all(np.diff([p.annual_return for p in result.frontier]) >= 0))
        self.assertTrue(np.all(np.diff([p.volatility for p in result.frontier]) >= -1e-8))
        np.testing.assert_allclose(result.frontier[-1].weights, [0, 1], atol=1e-8)

    def test_target_return_and_infeasible_target(self):
        weights = pa._minimum_variance_weights(
            self.mean.to_numpy() * 252, self.covariance.to_numpy() * 252, target=0.18
        )
        np.testing.assert_allclose(weights, [0.2, 0.8], atol=1e-8)
        with self.assertRaisesRegex(pa.PortfolioError, "not achievable"):
            pa._minimum_variance_weights(
                self.mean.to_numpy() * 252, self.covariance.to_numpy() * 252, target=0.3
            )

    def test_performance_aligns_both_axes_and_weights_by_ticker(self):
        weights = pd.Series([0.8, 0.2], index=["B", "A"])
        portfolio = pa.portfolio_performance(
            weights, self.mean, self.covariance.loc[["B", "A"], ["A", "B"]], 0.02
        )
        self.assertAlmostEqual(portfolio.annual_return, 0.18)
        self.assertAlmostEqual(portfolio.volatility, np.sqrt(0.2 ** 2 * 0.04 + 0.8 ** 2 * 0.09))
        with self.assertRaises(pa.PortfolioError):
            pa.portfolio_performance(np.array([0.2, 0.8]), self.mean, self.covariance, 0.02)

    def test_correlated_three_asset_tangency_solution(self):
        annual_mean = np.array([0.08, 0.13, 0.19])
        annual_covariance = np.array([[0.04, 0.01, 0.008], [0.01, 0.09, 0.025], [0.008, 0.025, 0.16]])
        mean = pd.Series(annual_mean / 252, index=["A", "B", "C"])
        covariance = pd.DataFrame(annual_covariance / 252, index=mean.index, columns=mean.index)
        expected = np.linalg.solve(annual_covariance, annual_mean - 0.02)
        expected /= expected.sum()
        result = pa.efficient_frontier(mean, covariance, 0.02)
        np.testing.assert_allclose(result.maximum_sharpe.weights, expected, atol=1e-6)

    def test_solutions_beat_an_independent_dense_allocation_grid(self):
        rng = np.random.default_rng(2026)
        grid = np.array([
            [a / 100, b / 100, 1 - (a + b) / 100]
            for a in range(101) for b in range(101 - a)
        ])
        for case in range(12):
            matrix = rng.normal(size=(3, 3))
            covariance = (matrix @ matrix.T + np.eye(3) * 0.1) * 0.01
            mean, rate = rng.uniform(-0.2, 0.4, 3), rng.uniform(0, 0.1)
            labels = ["A", "B", "C"]
            result = pa.efficient_frontier(
                pd.Series(mean / 252, index=labels),
                pd.DataFrame(covariance / 252, index=labels, columns=labels), rate, points=15,
            )
            grid_returns = grid @ mean
            grid_volatility = np.sqrt(np.einsum("ij,jk,ik->i", grid, covariance, grid))
            with self.subTest(case=case):
                self.assertLessEqual(result.minimum_variance.volatility, grid_volatility.min() + 1e-7)
                self.assertGreaterEqual(
                    result.maximum_sharpe.sharpe, ((grid_returns - rate) / grid_volatility).max() - 1e-6
                )
                for portfolio in result.frontier:
                    dominated = (
                        (grid_returns >= portfolio.annual_return + 1e-7)
                        & (grid_volatility < portfolio.volatility - 1e-7)
                    )
                    self.assertFalse(dominated.any())

    def test_single_asset_equal_returns_and_singular_risk(self):
        single = pa.efficient_frontier(self.mean.iloc[:1], self.covariance.iloc[:1, :1], 0.02)
        self.assertEqual(len(single.frontier), 1)
        np.testing.assert_array_equal(single.maximum_sharpe.weights, [1])
        equal = pa.efficient_frontier(self.mean * 0 + 0.1 / 252, self.covariance, 0.02)
        self.assertEqual(len(equal.frontier), 1)
        np.testing.assert_allclose(equal.maximum_sharpe.weights, [9 / 13, 4 / 13], atol=1e-7)
        singular = self.covariance * 0 + 0.04 / 252
        tied_risk = pa.efficient_frontier(self.mean, singular, 0.02)
        self.assertEqual(len(tied_risk.frontier), 1)
        np.testing.assert_allclose(tied_risk.minimum_variance.weights, [0, 1], atol=1e-7)

    def test_tied_maximum_return_endpoint_minimizes_risk(self):
        mean = pd.Series([0.05, 0.2, 0.2], index=["A", "B", "C"]) / 252
        covariance = pd.DataFrame(
            np.diag([0.16, 0.04, 0.09]) / 252, index=mean.index, columns=mean.index
        )
        result = pa.efficient_frontier(mean, covariance, 0.02)
        np.testing.assert_allclose(result.frontier[-1].weights, [0, 9 / 13, 4 / 13], atol=1e-7)

    def test_zero_volatility_does_not_invent_a_finite_sharpe_optimum(self):
        result = pa.efficient_frontier(self.mean, self.covariance * 0, 0.02)
        self.assertIsNone(result.maximum_sharpe)
        self.assertIsNone(result.minimum_variance.sharpe)
        self.assertEqual(len(result.frontier), 1)
        hedge_covariance = pd.DataFrame(
            [[0.04, -0.04], [-0.04, 0.04]], index=self.mean.index, columns=self.mean.index
        ) / 252
        hedge = pa.efficient_frontier(self.mean, hedge_covariance, 0.02)
        self.assertIsNone(hedge.maximum_sharpe)
        np.testing.assert_allclose(hedge.minimum_variance.weights, [0.5, 0.5], atol=1e-7)

    def test_non_positive_excess_return_uses_best_finite_sharpe(self):
        result = pa.efficient_frontier(self.mean, self.covariance, 0.3)
        np.testing.assert_allclose(result.maximum_sharpe.weights, [0, 1])
        self.assertAlmostEqual(result.maximum_sharpe.sharpe, -1 / 3)
        mixed = pa.efficient_frontier(self.mean, self.covariance, 0.15)
        np.testing.assert_allclose(mixed.maximum_sharpe.weights, [0, 1], atol=1e-7)

    @patch("portfolio_analysis.minimize")
    def test_failed_or_infeasible_solver_results_are_not_repaired(self, minimize):
        for result in (
            OptimizeResult(success=False, message="iteration limit"),
            OptimizeResult(success=True, x=np.array([0.5, 0.7])),
            OptimizeResult(success=True, x=np.array([-0.1, 1.1])),
            OptimizeResult(success=True, x=np.array([np.nan, np.nan])),
        ):
            with self.subTest(result=result), self.assertRaises(pa.PortfolioError):
                minimize.return_value = result
                pa.efficient_frontier(self.mean, self.covariance, 0.02)
        minimize.return_value = OptimizeResult(success=True, x=np.array([0.5, 0.5]))
        with self.assertRaisesRegex(pa.PortfolioError, "infeasible"):
            pa._minimum_variance_weights(
                self.mean.to_numpy() * 252, self.covariance.to_numpy() * 252, target=0.18
            )

    def test_invalid_covariance_is_not_treated_as_zero_risk(self):
        for matrix in (
            [[1, 2], [2, 1]], [[1, 1], [0, 1]], [[np.nan, 0], [0, 1]],
        ):
            with self.subTest(matrix=matrix), self.assertRaises(pa.PortfolioError):
                pa.efficient_frontier(
                    self.mean, pd.DataFrame(matrix, index=self.mean.index, columns=self.mean.index), 0.02
                )

    def test_random_samples_are_reproducible_and_bounded(self):
        short = pa.sample_portfolios(self.mean, self.covariance, 0.02, count=10)
        long = pa.sample_portfolios(self.mean, self.covariance, 0.02, count=20)
        for actual, expected in zip(short, long):
            pd.testing.assert_series_equal(actual.weights, expected.weights)
            self.assertAlmostEqual(actual.weights.sum(), 1)
        for count in (0, 10001, 1.5):
            with self.subTest(count=count), self.assertRaises(pa.PortfolioError):
                pa.sample_portfolios(self.mean, self.covariance, 0.02, count=count)


class HistoricalComparisonTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2024-01-01", periods=3)
        self.prices = pd.DataFrame({"A": [100, 200, 100], "B": [100, 100, 100]}, index=self.dates)
        self.market = pd.DataFrame({"INDEX": [100, 110, 99]}, index=self.dates)
        self.selected = pd.Series([0.5, 0.5], index=["A", "B"])
        self.optimized = pd.Series([1.0, 0.0], index=["A", "B"])

    def test_daily_rebalancing_and_compounding_share_a_zero_baseline(self):
        daily, cumulative = pa.historical_comparison(
            self.prices, self.market, self.selected, self.optimized, "Market"
        )
        np.testing.assert_allclose(daily["Selected Portfolio"], [0.5, -0.25])
        np.testing.assert_allclose(cumulative["Selected Portfolio"], [0, 0.5, 0.125])
        np.testing.assert_allclose(cumulative["Optimized Portfolio"], [0, 1, 0])
        np.testing.assert_allclose(cumulative["Market"], [0, 0.1, -0.01], atol=1e-12)
        self.assertEqual(cumulative.index[0], self.dates[0])

    def test_comparison_aligns_prices_before_calculating_returns(self):
        earlier = pd.DataFrame(
            {"INDEX": [100, 150, 165, 148.5]},
            index=self.dates.insert(0, self.dates[0] - pd.Timedelta(days=1)),
        )
        daily, cumulative = pa.historical_comparison(self.prices, earlier, self.selected, None, "Market")
        np.testing.assert_allclose(daily["Market"], [0.1, -0.1])
        self.assertEqual(cumulative.index[0], self.dates[0])
        self.assertNotIn("Optimized Portfolio", daily)

    def test_calendar_mismatches_are_not_silently_dropped(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        prices = pd.DataFrame({"A": [100, 110, 120, 130, 140]}, index=dates)
        market = pd.DataFrame({"INDEX": [100, 110, 130, 140]}, index=dates.delete(2))
        with self.assertRaisesRegex(pa.PortfolioError, "closing dates do not match"):
            pa.historical_comparison(prices, market, pd.Series([1.0], index=["A"]), None, "Market")
        market.index += pd.Timedelta(days=30)
        with self.assertRaisesRegex(pa.PortfolioError, "shared closing dates"):
            pa.historical_comparison(prices, market, pd.Series([1.0], index=["A"]), None, "Market")

    def test_alpha_uses_daily_excess_returns_and_is_not_sample_length_dependent(self):
        for repeats in (1, 2, 63):
            market = pd.Series(
                np.tile([-0.02, 0.01, 0.015, -0.005], repeats),
                index=pd.bdate_range("2024-01-01", periods=4 * repeats, tz="America/New_York"),
            )
            portfolio = 0.04 / 252 + 2 * (market - 0.04 / 252) + 0.0001
            with self.subTest(repeats=repeats):
                alpha, beta = pa.calculate_alpha_beta(portfolio, market, 0.04)
                self.assertAlmostEqual(alpha, 0.0252, places=10)
                self.assertAlmostEqual(beta, 2, places=10)
                self.assertIsNotNone(market.index.tz)
                self.assertIsNotNone(portfolio.index.tz)

    def test_beta_uses_matching_observations(self):
        dates = pd.bdate_range("2024-01-01", periods=4)
        market = pd.Series([-0.1, 0.2, -0.05, 0.1], index=dates)
        portfolio = 2 * market
        alpha, beta = pa.calculate_alpha_beta(portfolio.iloc[:3], market.iloc[1:], 0)
        self.assertAlmostEqual(alpha, 0)
        self.assertAlmostEqual(beta, 2)

    def test_alpha_beta_rejects_missing_zero_variance_and_insufficient_data(self):
        market = pd.Series([0.01, 0.02, -0.01], index=self.dates)
        for invalid in (
            market * 0, market.iloc[:1], market * np.nan,
            pd.concat([market, market.iloc[:1]]),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(pa.PortfolioError):
                pa.calculate_alpha_beta(market, invalid, 0.04)
        with self.assertRaisesRegex(pa.PortfolioError, "missing"):
            pa.weighted_returns(pd.DataFrame({"A": [np.nan]}), pd.Series([1], index=["A"]))


class FamaFrenchTests(unittest.TestCase):
    def test_parser_reads_daily_percentages_and_ignores_archive_footer(self):
        factors = pa.parse_fama_french_factors(factor_archive(
            "20240102, 1.00, -0.50, 0.25, 0.01\n"
            "20240103, -2.00, 0.75, -0.10, 0.02"
        ))
        self.assertEqual(factors.columns.tolist(), pa.FAMA_FRENCH_COLUMNS)
        self.assertEqual(factors.index.tolist(), [
            pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"),
        ])
        np.testing.assert_allclose(factors.iloc[0], [0.01, -0.005, 0.0025, 0.0001])

    @patch("portfolio_analysis.requests.get")
    def test_downloader_uses_official_archive_with_timeout(self, get):
        get.return_value.content = factor_archive(
            "20240102, 1.00, -0.50, 0.25, 0.01"
        )
        result = pa.download_fama_french_factors()
        self.assertEqual(len(result), 1)
        self.assertEqual(get.call_args.args[0], pa.FAMA_FRENCH_DAILY_URL)
        self.assertEqual(get.call_args.kwargs["timeout"], 30)
        get.return_value.raise_for_status.assert_called_once()

    def test_regression_recovers_alpha_and_factor_exposures(self):
        dates = pd.bdate_range("2024-01-01", periods=30)
        values = np.random.default_rng(2026).normal(0, 0.01, (len(dates), 3))
        factors = pd.DataFrame(values, index=dates, columns=["Mkt-RF", "SMB", "HML"])
        factors["RF"] = 0.0001
        portfolio = (
            factors["RF"] + 0.0002 + 1.2 * factors["Mkt-RF"]
            - 0.3 * factors["SMB"] + 0.5 * factors["HML"]
        )
        result = pa.fama_french_regression(portfolio, factors)
        self.assertAlmostEqual(result.annual_alpha, 0.0002 * 252)
        self.assertAlmostEqual(result.market_beta, 1.2)
        self.assertAlmostEqual(result.smb_beta, -0.3)
        self.assertAlmostEqual(result.hml_beta, 0.5)
        self.assertAlmostEqual(result.r_squared, 1)
        self.assertEqual(result.observations, 30)
        self.assertEqual(result.start_date, dates[0])
        self.assertEqual(result.end_date, dates[-1])

    def test_regression_rejects_insufficient_and_collinear_factor_history(self):
        dates = pd.bdate_range("2024-01-01", periods=6)
        factors = pd.DataFrame({
            "Mkt-RF": np.arange(6) / 100,
            "SMB": np.arange(6) / 100,
            "HML": np.arange(6) / 100,
            "RF": np.zeros(6),
        }, index=dates)
        portfolio = pd.Series(np.arange(6) / 100, index=dates)
        with self.assertRaisesRegex(pa.PortfolioError, "collinear"):
            pa.fama_french_regression(portfolio, factors)
        with self.assertRaisesRegex(pa.PortfolioError, "five shared"):
            pa.fama_french_regression(portfolio.iloc[:4], factors)
        with self.assertRaises(pa.PortfolioError):
            pa.parse_fama_french_factors(b"not a zip file")


if __name__ == "__main__":
    unittest.main()
