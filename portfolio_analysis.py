"""Validated, long-only portfolio calculations independent of Streamlit."""

from dataclasses import dataclass
from io import BytesIO, StringIO
import re
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import requests
from scipy.optimize import linprog, minimize
import yfinance as yf
from curl_cffi.requests.exceptions import RequestException as YahooRequestError
from yfinance.exceptions import YFException


TRADING_DAYS = 252
FEASIBILITY_TOL = 1e-7
ZERO_VOLATILITY = 1e-8
FAMA_FRENCH_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
FAMA_FRENCH_COLUMNS = ["Mkt-RF", "SMB", "HML", "RF"]


class PortfolioError(ValueError):
    """The supplied data or constraints cannot produce a reliable result."""


@dataclass(frozen=True)
class Portfolio:
    weights: pd.Series
    annual_return: float
    volatility: float
    sharpe: float | None


@dataclass(frozen=True)
class Optimization:
    frontier: list[Portfolio]
    minimum_variance: Portfolio
    maximum_sharpe: Portfolio | None


@dataclass(frozen=True)
class FactorRegression:
    annual_alpha: float
    market_beta: float
    smb_beta: float
    hml_beta: float
    r_squared: float | None
    observations: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp


def validate_tickers(values):
    values = values.split(",") if isinstance(values, str) else list(values)
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise PortfolioError("Enter at least one ticker; remove empty entries and trailing commas.")
    tickers = [value.strip().upper() for value in values]
    if any(not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=-]*", ticker) for ticker in tickers):
        raise PortfolioError("Tickers may contain letters, numbers, dots, hyphens, ^ or =, not spaces.")
    if len(set(tickers)) != len(tickers):
        raise PortfolioError("Each ticker must appear only once. Remove duplicate tickers.")
    return tickers


def normalize_weights(values, tickers):
    tickers = validate_tickers(tickers)
    if isinstance(values, pd.Series):
        if not values.index.is_unique or set(values.index) != set(tickers):
            raise PortfolioError("Weights must have exactly one entry for each ticker.")
        values = values.reindex(tickers)
    try:
        weights = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise PortfolioError("Every weight must be a finite, non-negative number.") from error
    if weights.shape != (len(tickers),):
        raise PortfolioError("Provide exactly one weight for each ticker.")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise PortfolioError("Every weight must be a finite, non-negative number.")
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise PortfolioError("At least one portfolio weight must be greater than zero.")
    return pd.Series(weights / total, index=tickers, name="Weight")


def read_portfolio_csv(file):
    try:
        frame = pd.read_csv(file, header=None, dtype=str)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise PortfolioError(f"Could not read the portfolio CSV: {error}") from error
    if len(frame) not in (1, 2):
        raise PortfolioError("CSV must have one row of tickers and, optionally, a second row of weights.")
    tickers = validate_tickers(frame.iloc[0].tolist())
    values = frame.iloc[1].tolist() if len(frame) == 2 else np.ones(len(tickers))
    return tickers, normalize_weights(values, tickers)


def validate_dates(start_date, end_date):
    if pd.isna(start_date) or pd.isna(end_date) or pd.Timestamp(start_date) >= pd.Timestamp(end_date):
        raise PortfolioError("Start Date must be before End Date. End Date is exclusive.")


def clean_prices(prices):
    """Use the shared history, but never fill or skip an internal missing close."""
    if not isinstance(prices, pd.DataFrame) or prices.empty or not prices.columns.is_unique:
        raise PortfolioError("No usable closing-price history was returned.")
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise PortfolioError("Closing prices must have valid trading dates.")
    prices = prices.copy()
    prices.index = prices.index.tz_localize(None).normalize()
    if prices.index.has_duplicates:
        raise PortfolioError("Closing-price history contains duplicate trading dates.")
    try:
        prices = prices.apply(pd.to_numeric, errors="raise").astype(float).sort_index()
    except (TypeError, ValueError) as error:
        raise PortfolioError("Closing-price history contains non-numeric values.") from error
    invalid = np.isinf(prices) | (prices <= 0)
    if invalid.any().any():
        names = ", ".join(prices.columns[invalid.any()])
        raise PortfolioError(f"Non-positive or infinite closing prices were returned for: {names}.")
    missing = prices.columns[prices.isna().all()]
    if len(missing):
        raise PortfolioError(f"No closing prices were returned for: {', '.join(missing)}.")
    start = max(prices[column].first_valid_index() for column in prices)
    end = min(prices[column].last_valid_index() for column in prices)
    shared = prices.loc[start:end].copy()
    if len(shared) < 3:
        raise PortfolioError("At least three shared closing prices (two daily returns) are required.")
    gaps = shared.columns[shared.isna().any()]
    if len(gaps):
        raise PortfolioError(
            f"Missing closing prices within the shared history for: {', '.join(gaps)}. "
            "Choose a shorter date range or compatible tickers; missing prices are not filled."
        )
    shared.attrs["trimmed_rows"] = len(prices) - len(shared)
    return shared


def download_prices(tickers, start_date, end_date):
    tickers = validate_tickers(tickers)
    validate_dates(start_date, end_date)
    try:
        data = yf.download(
            tickers, start=start_date, end=end_date, auto_adjust=True, progress=False,
            timeout=20,
        )
    except (YFException, YahooRequestError, requests.RequestException, ValueError) as error:
        raise PortfolioError(f"Could not download closing prices: {error}") from error
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise PortfolioError("No closing prices were returned. Check the tickers and date range.")
    if isinstance(data.columns, pd.MultiIndex):
        close_level = next(
            (level for level in range(data.columns.nlevels)
             if "Close" in data.columns.get_level_values(level)),
            None,
        )
        if close_level is None:
            raise PortfolioError("The price response does not contain closing prices.")
        prices = data.xs("Close", axis=1, level=close_level)
    elif len(tickers) == 1 and "Close" in data:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        raise PortfolioError("The price response does not identify each ticker's closing prices.")
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise PortfolioError(f"No closing prices were returned for: {', '.join(missing)}.")
    return clean_prices(prices.loc[:, tickers])


def get_risk_free_rate():
    try:
        history = yf.Ticker("^TNX").history(
            period="1d", auto_adjust=False, raise_errors=True, timeout=20
        )
    except (YFException, YahooRequestError, requests.RequestException, ValueError) as error:
        raise PortfolioError(f"Could not load the Treasury yield: {error}") from error
    if history.empty or "Close" not in history:
        raise PortfolioError("No current 10-year Treasury yield was returned.")
    try:
        rate = float(history["Close"].iloc[-1]) / 100
    except (TypeError, ValueError) as error:
        raise PortfolioError("The Treasury yield is not a valid number.") from error
    return validate_risk_free_rate(rate)


def get_asset_name(ticker):
    try:
        info = yf.Ticker(ticker).get_info()
    except (YFException, YahooRequestError, requests.RequestException, ValueError) as error:
        raise PortfolioError(f"Company name unavailable for {ticker}: {error}") from error
    if not isinstance(info, dict) or not info.get("shortName"):
        raise PortfolioError(f"Company name unavailable for {ticker}.")
    return info["shortName"]


def parse_fama_french_factors(content):
    """Parse Kenneth French's zipped daily US three-factor data into decimal returns."""
    if not isinstance(content, bytes) or not content:
        raise PortfolioError("The Fama-French factor download is empty.")
    try:
        with ZipFile(BytesIO(content)) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(csv_names) != 1:
                raise PortfolioError("The Fama-French archive must contain exactly one CSV file.")
            text = archive.read(csv_names[0]).decode("utf-8-sig")
    except (BadZipFile, KeyError, UnicodeDecodeError) as error:
        raise PortfolioError(f"Could not read the Fama-French factor archive: {error}") from error

    lines = text.splitlines()
    expected_header = ["", *FAMA_FRENCH_COLUMNS]
    header_index = next(
        (
            index for index, line in enumerate(lines)
            if [value.strip() for value in line.split(",")] == expected_header
        ),
        None,
    )
    if header_index is None:
        raise PortfolioError("The Fama-French factor CSV does not contain the expected columns.")
    try:
        frame = pd.read_csv(StringIO("\n".join(lines[header_index:])), dtype=str)
    except pd.errors.ParserError as error:
        raise PortfolioError(f"Could not parse the Fama-French factor CSV: {error}") from error
    frame.columns = [column.strip() for column in frame.columns]
    date_column = frame.columns[0]
    dates = frame[date_column].str.strip()
    frame = frame.loc[dates.str.fullmatch(r"\d{8}", na=False)].copy()
    if frame.empty:
        raise PortfolioError("The Fama-French factor CSV does not contain daily observations.")
    try:
        index = pd.to_datetime(frame.pop(date_column).str.strip(), format="%Y%m%d", errors="raise")
        factors = frame.loc[:, FAMA_FRENCH_COLUMNS].apply(pd.to_numeric, errors="raise") / 100
    except (KeyError, TypeError, ValueError) as error:
        raise PortfolioError(f"The Fama-French factor CSV contains invalid values: {error}") from error
    factors.index = pd.DatetimeIndex(index, name="Date")
    factors = factors.sort_index()
    if factors.index.has_duplicates or not np.isfinite(factors.to_numpy()).all():
        raise PortfolioError("The Fama-French factors require unique dates and finite values.")
    return factors


def download_fama_french_factors():
    try:
        response = requests.get(
            FAMA_FRENCH_DAILY_URL,
            headers={"User-Agent": "Merrimack-Portfolio-Analysis/1.0"},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise PortfolioError(f"Could not download Fama-French factors: {error}") from error
    return parse_fama_french_factors(response.content)


def validate_risk_free_rate(rate):
    message = "Enter an annual risk-free rate from 0 to 1 (for example, 0.04 for 4%)."
    try:
        rate = float(rate)
    except (TypeError, ValueError) as error:
        raise PortfolioError(message) from error
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise PortfolioError(message)
    return rate


def daily_returns(prices):
    return clean_prices(prices).pct_change(fill_method=None).iloc[1:]


def _annual_moments(mean_returns, cov_matrix):
    if not isinstance(mean_returns, pd.Series) or mean_returns.empty or not mean_returns.index.is_unique:
        raise PortfolioError("Mean returns must identify each ticker exactly once.")
    labels = mean_returns.index
    if (
        not isinstance(cov_matrix, pd.DataFrame)
        or not cov_matrix.index.is_unique or not cov_matrix.columns.is_unique
        or set(cov_matrix.index) != set(labels) or set(cov_matrix.columns) != set(labels)
    ):
        raise PortfolioError("The covariance matrix must match the portfolio tickers.")
    mean = mean_returns.to_numpy(dtype=float) * TRADING_DAYS
    covariance = cov_matrix.loc[labels, labels].to_numpy(dtype=float) * TRADING_DAYS
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise PortfolioError("Return estimates are incomplete or non-finite. Use more shared history.")
    scale = max(float(np.max(np.abs(covariance))), 1e-16)
    if (
        not np.allclose(covariance, covariance.T, rtol=1e-10, atol=scale * 1e-10)
        or np.linalg.eigvalsh(covariance).min() < -scale * 1e-10
    ):
        raise PortfolioError("The covariance matrix is not a valid risk estimate.")
    return labels, mean, covariance


def _aligned_weights(weights, labels):
    if (
        not isinstance(weights, pd.Series) or not weights.index.is_unique
        or set(weights.index) != set(labels)
    ):
        raise PortfolioError("Weights must be matched to the return data by ticker.")
    values = weights.reindex(labels).to_numpy(dtype=float)
    if (
        not np.isfinite(values).all() or (values < -FEASIBILITY_TOL).any()
        or abs(values.sum() - 1) > FEASIBILITY_TOL
    ):
        raise PortfolioError("Portfolio weights must be non-negative and sum to 100%.")
    return values


def _portfolio(weights, labels, mean, covariance, rate):
    annual_return = float(mean @ weights)
    volatility = float(np.sqrt(max(float(weights @ covariance @ weights), 0.0)))
    sharpe = (annual_return - rate) / volatility if volatility > ZERO_VOLATILITY else None
    return Portfolio(pd.Series(weights, index=labels), annual_return, volatility, sharpe)


def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    labels, mean, covariance = _annual_moments(mean_returns, cov_matrix)
    values = _aligned_weights(weights, labels)
    return _portfolio(values, labels, mean, covariance, validate_risk_free_rate(risk_free_rate))


def _solve_variance(covariance, initial, constraints, description):
    scale = max(float(np.max(np.diag(covariance))), 1e-16)
    scaled = covariance / scale
    result = minimize(
        lambda w: float(w @ scaled @ w), initial,
        jac=lambda w: 2 * scaled @ w, method="SLSQP",
        bounds=[(0, None)] * len(initial), constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise PortfolioError(f"Could not solve {description}: {result.message}")
    weights = np.asarray(result.x)
    if (
        weights.shape != initial.shape or not np.isfinite(weights).all()
        or (weights < -FEASIBILITY_TOL).any()
        or any(abs(constraint["fun"](weights)) > FEASIBILITY_TOL for constraint in constraints)
    ):
        raise PortfolioError(f"The solver returned infeasible weights for {description}.")
    return weights


def _minimum_variance_weights(mean, covariance, target=None, initial=None):
    count = len(mean)
    low, high = float(mean.min()), float(mean.max())
    if target is not None and not low - 1e-12 <= target <= high + 1e-12:
        raise PortfolioError("The target return is not achievable with non-negative, fully invested weights.")
    if count == 1:
        return np.ones(1)
    if target is not None and abs(target - high) <= 1e-12:
        eligible = np.flatnonzero(np.abs(mean - high) <= 1e-12)
        weights = np.zeros(count)
        weights[eligible] = _minimum_variance_weights(
            mean[eligible], covariance[np.ix_(eligible, eligible)]
        )
        return weights
    if initial is None:
        initial = np.full(count, 1 / count)
        if target is not None and high - low > 1e-12:
            fraction = (target - low) / (high - low)
            initial = np.zeros(count)
            initial[np.argmin(mean)] = 1 - fraction
            initial[np.argmax(mean)] = fraction
    constraints = [{
        "type": "eq", "fun": lambda w: w.sum() - 1,
        "jac": lambda w: np.ones(count),
    }]
    if target is not None and high - low > 1e-12:
        scaled_mean = (mean - low) / (high - low)
        scaled_target = (target - low) / (high - low)
        constraints.append({
            "type": "eq", "fun": lambda w: scaled_mean @ w - scaled_target,
            "jac": lambda w: scaled_mean,
        })
    if np.max(np.abs(covariance)) == 0:
        return np.eye(count)[np.argmax(mean)] if target is None else initial
    return _solve_variance(covariance, initial, constraints, "minimum variance")


def _efficient_start(mean, covariance):
    weights = _minimum_variance_weights(mean, covariance)
    scale = float(np.max(np.abs(covariance)))
    if not scale or np.ptp(mean) <= 1e-12:
        return weights
    eigenvalues, eigenvectors = np.linalg.eigh(covariance / scale)
    positive = eigenvalues > 1e-10
    if positive.all():
        return weights
    # Among equal-risk solutions, start at the one with the highest return.
    equalities = np.vstack([np.ones(len(mean)), eigenvectors[:, positive].T])
    result = linprog(
        -(mean - mean.min()) / np.ptp(mean),
        A_eq=equalities, b_eq=equalities @ weights,
        bounds=[(0, 1)] * len(mean), method="highs",
    )
    if not result.success:
        raise PortfolioError(f"Could not resolve equal-risk portfolios: {result.message}")
    if (
        not np.isfinite(result.x).all() or (result.x < -FEASIBILITY_TOL).any()
        or np.max(np.abs(equalities @ (result.x - weights))) > FEASIBILITY_TOL
    ):
        raise PortfolioError("The solver returned infeasible equal-risk portfolio weights.")
    return result.x


def _maximum_sharpe_weights(mean, covariance, rate):
    excess = mean - rate
    if excess.max() <= 0:
        volatility = np.sqrt(np.maximum(np.diag(covariance), 0))
        eligible = np.flatnonzero(volatility > ZERO_VOLATILITY)
        if not len(eligible):
            return None
        # With non-positive excess returns, diversification cannot beat the best vertex.
        best = eligible[np.argmax(excess[eligible] / volatility[eligible])]
        return np.eye(len(mean))[best]
    scaled_excess = excess / np.max(np.abs(excess))
    initial = np.zeros(len(mean))
    initial[np.argmax(excess)] = 1 / scaled_excess.max()
    constraints = [{
        "type": "eq", "fun": lambda y: scaled_excess @ y - 1,
        "jac": lambda y: scaled_excess,
    }]
    # Fix positive excess return, minimize variance, then recover fully invested weights.
    scaled_weights = _solve_variance(covariance, initial, constraints, "maximum Sharpe ratio")
    weights = scaled_weights / scaled_weights.sum()
    if np.sqrt(max(float(weights @ covariance @ weights), 0)) <= ZERO_VOLATILITY:
        return None
    return weights


def efficient_frontier(mean_returns, cov_matrix, risk_free_rate, points=50):
    if not isinstance(points, int) or isinstance(points, bool) or points < 2:
        raise PortfolioError("At least two frontier points must be requested.")
    labels, mean, covariance = _annual_moments(mean_returns, cov_matrix)
    rate = validate_risk_free_rate(risk_free_rate)
    minimum_weights = _efficient_start(mean, covariance)
    minimum = _portfolio(minimum_weights, labels, mean, covariance, rate)
    frontier = [minimum]
    if float(mean.max()) - minimum.annual_return > 1e-12:
        endpoint = _minimum_variance_weights(mean, covariance, float(mean.max()))
        targets = np.linspace(minimum.annual_return, float(mean.max()), points)
        for fraction, target in zip(np.linspace(0, 1, points)[1:-1], targets[1:-1]):
            initial = (1 - fraction) * minimum_weights + fraction * endpoint
            weights = _minimum_variance_weights(mean, covariance, float(target), initial)
            frontier.append(_portfolio(weights, labels, mean, covariance, rate))
        frontier.append(_portfolio(endpoint, labels, mean, covariance, rate))
    sharpe_weights = _maximum_sharpe_weights(mean, covariance, rate)
    maximum = (
        _portfolio(sharpe_weights, labels, mean, covariance, rate)
        if sharpe_weights is not None else None
    )
    return Optimization(frontier, minimum, maximum)


def sample_portfolios(mean_returns, cov_matrix, risk_free_rate, count=5000):
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 10000:
        raise PortfolioError("Choose between 1 and 10,000 random portfolio samples.")
    labels, mean, covariance = _annual_moments(mean_returns, cov_matrix)
    rate = validate_risk_free_rate(risk_free_rate)
    weights = np.random.default_rng(0).dirichlet(np.ones(len(labels)), size=count)
    return [_portfolio(weight, labels, mean, covariance, rate) for weight in weights]


def weighted_returns(returns, weights):
    values = _aligned_weights(weights, returns.columns)
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise PortfolioError("Portfolio returns contain missing or non-finite values.")
    return pd.Series(returns.to_numpy() @ values, index=returns.index)


def historical_comparison(prices, benchmark_prices, selected_weights, optimized_weights, benchmark_name):
    prices, benchmark_prices = clean_prices(prices), clean_prices(benchmark_prices)
    if benchmark_prices.shape[1] != 1:
        raise PortfolioError("Select exactly one benchmark for the comparison.")
    start = max(prices.index[0], benchmark_prices.index[0])
    end = min(prices.index[-1], benchmark_prices.index[-1])
    prices, benchmark_prices = prices.loc[start:end], benchmark_prices.loc[start:end]
    if len(prices) < 3 or len(benchmark_prices) < 3:
        raise PortfolioError("The portfolio and benchmark need at least three shared closing dates.")
    if not prices.index.equals(benchmark_prices.index):
        raise PortfolioError(
            "Portfolio and benchmark closing dates do not match. Choose a shorter date range "
            "or a market with the same trading calendar; unmatched days are not skipped."
        )
    returns = daily_returns(prices)
    comparison = pd.DataFrame(index=returns.index)
    if optimized_weights is not None:
        comparison["Optimized Portfolio"] = weighted_returns(returns, optimized_weights)
    comparison["Selected Portfolio"] = weighted_returns(returns, selected_weights)
    comparison[benchmark_name] = daily_returns(benchmark_prices).iloc[:, 0]
    cumulative = (1 + comparison).cumprod() - 1
    baseline = pd.DataFrame(0.0, index=prices.index[:1], columns=comparison.columns)
    return comparison, pd.concat([baseline, cumulative])


def calculate_alpha_beta(portfolio_returns, market_returns, risk_free_rate):
    rate = validate_risk_free_rate(risk_free_rate)
    series = []
    for values in (portfolio_returns, market_returns):
        if not isinstance(values, pd.Series) or not isinstance(values.index, pd.DatetimeIndex):
            raise PortfolioError("Alpha and beta require dated daily return series.")
        values = values.copy()
        values.index = values.index.tz_localize(None).normalize()
        if values.index.has_duplicates or values.index.hasnans or not np.isfinite(values).all():
            raise PortfolioError("Alpha and beta require unique dates and complete finite returns.")
        series.append(values)
    combined = pd.concat(series, axis=1, join="inner").sort_index()
    if len(combined) < 2:
        raise PortfolioError("Alpha and beta need at least two shared daily returns.")
    portfolio, market = combined.iloc[:, 0], combined.iloc[:, 1]
    centered_market = market - market.mean()
    squared_market = float(centered_market @ centered_market)
    if squared_market <= 1e-20:
        raise PortfolioError("Alpha and beta are undefined because the benchmark has zero volatility.")
    beta = float(centered_market @ (portfolio - portfolio.mean()) / squared_market)
    daily_rate = rate / TRADING_DAYS
    alpha = float(((portfolio.mean() - daily_rate) - beta * (market.mean() - daily_rate)) * TRADING_DAYS)
    return alpha, beta


def fama_french_regression(portfolio_returns, factors):
    """Fit Rp - Rf = alpha + beta_m(Mkt-RF) + beta_s(SMB) + beta_h(HML)."""
    if not isinstance(portfolio_returns, pd.Series) or not isinstance(
        portfolio_returns.index, pd.DatetimeIndex
    ):
        raise PortfolioError("Fama-French analysis requires dated daily portfolio returns.")
    portfolio = portfolio_returns.copy()
    portfolio.index = portfolio.index.tz_localize(None).normalize()
    if (
        portfolio.index.has_duplicates or portfolio.index.hasnans
        or not np.isfinite(portfolio.to_numpy(dtype=float)).all()
    ):
        raise PortfolioError(
            "Fama-French analysis requires unique dates and complete finite portfolio returns."
        )
    if (
        not isinstance(factors, pd.DataFrame)
        or not isinstance(factors.index, pd.DatetimeIndex)
        or factors.index.has_duplicates or factors.index.hasnans
        or any(column not in factors for column in FAMA_FRENCH_COLUMNS)
    ):
        raise PortfolioError(
            "Fama-French factors require unique dates and Mkt-RF, SMB, HML, and RF columns."
        )
    factor_data = factors.loc[:, FAMA_FRENCH_COLUMNS].copy()
    factor_data.index = factor_data.index.tz_localize(None).normalize()
    try:
        factor_data = factor_data.astype(float)
    except (TypeError, ValueError) as error:
        raise PortfolioError("Fama-French factors must be numeric.") from error
    if not np.isfinite(factor_data.to_numpy()).all():
        raise PortfolioError("Fama-French factors must be complete and finite.")

    combined = pd.concat(
        [portfolio.rename("Portfolio"), factor_data], axis=1, join="inner"
    ).sort_index()
    if len(combined) < 5:
        raise PortfolioError(
            "Fama-French analysis needs at least five shared daily return observations."
        )
    design = np.column_stack(
        [np.ones(len(combined)), combined[["Mkt-RF", "SMB", "HML"]].to_numpy()]
    )
    excess_returns = (combined["Portfolio"] - combined["RF"]).to_numpy()
    coefficients, _, rank, _ = np.linalg.lstsq(design, excess_returns, rcond=None)
    if rank < design.shape[1]:
        raise PortfolioError(
            "Fama-French regression is undefined because the factor history is collinear."
        )
    fitted = design @ coefficients
    residual_sum_squares = float(np.sum((excess_returns - fitted) ** 2))
    total_sum_squares = float(np.sum((excess_returns - excess_returns.mean()) ** 2))
    r_squared = (
        1 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 1e-20 else None
    )
    return FactorRegression(
        annual_alpha=float(coefficients[0] * TRADING_DAYS),
        market_beta=float(coefficients[1]),
        smb_beta=float(coefficients[2]),
        hml_beta=float(coefficients[3]),
        r_squared=r_squared,
        observations=len(combined),
        start_date=combined.index[0],
        end_date=combined.index[-1],
    )
