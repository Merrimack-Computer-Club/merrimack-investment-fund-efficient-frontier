# Portfolio Optimization with Efficient Frontier

This Streamlit application allows users to input a portfolio of stock tickers, select custom portfolio weights via interactive sliders, and visualize the portfolio's performance on the Efficient Frontier. The app provides insights into risk and return trade-offs and helps users find optimal portfolio allocations.

## Features

- **Input Stock Tickers**: Users can enter a list of stock symbols (e.g., AAPL, MSFT, TSLA) to analyze.
- **Adjust Portfolio Weights**: Dynamic sliders allow users to specify the weight of each stock in the portfolio.
- **Efficient Frontier Calculation**: The app computes the Efficient Frontier, which is a set of optimal portfolios that provide the best possible return for a given level of risk.
- **Custom Portfolio Comparison**: Users can compare their custom portfolio to the Efficient Frontier.
- **Interactive Visualization**: The results are displayed on an interactive graph using Plotly, where the Efficient Frontier and the custom portfolio are plotted.

## Getting Started

### Step 1: Clone the Repository

First, clone this repository to your local machine.

### Step 2: Install Dependencies

Install the required Python packages using `pip`.

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

To start the Streamlit application, run the following command in your terminal:

```bash
streamlit run efficient_frontier.py
```

This will start a local web server and open the app in your browser.

### Step 4: Use the App

1. **Enter Tickers**: Input a list of stock tickers (comma-separated) into the text box. For example: `AAPL, MSFT, TSLA`.
2. **Set Date Range**: Specify the start and end dates for historical data fetching.
3. **Adjust Weights**: Use the sliders to assign weights to each stock in your portfolio. The total weight will be automatically normalized to sum to 1.
4. **Update Portfolio**: Click the "Update Portfolio" button to calculate the Efficient Frontier and see how your custom portfolio compares.
5. **View Results**: The plot will display the Efficient Frontier and highlight your custom portfolio with its corresponding risk and return.

## Example

Here’s a quick example of how to use the app:

- Input the tickers `AAPL, MSFT, TSLA` in the ticker field.
- Set the portfolio weights using the sliders (e.g., 40% AAPL, 30% MSFT, 30% TSLA).
- Click "Update Portfolio" to visualize the portfolio’s performance against the Efficient Frontier.

## Analyst Recommendations

Open **analyst recommendations** in the sidebar and select a market index.
The page compares the individual companies in that index, not the index itself,
with their mean analyst price targets from Yahoo Finance.

- Supported lists: S&P 500, NASDAQ-100, Russell 1000, Dow Jones, FTSE 100,
  Nikkei 225, and SSE Composite. US/UK constituents come from Wikipedia, Nikkei
  constituents from TopForeignStocks, and SSE constituents from China Securities
  Index's downloadable list. The Nikkei source is dated January 2024 and may
  not reflect subsequent index changes; source links are shown on the page.
- Symbols are converted to Yahoo Finance format, including US share classes
  (`BRK.B` becomes `BRK-B`), London (`.L`), Tokyo (`.T`), and Shanghai (`.SS`).
- **Current Price** is the latest closing price, not a live quote. Price changes
  exclude dividends and use the last available close on or before the comparison
  date. YTD starts with the previous year's final close.
- Missing analyst coverage or insufficient history appears as a blank value,
  not a zero or an invented recommendation. Prices remain visible when a target
  request fails, and request failures are shown separately.
- Prices and targets are in each stock's quoted currency/unit (for example,
  London prices may be quoted in pence). They are not converted to a common currency.
- Constituents are cached for one day and stock data for one hour. **Refresh data**
  clears these caches. Large indexes can take several minutes; progress is shown.
  If Yahoo Finance limits requests, wait before retrying.

### Regression tests

Run the offline tests after installing `requirements.txt`:

```bash
python -m unittest discover -s tests -v
```

## Screenshots

![App Screenshot](assets/merrimack-smf-application-screenshot.png)  <!-- Add a screenshot of your app here -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### Additional Instructions:
- Replace `<repository-url>` with your actual repository URL.
- Replace `<repository-folder>` with the name of your cloned repository.
- Add an actual screenshot by placing it in the same folder as `README.md` and calling it `screenshot.png`, or modify the image path to reflect your file's name.
- Ensure you generate the `requirements.txt` file for all package dependencies:
  ```bash
  pip freeze > requirements.txt
  ```

This markdown is now ready to be copied and used as your `README.md` file.
