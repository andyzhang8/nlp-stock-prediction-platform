import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_data_with_interval(ticker, start_date, end_date, interval):
    try:
        df = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval=interval,
        )
        if not df.empty:
            df['Ticker'] = ticker
        return df
    except Exception as e:
        print(f"Error fetching {interval} data for {ticker}: {e}")
        return pd.DataFrame()

def fetch_stock_data(tickers, start_date, end_date):
    all_data = []

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + timedelta(days=60), end_date)

            interval = "1h" if (end_date - current_start).days <= 730 else "1d"

            df = fetch_data_with_interval(ticker, current_start, current_end, interval)
            if not df.empty:
                all_data.append(df)

            current_start = current_end
            time.sleep(1)

    if all_data:
        combined_data = pd.concat(all_data)
        combined_data.reset_index(inplace=True)
        return combined_data
    else:
        print("No data was fetched.")
        return pd.DataFrame()

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
start_date = datetime.now() - timedelta(days=365 * 10)
end_date = datetime.now()

stock_data = fetch_stock_data(tickers, start_date, end_date)

if not stock_data.empty:
    stock_data.to_csv("stock_prices.csv", index=False)
    print("Dataset saved to stock_prices.csv")
else:
    print("No data to save.")
