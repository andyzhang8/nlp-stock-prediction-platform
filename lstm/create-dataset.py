import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_data_with_interval(ticker, start_date, end_date, interval, retries=3):
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval,
            )
            if not df.empty:
                df['Ticker'] = ticker
                df['DateTime'] = df.index
                df.reset_index(drop=True, inplace=True)
                return df
        except Exception as e:
            print(f"Error fetching {interval} data for {ticker} on attempt {attempt + 1}: {e}")
        time.sleep(1)  
    print(f"Failed to fetch {interval} data for {ticker} after {retries} retries.")
    return pd.DataFrame()

def fetch_stock_data(tickers, start_date, end_date):
    all_data = []
    failed_tickers = []

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        current_start = start_date
        has_data = False

        while current_start < end_date:
            current_end = min(current_start + timedelta(days=60), end_date)

            interval = "1h" if (end_date - current_start).days <= 730 else "1d"

            df = fetch_data_with_interval(ticker, current_start, current_end, interval)
            if not df.empty:
                all_data.append(df)
                has_data = True

            current_start = current_end
            time.sleep(1)

        if not has_data:
            failed_tickers.append(ticker)

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data.reset_index(drop=True, inplace=True)

        combined_data.dropna(inplace=True)
        combined_data = combined_data.astype({
            'Open': 'float64',
            'High': 'float64',
            'Low': 'float64',
            'Close': 'float64',
            'Volume': 'float64'
        })

        combined_data = combined_data[['DateTime', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
        combined_data.sort_values(by=['DateTime', 'Ticker'], inplace=True) 

        if failed_tickers:
            print(f"The following tickers had no data: {', '.join(failed_tickers)}")

        return combined_data
    else:
        print("No data was fetched.")
        return pd.DataFrame()

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
start_date = datetime.now() - timedelta(days=365 * 10)
end_date = datetime.now()

stock_data = fetch_stock_data(tickers, start_date, end_date)

try:
    if not stock_data.empty:
        stock_data.to_csv("stock_prices_cleaned.csv", index=False)
        print("Cleaned dataset saved to stock_prices_cleaned.csv")
    else:
        print("No data to save.")
except Exception as e:
    print(f"An error occurred while saving data: {e}")
    if 'all_data' in locals() and all_data:
        backup_data = pd.concat(all_data, ignore_index=True)
        backup_data.to_csv("stock_prices_partial.csv", index=False)
        print("Partial data saved to stock_prices_partial.csv due to an error.")
