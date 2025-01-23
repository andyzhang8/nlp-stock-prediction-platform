import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_hourly_data(tickers, start_date, end_date):
    all_data = []

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + timedelta(days=60), end_date)

            try:
                df = yf.download(
                    ticker,
                    start=current_start.strftime('%Y-%m-%d'),
                    end=current_end.strftime('%Y-%m-%d'),
                    interval="1h",
                )

                if not df.empty:
                    df['Ticker'] = ticker
                    all_data.append(df)

            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")

            current_start = current_end

            # delay for rate limit
            time.sleep(1)

    # Conglomerate
    if all_data:
        combined_data = pd.concat(all_data)
        combined_data.reset_index(inplace=True)
        return combined_data
    else:
        print("No data was fetched.")
        return pd.DataFrame()

# parameters
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
start_date = datetime.now() - timedelta(days=365 * 10)
end_date = datetime.now()

hourly_data = fetch_hourly_data(tickers, start_date, end_date)

if not hourly_data.empty:
    hourly_data.to_csv("hourly_stock_prices.csv", index=False)
    print("Dataset saved to hourly_stock_prices.csv")
else:
    print("No data to save.")