import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_alpha_vantage_hourly(ticker, api_key):
    all_data = []
    current_end = datetime.now()
    current_start = current_end - timedelta(days=60)

    while current_start > datetime.now() - timedelta(days=365 * 10):
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": ticker,
            "interval": "60min",
            "apikey": api_key,
            "outputsize": "full",
            "datatype": "json"
        }
        
        print(f"Fetching data for {ticker} from {current_start} to {current_end}...")

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if "Time Series (60min)" not in data:
                print(f"Error fetching data for {ticker}: {data.get('Note', data.get('Error Message', 'Unknown error'))}")
                break

            timeseries = data["Time Series (60min)"]
            df = pd.DataFrame.from_dict(timeseries, orient="index")
            df = df.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume"
            })
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            df = df[(df.index >= current_start) & (df.index <= current_end)]

            if not df.empty:
                all_data.append(df)

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            break

        current_end = current_start
        current_start = current_end - timedelta(days=60)

        time.sleep(12)

    if all_data:
        combined_data = pd.concat(all_data)
        return combined_data
    else:
        print("No data was fetched.")
        return pd.DataFrame()

api_key = "YOUR_ALPHA_VANTAGE_API_KEY"
tickers = ["AAPL", "MSFT", "GOOGL"]

all_ticker_data = []

for ticker in tickers:
    data = fetch_alpha_vantage_hourly(ticker, api_key)
    if not data.empty:
        data["Ticker"] = ticker
        all_ticker_data.append(data)

if all_ticker_data:
    combined_data = pd.concat(all_ticker_data)
    combined_data.reset_index(inplace=True)
    combined_data.rename(columns={"index": "Datetime"}, inplace=True)
    combined_data.to_csv("alpha_vantage_hourly_data.csv", index=False)
    print("Data saved to alpha_vantage_hourly_data.csv")
else:
    print("No data to save.")
