import requests
import pandas as pd
import time


# class StockData:

#     def __init__(self):
#         pass

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

df = pd.read_csv("data\\tickers.csv")
tickers = df.Company
for ticker in tickers:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1h&range=1d"
    response = requests.get(url, headers=headers)
    data = response.json()
    if data is None:
        print(ticker)
        continue

    data = data['chart']['result'][0]
    timestamp = data['timestamp']

    indicators = data['indicators']['quote'][0]

    time.sleep(1)

    # print(f"{ticker}: {prices}")