import requests
import pandas as pd
import time
from enum import Enum
import logging

from typing import List, Dict, Any

from db_helper import DB

class Endpoints(Enum):
    STOCK_DATA = "https://query1.finance.yahoo.com/v8/finance/chart/"
    QUOTE_SUMMARY = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/"
# AAPL?modules=assetProfile,summaryDetail,financialData,defaultKeyStatistics


class StockData:

    def __init__(self, csv_file="tickers"):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        self.df = pd.read_csv(f"data/{csv_file}.csv")
        self.tickers = self.df.Company

        logging.basicConfig(
            level=logging.DEBUG,  # Set the minimum log level
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
            handlers=[
                logging.FileHandler("app.log"),  # Log to a file
                logging.StreamHandler()  # Log to the console
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.db = DB(collection="trends")
        self.collection = self.db.collection


    def get_data(self, ticker: str, endpoint: Endpoints, interval: str, range: str) -> Dict[str, Any]:
        """
        Fetches data for the given tickers from the specified endpoint.

        Args:
            tickers (List[str]): A list of ticker symbols.
            endpoint (Endpoint): The API endpoint to fetch data from.
            interval (str): The interval for the data (e.g., "1d", "1h").
            range (str): The range for the data (e.g., "1mo", "1y").

        Returns:
            Dict[str, Any]: A dictionary containing the fetched data.
        """
        out = {}
    
        query_url = endpoint.value + f"{ticker}?interval={interval}&range={range}"
        response = requests.get(query_url, headers=self.headers)

        if response.status_code != 200:
            self.logger.error(f"Query for {ticker} failed on {endpoint}!")
            return {}

        data = response.json()['chart']['result']

        if data is None:
            self.logger.warning(f"Ticker {ticker} not found on {endpoint}!")
            return {}
        
        data = data[0]

        timestamp = data['timestamp']
        indicators = data['indicators']['quote'][0]

        for key, values in indicators.items():
            indicators[key] = [round(value, 2) for value in values]

        indicators['timestamps'] = timestamp

        out[ticker] = indicators

        return out

    def update_stock_data(self, new_data):
        """
        Updates the MongoDB collection with stock data. Appends data if the company exists, otherwise creates a new entry.

        :param db: The MongoDB database instance.
        :param collection_name: The name of the MongoDB collection.
        :param new_data: Dictionary containing stock data keyed by company name.
        """
        
        for company, data in new_data.items():
            existing_entry = self.collection.find_one({"ticker": company})

            if existing_entry:
                updated_data = {}
                for key in ["timestamps", "volume", "high", "low", "open", "close"]:
                    if key in data:
                        updated_values = existing_entry[key] + data[key]
                        sorted_data = sorted(
                            zip(updated_values, *[existing_entry[k] + data[k] for k in data if k != "timestamps"]),
                            key=lambda x: x[0]
                        )
                        sorted_values = list(zip(*sorted_data))
                        updated_data[key] = list(sorted_values[0])
                        for i, field in enumerate(["volume", "high", "low", "open", "close"], start=1):
                            updated_data[field] = list(sorted_values[i])
                
                self.collection.update_one(
                    {"ticker": company},
                    {"$set": updated_data}
                )
            else:
                data["ticker"] = company
                self.collection.insert_one(data)
        
    def update_db(self):
        for ticker in self.tickers:
            self.update_stock_data(self.get_data(ticker, Endpoints.STOCK_DATA, "1h", "1d"))
            time.sleep(1)
            



if __name__ == '__main__':
    stock = StockData()
    # print(stock.get_data('AAPL', Endpoints.STOCK_DATA, "1h", "1d"))
    stock.update_db()

