import requests
import pandas as pd
import time
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
            level=logging.WARNING,  # Set the minimum log level
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
        
        data: Dict = data[0]
       
        if any(key not in data for key in ['timestamp', 'indicators']):
            self.logger.warning(f"Timestamp and indicators not found for {ticker}!")
            return {}
    
        timestamp: int = data['timestamp']
        if not all(isinstance(value, int) for value in timestamp):
            self.logger.warning(f"Invalid timetamp data from {ticker}!")
            return {}
        
        indicators: Dict = data['indicators']['quote'][0]
        if not self._validate_data_format(indicators):
            self.logger.warning(f"Invalid price data from {ticker}!")
            return {}
        
        for key, values in indicators.items():
            indicators[key] = [round(value, 2) if value is not None else -1 for value in values]


        indicators['timestamps'] = timestamp
        out[ticker] = indicators

        return out

    def update_stock_data(self, new_data: Dict):
        """
        Updates the MongoDB collection with stock data. Uses timestamps as the primary keys.
        Each timestamp contains a dictionary of companies with their stock data.

        :param new_data: Dictionary containing stock data keyed by company name.
        """
        
        for company, data in new_data.items():
            for i, timestamp in enumerate(data["timestamps"]):
                company_data = {
                    "volume": data["volume"][i],
                    "high": data["high"][i],
                    "low": data["low"][i],
                    "open": data["open"][i],
                    "close": data["close"][i],
                }

                existing_entry = self.collection.find_one({"_id": timestamp})

                if existing_entry:
                    existing_entry["companies"][company] = company_data
                    self.collection.update_one(
                        {"_id": timestamp},
                        {"$set": {"companies": existing_entry["companies"]}}
                    )
                else:
                    self.collection.insert_one({
                        "_id": timestamp,
                        "companies": {company: company_data}
                    })


        
    def update_db(self, num_workers=4):
        start_time = time.time()
        aggregated_data = {}
        lock = threading.Lock()
        
        def threaded_fetch_data(ticker):
            with lock:
                data = self.get_data(ticker, Endpoints.STOCK_DATA, "1h", "1y")
                time.sleep(0.5)
            return ticker, data

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_ticker = {
                executor.submit(threaded_fetch_data, ticker): ticker for ticker in self.tickers
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                ticker, data = future.result()
                aggregated_data[ticker] = data

                if len(aggregated_data) == num_workers:
                    self.update_stock_data(aggregated_data)
                    aggregated_data = {}

        if aggregated_data:
            self.update_stock_data(aggregated_data)

            

        duration = time.time() - start_time
        logging.info(f"Parsed {len(self.tickers)} tickers in {duration:.2f}s")


    def _validate_data_format(self, data):
        required_keys = ['open', 'high', 'volume', 'low', 'close']
        
        if not all(key in data for key in required_keys):
            return False
        
        for key in required_keys:
            values = data[key]
            if not isinstance(values, list):
                return False
            
        return True
                



if __name__ == '__main__':
    stock = StockData()
    # print(stock.get_data('NVDA', Endpoints.STOCK_DATA, "1h", "1d"))
    stock.update_db()

