import torch
from torch.utils.data import Dataset
import numpy as np
from bson import decode_all
import pandas as pd

class StockDataset(Dataset):
    def __init__(self, data_path, seq_length=50, target_col='close'):
        with open(data_path, 'rb') as file:
            self.df: pd.DataFrame = pd.json_normalize(decode_all(file.read()))

        self.seq_length = seq_length
        self.companies = []
        seen_companies = set()

        for col in self.df.columns:
            if 'companies.' in col:
                company_name = col.split('.')[1]
                if company_name not in seen_companies:
                    self.companies.append(company_name)
                    seen_companies.add(company_name)

        self.company_mapping = {ticker: idx for idx, ticker in enumerate(self.companies)}

        self.feature_mins = {}
        self.feature_maxs = {}
        for company in self.companies:
            columns = [f"companies.{company}.open", f"companies.{company}.high",
                       f"companies.{company}.low", f"companies.{company}.close"]
            for col in columns:
                self.feature_mins[col] = self.df[col].min()
                self.feature_maxs[col] = self.df[col].max()
        


    def __len__(self):
        return len(self.df) - self.seq_length

    def __getitem__(self, idx):
        company_idx = idx % len(self.companies)
        time_idx = idx // len(self.companies)

        company = self.companies[company_idx]
        company_id = self.company_mapping[company]

        columns = [f"companies.{company}.open", f"companies.{company}.high",
                   f"companies.{company}.low", f"companies.{company}.close"]
        
    
        x = self.df.iloc[time_idx:time_idx + self.seq_length][columns].values
        x_scaled = (x - np.array([self.feature_mins[col] for col in columns])) / \
                   (np.array([self.feature_maxs[col] for col in columns]) - np.array([self.feature_mins[col] for col in columns]))

        # Extract and scale the target
        y = self.df.iloc[time_idx + self.seq_length][f"companies.{company}.close"]
        y_scaled = (y - self.feature_mins[f"companies.{company}.close"]) / \
                   (self.feature_maxs[f"companies.{company}.close"] - self.feature_mins[f"companies.{company}.close"])


        return (
            torch.tensor(company_id, dtype=torch.int),  
            torch.tensor(x_scaled, dtype=torch.float32),  
            torch.tensor(y_scaled, dtype=torch.float32)
        )




if __name__ == '__main__':
    ds = StockDataset("data/db_dump/stocks/trends.bson")
    for i in range(1):
        print(ds[i])