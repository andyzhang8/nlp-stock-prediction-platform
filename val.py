import torch
import torch.nn as nn
import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

sys.path.append(os.path.expanduser('~/nlp-stock-prediction-platform'))

from models.lstm2.model import StockLSTM
from data.stock_dataloader import StockDataset

MODEL_PATH = os.path.expanduser('~/nlp-stock-prediction-platform/models/lstm2/best_model.pth')
DATA_PATH = "data/db_dump/stocks/trends.bson"
SEQ_LENGTH = 50
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(num_companies, embedding_dim=32, input_dim=4, hidden_dim=64, num_layers=2, dropout=0.2):
    model = StockLSTM(num_companies, embedding_dim, input_dim, hidden_dim, num_layers, dropout)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("model loaded")
    return model

def get_test_data():
    dataset = StockDataset(DATA_PATH, seq_length=SEQ_LENGTH)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    tickers = []
    actual_prices = []
    
    for i in range(len(dataset)):
        company_idx = i % len(dataset.companies)
        company = dataset.companies[company_idx]
        tickers.append(company)

        target_col = f"companies.{company}.close"
        actual_price = dataset.df.iloc[i + dataset.seq_length][target_col]
        actual_prices.append(actual_price)

    num_companies = len(dataset.companies)
    return test_loader, tickers, actual_prices, dataset, num_companies

def run_inference(model, test_loader, dataset):
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            company_ids = batch[0].to(DEVICE)
            features = batch[1].to(DEVICE)

            outputs = model(company_ids, features)  
            scaled_preds = outputs.cpu().numpy()

            for i, company_id in enumerate(company_ids.cpu().numpy()):
                company = dataset.companies[company_id]
                target_col = f"companies.{company}.close"
                min_price = dataset.feature_mins[target_col]
                max_price = dataset.feature_maxs[target_col]
                
                original_price = scaled_preds[i] * (max_price - min_price) + min_price
                predictions.append(original_price)

    return predictions

if __name__ == "__main__":
    test_loader, tickers, actual_prices, dataset, num_companies = get_test_data()
    model = load_model(num_companies)

    predictions = run_inference(model, test_loader, dataset)

    results_df = pd.DataFrame({
        "Ticker": tickers[:len(predictions)],  
        "Actual_Price": actual_prices[:len(predictions)],  
        "Predicted_Price": predictions
    })

    print("\nPredicted vs. Actual:")
    print(results_df.head(10))

    results_df.to_csv("stock_predictions_comparison.csv", index=False)
    print("\nPredictions saved to stock_predictions_comparison.csv")
