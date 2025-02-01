import torch
import torch.nn as nn
import os
import sys
import numpy as np
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
    print("Model loaded for inference.")
    return model

def get_test_data():
    dataset = StockDataset(DATA_PATH, seq_length=SEQ_LENGTH)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_companies = len(dataset.companies)  
    return test_loader, num_companies

def run_inference(model, test_loader):
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            company_ids = batch[0].to(DEVICE)
            features = batch[1].to(DEVICE)

            outputs = model(company_ids, features)  
            
            predictions.extend(outputs.cpu().numpy())  

    return predictions

if __name__ == "__main__":
    test_loader, num_companies = get_test_data()
    model = load_model(num_companies)

    predictions = run_inference(model, test_loader)

    print("\nPredicted Stock Prices:")
    for i, pred in enumerate(predictions[:10]): 
        print(f"Sample {i+1}: Predicted Price = {pred:.4f}")

    np.savetxt("stock_predictions.csv", predictions, delimiter=",", header="Predicted_Stock_Price", comments="")
    print("\nPredictions saved to stock_predictions.csv")
