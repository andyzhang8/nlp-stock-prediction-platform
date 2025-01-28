import os
import glob
import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from train import LSTMModel, StocksDataset

def evaluate(model, data_loader, device='cpu'):
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.unsqueeze(-1).to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate trained model on a test set.")
    parser.add_argument('--test_dir', type=str, required=True, 
                        help='Directory containing CSV files for testing')
    parser.add_argument('--model_path', type=str, default='model.pth', 
                        help='Path to the saved model weights')
    parser.add_argument('--sequence_length', type=int, default=60, 
                        help='Sequence length used in training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = StocksDataset(args.test_dir, sequence_length=args.sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = LSTMModel(input_size=1, hidden_size=64, num_layers=1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    test_loss = evaluate(model, test_loader, device=device)
    print(f"Test MSE Loss: {test_loss:.6f}")
