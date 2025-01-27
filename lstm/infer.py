import torch
import pandas as pd
import argparse
from torch.utils.data import DataLoader, Dataset
from train import LSTMModel

class StockDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        inputs = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + self.sequence_length]
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def load_data_from_csv(csv_path, sequence_length, column_name='close'):
    df = pd.read_csv(csv_path)
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the CSV file.")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    stock_prices = df[column_name].values
    normalized_prices = (stock_prices - stock_prices.mean()) / stock_prices.std()

    dataset = StockDataset(normalized_prices, sequence_length)
    return DataLoader(dataset, batch_size=1, shuffle=False), stock_prices.mean(), stock_prices.std()


def test_model(model_path, data_loader, device='cpu'):
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, target in data_loader:
            inputs = inputs.unsqueeze(-1).to(device) 
            target = target.to(device)

            output = model(inputs)
            predictions.append(output.item())
            actuals.append(target.item())

    return predictions, actuals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test stock prediction model.")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model.')
    parser.add_argument('--sequence_length', type=int, default=4320, help='Number of hours (time steps) for prediction.')
    parser.add_argument('--column_name', type=str, default='close', help='Column name containing stock prices.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader, mean, std = load_data_from_csv(
        args.csv_path, args.sequence_length, column_name=args.column_name
    )

    predictions, actuals = test_model(args.model_path, test_loader, device)

    predictions = [p * std + mean for p in predictions]
    actuals = [a * std + mean for a in actuals]

    for i, (pred, actual) in enumerate(zip(predictions, actuals)):
        print(f"Prediction {i + 1}: {pred:.2f}, Actual: {actual:.2f}")
