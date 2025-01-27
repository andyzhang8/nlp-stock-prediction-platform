import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

from load import load_all_data
from model import StockGRU

def train_model(
    directory_path,        
    sequence_length=48,    
    batch_size=32,
    epochs=10,
    lr=1e-3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X_train, X_test, y_train, y_test = load_all_data(
        directory_path=directory_path,
        sequence_length=sequence_length
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = X_train.shape[2]
    model = StockGRU(input_size=input_size).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        accuracy = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Test Acc: {accuracy:.4f}")
    
    torch.save(model.state_dict(), "stock_gru.pth")
    
    model.eval()
    all_labels = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            for p in probs:
                if p >= 0.8:
                    all_labels.append("STRONG BUY")
                elif p >= 0.6:
                    all_labels.append("BUY")
                elif p >= 0.4:
                    all_labels.append("HOLD")
                else:
                    all_labels.append("SELL")

    print("Example multi-bucket recommendations on the test set:")
    print(all_labels[:20])

    return model

if __name__ == "__main__":
    directory_path = "./2021" 
    trained_model = train_model(
        directory_path=directory_path, 
        sequence_length=48, 
        epochs=10, 
        lr=1e-3
    )
