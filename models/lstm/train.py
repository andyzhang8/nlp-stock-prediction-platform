import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.expanduser('~/nlp-stock-prediction-platform'))
from data.stock_dataloader import StockDataset
from models.lstm.model import StockLSTM


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch in train_loader:
            company_ids = batch[0].to(device)
            features = batch[1].to(device)
            targets = batch[2].to(device)

            optimizer.zero_grad()

            outputs = model(company_ids, features)
            loss = criterion(outputs, targets)


            loss.backward()
            optimizer.step()


            epoch_train_loss += loss.item()


        avg_train_loss = epoch_train_loss / len(train_loader)


        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                company_ids = batch[0].to(device)
                features = batch[1].to(device)
                targets = batch[2].to(device)

                outputs = model(company_ids, features)
                loss = criterion(outputs, targets)

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")

    print("Training complete")


if __name__ == "__main__":
    data_path = "data/db_dump/stocks/trends.bson"
    seq_length = 50
    embedding_dim = 32
    input_dim = 4
    hidden_dim = 64
    num_layers = 2
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    validation_split = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = StockDataset(data_path, seq_length=seq_length)

    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_companies = len(dataset.companies)

    model = StockLSTM(num_companies, embedding_dim, input_dim, hidden_dim, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)