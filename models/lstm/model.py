import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, num_companies, embedding_dim=32, input_dim=4, hidden_dim=64, num_layers=2):
        super(StockLSTM, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=num_companies, embedding_dim=embedding_dim)

        self.lstm = nn.LSTM(input_size=input_dim + embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, company_id, x: torch.Tensor):
        company_embedding: torch.Tensor = self.embedding(company_id)

        company_embedding = company_embedding.unsqueeze(1).repeat(1, x.shape[1], 1)  # (batch_size, seq_length, embedding_dim)

        x = torch.cat((x, company_embedding), dim=2)

        lstm_out, _ = self.lstm(x)

        output: torch.Tensor = self.fc(lstm_out[:, -1, :])  

        return output.squeeze(1)

