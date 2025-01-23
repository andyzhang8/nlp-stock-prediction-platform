import torch
import torch.nn as nn

class LSTMLightweight(nn.Module):
    def __init__(
        self, 
        input_size=1, 
        hidden_size=16, 
        num_layers=2,
        output_size=1, 
        dropout=0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.linear_in = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )
        
        self.fc_hidden = nn.Linear(hidden_size, hidden_size // 2)
        self.fc_out = nn.Linear(hidden_size // 2, output_size)

        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        x = h_n[-1, :, :] 
        
        x = self.fc_hidden(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc_out(x)
        return out.squeeze(-1)

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
        
        # linear layers
        nn.init.kaiming_normal_(self.fc_hidden.weight)
        nn.init.constant_(self.fc_hidden.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0.0)
