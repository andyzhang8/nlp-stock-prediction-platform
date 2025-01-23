import torch
import torch.nn as nn

class LSTMLightweight(nn.Module):
    def __init__(
        self, 
        input_size=1, 
        hidden_size=16, 
        num_layers=1, 
        output_size=1, 
        dropout=0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.linear_in = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        # single layer because we're poor
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_size, output_size)

        # comment out if need simpler model
        self.init_weights()



    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.linear_in(x)
        x = self.relu(x)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1, :, :]
        
        x = self.dropout(x)
        out = self.linear_out(x)
        return out.squeeze(-1)



    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)