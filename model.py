import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = torch.unsqueeze(x, dim=1)
        x = self.linear(x)
        return x


