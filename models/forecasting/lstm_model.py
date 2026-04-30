import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class UnivariateLSTM(nn.Module):
    """
    LSTM model for univariate timeseries forecasting.
    Predicts the next 5 days of returns given a lookback window of historical returns.
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, output_size: int = 5):
        super(UnivariateLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Batch_first=True -> Input shape: (batch_size, sequence_length, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer maps hidden state to a 5-day forecast
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (batch, seq_len, 1)
        :return: Tensor of shape (batch, 5) representing next 5 return predictions
        """
        # Initialize hidden state (h0, c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output of the last time step
        out = out[:, -1, :] 
        
        # Map to 5-day forecast
        out = self.fc(out)
        return out

def create_sequences(data: np.ndarray, seq_length: int, forecast_horizon: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts 1D array of returns into (X, y) sequences for training.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        x = data[i:(i + seq_length)]
        y = data[(i + seq_length):(i + seq_length + forecast_horizon)]
        xs.append(x)
        ys.append(y)
    return np.array(xs)[..., np.newaxis], np.array(ys)
