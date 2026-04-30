import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

class MultiAssetTransformer(nn.Module):
    """
    Transformer predicting 1-step forward returns jointly across N assets.
    Leverages shared attention mechanism to capture inter-asset relationships.
    """
    def __init__(self, num_assets: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super(MultiAssetTransformer, self).__init__()
        
        self.num_assets = num_assets
        
        # Expand raw (num_assets) returns into higher d_model embedding space
        self.embedding_layer = nn.Linear(num_assets, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Decode back into N return predictions
        self.decoder = nn.Linear(d_model, num_assets)
        
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        :param src: Input tensor of shape (batch_size, seq_len, num_assets)
        :return: Output tensor of shape (batch_size, num_assets)
        """
        # Transformer in PyTorch expects (seq_len, batch_size, features) by default
        src = src.transpose(0, 1) 
        
        src = self.embedding_layer(src)
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src)
        
        # We only care about predicting the next timestep, so take the final sequence output element
        output_last = output[-1, :, :]
        
        predictions = self.decoder(output_last)
        return predictions
