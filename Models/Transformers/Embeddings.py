import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine

# Linear Positional Encoder
class LinearPositionalEncoder(nn.Module):
    def __init__(self, input_dim, d_model, dropout=0.1):
        super(LinearPositionalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.encoder(x)

# Time-Adaptive Positional Encoding (tAPE)
class tAPE(nn.Module):
    def __init__(self, input_dim, d_model=64, dropout=0.1, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.scale_factor = scale_factor

        # Linear transformation to match input_dim to d_model
        self.feature_projection = nn.Linear(input_dim, d_model)

    def _generate_positional_encoding(self, seq_len):
        # Generate the positional encodings
        pe = torch.zeros(seq_len, self.d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin((position * div_term) * (self.d_model / seq_len))
        pe[:, 1::2] = torch.cos((position * div_term) * (self.d_model / seq_len))
        return self.scale_factor * pe.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [seq_len, input_dim]
        """
        # Project input features to d_model using a linear layer
        x = self.feature_projection(x)  # Shape: [seq_len, d_model]

        # Generate positional encoding for the sequence length
        seq_len = x.size(1)
        pe = self._generate_positional_encoding(seq_len).to(x.device)

        # Add positional encoding to the input features and apply dropout
        x = x + pe
        return self.dropout(x)
