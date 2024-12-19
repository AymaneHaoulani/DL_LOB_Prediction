import torch
import torch.nn as nn
import numpy as np

# Model Definition
class TimeSeriesTransformerWrapper(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes,
                 d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super(TimeSeriesTransformerWrapper, self).__init__()

        # Transformer Encoder
        self.pos_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            dim_feedforward=d_model * 4
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=n_layers
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):

        # Positional encoding and transformation
        x = self.pos_encoder(x)

        # Transpose for transformer (seq_len, batch_size, features)
        x = x.transpose(0, 1)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Take the last time step and classify
        x = x[-1, :, :]

        # Classification
        return self.classifier(x)


# Training Configuration
class TrainingConfig:
    def __init__(self,data):
        # Model Hyperparameters
        features = [col for col in data.columns if col not in ['label', 'UpdateID', 'Timestamp']]
        self.input_dim = len(features)  # Number of input features
        self.seq_len = 10  # Sequence length
        self.num_classes = 3  # D, S, U

        # Training Hyperparameters
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.batch_size = 64
        self.num_epochs = 2

        # Device Configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
