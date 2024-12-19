from Models.Transformers.Embeddings import tAPE
import torch
import torch.nn as nn


class BiTranWrapper(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes,
                 d_model=64, n_heads=4, n_layers=2, dropout=0.1,
                 use_tape=True, scale_factor=1.0):
        super(BiTranWrapper, self).__init__()

        self.use_tape = use_tape

        # Embedding Layer (Separates input mapping from positional encoding)
        self.embed_layer = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # Positional Encoding (using tAPE or fallback to simpler encoding)
        if use_tape:
            self.positional_encoder = tAPE(
                input_dim=d_model,  # d_model after embedding
                d_model=d_model,
                dropout=dropout,
                scale_factor=scale_factor
            )
        else:
            self.positional_encoder = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            )

        # Transformer encoder layers
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
        # Step 1: Embedding (map input to d_model)
        x = self.embed_layer(x)  # Shape: (batch_size, seq_len, d_model)

        # Step 2: Positional Encoding (tAPE or default)
        x = self.positional_encoder(x)

        # Step 3: Transpose for Transformer (seq_len, batch_size, features)
        x = x.transpose(0, 1)

        # Step 4: Transformer Encoder
        x = self.transformer_encoder(x)

        # Step 5: Take the last time step and classify
        x = x[-1, :, :]  # Take the last time step's features

        # Step 6: Classification
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
