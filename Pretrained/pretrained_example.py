import torch
from Models.Transformers.tAPETransformer.model import *

def load_pretrained_model(file_path, data):
    
    # Configuration
    config = TrainingConfig(data)
    # We set the sequence length at 10 (common choice in LOB data experiences)
    config.seq_len = 10
    # Optimal batch size of 64
    config.batch_size = 64

    model = BiTranWrapper(
            input_dim=config.input_dim,
            seq_len=config.seq_len,
            num_classes=config.num_classes,
            use_tape=True,
            n_heads=8
        )
    
    model.load_state_dict(torch.load(file_path))
    return model