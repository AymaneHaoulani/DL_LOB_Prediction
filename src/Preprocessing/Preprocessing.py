from sklearn.model_selection import train_test_split
import numpy as np
import torch
from LabelGeneration import *
from FeatureEngineering import *

def create_sequences(X, y, seq_length):
        """
        Create sequences for time series analysis with sliding window

        Args:
        X (np.ndarray): Normalized features
        y (np.ndarray): Labels
        seq_length (int): Length of each sequence

        Returns:
        Tuple of numpy arrays: (sequences, corresponding labels)
        """
        sequences = []
        labels = []

        # Sliding window with step of 1
        for i in range(len(X) - seq_length + 1):
            # Sequence of features
            seq = X[i:i+seq_length]
            # Label is the last label in the sequence
            label = y[i+seq_length-1]

            sequences.append(seq)
            labels.append(label)

        return np.array(sequences), np.array(labels)


def prepare_market_data(data, sequence_length=10, test_size=0.3):
    """
    Prepare market data for time series analysis with crypto limit order book features.

    Args:
    data (pd.DataFrame): Preprocessed dataframe with features and labels
    sequence_length (int): Length of sequence window
    test_size (float): Proportion of data for testing
    random_state (int): Random seed for reproducibility

    Returns:
    Tuple of PyTorch tensors: (X_train_seq, X_test_seq, y_train_seq, y_test_seq)
    """
    # Remove non-feature columns
    features = [col for col in data.columns if col not in ['label', 'UpdateID', 'Timestamp']]
    X = data[features].values
    y = data['label'].values

    # Normalize features
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1e-8, std)
    X_normalized = (X - mean) / std

    # Create sequences
    X_seq, y_seq = create_sequences(X_normalized, y, sequence_length)

    # Handling edge cases when test_size is 0 or 1
    if test_size == 0:
        X_train_seq, y_train_seq = X_seq, y_seq
        X_test_seq, y_test_seq = np.array([]), np.array([])  # No test data
    elif test_size == 1:
        X_train_seq, y_train_seq = np.array([]), np.array([])  # No training data
        X_test_seq, y_test_seq = X_seq, y_seq
    else:
        # Split into train and test sets
        test_rows_size = int(len(X_seq) * test_size)

        # Training data (remaining data excluding test portion)
        X_train_seq, y_train_seq = X_seq[:-test_rows_size], y_seq[:-test_rows_size]

        # Test data (last portion of data)
        X_test_seq, y_test_seq = X_seq[-test_rows_size:], y_seq[-test_rows_size:]

    # Convert to PyTorch tensors
    return (
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(X_test_seq, dtype=torch.float32),
        torch.tensor(y_train_seq, dtype=torch.long),
        torch.tensor(y_test_seq, dtype=torch.long)
    )



def create_dataloaders(X_train_seq, X_test_seq, y_train_seq, y_test_seq, batch_size=64):
    """
    Create PyTorch DataLoaders for training and testing.

    Args:
    X_train_seq (torch.Tensor): Training feature sequences
    X_test_seq (torch.Tensor): Testing feature sequences
    y_train_seq (torch.Tensor): Training labels
    y_test_seq (torch.Tensor): Testing labels
    batch_size (int): Batch size for DataLoaders

    Returns:
    Tuple of PyTorch DataLoaders: (train_loader, test_loader)
    """
    # Check if training data is available
    if len(X_train_seq) == 0 or len(y_train_seq) == 0:
        train_loader = None
    else:
        train_dataset = torch.utils.data.TensorDataset(X_train_seq, y_train_seq)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,  # Adjust based on your system
            pin_memory=True
        )

    # Check if testing data is available
    if len(X_test_seq) == 0 or len(y_test_seq) == 0:
        test_loader = None
    else:
        test_dataset = torch.utils.data.TensorDataset(X_test_seq, y_test_seq)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    return train_loader, test_loader


def prepare_data(data, sequence_length=10, batch_size=64, test_size=0.3):
    # Ensure data is preprocessed with feature engineering and label encoding
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = prepare_market_data(data, sequence_length=sequence_length, test_size=test_size)

    train_loader, test_loader = create_dataloaders(X_train_seq, X_test_seq, y_train_seq, y_test_seq, batch_size=batch_size)

    return train_loader, test_loader


# Function to split the dataset into train and test sets
def split_data(data, train_size=0.8):
    split_idx = int(len(data) * train_size)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data

def feature_engineering_and_labels_generation(data, k):
    """
    Generate labels for the given data using midprice variations.

    Args:
    data (pd.DataFrame): The input DataFrame.
    k (int): The window size for calculating midprice variation.

    Returns:
    pd.DataFrame: A new DataFrame with generated labels and added features.
    """
    # Create a copy of the data to avoid modifying a slice
    data = data.copy()

    # Generate labels and features
    theta = get_best_theta(k)
    data['label'] = get_midprice_variation_column(data, k, theta)
    data = add_features(data)

    # Apply label encoding
    data['label'] = label_encoding(data['label'])

    return data