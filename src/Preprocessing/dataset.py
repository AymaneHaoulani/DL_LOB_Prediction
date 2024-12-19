import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from src.Preprocessing.FeatureEngineering import *
from src.Preprocessing.LabelGeneration import *
from src.Preprocessing.Preprocessing import *
from Models.Others.model_architectures import *

class StreamingDataset(Dataset):
    """
    This class is used to create an iterator that retrieves the data in chunks to save memory usage.
    The class is used for the LSTM and CNN-LSTM architecture
    """
    def __init__(self, file_path, horizon, theta, sequence_length, batch_size, device, scaler=None, mode='train', chunk_size=64*2000, feature_expansion=True):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.scaler = scaler
        self.mode = mode
        self.features = None
        self.chunk_size = chunk_size
        self.data_iter = pd.read_csv(file_path, chunksize=chunk_size)
        self.buffer = pd.DataFrame()
        self.end_of_file = False
        self.total_samples = 0
        self.feature_expansion = feature_expansion
        self.horizon = horizon
        self.theta = theta
        self.batch_size = batch_size
        self.device = device
        self.load_features()

    def get_features(self):
        return self.features

    def load_features(self):
        row = pd.read_csv(self.file_path, nrows=1)
        processed_data = first_preprocessing_step(row, self.horizon, self.theta, self.feature_expansion)
        self.features = [col for col in processed_data.columns if col not in ['label', 'Update ID', 'Timestamp']]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        pass

    # The method returns an iterator that iterates through the data in chunks and batches.
    def get_batches(self):
        batch_X, batch_y = [], []
        while not self.end_of_file:
            try:
                chunk = next(self.data_iter)
            except StopIteration:
                self.end_of_file = True
                chunk = pd.DataFrame()
            except OSError as e:
                print(f"Error reading chunk: {e}")
                break

            data_chunk = pd.concat([self.buffer, chunk], ignore_index=True)

            if chunk.empty:
                self.end_of_file = True

            
            data_chunk = first_preprocessing_step(data_chunk, self.horizon, self.theta, self.feature_expansion)

            num_sequences = len(data_chunk) - self.sequence_length - self.horizon + 1

            if num_sequences <= 0:
                self.buffer = data_chunk
                continue

            X = data_chunk[self.features].values
            y = data_chunk['label'].values

            if self.scaler is not None:
                if self.mode == 'train':
                    self.scaler.partial_fit(X)
                X = self.scaler.transform(X)

            for i in range(num_sequences):
                X_seq = X[i:i + self.sequence_length]
                y_seq = y[i + self.sequence_length - 1]

                batch_X.append(X_seq)
                batch_y.append(y_seq)

                if len(batch_X) == self.batch_size:
                    batch_X_np = np.array(batch_X, dtype=np.float32)
                    batch_y_np = np.array(batch_y, dtype=np.int64)
                    yield torch.from_numpy(batch_X_np).to(self.device), torch.from_numpy(batch_y_np).to(self.device)
                    batch_X, batch_y = [], []

            # Yield last partial batch if any
            if len(batch_X) > 0:
                batch_X_np = np.array(batch_X, dtype=np.float32)
                batch_y_np = np.array(batch_y, dtype=np.int64)
                yield torch.from_numpy(batch_X_np).to(self.device), torch.from_numpy(batch_y_np).to(self.device)
                batch_X, batch_y = [], []

            self.buffer = data_chunk.iloc[-(self.sequence_length + self.horizon - 1):].copy()
            self.total_samples += num_sequences

class StreamingDatasetLogReg(Dataset):
    """
    This class is used to create an iterator that retrieves the data in chunks to save memory usage.
    The class is used for the logistic regression architecture
    """
    def __init__(self, file_path, horizon, theta, sequence_length, batch_size, device, scaler=None, mode='train', chunk_size=64*2000, feature_expansion=True):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.scaler = scaler
        self.mode = mode
        self.features = None
        self.chunk_size = chunk_size
        self.data_iter = pd.read_csv(file_path, chunksize=chunk_size)
        self.buffer = pd.DataFrame()
        self.end_of_file = False
        self.total_samples = 0
        self.feature_expansion = feature_expansion
        self.horizon = horizon
        self.theta = theta
        self.batch_size = batch_size
        self.device = device
        self.load_features()

    def get_features(self):
        return self.features

    def load_features(self):
        row = pd.read_csv(self.file_path, nrows=1)
        processed_data = first_preprocessing_step(row, self.horizon, self.theta, self.feature_expansion)
        self.features = [col for col in processed_data.columns if col not in ['label', 'Update ID', 'Timestamp']]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        pass

    def get_batches(self):
        batch_X, batch_y = [], []
        while not self.end_of_file:
            try:
                chunk = next(self.data_iter)
            except StopIteration:
                self.end_of_file = True
                chunk = pd.DataFrame()
            except OSError as e:
                print(f"Error reading chunk: {e}")
                break

            data_chunk = pd.concat([self.buffer, chunk], ignore_index=True)

            if chunk.empty:
                self.end_of_file = True

            data_chunk = first_preprocessing_step(data_chunk, self.horizon, self.theta, self.feature_expansion)

            num_sequences = len(data_chunk) - self.sequence_length - self.horizon + 1

            if num_sequences <= 0:
                self.buffer = data_chunk
                continue

            X = data_chunk[self.features].values
            y = data_chunk['label'].values

            if self.scaler is not None:
                if self.mode == 'train':
                    self.scaler.partial_fit(X)
                X = self.scaler.transform(X)

            feature_dim = len(self.features)
            for i in range(num_sequences):
                X_seq = X[i:i + self.sequence_length]
                y_seq = y[i + self.sequence_length - 1]

                X_flat = X_seq.flatten()

                batch_X.append(X_flat)
                batch_y.append(y_seq)

                if len(batch_X) == self.batch_size:
                    batch_X_np = np.array(batch_X, dtype=np.float32)
                    batch_y_np = np.array(batch_y, dtype=np.int64)
                    yield torch.from_numpy(batch_X_np).to(self.device), torch.from_numpy(batch_y_np).to(self.device)
                    batch_X, batch_y = [], []

            # Yield last partial batch if any
            if len(batch_X) > 0:
                batch_X_np = np.array(batch_X, dtype=np.float32)
                batch_y_np = np.array(batch_y, dtype=np.int64)
                yield torch.from_numpy(batch_X_np).to(self.device), torch.from_numpy(batch_y_np).to(self.device)
                batch_X, batch_y = [], []

            self.buffer = data_chunk.iloc[-(self.sequence_length + self.horizon - 1):].copy()
            self.total_samples += num_sequences


def compute_class_weights_streaming(dataset, device):
    y_train = []
    batch_generator = dataset.get_batches()
    for X_batch, y_batch in batch_generator:
        y_train.extend(y_batch.cpu().numpy())
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    return class_weights
