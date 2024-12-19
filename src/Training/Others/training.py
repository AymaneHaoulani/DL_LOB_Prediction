import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import seaborn as sns
from tqdm import tqdm
import itertools
import random
import os
import joblib


from src.Preprocessing.FeatureEngineering import *
from src.Preprocessing.LabelGeneration import *
from src.Preprocessing.Preprocessing import *
from src.Preprocessing.dataset import *
from Models.Others.model_architectures import *


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

save_base_dir = '../trained_models/model'
os.makedirs(save_base_dir, exist_ok=True)

lstm_models_dir = os.path.join(save_base_dir, 'LSTM')
cnn_lstm_models_dir = os.path.join(save_base_dir, 'CNN_LSTM')
logreg_models_dir = os.path.join(save_base_dir, 'LogReg')

os.makedirs(lstm_models_dir, exist_ok=True)
os.makedirs(cnn_lstm_models_dir, exist_ok=True)
os.makedirs(logreg_models_dir, exist_ok=True)

train_set_path = './data/train.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_model(model, train_dataset, criterion, optimizer, device, num_epochs=2, max_grad_norm=1.0):
    for epoch in range(1, num_epochs+1):
        model.train()
        train_dataset.data_iter = pd.read_csv(train_dataset.file_path, chunksize=train_dataset.chunk_size)
        train_dataset.buffer = pd.DataFrame()
        train_dataset.end_of_file = False

        train_losses = []
        batch_iterator = train_dataset.get_batches()
        progress_bar = tqdm(batch_iterator, desc=f'Epoch {epoch}/{num_epochs}')

        for X_batch, y_batch in progress_bar:
            # Ensure model and batches are on the same device
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_train_loss = np.mean(train_losses)
        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

def evaluate(model, dataset, device, criterion=None):
    model.eval()
    losses = []
    all_preds, all_targets = [], []
    with torch.no_grad():
        dataset.data_iter = pd.read_csv(dataset.file_path, chunksize=dataset.chunk_size)
        dataset.buffer = pd.DataFrame()
        dataset.end_of_file = False
        for X_batch, y_batch in dataset.get_batches():
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            if criterion is not None:
                loss = criterion(outputs, y_batch)
                losses.append(loss.item())
    avg_loss = np.mean(losses) if len(losses) > 0 else None
    return all_targets, all_preds, avg_loss

def run_full_pipeline_for_horizon(horizon, train_path, test_path, device, sequence_length=10, batch_size=64, num_epochs=2):
    scaler = StandardScaler()
    print(f"\n--- Horizon = {horizon} ---")
    print("Training on train.csv")

    train_dataset = StreamingDataset(
        horizon=horizon,
        theta=get_best_theta(horizon),
        file_path=train_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        device=device,
        mode='train',
        scaler=scaler
    )

    weights = compute_class_weights_streaming(train_dataset, device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    input_size = len(train_dataset.get_features())
    output_size = 3
    hidden_size = 50
    num_layers = 2
    dropout_rate = 0.2

    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        device=device,
        num_layers=num_layers,
        dropout=dropout_rate
    )

    model.apply(init_weights)
    model.to(device)  # Move model to the device

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(
        model=model,
        train_dataset=train_dataset,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs
    )

    # Define the path to save the model
    model_save_path = os.path.join(lstm_models_dir, f'lstm_horizon_{horizon}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"LSTM model saved at: {model_save_path}")

    # Define the path to save the scaler
    scaler_save_path = os.path.join(lstm_models_dir, f'scaler_lstm_horizon_{horizon}.pkl')
    joblib.dump(scaler, scaler_save_path)
    print(f"LSTM scaler saved at: {scaler_save_path}")


def run_full_pipeline_for_horizon_cnn_lstm(horizon, train_path, test_path, device, sequence_length=10, batch_size=64, num_epochs=2):

    # Initialize scaler
    scaler = StandardScaler()

    print(f"\n--- CNN_LSTM Model | Horizon = {horizon} ---")
    print("Training on train.csv")

    # Create training dataset
    train_dataset = StreamingDataset(
        horizon=horizon,
        theta=get_best_theta(horizon),
        file_path=train_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        device=device,
        mode='train',
        scaler=scaler
    )

    # Define loss function
    # For multi-class classification, CrossEntropyLoss is suitable
    weights = compute_class_weights_streaming(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=weights)


    # Model parameters
    input_size = len(train_dataset.get_features())
    output_size = 3  # Assuming three classes: D, S, U
    lstm_hidden_dim_1 = 512
    lstm_hidden_dim_2 = 320
    dropout_rate = 0.5

    # Create the CNN_LSTM model
    model = CNN_LSTM_Model(
        input_channels=input_size,
        lstm_hidden_dim=lstm_hidden_dim_1,
        lstm_hidden_dim_2=lstm_hidden_dim_2,
        num_classes=output_size,
        dropout_rate=dropout_rate
    )

    # Apply the weight initialization
    model.apply(init_weights)

    # Move model to the specified device
    model.to(device)

    # Define optimizer with weight decay for L2 regularization (optional)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Train the model
    train_model(
        model=model,
        train_dataset=train_dataset,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs
    )

    # Define the path to save the model
    model_save_path = os.path.join(cnn_lstm_models_dir, f'cnn_lstm_horizon_{horizon}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"CNN_LSTM model saved at: {model_save_path}")

    # Define the path to save the scaler
    scaler_save_path = os.path.join(cnn_lstm_models_dir, f'scaler_cnn_lstm_horizon_{horizon}.pkl')
    joblib.dump(scaler, scaler_save_path)
    print(f"CNN_LSTM scaler saved at: {scaler_save_path}")

def run_full_pipeline_for_horizon_logreg(horizon, train_path, test_path, device, sequence_length=10, batch_size=64, num_epochs=2):
    # Initialize scaler
    scaler = StandardScaler()

    print(f"\n--- Logistic Regression | Horizon = {horizon} ---")
    print("Training on train.csv")

    # Create training dataset (Logistic Regression version)
    train_dataset = StreamingDatasetLogReg(
        horizon=horizon,
        theta=get_best_theta(horizon),
        file_path=train_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        device=device,
        mode='train',
        scaler=scaler
    )

    # Compute class weights
    weights = compute_class_weights_streaming(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Determine input dimension for logistic regression
    num_features = len(train_dataset.get_features())
    input_dim = sequence_length * num_features
    output_dim = 3  # D, S, U

    # Initialize logistic regression model
    model = LogisticRegressionModel(input_dim=input_dim, output_dim=output_dim)
    model.apply(init_weights)
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train model
    train_model(
        model=model,
        train_dataset=train_dataset,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs
    )

    # You can define a directory for saving logistic regression models if you like
    logreg_models_dir = os.path.join(save_base_dir, 'LogReg')
    os.makedirs(logreg_models_dir, exist_ok=True)

    # Save the logistic regression model
    model_save_path = os.path.join(logreg_models_dir, f'logreg_horizon_{horizon}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Logistic Regression model saved at: {model_save_path}")

    # Save the scaler
    scaler_save_path = os.path.join(logreg_models_dir, f'scaler_logreg_horizon_{horizon}.pkl')
    joblib.dump(scaler, scaler_save_path)
    print(f"Logistic Regression scaler saved at: {scaler_save_path}")




# Logistic Regression Training

horizons = [10, 20, 50, 100]

for h in horizons:
    print(f"\nTraining Logistic Regression model for horizon = {h}...")
    run_full_pipeline_for_horizon_logreg(
        horizon=h,
        train_path=train_set_path,
        test_path=test_set_path,
        device=device,
        sequence_length=10,
        batch_size=64,
        num_epochs=2
    )
    

# LSTM Training
horizons = [10, 20, 50, 100]

for h in horizons:
    print(f"\nTraining experiment for horizon = {h}...")
    run_full_pipeline_for_horizon(
        horizon=h,
        train_path=train_set_path,
        test_path=test_set_path,
        device=device,
        sequence_length=10,
        batch_size=64,
        num_epochs=2
    )

# CNN-LSTM Training

horizons = [10, 20, 50, 100]

for h in horizons:
    print(f"\n=== Training CNN_LSTM Model for Horizon = {h} ===")
    run_full_pipeline_for_horizon_cnn_lstm(
        horizon=h,
        train_path=train_set_path,
        test_path=test_set_path,
        device=device,
        sequence_length=10,
        batch_size=64,
        num_epochs=2
    )
