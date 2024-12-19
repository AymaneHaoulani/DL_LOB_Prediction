# make_predictions.py

import os
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random

import sys

# Suppress specific warnings
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from src.Preprocessing.FeatureEngineering import *
from src.Preprocessing.LabelGeneration import *
from src.Preprocessing.Preprocessing import *
from src.Preprocessing.dataset import *


horizons = [10, 20, 50, 100]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_base_dir = '../trained_models/model'
test_data_path = '../data/test.csv'
cm_save_dir = '../trained_models/model'


horizons, save_base_dir, test_data_path, device, cm_save_dir

def make_predictions(model_type, horizon, save_base_dir, test_data_path, device, sequence_length=10, batch_size=64):

    if model_type == 'LSTM':
        model_dir = os.path.join(save_base_dir, 'LSTM')
        model_filename = f'lstm_horizon_{horizon}.pth'
        scaler_filename = f'scaler_lstm_horizon_{horizon}.pkl'
        ModelClass = LSTMModel
        hidden_size = 50
        num_layers = 2
        dropout = 0.2
        output_size = 3
        dataset_class = StreamingDataset

    elif model_type == 'CNN_LSTM':
        model_dir = os.path.join(save_base_dir, 'CNN_LSTM')
        model_filename = f'cnn_lstm_horizon_{horizon}.pth'
        scaler_filename = f'scaler_cnn_lstm_horizon_{horizon}.pkl'
        ModelClass = CNN_LSTM_Model
        lstm_hidden_dim_1 = 512
        lstm_hidden_dim_2 = 320
        dropout_rate = 0.5
        num_classes = 3
        dataset_class = StreamingDataset

    elif model_type == 'logreg':
        model_dir = os.path.join(save_base_dir, 'LogReg')
        model_filename = f'logreg_horizon_{horizon}.pth'
        scaler_filename = f'scaler_logreg_horizon_{horizon}.pkl'
        ModelClass = LogisticRegressionModel
        output_dim = 3
        dataset_class = StreamingDatasetLogReg
    else:
        raise ValueError("model_type must be 'LSTM', 'CNN_LSTM', or 'logreg'.")

    model_path = os.path.join(model_dir, model_filename)
    scaler_path = os.path.join(model_dir, scaler_filename)

    if not os.path.exists(model_path):
        print(f"Model file does not exist: {model_path}")
        return None, None
    if not os.path.exists(scaler_path):
        print(f"Scaler file does not exist: {scaler_path}")
        return None, None

    # Load the scaler
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from {scaler_path}")

    # Determine input size/features
    theta = get_best_theta(horizon)
    sample_processed = first_preprocessing_step(pd.read_csv(test_data_path, nrows=1), horizon, theta, feature_expansion=True)
    features = [col for col in sample_processed.columns if col not in ['label', 'Update ID', 'Timestamp']]

    # Load model
    if model_type == 'LSTM':
        input_size = len(features)
        model = ModelClass(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            device=device,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type == 'CNN_LSTM':
        input_channels = len(features)
        model = ModelClass(
            input_channels=input_channels,
            lstm_hidden_dim=lstm_hidden_dim_1,
            lstm_hidden_dim_2=lstm_hidden_dim_2,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
    elif model_type == 'logreg':
        # For logistic regression: input_dim = sequence_length * number_of_features
        num_features = len(features)
        input_dim = sequence_length * num_features
        model = ModelClass(input_dim=input_dim, output_dim=output_dim)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"{model_type} model loaded from {model_path}")

    # Initialize the dataset for test data
    test_dataset = dataset_class(
        file_path=test_data_path,
        horizon=horizon,
        theta=theta,
        sequence_length=sequence_length,
        batch_size=batch_size,
        device=device,
        scaler=scaler,
        mode='test',
        feature_expansion=True
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_dataset.get_batches(), desc=f"Making Predictions {model_type} Horizon {horizon}"):
            outputs = model(X_batch)
            if model_type == 'CNN_LSTM':
                _, predicted = torch.max(outputs.data, 1)
            elif model_type == 'LSTM':
                _, predicted = torch.max(outputs.data, 1)
            elif model_type == 'logreg':
                _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    results_df = pd.DataFrame({
        'True_Label': all_targets,
        f'Predicted_Label_{model_type}_horizon_{horizon}': all_preds
    })

    return results_df, {0: 'D', 1: 'S', 2: 'U'}


def evaluate_predictions(results_df, model_type, horizon, label_mapping, save_dir=None):
    """
    Evaluates predictions by calculating accuracy and weighted F1 score,
    and prints the classification report.
    """
    y_true = results_df['True_Label']
    y_pred = results_df[f'Predicted_Label_{model_type}_horizon_{horizon}']

    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"--- {model_type} Model | Horizon = {horizon} ---")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Weighted F1 Score: {weighted_f1:.6f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['D', 'S', 'U']))
    print("\n" + "-"*50 + "\n")

def test_lstm_model(sequence_length=10, batch_size=64):
    """
    Tests the LSTM model across different horizons, evaluates predictions,
    calculates metrics, plots the summary, and saves the summary metrics to a CSV file.
    """
    model_type = 'LSTM'
    print(f"Starting tests for {model_type} model.\n")

    # Initialize an empty DataFrame to accumulate summary metrics
    summary_metrics = pd.DataFrame(columns=['Model_Type', 'Horizon', 'Accuracy', 'Weighted_F1'])

    for horizon in horizons:
        print(f"=== Predictions for {model_type} Model | Horizon = {horizon} ===")
        results_df, label_mapping = make_predictions(
            model_type=model_type,
            horizon=horizon,
            save_base_dir=save_base_dir,
            test_data_path=test_data_path,
            device=device,
            sequence_length=sequence_length,
            batch_size=batch_size
        )

        if results_df is not None and not results_df.empty:
            # Evaluate predictions and save confusion matrices
            evaluate_predictions(results_df, model_type, horizon, label_mapping, cm_save_dir)

            # Calculate metrics
            y_true = results_df['True_Label']
            y_pred = results_df[f'Predicted_Label_{model_type}_horizon_{horizon}']

            accuracy = accuracy_score(y_true, y_pred)
            weighted_f1 = f1_score(y_true, y_pred, average='weighted')

            # Append the metrics to the summary DataFrame
            new_entry = pd.DataFrame([{
                'Model_Type': model_type,
                'Horizon': horizon,
                'Accuracy': accuracy,
                'Weighted_F1': weighted_f1
            }])
            summary_metrics = pd.concat([summary_metrics, new_entry], ignore_index=True)
        else:
            print(f"No results to evaluate for {model_type} model at horizon {horizon}.\n")

    # Print Metrics Summary
    print(f"=== Metrics Summary - {model_type} ===")
    print(summary_metrics.to_string(index=False))
    print("\n" + "="*50 + "\n")

    # Define the path to save the summary metrics
    summary_save_path = os.path.join(save_base_dir, f'{model_type.lower()}_summary_metrics.csv')

    # Save the summary metrics to a CSV file
    summary_metrics.to_csv(summary_save_path, index=False)
    print(f"{model_type} metrics summary saved at: {summary_save_path}\n")

def test_cnn_lstm_model(sequence_length=10, batch_size=64):
    """
    Tests the CNN_LSTM model across different horizons, evaluates predictions,
    calculates metrics, plots the summary, and saves the summary metrics to a CSV file.
    """
    model_type = 'CNN_LSTM'
    print(f"Starting tests for {model_type} model.\n")

    # Initialize an empty DataFrame to accumulate summary metrics
    summary_metrics = pd.DataFrame(columns=['Model_Type', 'Horizon', 'Accuracy', 'Weighted_F1'])

    for horizon in horizons:
        print(f"=== Predictions for {model_type} Model | Horizon = {horizon} ===")
        results_df, label_mapping = make_predictions(
            model_type=model_type,
            horizon=horizon,
            save_base_dir=save_base_dir,
            test_data_path=test_data_path,
            device=device,
            sequence_length=sequence_length,
            batch_size=batch_size
        )

        if results_df is not None and not results_df.empty:
            # Evaluate predictions and save confusion matrices
            evaluate_predictions(results_df, model_type, horizon, label_mapping, cm_save_dir)

            # Calculate metrics
            y_true = results_df['True_Label']
            y_pred = results_df[f'Predicted_Label_{model_type}_horizon_{horizon}']

            accuracy = accuracy_score(y_true, y_pred)
            weighted_f1 = f1_score(y_true, y_pred, average='weighted')

            # Append the metrics to the summary DataFrame
            new_entry = pd.DataFrame([{
                'Model_Type': model_type,
                'Horizon': horizon,
                'Accuracy': accuracy,
                'Weighted_F1': weighted_f1
            }])
            summary_metrics = pd.concat([summary_metrics, new_entry], ignore_index=True)
        else:
            print(f"No results to evaluate for {model_type} model at horizon {horizon}.\n")

    # Print Metrics Summary
    print(f"=== Metrics Summary - {model_type} ===")
    print(summary_metrics.to_string(index=False))
    print("\n" + "="*50 + "\n")

    # Define the path to save the summary metrics
    summary_save_path = os.path.join(save_base_dir, f'{model_type.lower()}_summary_metrics.csv')

    # Save the summary metrics to a CSV file
    summary_metrics.to_csv(summary_save_path, index=False)
    print(f"{model_type} metrics summary saved at: {summary_save_path}\n")

def test_logistic_regression_model(sequence_length=10, batch_size=64):
    """
    Tests the Logistic Regression model across different horizons, evaluates predictions,
    calculates metrics, plots the summary, and saves the summary metrics to a CSV file.
    """
    model_type = 'logreg'
    print(f"Starting tests for {model_type} model.\n")

    # Initialize an empty DataFrame to accumulate summary metrics
    summary_metrics = pd.DataFrame(columns=['Model_Type', 'Horizon', 'Accuracy', 'Weighted_F1'])

    for horizon in horizons:
        print(f"=== Predictions for {model_type} Model | Horizon = {horizon} ===")
        results_df, label_mapping = make_predictions(
            model_type=model_type,
            horizon=horizon,
            save_base_dir=save_base_dir,
            test_data_path=test_data_path,
            device=device,
            sequence_length=sequence_length,
            batch_size=batch_size
        )

        if results_df is not None and not results_df.empty:
            # Evaluate predictions and save confusion matrices
            evaluate_predictions(results_df, model_type, horizon, label_mapping, cm_save_dir)

            # Calculate metrics
            y_true = results_df['True_Label']
            y_pred = results_df[f'Predicted_Label_{model_type}_horizon_{horizon}']

            accuracy = accuracy_score(y_true, y_pred)
            weighted_f1 = f1_score(y_true, y_pred, average='weighted')

            # Append the metrics to the summary DataFrame
            new_entry = pd.DataFrame([{
                'Model_Type': model_type,
                'Horizon': horizon,
                'Accuracy': accuracy,
                'Weighted_F1': weighted_f1
            }])
            summary_metrics = pd.concat([summary_metrics, new_entry], ignore_index=True)
        else:
            print(f"No results to evaluate for {model_type} model at horizon {horizon}.\n")

    # Print Metrics Summary
    print(f"=== Metrics Summary - {model_type} ===")
    print(summary_metrics.to_string(index=False))
    print("\n" + "="*50 + "\n")

    # Define the path to save the summary metrics
    summary_save_path = os.path.join(save_base_dir, f'{model_type}_summary_metrics.csv')

    # Save the summary metrics to a CSV file
    summary_metrics.to_csv(summary_save_path, index=False)
    print(f"{model_type.capitalize()} metrics summary saved at: {summary_save_path}\n")
