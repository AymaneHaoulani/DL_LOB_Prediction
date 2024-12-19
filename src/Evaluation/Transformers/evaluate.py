from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch.nn as nn
from src.Preprocessing.Preprocessing import *

def evaluate_chunk(model, test_data, config, chunk_size):
    """Evaluate the model on a single chunk of data."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()

    # Prepare DataLoader for the current chunk
    _, test_loader = prepare_data(test_data, sequence_length=config.seq_len, batch_size=config.batch_size,test_size=1)

    # Evaluate over batches in the chunk
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(config.device), batch_y.to(config.device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            # Collect predictions and labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')

    return all_preds, all_labels, total_loss, accuracy, f1, recall, precision


def evaluate_model_in_chunks(model, val_data, config, chunk_size=10000):
    """Evaluate the model in chunks and calculate overall metrics."""
    num_chunks = len(val_data) // chunk_size
    if len(val_data) % chunk_size != 0:
        num_chunks += 1  # account for any leftover data

    total_loss = 0
    total_correct = 0
    total_total = 0
    all_preds = []
    all_labels = []

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(val_data))
        chunk = val_data[start_idx:end_idx]

        # Evaluate on the current chunk
        chunk_preds, chunk_labels, chunk_loss, chunk_accuracy, chunk_f1, chunk_recall, chunk_precision = evaluate_chunk(
            model, chunk, config, chunk_size
        )

        # Accumulate metrics across chunks
        total_loss += chunk_loss
        total_correct += chunk_accuracy * len(chunk)
        total_total += len(chunk)

        # Collect predictions and labels for overall metrics
        all_preds.extend(chunk_preds)
        all_labels.extend(chunk_labels)
        print(f"\nEvaluation on chunk {chunk_idx} finished")


    overall_accuracy = total_correct / total_total
    overall_f1 = f1_score(all_labels, all_preds, average='weighted')
    overall_recall = recall_score(all_labels, all_preds, average='weighted')
    overall_precision = precision_score(all_labels, all_preds, average='weighted')

    # Print overall metrics
    print(f'Overall Accuracy: {overall_accuracy:.2f}%')
    print(f'Overall F1 Score: {overall_f1:.4f}')
    print(f'Overall Recall: {overall_recall:.4f}')
    print(f'Overall Precision: {overall_precision:.4f}')

    return {
        'loss': total_loss / total_total,
        'accuracy': overall_accuracy,
        'f1': overall_f1,
        'recall': overall_recall,
        'precision': overall_precision
    }
