import torch
import torch.optim as optim
import torch.nn as nn
from src.Preprocessing.Preprocessing import prepare_data

# Training Utilities
def train_chunk(model, train_data, config, chunk_size, class_weights):
    """Handles training for each chunk of data."""
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(config.device))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Prepare DataLoader for the current chunk
    train_loader, _ = prepare_data(train_data, sequence_length=config.seq_len, batch_size=config.batch_size, test_size=0)

    # Training Loop for current chunk
    total_train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(config.device), batch_y.to(config.device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()

    return model, total_train_loss, train_correct, train_total


def train_model_in_chunks(model, class_weights, train_data, val_data, config, chunk_size=10000):
    """Handles training in chunks, with evaluation after each epoch."""

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Split the training data into chunks
    num_chunks = len(train_data) // chunk_size
    if len(train_data) % chunk_size != 0:
        num_chunks += 1  # account for any leftover data

    # Training Loop (over epochs)
    for epoch in range(config.num_epochs):
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        # Process each chunk of data
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(train_data))
            chunk = train_data[start_idx:end_idx]

            # Train on the current chunk
            model, chunk_train_loss, chunk_train_correct, chunk_train_total = train_chunk(
                model, chunk, config, chunk_size, class_weights
            )

            # Accumulate metrics across chunks
            total_train_loss += chunk_train_loss
            train_correct += chunk_train_correct
            train_total += chunk_train_total
            print(f"\nTraining on chunk {chunk_idx} finished")

        # Validation after each epoch
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            _, val_loader = prepare_data(val_data, sequence_length=config.seq_len, batch_size=config.batch_size, test_size=1)
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(config.device), batch_y.to(config.device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        # Calculate Accuracy
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total

        # Print Epoch Summary
        print(f'Epoch [{epoch+1}/{config.num_epochs}]')
        print(f'Train Loss: {total_train_loss/len(train_data):.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss/len(val_data):.4f}, Accuracy: {val_accuracy:.2f}%')

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

    return model