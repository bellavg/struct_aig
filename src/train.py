# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Performs one full training epoch.

    Args:
        model (torch.nn.Module): The GraphTransformer model to train.
        loader (DataLoader): The DataLoader for the training set.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        criterion (torch.nn.Module): The loss function (e.g., MSELoss).
        device (torch.device): The device to train on (e.g., 'cuda' or 'cpu').

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(tqdm(loader, desc="Training")):
        data = data.to(device)
        optimizer.zero_grad()

        # The model predicts the 16-dimensional structural vector
        out = model(data)

        # Reshape the ground truth tensor to match the output shape
        # data.y is [batch_size * 16], we need [batch_size, 16]
        y = data.y.view(data.num_graphs, -1)

        # The ground truth is stored in data.y
        loss = criterion(out, y)
        # --- MONITORING PRINTS ---
        # 1. Print loss for every 10 batches
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Batch Loss: {loss.item():.4f}")

        # 2. For the first batch of each epoch, print a sample of predictions vs ground truth
        if batch_idx == 0:
            print("--- Sample Predictions vs. Ground Truth (Epoch {}) ---".format(epoch))
            # Print for the first 3 graphs in the batch
            num_samples = min(data.num_graphs, 3)
            for i in range(num_samples):
                print(f"Sample {i + 1} Pred: {[f'{val:.2f}' for val in out[i].tolist()]}")
                print(f"Sample {i + 1} True: {[f'{val:.2f}' for val in y[i].tolist()]}")
            print("----------------------------------------------------")
        # --- END MONITORING ---

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on a given dataset.

    Args:
        model (torch.nn.Module): The GraphTransformer model to evaluate.
        loader (DataLoader): The DataLoader for the validation or test set.
        criterion (torch.nn.Module): The loss function (e.g., MSELoss).
        device (torch.device): The device to evaluate on.

    Returns:
        float: The average evaluation loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating"):
            data = data.to(device)
            out = model(data)

            # Reshape the ground truth tensor to match the output shape
            y = data.y.view(data.num_graphs, -1)

            loss = criterion(out, y)
            total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)