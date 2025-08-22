# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(model, loader, optimizer, criterion, device):
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
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()

        # The model predicts the 16-dimensional structural vector
        out = model(data)

        # The ground truth is stored in data.y
        loss = criterion(out, data.y)

        loss.backward()
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
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

