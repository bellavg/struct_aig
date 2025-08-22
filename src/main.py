# set up train and testing

# check for processed files first. have flag that says reprocess just in case the data is updated

# part 1

# TODO run with example data
# TODO make into pytorch lightning
# TODO add in padding for mask and x (check other laptop to see what needs padding), test all this to make sure it still works
# TODO full training of encoder? - check if makes sense.
# TODO make model Siamese and add difference vector

# -*- coding: utf-8 -*-
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import MSELoss
import os

# --- Local Imports ---
from src.data import AIGDataset, collate_aig
from src.dag_transformer import GraphTransformer
from src.train import train_epoch, evaluate


def main(args):
    """
    Main script to handle data loading, model training, and evaluation.
    """
    # --- 1. Setup and Device Configuration ---
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directory for saving models if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # --- 2. Load and Split Dataset ---
    print("Loading dataset...")
    full_dataset = AIGDataset(processed_file_path=args.dataset_path)

    # Define split sizes
    num_graphs = len(full_dataset)
    train_size = int(args.train_split * num_graphs)
    val_size = num_graphs - train_size

    print(f"Total graphs: {num_graphs}, Training: {train_size}, Validation: {val_size}")

    # Perform the split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_aig
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_aig
    )

    # --- 3. Initialize Model, Optimizer, and Loss Function ---
    print("Initializing model...")
    model = GraphTransformer(
        in_size=4,  # [const, pi, gate, po]
        num_edge_features=2,  # [regular, inverted]
        out_size=16,  # Target vector size
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = MSELoss()  # Mean Squared Error is suitable for regression

    # --- 4. Training Loop ---
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path} (Val Loss: {best_val_loss:.4f})")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Graph Transformer for AIG structural attribute regression.')

    # --- Dataset and Saving Arguments ---
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the processed .pkl.gz AIG dataset file.')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--train_split', type=float, default=0.8, help='Proportion of the dataset to use for training.')

    # --- Model Hyperparameters ---
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model embeddings.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer encoder layers.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')

    # --- Training Arguments ---
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--device', type=int, default=0, help='CUDA device index to use.')

    args = parser.parse_args()
    main(args)
