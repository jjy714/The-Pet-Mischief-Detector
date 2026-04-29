from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.hybrid.dataset import HybridDataset, collate_hybrid
from models.hybrid.model import HybridMischiefModel


def train_hybrid(
    cache_dir: Path,
    checkpoint_path: str,
    epochs: int = 80,
    lr: float = 5e-4,
    batch_size: int = 4,
    val_fraction: float = 0.2,
    device: str = "cpu",
) -> None:
    """
    Train HybridMischiefModel on pre-computed cached samples.

    Args:
        cache_dir:        Directory containing .pt cache files from preprocessing.
        checkpoint_path:  Where to save the best model weights.
        epochs:           Total training epochs.
        lr:               AdamW learning rate.
        batch_size:       Samples per gradient step (PyG handles variable graphs).
        val_fraction:     Fraction of data held out for validation.
        device:           "cpu" or "cuda".
    """
    dataset = HybridDataset(cache_dir)
    n_val   = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,  collate_fn=collate_hybrid,
    )
    val_loader = DataLoader(
        val_set,   batch_size=batch_size, shuffle=False, collate_fn=collate_hybrid,
    )

    model     = HybridMischiefModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for global_feat, graph_batch, labels in train_loader:
            global_feat  = global_feat.to(device)
            graph_batch  = graph_batch.to(device)
            labels       = labels.to(device)
            pred = model(global_feat, graph_batch)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for global_feat, graph_batch, labels in val_loader:
                global_feat  = global_feat.to(device)
                graph_batch  = graph_batch.to(device)
                labels       = labels.to(device)
                val_loss += criterion(model(global_feat, graph_batch), labels).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    print(f"Best val loss: {best_val_loss:.4f}  → saved to {checkpoint_path}")
