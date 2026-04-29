from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.gnn.dataset import ClipDataset, FrameLoader
from models.gnn.model import IN_DIM, MischiefGNN


def _collate_fn(batch: list) -> tuple[list, torch.Tensor]:
    # batch_size=1; unpack the single (graphs_list, label) pair
    graphs_list, labels = zip(*batch)
    return list(graphs_list[0]), torch.stack(labels)


def train_gnn(
    clips: list[dict],
    frame_loader: FrameLoader,
    checkpoint_path: str,
    epochs: int = 50,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    device: str = "cpu",
) -> None:
    dataset = ClipDataset(clips, frame_loader)
    n_val   = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,  collate_fn=_collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False, collate_fn=_collate_fn)

    model     = MischiefGNN(in_dim=IN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for graphs, label in train_loader:
            label  = label.to(device)
            graphs = [g.to(device) for g in graphs]
            pred   = model(graphs)
            loss   = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for graphs, label in val_loader:
                label  = label.to(device)
                graphs = [g.to(device) for g in graphs]
                val_loss += criterion(model(graphs), label).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    print(f"Best val loss: {best_val_loss:.4f}  → saved to {checkpoint_path}")
