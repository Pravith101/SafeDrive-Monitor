"""Train and evaluate the SafeDrive temporal GRU on preprocessed arrays."""
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TemporalGRU(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64,
                 num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=0.2 if num_layers > 1 else 0.0)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(),
                                        nn.Dropout(0.2), nn.Linear(32, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.classifier(output[:, -1, :])


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 40, lr: float = 1e-3, device: str = "cpu") -> nn.Module:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    best_loss, best_weights, stale = float("inf"), None, 0
    patience = 9
    for epoch in range(epochs):
        model.train()
        loss_sum = correct = count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            count += y.size(0)
        val_loss, val_correct, val_count = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1:02d}/{epochs} train_loss={loss_sum/max(count,1):.4f} "
              f"train_acc={100*correct/max(count,1):.2f}% val_loss={val_loss:.4f} "
              f"val_acc={100*val_correct/max(val_count,1):.2f}% "
              f"lr={optimizer.param_groups[0]['lr']:.2g}")
        if val_loss < best_loss:
            best_loss, best_weights, stale = val_loss, {k: v.detach().cpu().clone()
                                                        for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping: validation loss did not improve.")
                break
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu"):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / max(total, 1), correct, total


def main() -> None:
    set_seed(42)
    x = np.load(ROOT / "data/processed/sequences.npy").astype(np.float32)
    y = np.load(ROOT / "data/processed/labels.npy").astype(np.int64)
    groups = np.load(ROOT / "data/processed/groups.npy").astype(str)
    if x.ndim != 3 or x.shape[1:] != (30, 5) or len(x) != len(y) or len(y) != len(groups):
        raise ValueError(f"Expected X=(N,30,5), y=(N,), groups=(N,), got {x.shape}, {y.shape}, {groups.shape}")
    if len(np.unique(y)) != 2:
        raise ValueError("Training data must contain both labels 0 and 1.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(x, y, groups))
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
        raise ValueError("Video-level split must contain both labels in train and validation. "
                         "Add more labeled videos or adjust the split.")
    data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    train_set = torch.utils.data.Subset(data, train_idx.tolist())
    val_set = torch.utils.data.Subset(data, val_idx.tolist())
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TemporalGRU()
    model = train_model(model, train_loader, val_loader, epochs=40, device=device)
    _, correct, total = evaluate(model, val_loader, device)
    accuracy = 100 * correct / max(total, 1)
    weights = ROOT / "weights"
    weights.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": 5, "hidden_dim": 64,
                "num_layers": 2, "num_classes": 2, "val_accuracy": accuracy},
               weights / "temporal_gru.pth")
    print(f"Best-checkpoint validation accuracy: {accuracy:.2f}%")
    print(f"Saved {weights / 'temporal_gru.pth'}")


if __name__ == "__main__":
    main()
