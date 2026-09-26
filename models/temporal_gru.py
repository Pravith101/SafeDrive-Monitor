"""Legacy UTA-RLDD endpoint GRU; the current FL3D model is driver_state_cnn.py."""
from pathlib import Path
import json
import random

import numpy as np
import torch
import joblib
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from data.preprocess_activity import LABELS

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("ear", "mar", "pitch", "yaw", "roll")
CHECKPOINT_VERSION = 1


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
    processed = ROOT / "data/processed"
    required = [processed / name for name in ("sequences.npy", "labels.npy", "groups.npy", "metadata.json",
                                                "feature_scaler.joblib")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing preprocessed artifacts: " + ", ".join(missing) +
                                ". Run python data/preprocess_activity.py first.")
    x = np.load(required[0]).astype(np.float32)
    y = np.load(required[1]).astype(np.int64)
    groups = np.load(required[2]).astype(str)
    metadata = json.loads(required[3].read_text(encoding="utf-8"))
    scaler = joblib.load(required[4])
    if getattr(scaler, "n_features_in_", None) != len(FEATURES):
        raise ValueError("Preprocessing scaler has incompatible feature count")
    sequence_length = metadata.get("sequence_length")
    expected_labels = {str(k): value for k, value in LABELS.items()}
    if (metadata.get("feature_order") != list(FEATURES) or
            metadata.get("labels") != expected_labels or not isinstance(sequence_length, int)):
        raise ValueError("Preprocessing metadata has incompatible features, labels, or sequence length")
    if x.ndim != 3 or x.shape[1:] != (sequence_length, len(FEATURES)) or len(x) != len(y) or len(y) != len(groups):
        raise ValueError(f"Expected X=(N,{sequence_length},{len(FEATURES)}), matching y and groups; "
                         f"got {x.shape}, {y.shape}, {groups.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Training features contain NaN or infinite values")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training data must contain both labels 0 and 1.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=metadata.get("validation_fraction", 0.2),
                                 random_state=metadata.get("split_seed", 42))
    train_idx, val_idx = next(splitter.split(x, y, groups))
    actual_train_groups = sorted(np.unique(groups[train_idx]).tolist())
    if metadata.get("scaler_fit_groups") != actual_train_groups:
        raise ValueError("Scaler training-video groups do not match this train/validation split; rerun preprocessing")
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
        raise ValueError("Preprocessing scaler contains non-finite parameters")
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
        raise ValueError("Video-level split must contain both labels in train and validation. "
                         "Add more labeled videos or adjust the split.")
    data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    train_set = torch.utils.data.Subset(data, train_idx.tolist())
    val_set = torch.utils.data.Subset(data, val_idx.tolist())
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TemporalGRU(input_dim=len(FEATURES))
    model = train_model(model, train_loader, val_loader, epochs=40, device=device)
    _, correct, total = evaluate(model, val_loader, device)
    accuracy = 100 * correct / max(total, 1)
    weights = ROOT / "weights"
    weights.mkdir(exist_ok=True)
    torch.save({"format_version": CHECKPOINT_VERSION, "state_dict": model.state_dict(),
                "input_dim": len(FEATURES), "hidden_dim": 64, "num_layers": 2,
                "num_classes": 2, "sequence_length": sequence_length,
                "feature_order": list(FEATURES), "labels": metadata.get("labels"),
                "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                "split_seed": metadata.get("split_seed"),
                "validation_fraction": metadata.get("validation_fraction"),
                "val_accuracy": accuracy},
               weights / "temporal_gru.pth")
    print(f"Best-checkpoint validation accuracy: {accuracy:.2f}%")
    print(f"Saved {weights / 'temporal_gru.pth'}")


if __name__ == "__main__":
    main()
