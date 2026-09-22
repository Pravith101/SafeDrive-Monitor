import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class TemporalDrowsinessGRU(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, num_layers: int = 2, num_classes: int = 2):
        super(TemporalDrowsinessGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        return self.classifier(last_step)

def train_model(model: nn.Module, loader: DataLoader, epochs: int = 5, lr: float = 0.001, device: str = "cpu") -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {total_loss / len(loader):.4f}")

def evaluate_model(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            preds = model(sequences).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = (correct / total) * 100.0 if total > 0 else 0.0
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Synthetic demo verification run
    print(f"Running pipeline self-check on device: {device}")
    synthetic_inputs = np.random.rand(100, 30, 5).astype(np.float32)
    synthetic_labels = np.random.randint(0, 2, size=(100,)).astype(np.int64)

    from sequence_dataset import SequenceDataset
    dataset = SequenceDataset(synthetic_inputs, synthetic_labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = TemporalDrowsinessGRU(input_dim=5, hidden_dim=32, num_layers=2)
    train_model(model, train_loader, epochs=3, device=device)
    evaluate_model(model, train_loader, device=device)
    
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/temporal_gru.pth")
    print("Model checkpoint saved to weights/temporal_gru.pth")