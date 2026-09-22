import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceDataset(Dataset):
    """
    Sliding window sequence dataset for temporal risk evaluation.
    Enforces strict dimension and value boundaries.
    """
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, expected_features: int = 5):
        if not isinstance(sequences, np.ndarray) or not isinstance(labels, np.ndarray):
            raise TypeError("Sequences and labels must be NumPy ndarrays.")

        if sequences.ndim != 3:
            raise ValueError(f"Sequences must have 3 dimensions (N, seq_len, features). Received: {sequences.shape}")

        if sequences.shape[2] != expected_features:
            raise ValueError(f"Expected {expected_features} features per step, found {sequences.shape[2]}")

        if len(sequences) != len(labels):
            raise ValueError(f"Sample count mismatch: {len(sequences)} sequences vs {len(labels)} labels.")

        if np.isnan(sequences).any():
            raise ValueError("Input sequences contain NaN values.")

        unique_labels = set(np.unique(labels))
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Binary classification requires labels in {{0, 1}}. Found: {unique_labels}")

        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]