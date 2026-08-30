import torch
from torch.utils.data import Dataset
import numpy as np

class DrowsinessSequenceDataset(Dataset):
    def __init__(self, sequences, labels, sequence_length=30):
        """
        sequences: List or array of numerical frames (EAR, MAR, Pitch, Yaw, Roll)
        labels: List of integer labels (0: Alert, 1: Drowsy, 2: Impaired)
        sequence_length: Number of frames per time-series window (default 1 sec at 30fps)
        """
        self.sequences = sequences
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Retrieve the sequence and its corresponding label
        sequence_data = self.sequences[idx]
        label_data = self.labels[idx]
        
        # Convert to PyTorch tensors for the GRU model
        seq_tensor = torch.tensor(sequence_data, dtype=torch.float32)
        label_tensor = torch.tensor(label_data, dtype=torch.long)
        
        return seq_tensor, label_tensor