import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class SafeDriveLSTM(nn.Module):
    """
    LSTM network for temporal sequence processing of 6-axis sensor data.
    """
    def __init__(self, input_size: int = 6, hidden_size: int = 64, num_layers: int = 2, num_classes: int = 2) -> None:
        super(SafeDriveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Isolate the hidden state of the final time step
        out = out[:, -1, :] 
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class SensorPredictor:
    def __init__(self, model_path: str = None, device: str = 'cpu') -> None:
        self.device = torch.device(device)
        self.model = SafeDriveLSTM().to(self.device)
        
        if model_path:
            self._load_weights(model_path)
        else:
            self.model.eval() 

    def _load_weights(self, path: str) -> None:
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load sensor model weights: {e}")

    def preprocess(self, sensor_window: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(sensor_window).float()
        
        # Z-score normalization for sensor stability
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True) + 1e-5
        tensor = (tensor - mean) / std
        
        # Add batch dimension: (1, seq_len, 6)
        tensor = tensor.unsqueeze(0) 
        return tensor.to(self.device)

    def predict(self, sensor_window: np.ndarray) -> Tuple[str, float]:
        if sensor_window is None or sensor_window.size == 0:
            raise ValueError("Invalid sensor window provided for inference.")
            
        tensor = self.preprocess(sensor_window)
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        status = "IMPAIRED" if predicted.item() == 1 else "NORMAL"
        return status, float(confidence.item())