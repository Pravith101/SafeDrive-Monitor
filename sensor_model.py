import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class SensorPredictor:
    def __init__(self, model_path: str = None, device: str = 'cpu') -> None:
        self.device = torch.device(device)
        self.model = self._load_model(model_path) if model_path else None

    def _load_model(self, path: str) -> nn.Module:
        try:
            model = torch.load(path, map_location=self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load sensor model: {e}")

    def preprocess(self, sensor_data: np.ndarray) -> torch.Tensor:
        # Expected shape: (sequence_length, num_features) 
        # e.g., 50 consecutive reads of AccX, AccY, AccZ, GyroX, GyroY, GyroZ
        tensor = torch.from_numpy(sensor_data).float()
        tensor = tensor.unsqueeze(0) 
        return tensor.to(self.device)

    def predict(self, sensor_window: np.ndarray) -> Tuple[str, float]:
        if sensor_window is None or sensor_window.size == 0:
            raise ValueError("Invalid sensor window provided for inference.")
            
        # Mock prediction for pipeline testing
        if self.model is None:
            return "NORMAL", 0.88
            
        tensor = self.preprocess(sensor_window)
        with torch.no_grad():
            output = self.model(tensor)
            confidence, predicted = torch.max(output, 1)
            
        status = "IMPAIRED" if predicted.item() == 1 else "NORMAL"
        return status, float(confidence.item())