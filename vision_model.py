import cv2
import torch
import numpy as np
from typing import Tuple

class VisionPredictor:
    def __init__(self, model_path: str = None, device: str = 'cpu') -> None:
        self.device = torch.device(device)
        self.model = self._load_model(model_path) if model_path else None

    def _load_model(self, path: str) -> torch.nn.Module:
        try:
            model = torch.load(path, map_location=self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load vision model: {e}")

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (224, 224))
        normalized = resized / 255.0
        
        tensor = torch.from_numpy(normalized).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0) 
        return tensor.to(self.device)

    def predict(self, frame: np.ndarray) -> Tuple[str, float]:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided for inference.")
        
        # Mock prediction for pipeline testing until weights are trained
        if self.model is None:
            return "NORMAL", 0.92
            
        tensor = self.preprocess(frame)
        with torch.no_grad():
            output = self.model(tensor)
            confidence, predicted = torch.max(output, 1)
            
        status = "DROWSY" if predicted.item() == 1 else "NORMAL"
        return status, float(confidence.item())