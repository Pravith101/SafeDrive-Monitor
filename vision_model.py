import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class SafeDriveCNN(nn.Module):
    """
    Lightweight Convolutional Neural Network optimized for spatial facial feature extraction.
    """
    def __init__(self, num_classes: int = 2) -> None:
        super(SafeDriveCNN, self).__init__()
        # Input: 1 channel (grayscale), 224x224 image
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(-1, 64 * 28 * 28) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class VisionPredictor:
    def __init__(self, model_path: str = None, device: str = 'cpu') -> None:
        self.device = torch.device(device)
        self.model = SafeDriveCNN().to(self.device)
        
        if model_path:
            self._load_weights(model_path)
        else:
            self.model.eval() # Running initialized weights for pipeline testing

    def _load_weights(self, path: str) -> None:
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load vision model weights: {e}")

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
            
        tensor = self.preprocess(frame)
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        status = "DROWSY" if predicted.item() == 1 else "NORMAL"
        return status, float(confidence.item())