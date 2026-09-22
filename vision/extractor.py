import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class FrameMetrics:
    ear: float
    mar: float
    pitch: float
    yaw: float
    roll: float

class VisionExtractor:
    def __init__(self):
        # MediaPipe landmark indices for left and right eye corners
        self.LEFT_EYE_OUTER = 33
        self.RIGHT_EYE_OUTER = 263

    def compute_ear(self, landmarks, eye_indices: List[int], img_w: int, img_h: int) -> float:
        points = [np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h]) for idx in eye_indices]
        # Vertical distances
        v1 = np.linalg.norm(points[1] - points[5])
        v2 = np.linalg.norm(points[2] - points[4])
        # Horizontal distance
        h = np.linalg.norm(points[0] - points[3])
        return float((v1 + v2) / (2.0 * (h + 1e-6)))

    def compute_mar(self, landmarks, mouth_indices: List[int], img_w: int, img_h: int) -> float:
        points = [np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h]) for idx in mouth_indices]
        v = np.linalg.norm(points[1] - points[3])
        h = np.linalg.norm(points[0] - points[2])
        return float(v / (h + 1e-6))

    def compute_roll(self, left_eye_corner: Tuple[float, float], right_eye_corner: Tuple[float, float]) -> float:
        dx = right_eye_corner[0] - left_eye_corner[0]
        dy = right_eye_corner[1] - left_eye_corner[1]
        angle_rad = math.atan2(dy, dx)
        return float(math.degrees(angle_rad))

    def extract_metrics(self, landmarks, img_w: int, img_h: int) -> FrameMetrics:
        # Outer eye landmark coordinates for roll calculation
        left_corner = (landmarks[self.LEFT_EYE_OUTER].x * img_w, landmarks[self.LEFT_EYE_OUTER].y * img_h)
        right_corner = (landmarks[self.RIGHT_EYE_OUTER].x * img_w, landmarks[self.RIGHT_EYE_OUTER].y * img_h)
        
        calculated_roll = self.compute_roll(left_corner, right_corner)

        # 6-point indices for EAR & 4-point indices for MAR
        left_indices = [33, 160, 158, 133, 153, 144]
        right_indices = [362, 385, 387, 263, 373, 380]
        mouth_indices = [61, 81, 291, 178]

        ear_l = self.compute_ear(landmarks, left_indices, img_w, img_h)
        ear_r = self.compute_ear(landmarks, right_indices, img_w, img_h)
        avg_ear = (ear_l + ear_r) / 2.0
        mar = self.compute_mar(landmarks, mouth_indices, img_w, img_h)

        return FrameMetrics(
            ear=avg_ear,
            mar=mar,
            pitch=0.0,
            yaw=0.0,
            roll=calculated_roll
        )