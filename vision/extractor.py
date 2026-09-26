"""Facial geometry features in the order EAR, MAR, pitch, yaw, roll.

Pose angles are OpenCV solvePnP Euler angles in degrees, relative to the
camera-facing canonical face model. Roll is the eye-line angle for consistency
with the original feature definition.
"""
import math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class FrameMetrics:
    ear: float
    mar: float
    pitch: float
    yaw: float
    roll: float


class VisionExtractor:
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    FEATURES = ("ear", "mar", "pitch", "yaw", "roll")

    # Approximate 3D face points paired with MediaPipe FaceMesh landmarks.
    _POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)
    _FACE_MODEL = np.asarray([
        (0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1),
    ], dtype=np.float64)

    def compute_ear(self, landmarks, eye_indices: List[int], img_w: int, img_h: int) -> float:
        points = [np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h]) for idx in eye_indices]
        v1 = np.linalg.norm(points[1] - points[5])
        v2 = np.linalg.norm(points[2] - points[4])
        h = np.linalg.norm(points[0] - points[3])
        return float((v1 + v2) / (2.0 * (h + 1e-6)))

    def compute_mar(self, landmarks, mouth_indices: List[int], img_w: int, img_h: int) -> float:
        points = [np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h]) for idx in mouth_indices]
        return float(np.linalg.norm(points[1] - points[3]) / (np.linalg.norm(points[0] - points[2]) + 1e-6))

    def compute_roll(self, left_eye_corner: Tuple[float, float], right_eye_corner: Tuple[float, float]) -> float:
        return float(math.degrees(math.atan2(right_eye_corner[1] - left_eye_corner[1],
                                             right_eye_corner[0] - left_eye_corner[0])))

    def compute_pose(self, landmarks, img_w: int, img_h: int) -> Tuple[float, float]:
        image_points = np.asarray([(landmarks[i].x * img_w, landmarks[i].y * img_h)
                                   for i in self._POSE_LANDMARKS], dtype=np.float64)
        focal = float(img_w)
        camera = np.asarray([[focal, 0, img_w / 2], [0, focal, img_h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rotation, _ = cv2.solvePnP(self._FACE_MODEL, image_points, camera,
                                       np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError("Could not estimate face pose from landmarks")
        matrix, _ = cv2.Rodrigues(rotation)
        sy = math.hypot(matrix[0, 0], matrix[1, 0])
        if sy < 1e-6:
            x = math.atan2(-matrix[1, 2], matrix[1, 1])
            y = math.atan2(-matrix[2, 0], sy)
        else:
            x = math.atan2(matrix[2, 1], matrix[2, 2])
            y = math.atan2(-matrix[2, 0], sy)
        return float(math.degrees(x)), float(math.degrees(y))

    def extract_metrics(self, landmarks, img_w: int, img_h: int) -> FrameMetrics:
        left = (landmarks[self.LEFT_EYE_OUTER].x * img_w, landmarks[self.LEFT_EYE_OUTER].y * img_h)
        right = (landmarks[self.RIGHT_EYE_OUTER].x * img_w, landmarks[self.RIGHT_EYE_OUTER].y * img_h)
        left_indices = [33, 160, 158, 133, 153, 144]
        right_indices = [362, 385, 387, 263, 373, 380]
        mouth_indices = [61, 81, 291, 178]
        ear = (self.compute_ear(landmarks, left_indices, img_w, img_h) +
               self.compute_ear(landmarks, right_indices, img_w, img_h)) / 2.0
        pitch, yaw = self.compute_pose(landmarks, img_w, img_h)
        values = (ear, self.compute_mar(landmarks, mouth_indices, img_w, img_h), pitch, yaw,
                  self.compute_roll(left, right))
        if not np.isfinite(values).all():
            raise ValueError("Face landmarks produced non-finite features")
        return FrameMetrics(*map(float, values))
