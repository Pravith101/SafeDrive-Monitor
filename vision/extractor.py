import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class FrameMetrics:
    ear: float
    mar: float
    pitch: float
    yaw: float
    roll: float

class VisionExtractor:
    def __init__(self):
        # Using the standard import now that Python 3.11 is active
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.left_eye = [362, 385, 386, 263, 374, 380]
        self.right_eye = [33, 159, 158, 133, 153, 145]
        self.mouth = [78, 81, 13, 311, 308, 178, 14, 402]

    def _compute_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _compute_ear(self, landmarks, eye_indices):
        h_dist = self._compute_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
        v_dist1 = self._compute_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
        v_dist2 = self._compute_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
        if h_dist == 0:
            return 0.0
        return (v_dist1 + v_dist2) / (2.0 * h_dist)

    def _compute_mar(self, landmarks):
        h_dist = self._compute_distance(landmarks[78], landmarks[308])
        v_dist = self._compute_distance(landmarks[13], landmarks[14])
        if h_dist == 0:
            return 0.0
        return v_dist / h_dist

    def process_frame(self, frame) -> Optional[FrameMetrics]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        coords = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(face_landmarks.landmark)}
        
        left_ear = self._compute_ear(coords, self.left_eye)
        right_ear = self._compute_ear(coords, self.right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = self._compute_mar(coords)
        
        nose = coords[1]
        left_cheek = coords[234]
        right_cheek = coords[454]
        top_head = coords[10]
        bottom_chin = coords[152]
        
        face_width = self._compute_distance(left_cheek, right_cheek)
        face_height = self._compute_distance(top_head, bottom_chin)
        
        yaw = ((nose[0] - left_cheek[0]) / face_width - 0.5) * 100 if face_width > 0 else 0.0
        pitch = ((nose[1] - top_head[1]) / face_height - 0.5) * 100 if face_height > 0 else 0.0
        roll = 0.0
        
        return FrameMetrics(ear=avg_ear, mar=mar, pitch=pitch, yaw=yaw, roll=roll)