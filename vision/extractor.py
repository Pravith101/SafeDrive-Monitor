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
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.left_eye = [362, 385, 386, 263, 374, 380]
        self.right_eye = [33, 159, 158, 133, 153, 145]
        self.mouth = [78, 81, 13, 311, 308, 178, 14, 402]
        
    def _compute_ratio(self, landmarks, indices, horizontal_pairs, vertical_pairs):
        width = np.linalg.norm(landmarks[indices[horizontal_pairs[0]]] - landmarks[indices[horizontal_pairs[1]]])
        height = sum(np.linalg.norm(landmarks[indices[p1]] - landmarks[indices[p2]]) for p1, p2 in vertical_pairs) / len(vertical_pairs)
        return height / width if width > 0 else 0.0

    def process_frame(self, frame: np.ndarray) -> Optional[FrameMetrics]:
        results = self.mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
            
        mesh = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        landmarks = np.array([[p.x * w, p.y * h] for p in mesh])
        
        left_ear = self._compute_ratio(landmarks, self.left_eye, (0, 3), ((1, 5), (2, 4)))
        right_ear = self._compute_ratio(landmarks, self.right_eye, (0, 3), ((1, 5), (2, 4)))
        ear = (left_ear + right_ear) / 2.0
        
        mar = self._compute_ratio(landmarks, self.mouth, (0, 4), ((1, 7), (2, 6), (3, 5)))
        
        face_3d = np.array([
            [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
        ], dtype=np.float64)
        
        face_2d = np.array([
            landmarks[1], landmarks[152], landmarks[226], 
            landmarks[446], landmarks[57], landmarks[287]
        ], dtype=np.float64)
        
        focal_length = 1 * w
        cam_matrix = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]])
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs)
        if not success:
            return FrameMetrics(ear, mar, 0.0, 0.0, 0.0)
            
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        return FrameMetrics(ear, mar, angles[0] * 360, angles[1] * 360, angles[2] * 360)