"""Webcam inference using the exact scaler and architecture from training."""
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch

from models.temporal_gru import TemporalGRU
from vision.extractor import VisionExtractor

ROOT = Path(__file__).resolve().parent
WINDOW = 30


def run_live_monitor(camera_index: int = 0) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = ROOT / "weights/temporal_gru.pth"
    scaler_path = ROOT / "data/processed/feature_scaler.joblib"
    if not checkpoint_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Missing model or scaler. Run preprocessing and training first.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TemporalGRU(input_dim=checkpoint.get("input_dim", 5),
                        hidden_dim=checkpoint.get("hidden_dim", 64),
                        num_layers=checkpoint.get("num_layers", 2),
                        num_classes=checkpoint.get("num_classes", 2))
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.to(device).eval()
    scaler = joblib.load(scaler_path)
    extractor = VisionExtractor()
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                min_detection_confidence=0.5)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        face_mesh.close()
        raise RuntimeError(f"Could not open webcam index {camera_index}.")
    buffer = deque(maxlen=WINDOW)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            text, color = "Tracking driver...", (0, 200, 0)
            if result.multi_face_landmarks:
                h, w = frame.shape[:2]
                m = extractor.extract_metrics(result.multi_face_landmarks[0].landmark, w, h)
                buffer.append([m.ear, m.mar, m.pitch, m.yaw, m.roll])
                if len(buffer) == WINDOW:
                    raw = np.asarray(buffer, dtype=np.float32)
                    scaled = scaler.transform(raw).astype(np.float32)
                    tensor = torch.from_numpy(scaled).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        prediction = int(model(tensor).argmax(dim=1).item())
                    text, color = (("DROWSY WARNING", (0, 0, 255)) if prediction == 1
                                   else ("Driver Alert", (0, 200, 0)))
            else:
                buffer.clear()
                text, color = "Face Lost", (0, 165, 255)
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.imshow("SafeDrive Monitor (Live Inference)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_monitor()
