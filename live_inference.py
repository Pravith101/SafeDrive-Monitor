"""Webcam inference using the preprocessing scaler and matching GRU checkpoint."""
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch

from models.temporal_gru import TemporalGRU, CHECKPOINT_VERSION
from vision.extractor import VisionExtractor

ROOT = Path(__file__).resolve().parent
FEATURES = VisionExtractor.FEATURES


def load_model_artifacts(checkpoint_path: str | Path, scaler_path: str | Path, device="cpu"):
    checkpoint_path, scaler_path = Path(checkpoint_path), Path(scaler_path)
    for path, name in ((checkpoint_path, "model checkpoint"), (scaler_path, "feature scaler")):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}. Run preprocessing and training first.")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # Older supported PyTorch versions do not expose weights_only.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        raise ValueError(f"Could not load model checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a metadata dictionary; retrain with models/temporal_gru.py")
    expected = {"format_version": CHECKPOINT_VERSION, "input_dim": len(FEATURES),
                "feature_order": list(FEATURES), "num_classes": 2}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Checkpoint {key} mismatch: expected {value!r}, got {checkpoint.get(key)!r}")
    for key in ("sequence_length", "hidden_dim", "num_layers"):
        value = checkpoint.get(key)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"Checkpoint has invalid {key}")
    sequence_length = checkpoint["sequence_length"]
    labels = checkpoint.get("labels")
    if not isinstance(labels, dict) or set(labels) != {"0", "1"} or not all(
            isinstance(text, str) and text.strip() for text in labels.values()):
        raise ValueError("Checkpoint is missing the two-class label mapping; retrain with models/temporal_gru.py")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("Checkpoint is missing state_dict; retrain with models/temporal_gru.py")
    model = TemporalGRU(input_dim=checkpoint["input_dim"], hidden_dim=checkpoint["hidden_dim"],
                        num_layers=checkpoint["num_layers"], num_classes=checkpoint["num_classes"])
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        scaler = joblib.load(scaler_path)
    except Exception as exc:
        raise ValueError(f"Incompatible model or scaler artifacts: {exc}") from exc
    if getattr(scaler, "n_features_in_", None) != len(FEATURES):
        raise ValueError(f"Scaler must be fitted on {len(FEATURES)} ordered features")
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
        raise ValueError("Scaler contains non-finite parameters")
    try:
        saved_mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
        saved_scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint is missing scaler compatibility metadata; retrain the model") from exc
    if saved_mean.shape != (len(FEATURES),) or saved_scale.shape != (len(FEATURES),):
        raise ValueError("Checkpoint scaler metadata has incompatible feature count")
    if not np.array_equal(saved_mean, scaler.mean_) or not np.array_equal(saved_scale, scaler.scale_):
        raise ValueError("Checkpoint and scaler artifacts do not match; rerun preprocessing and training")
    model.to(device).eval()
    return model, scaler, sequence_length, checkpoint


def run_live_monitor(camera_index: int = 0) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, window, checkpoint = load_model_artifacts(
        ROOT / "weights/temporal_gru.pth", ROOT / "data/processed/feature_scaler.joblib", device)
    buffer = deque(maxlen=window)
    cap = None
    face_mesh = None
    try:
        face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                    min_detection_confidence=0.5)
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {camera_index}.")
        extractor = VisionExtractor()
        labels = checkpoint["labels"]
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            message, color = "Tracking driver...", (0, 200, 0)
            if result.multi_face_landmarks:
                h, w = frame.shape[:2]
                try:
                    metrics = extractor.extract_metrics(result.multi_face_landmarks[0].landmark, w, h)
                    buffer.append([getattr(metrics, feature) for feature in FEATURES])
                    if len(buffer) == window:
                        raw = np.asarray(buffer, dtype=np.float32)
                        scaled = scaler.transform(raw).astype(np.float32)
                        with torch.inference_mode():
                            prediction = int(model(torch.from_numpy(scaled).unsqueeze(0).to(device)).argmax(1).item())
                        message = labels[str(prediction)]
                        color = (0, 0, 255) if prediction == 1 else (0, 200, 0)
                except (ValueError, cv2.error, FloatingPointError):
                    buffer.clear()
                    message, color = "Face pose unavailable", (0, 165, 255)
            else:
                buffer.clear()
                message, color = "Face Lost", (0, 165, 255)
            cv2.putText(frame, message, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("SafeDrive Monitor (Live Inference)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if cap is not None:
            cap.release()
        if face_mesh is not None:
            face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_monitor()
