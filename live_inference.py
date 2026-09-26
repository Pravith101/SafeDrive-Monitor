"""Webcam and still-image inference for the FL3D driver-state CNN."""
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
import argparse

from models.temporal_gru import TemporalGRU, CHECKPOINT_VERSION
from models.driver_state_cnn import (CHECKPOINT_FORMAT, ID_TO_CLASS, IMAGE_SIZE,
                                     MOUTH_GATE_VERSION, MOUTH_OPEN_RATIO_THRESHOLD,
                                     DriverStateCNN)
from vision.extractor import VisionExtractor

ROOT = Path(__file__).resolve().parent
FEATURES = VisionExtractor.FEATURES
DEFAULT_MOUTH_OPEN_RATIO_THRESHOLD = MOUTH_OPEN_RATIO_THRESHOLD


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


def load_driver_state_model(checkpoint_path: str | Path, device="cpu"):
    """Load the trained FL3D frame classifier used by the current webcam demo."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing driver-state checkpoint: {checkpoint_path}. "
                                "Run models/driver_state_cnn.py first.")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Checkpoint is not a compatible SafeDrive FL3D CNN checkpoint")
    if checkpoint.get("image_size") != IMAGE_SIZE or checkpoint.get("class_to_id") != {
            "alert": 0, "microsleep": 1, "yawning": 2}:
        raise ValueError("Checkpoint image size or class map is incompatible")
    if (checkpoint.get("normalization_mean") != [0.5, 0.5, 0.5] or
            checkpoint.get("normalization_std") != [0.5, 0.5, 0.5]):
        raise ValueError("Checkpoint normalization metadata is incompatible")
    if checkpoint.get("mouth_gate_version") != MOUTH_GATE_VERSION:
        raise ValueError("Checkpoint is missing the compatible calibrated mouth-opening gate")
    threshold = checkpoint.get("mouth_open_ratio_threshold")
    if (not isinstance(threshold, (float, int)) or not np.isfinite(threshold) or
            threshold <= 0):
        raise ValueError("Checkpoint is missing a valid calibrated mouth-opening threshold")
    model = DriverStateCNN(num_classes=len(ID_TO_CLASS))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    bias = np.asarray(checkpoint.get("decision_bias", [0.0, 0.0, 0.0]), dtype=np.float32)
    if bias.shape != (len(ID_TO_CLASS),) or not np.isfinite(bias).all():
        raise ValueError("Checkpoint contains an invalid class decision bias")
    model.decision_bias = torch.as_tensor(bias, dtype=torch.float32, device=device)
    model.mouth_open_ratio_threshold = float(threshold)
    return model, checkpoint


def mouth_aperture_ratio(landmarks, frame_width: int, frame_height: int) -> float:
    """Measure inner-lip opening relative to mouth width in frame pixel space."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive")

    def point(index):
        landmark = landmarks[index]
        return np.asarray((landmark.x * frame_width, landmark.y * frame_height), dtype=np.float32)

    mouth_width = float(np.linalg.norm(point(61) - point(291)))
    if not np.isfinite(mouth_width) or mouth_width < 1.0:
        raise ValueError("Mouth landmarks are degenerate")
    aperture = float(np.linalg.norm(point(13) - point(14)) / mouth_width)
    if not np.isfinite(aperture):
        raise ValueError("Mouth aperture is not finite")
    return aperture


def apply_mouth_consistency_gate(state: str, aperture_ratio: float,
                                 threshold: float = DEFAULT_MOUTH_OPEN_RATIO_THRESHOLD) -> str:
    """Abstain on yawning predictions without the validated open-mouth cue."""
    if state == "yawning" and aperture_ratio < threshold:
        return "uncertain"
    return state


def prepare_face_tensor(frame, landmarks):
    """Crop and normalize a detected face to the FL3D classifier input."""
    height, width = frame.shape[:2]
    points = np.asarray([(point.x * width, point.y * height) for point in landmarks], dtype=np.float32)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    face_width, face_height = max(x_max - x_min, 1), max(y_max - y_min, 1)
    x_min = max(0, int(x_min - face_width * 0.12))
    x_max = min(width, int(x_max + face_width * 0.12))
    y_min = max(0, int(y_min - face_height * 0.10))
    y_max = min(height, int(y_max + face_height * 0.12))
    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        raise ValueError("Face landmarks do not overlap the camera frame")
    rgb = cv2.cvtColor(cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    array = rgb.astype(np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.inference_mode()
def predict_driver_state(model, frame, landmarks, device="cpu"):
    tensor = prepare_face_tensor(frame, landmarks).to(device)
    logits = model(tensor)[0] + getattr(model, "decision_bias", 0.0)
    probabilities = torch.softmax(logits, dim=0).cpu().numpy()
    label_id = int(probabilities.argmax())
    state = ID_TO_CLASS[str(label_id)]
    aperture = mouth_aperture_ratio(landmarks, frame.shape[1], frame.shape[0])
    state = apply_mouth_consistency_gate(
        state, aperture, getattr(model, "mouth_open_ratio_threshold", DEFAULT_MOUTH_OPEN_RATIO_THRESHOLD))
    return state, probabilities


def run_driver_state_monitor(camera_index: int = 0) -> None:
    """Run face-frame classification on webcam video with a short vote smoother."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_driver_state_model(ROOT / "weights/driver_state_cnn.pth", device)
    from collections import deque
    votes = deque(maxlen=9)
    cap = None
    face_mesh = None
    try:
        face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                    min_detection_confidence=0.5)
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {camera_index}.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_face_landmarks:
                try:
                    state, probabilities = predict_driver_state(
                        model, frame, result.multi_face_landmarks[0].landmark, device)
                    votes.append(probabilities)
                    averaged = np.mean(votes, axis=0)
                    state = ID_TO_CLASS[str(int(averaged.argmax()))]
                    landmarks = result.multi_face_landmarks[0].landmark
                    aperture = mouth_aperture_ratio(landmarks, frame.shape[1], frame.shape[0])
                    state = apply_mouth_consistency_gate(
                        state, aperture,
                        getattr(model, "mouth_open_ratio_threshold", DEFAULT_MOUTH_OPEN_RATIO_THRESHOLD))
                    color = (0, 165, 255) if state == "uncertain" else (
                        (0, 0, 255) if state != "alert" else (0, 200, 0))
                    message = "UNCERTAIN: mouth cue not confirmed" if state == "uncertain" else state.upper()
                except (ValueError, cv2.error, FloatingPointError):
                    votes.clear()
                    message, color = "Face crop unavailable", (0, 165, 255)
            else:
                votes.clear()
                message, color = "Face lost", (0, 165, 255)
            cv2.putText(frame, message, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("SafeDrive Monitor (FL3D CNN)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if cap is not None:
            cap.release()
        if face_mesh is not None:
            face_mesh.close()
        cv2.destroyAllWindows()


def run_image_inference(image_path: str | Path) -> None:
    """Run the same face crop and consistency gate on one still image."""
    image_path = Path(image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_driver_state_model(ROOT / "weights/driver_state_cnn.pth", device)
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                         min_detection_confidence=0.5) as face_mesh:
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        print("prediction=face_not_found")
        return
    landmarks = result.multi_face_landmarks[0].landmark
    raw_tensor = prepare_face_tensor(frame, landmarks).to(device)
    with torch.inference_mode():
        logits = model(raw_tensor)[0] + getattr(model, "decision_bias", 0.0)
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()
    raw_state = ID_TO_CLASS[str(int(probabilities.argmax()))]
    aperture = mouth_aperture_ratio(landmarks, frame.shape[1], frame.shape[0])
    state = apply_mouth_consistency_gate(raw_state, aperture, model.mouth_open_ratio_threshold)
    print(f"raw_model_prediction={raw_state}")
    print(f"mouth_aperture_ratio={aperture:.4f}; calibrated_threshold={model.mouth_open_ratio_threshold:.4f}")
    print(f"prediction={state}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeDrive FL3D CNN inference")
    parser.add_argument("--image", type=Path, help="Run one still image without opening a webcam")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    cli_args = parser.parse_args()
    if cli_args.image:
        run_image_inference(cli_args.image)
    else:
        run_driver_state_monitor(cli_args.camera)
