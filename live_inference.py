"""Webcam and still-image inference for the FL3D driver-state CNN."""
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
import argparse
import time
import threading

from models.temporal_gru import TemporalGRU, CHECKPOINT_VERSION
from models.driver_state_cnn import (CHECKPOINT_FORMAT, ID_TO_CLASS, IMAGE_SIZE,
                                     SCRATCH_ARCHITECTURE,
                                     EYE_CLOSED_RATIO_THRESHOLD, EYE_GATE_VERSION,
                                     MOUTH_GATE_VERSION, MOUTH_OPEN_RATIO_THRESHOLD,
                                     DriverStateCNN, TRANSFER_ARCHITECTURE)
from vision.extractor import VisionExtractor

ROOT = Path(__file__).resolve().parent
FEATURES = VisionExtractor.FEATURES
DEFAULT_MOUTH_OPEN_RATIO_THRESHOLD = MOUTH_OPEN_RATIO_THRESHOLD
MICROSLEEP_SCORE_THRESHOLD = 0.90
MICROSLEEP_HOLD_SECONDS = 1.0
ALERT_REPEAT_SECONDS = 4.0
MAX_ALERT_EVIDENCE_GAP_SECONDS = 0.5
STATE_CUE_COOLDOWN_SECONDS = 3.0
STATE_TONE_PATTERNS = {
    "microsleep": ((1040, 180), (780, 180), (1040, 320)),
    "yawning": ((660, 110), (880, 160)),
    "uncertain": ((440, 140),),
}


class StateSoundPolicy:
    """Play brief cues on entry to validated non-alert states, with a cooldown."""

    def __init__(self, cooldown_seconds=STATE_CUE_COOLDOWN_SECONDS):
        self.cooldown_seconds = cooldown_seconds
        self.last_state = None
        self.last_sound_at = None

    def update(self, state: str, now: float) -> str | None:
        if not np.isfinite(now):
            self.last_state = None
            return None
        entered = state != self.last_state
        self.last_state = state
        if not entered or state not in {"yawning", "uncertain"}:
            return None
        if (self.last_sound_at is not None and
                now - self.last_sound_at < self.cooldown_seconds):
            return None
        self.last_sound_at = now
        return state


class DriverAlertPolicy:
    """Require strong model and eye-closure evidence to trigger an audible alert."""

    def __init__(self, score_threshold=MICROSLEEP_SCORE_THRESHOLD,
                 hold_seconds=MICROSLEEP_HOLD_SECONDS,
                 repeat_seconds=ALERT_REPEAT_SECONDS,
                 max_evidence_gap=MAX_ALERT_EVIDENCE_GAP_SECONDS):
        self.score_threshold = score_threshold
        self.hold_seconds = hold_seconds
        self.repeat_seconds = repeat_seconds
        self.max_evidence_gap = max_evidence_gap
        self.evidence_since = None
        self.last_supported_at = None
        self.last_sound = None

    def update(self, microsleep_score: float, eye_ratio: float,
               eye_closed_threshold: float, now: float) -> tuple[bool, bool]:
        """Return (alert_active, play_sound) for this frame."""
        supported = (np.isfinite(now) and np.isfinite(microsleep_score) and
                     microsleep_score >= self.score_threshold and
                     np.isfinite(eye_ratio) and eye_ratio < eye_closed_threshold)
        if not supported:
            self.evidence_since = None
            self.last_supported_at = None
            self.last_sound = None
            return False, False
        gap = None if self.last_supported_at is None else now - self.last_supported_at
        if (self.evidence_since is None or gap is None or gap < 0 or
                gap > self.max_evidence_gap):
            self.evidence_since = now
            self.last_sound = None
        self.last_supported_at = now
        active = now - self.evidence_since >= self.hold_seconds
        play_sound = active and (self.last_sound is None or
                                 now - self.last_sound >= self.repeat_seconds)
        if play_sound:
            self.last_sound = now
        return active, play_sound


def play_state_sound(state: str) -> None:
    """Play the selected state cue without blocking webcam frame processing."""
    pattern = STATE_TONE_PATTERNS.get(state)
    if pattern is None:
        return

    def play_pattern():
        try:
            import winsound
            for frequency, duration_ms in pattern:
                winsound.Beep(frequency, duration_ms)
                time.sleep(0.07)
        except (ImportError, RuntimeError):
            print("\\a" * len(pattern), end="", flush=True)

    threading.Thread(target=play_pattern, daemon=True).start()


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
    sample_fps = checkpoint.get("sample_fps", 1.0)
    if not isinstance(sample_fps, (int, float)) or not np.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("Checkpoint has invalid sample_fps")
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
    sample_fps = float(checkpoint.get("sample_fps", 1.0))
    last_sample = 0.0
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
                    now = time.monotonic()
                    if now - last_sample >= 1.0 / sample_fps:
                        metrics = extractor.extract_metrics(result.multi_face_landmarks[0].landmark, w, h)
                        buffer.append([getattr(metrics, feature) for feature in FEATURES])
                        last_sample = now
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


def load_uta_rldd_model(checkpoint_path: str | Path, device="cpu"):
    """Load the cloud-trained three-state UTA-RLDD model and fold-calibrated threshold."""
    from models.uta_rldd_cloud import CLASS_NAMES, FEATURES, UtaGRU

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing UTA-RLDD checkpoint: {checkpoint_path}. "
                                "Run the Kaggle UTA-RLDD notebook and download uta_rldd_final.pth.")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "safedrive-uta-rldd-gru-v1":
        raise ValueError("Checkpoint is not a compatible UTA-RLDD temporal model")
    if (checkpoint.get("feature_order") != list(FEATURES) or
            checkpoint.get("sequence_length") != 30 or checkpoint.get("num_classes") != len(CLASS_NAMES)):
        raise ValueError("UTA-RLDD checkpoint feature order, sequence length, or classes are incompatible")
    mean = np.asarray(checkpoint.get("scaler_mean"), dtype=np.float32)
    scale = np.asarray(checkpoint.get("scaler_scale"), dtype=np.float32)
    threshold = checkpoint.get("drowsy_warning_threshold")
    if (mean.shape != (len(FEATURES),) or scale.shape != mean.shape or
            not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any() or
            not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("UTA-RLDD checkpoint normalization or warning threshold is invalid")
    model = UtaGRU(input_dim=len(FEATURES), hidden_dim=int(checkpoint["hidden_dim"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint, mean, scale


def run_uta_rldd_monitor(camera_index: int = 0) -> None:
    """Display research-only UTA-RLDD scores; this model is not alert-qualified."""
    from models.uta_rldd_cloud import CLASS_NAMES, FEATURES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, scaler_mean, scaler_scale = load_uta_rldd_model(
        ROOT / "weights/uta_rldd_final.pth", device)
    extractor = VisionExtractor()
    sequence_length = int(checkpoint["sequence_length"])
    feature_buffer = deque(maxlen=sequence_length)
    sample_fps = float(checkpoint.get("sample_fps", 1.0))
    last_sample = 0.0
    window_frames = 0
    current_message, current_color = "Tracking driver...", (0, 200, 0)
    print("UTA-RLDD model is research-only; it does not issue driver alerts or audible warnings.")
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
            if not result.multi_face_landmarks:
                feature_buffer.clear()
                window_frames = 0
                current_message, current_color = "Face lost - checking again", (0, 165, 255)
            else:
                now = time.monotonic()
                if now - last_sample >= 1.0 / sample_fps:
                    h, w = frame.shape[:2]
                    try:
                        metrics = extractor.extract_metrics(result.multi_face_landmarks[0].landmark, w, h)
                        values = np.asarray([getattr(metrics, feature) for feature in FEATURES], dtype=np.float32)
                        if not np.isfinite(values).all():
                            raise ValueError("Face feature vector is non-finite")
                        feature_buffer.append(values)
                        window_frames += 1
                        last_sample = now
                        if len(feature_buffer) == sequence_length:
                            if window_frames >= sequence_length:
                                sequence = (np.asarray(feature_buffer, dtype=np.float32) - scaler_mean) / scaler_scale
                                tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
                                with torch.inference_mode():
                                    probabilities = torch.softmax(model(tensor)[0], dim=0).cpu().numpy()
                                prediction = int(probabilities.argmax())
                                drowsy_score = float(probabilities[2])
                                if drowsy_score >= float(checkpoint["drowsy_warning_threshold"]):
                                    current_message, current_color = "RESEARCH FLAG - NOT ALERT", (0, 165, 255)
                                elif prediction == 2:
                                    current_message, current_color = "Possible drowsiness - monitor", (0, 165, 255)
                                else:
                                    current_message = CLASS_NAMES[prediction].replace("_", " ").upper()
                                    current_color = (0, 200, 0) if prediction == 0 else (0, 165, 255)
                                window_frames = 0
                    except (ValueError, cv2.error, FloatingPointError):
                        feature_buffer.clear()
                        window_frames = 0
                        current_message, current_color = "Face features unavailable", (0, 165, 255)
            cv2.putText(frame, current_message, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        current_color, 2)
            if len(feature_buffer) < sequence_length:
                cv2.putText(frame, f"Collecting features: {len(feature_buffer)}/{sequence_length}s",
                            (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("SafeDrive Monitor (UTA-RLDD)", frame)
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
    architecture = checkpoint.get("architecture", SCRATCH_ARCHITECTURE)
    if architecture not in {SCRATCH_ARCHITECTURE, TRANSFER_ARCHITECTURE}:
        raise ValueError("Checkpoint architecture is incompatible")
    try:
        normalization_mean = np.asarray(checkpoint.get("normalization_mean", [0.5] * 3), dtype=np.float32)
        normalization_std = np.asarray(checkpoint.get("normalization_std", [0.5] * 3), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint normalization metadata is incompatible") from exc
    if (normalization_mean.shape != (3,) or normalization_std.shape != (3,) or
            not np.isfinite(normalization_mean).all() or not np.isfinite(normalization_std).all() or
            (normalization_std <= 0).any()):
        raise ValueError("Checkpoint normalization metadata is incompatible")
    if checkpoint.get("mouth_gate_version") != MOUTH_GATE_VERSION:
        raise ValueError("Checkpoint is missing the compatible calibrated mouth-opening gate")
    threshold = checkpoint.get("mouth_open_ratio_threshold")
    if (not isinstance(threshold, (float, int)) or not np.isfinite(threshold) or
            threshold <= 0):
        raise ValueError("Checkpoint is missing a valid calibrated mouth-opening threshold")
    if checkpoint.get("eye_gate_version") != EYE_GATE_VERSION:
        raise ValueError("Checkpoint is missing the compatible calibrated eye-closure gate")
    eye_threshold = checkpoint.get("eye_closed_ratio_threshold")
    if (not isinstance(eye_threshold, (float, int)) or not np.isfinite(eye_threshold) or
            eye_threshold <= 0):
        raise ValueError("Checkpoint is missing a valid calibrated eye-closure threshold")
    model = DriverStateCNN(num_classes=len(ID_TO_CLASS), architecture=architecture)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    model.normalization_mean = tuple(normalization_mean.tolist())
    model.normalization_std = tuple(normalization_std.tolist())
    bias = np.asarray(checkpoint.get("decision_bias", [0.0, 0.0, 0.0]), dtype=np.float32)
    if bias.shape != (len(ID_TO_CLASS),) or not np.isfinite(bias).all():
        raise ValueError("Checkpoint contains an invalid class decision bias")
    model.decision_bias = torch.as_tensor(bias, dtype=torch.float32, device=device)
    model.mouth_open_ratio_threshold = float(threshold)
    model.eye_closed_ratio_threshold = float(eye_threshold)
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


def eye_aspect_ratio(landmarks, frame_width: int, frame_height: int) -> float:
    """Average the standard six-point eye aspect ratio for both eyes."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive")

    def point(index):
        landmark = landmarks[index]
        return np.asarray((landmark.x * frame_width, landmark.y * frame_height), dtype=np.float32)

    def ratio(horizontal, vertical_a, vertical_b):
        width = float(np.linalg.norm(point(horizontal[0]) - point(horizontal[1])))
        if not np.isfinite(width) or width < 1.0:
            raise ValueError("Eye landmarks are degenerate")
        height_a = float(np.linalg.norm(point(vertical_a[0]) - point(vertical_a[1])))
        height_b = float(np.linalg.norm(point(vertical_b[0]) - point(vertical_b[1])))
        return (height_a + height_b) / (2.0 * width)

    left = ratio((33, 133), (160, 144), (158, 153))
    right = ratio((362, 263), (385, 380), (387, 373))
    value = float((left + right) / 2.0)
    if not np.isfinite(value):
        raise ValueError("Eye aspect ratio is not finite")
    return value


def apply_mouth_consistency_gate(state: str, aperture_ratio: float,
                                 threshold: float = DEFAULT_MOUTH_OPEN_RATIO_THRESHOLD) -> str:
    """Abstain on yawning predictions without the validated open-mouth cue."""
    if state == "yawning" and aperture_ratio < threshold:
        return "uncertain"
    return state


def apply_eye_consistency_gate(state: str, aspect_ratio: float,
                                threshold: float = EYE_CLOSED_RATIO_THRESHOLD) -> str:
    """Abstain from alert only for strong eye-closure evidence."""
    if state == "alert" and aspect_ratio < threshold:
        return "uncertain"
    return state


def prepare_face_tensor(frame, landmarks, normalization_mean=(0.5, 0.5, 0.5),
                       normalization_std=(0.5, 0.5, 0.5)):
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
    array = (array - np.asarray(normalization_mean, dtype=np.float32)) / np.asarray(
        normalization_std, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.inference_mode()
def predict_driver_state(model, frame, landmarks, device="cpu"):
    tensor = prepare_face_tensor(frame, landmarks, getattr(model, "normalization_mean", (0.5,) * 3),
                                 getattr(model, "normalization_std", (0.5,) * 3)).to(device)
    logits = model(tensor)[0] + getattr(model, "decision_bias", 0.0)
    probabilities = torch.softmax(logits, dim=0).cpu().numpy()
    label_id = int(probabilities.argmax())
    state = ID_TO_CLASS[str(label_id)]
    aperture = mouth_aperture_ratio(landmarks, frame.shape[1], frame.shape[0])
    state = apply_mouth_consistency_gate(
        state, aperture, getattr(model, "mouth_open_ratio_threshold", DEFAULT_MOUTH_OPEN_RATIO_THRESHOLD))
    state = apply_eye_consistency_gate(
        state, eye_aspect_ratio(landmarks, frame.shape[1], frame.shape[0]),
        getattr(model, "eye_closed_ratio_threshold", EYE_CLOSED_RATIO_THRESHOLD))
    return state, probabilities


def run_driver_state_monitor(camera_index: int = 0) -> None:
    """Run conservative webcam classification and sustained drowsiness warnings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_driver_state_model(ROOT / "weights/driver_state_cnn.pth", device)
    from collections import deque
    votes = deque(maxlen=9)
    alert_policy = DriverAlertPolicy()
    state_sound_policy = StateSoundPolicy()
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
                    raw_state = ID_TO_CLASS[str(int(averaged.argmax()))]
                    landmarks = result.multi_face_landmarks[0].landmark
                    eye_ratio = eye_aspect_ratio(landmarks, frame.shape[1], frame.shape[0])
                    aperture = mouth_aperture_ratio(landmarks, frame.shape[1], frame.shape[0])
                    state = apply_mouth_consistency_gate(
                        raw_state, aperture,
                        getattr(model, "mouth_open_ratio_threshold", DEFAULT_MOUTH_OPEN_RATIO_THRESHOLD))
                    state = apply_eye_consistency_gate(
                        state, eye_ratio,
                        getattr(model, "eye_closed_ratio_threshold", EYE_CLOSED_RATIO_THRESHOLD))
                    warning_active, play_sound = alert_policy.update(
                        float(averaged[1]), eye_ratio,
                        getattr(model, "eye_closed_ratio_threshold", EYE_CLOSED_RATIO_THRESHOLD),
                        time.monotonic())
                    cue = state_sound_policy.update(
                        "microsleep" if warning_active else state, time.monotonic())
                    if play_sound:
                        play_state_sound("microsleep")
                    elif cue:
                        play_state_sound(cue)
                    if warning_active:
                        message, color = "DROWSINESS WARNING", (0, 0, 255)
                    elif raw_state == "microsleep":
                        message, color = "Possible microsleep - checking evidence", (0, 165, 255)
                    elif state == "uncertain":
                        color = (0, 165, 255)
                        cue = "mouth cue absent" if raw_state == "yawning" else "eyes appear closed"
                        message = f"UNCERTAIN: {cue}"
                    else:
                        message = state.upper()
                        color = (0, 200, 0) if state == "alert" else (255, 180, 0)
                except (ValueError, cv2.error, FloatingPointError):
                    votes.clear()
                    alert_policy.update(0.0, 1.0, EYE_CLOSED_RATIO_THRESHOLD, time.monotonic())
                    cue = state_sound_policy.update("uncertain", time.monotonic())
                    if cue:
                        play_state_sound(cue)
                    message, color = "Face crop unavailable", (0, 165, 255)
            else:
                votes.clear()
                alert_policy.update(0.0, 1.0, EYE_CLOSED_RATIO_THRESHOLD, time.monotonic())
                cue = state_sound_policy.update("uncertain", time.monotonic())
                if cue:
                    play_state_sound(cue)
                message, color = "Face lost", (0, 165, 255)
            cv2.putText(frame, message, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if message == "DROWSINESS WARNING":
                cv2.putText(frame, "Pull over safely", (20, 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
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
    raw_tensor = prepare_face_tensor(frame, landmarks, model.normalization_mean,
                                     model.normalization_std).to(device)
    with torch.inference_mode():
        logits = model(raw_tensor)[0] + getattr(model, "decision_bias", 0.0)
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()
    raw_state = ID_TO_CLASS[str(int(probabilities.argmax()))]
    aperture = mouth_aperture_ratio(landmarks, frame.shape[1], frame.shape[0])
    state = apply_mouth_consistency_gate(raw_state, aperture, model.mouth_open_ratio_threshold)
    eye_ratio = eye_aspect_ratio(landmarks, frame.shape[1], frame.shape[0])
    state = apply_eye_consistency_gate(
        state, eye_ratio, model.eye_closed_ratio_threshold)
    print(f"raw_model_prediction={raw_state}")
    print(f"mouth_aperture_ratio={aperture:.4f}; calibrated_threshold={model.mouth_open_ratio_threshold:.4f}")
    print(f"eye_aspect_ratio={eye_ratio:.4f}; "
          f"calibrated_threshold={model.eye_closed_ratio_threshold:.4f}")
    print(f"prediction={state}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeDrive webcam inference")
    parser.add_argument("--image", type=Path, help="Run one still image without opening a webcam")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--temporal", action="store_true",
                        help="Use the UTA-RLDD temporal GRU checkpoint instead of the FL3D CNN")
    parser.add_argument("--uta-rldd", action="store_true",
                        help="Display research-only UTA-RLDD scores (no driver alerts or sound)")
    cli_args = parser.parse_args()
    if cli_args.image:
        run_image_inference(cli_args.image)
    elif cli_args.temporal:
        run_live_monitor(cli_args.camera)
    elif cli_args.uta_rldd:
        run_uta_rldd_monitor(cli_args.camera)
    else:
        run_driver_state_monitor(cli_args.camera)
