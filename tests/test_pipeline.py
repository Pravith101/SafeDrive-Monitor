import numpy as np
import pytest
import torch
import math
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from data.preprocess_activity import contiguous_windows, grouped_split_indices, label_for_video
from live_inference import load_model_artifacts
from models.temporal_gru import TemporalGRU, CHECKPOINT_VERSION
from models.driver_state_cnn import (CLASS_TO_ID, CHECKPOINT_FORMAT,
                                     EYE_CLOSED_RATIO_THRESHOLD, EYE_GATE_VERSION,
                                     MOUTH_GATE_VERSION, MOUTH_OPEN_RATIO_THRESHOLD,
                                     DriverStateCNN, split_records)
from live_inference import (load_driver_state_model, prepare_face_tensor,
                            eye_aspect_ratio, mouth_aperture_ratio,
                            apply_eye_consistency_gate, apply_mouth_consistency_gate,
                            predict_driver_state)
from vision.extractor import VisionExtractor

torch.set_num_threads(min(4, torch.get_num_threads()))


@pytest.mark.parametrize(("name", "expected"), [
    ("0.mov", 0),
    ("10.MOV", 1),
    ("5.mp4", None),
    ("s01_o01_a02_rgb.mp4", None),
    ("not_a_video.txt", None),
])
def test_uta_rldd_labels_use_verified_filename_codes(name, expected):
    assert label_for_video(name) == expected


def test_windows_do_not_cross_face_detection_gaps():
    frames = [0, 1, 2, None, 3, 4, 5, 6]
    assert contiguous_windows(frames, sequence_length=3, stride=1) == [[0, 1, 2], [3, 4, 5], [4, 5, 6]]


def test_window_parameters_must_be_positive():
    with pytest.raises(ValueError):
        contiguous_windows([1, 2], 0)


def test_group_split_keeps_participants_disjoint_and_classes_in_both_sides():
    groups = np.repeat(["participant-01", "participant-02", "participant-03", "participant-04"], 2)
    labels = np.tile([0, 1], 4)
    train, validation = grouped_split_indices(groups, labels, test_size=0.5, seed=3)
    assert set(groups[train]).isdisjoint(groups[validation])
    assert set(labels[train]) == set(labels[validation]) == {0, 1}


def test_group_split_rejects_one_class_validation_partition():
    groups = np.array(["a", "a", "b", "b"])
    labels = np.array([0, 0, 0, 1])
    with pytest.raises(ValueError, match="both classes"):
        grouped_split_indices(groups, labels, test_size=0.5)


def test_pose_estimation_uses_landmarks_and_returns_angles(monkeypatch):
    landmarks = [type("Point", (), {"x": 0.5, "y": 0.5})() for _ in range(468)]
    points = [(0.51, 0.45), (0.50, 0.75), (0.35, 0.42), (0.65, 0.44),
              (0.42, 0.60), (0.58, 0.60)]
    for idx, (x, y) in zip(VisionExtractor._POSE_LANDMARKS, points):
        landmarks[idx].x, landmarks[idx].y = x, y
    monkeypatch.setattr("vision.extractor.cv2.solvePnP",
                        lambda *args, **kwargs: (True, np.zeros((3, 1)), np.zeros((3, 1))))
    pitch, yaw = VisionExtractor().compute_pose(landmarks, 640, 480)
    assert pitch == pytest.approx(0.0)
    assert yaw == pytest.approx(0.0)

    rotation = np.array([[0.2], [0.0], [0.0]], dtype=np.float64)
    monkeypatch.setattr("vision.extractor.cv2.solvePnP",
                        lambda *args, **kwargs: (True, rotation, np.zeros((3, 1))))
    pitch, yaw = VisionExtractor().compute_pose(landmarks, 640, 480)
    assert pitch == pytest.approx(math.degrees(0.2), abs=1e-5)
    assert yaw == pytest.approx(0.0, abs=1e-5)
    rotation = np.array([[0.0], [-0.15], [0.0]], dtype=np.float64)
    pitch, yaw = VisionExtractor().compute_pose(landmarks, 640, 480)
    assert pitch == pytest.approx(0.0, abs=1e-5)
    assert yaw == pytest.approx(math.degrees(-0.15), abs=1e-5)
    assert VisionExtractor().compute_roll((0.0, 0.0), (1.0, 1.0)) == pytest.approx(45.0)


def test_checkpoint_round_trip_and_incompatible_feature_order(tmp_path):
    model_path = tmp_path / "model.pth"
    scaler_path = tmp_path / "scaler.joblib"
    scaler = StandardScaler().fit(np.arange(20, dtype=np.float32).reshape(4, 5))
    import joblib
    joblib.dump(scaler, scaler_path)
    model = TemporalGRU()
    checkpoint = {"format_version": CHECKPOINT_VERSION, "state_dict": model.state_dict(),
                  "input_dim": 5, "hidden_dim": 64, "num_layers": 2, "num_classes": 2,
                  "sequence_length": 7, "feature_order": ["ear", "mar", "pitch", "yaw", "roll"],
                  "labels": {"0": "safe", "1": "drowsy"},
                  "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()}
    torch.save(checkpoint, model_path)
    loaded, loaded_scaler, window, _ = load_model_artifacts(model_path, scaler_path)
    assert window == 7
    assert loaded.training is False
    assert loaded_scaler.n_features_in_ == 5
    with torch.inference_mode():
        logits = loaded(torch.zeros(1, window, 5))
    assert logits.shape == (1, 2)

    incompatible_scaler = StandardScaler().fit(np.arange(20, dtype=np.float32).reshape(4, 5) + 1)
    joblib.dump(incompatible_scaler, scaler_path)
    with pytest.raises(ValueError, match="do not match"):
        load_model_artifacts(model_path, scaler_path)
    joblib.dump(scaler, scaler_path)

    checkpoint["feature_order"] = ["mar", "ear", "pitch", "yaw", "roll"]
    torch.save(checkpoint, model_path)
    with pytest.raises(ValueError, match="feature_order mismatch"):
        load_model_artifacts(model_path, scaler_path)


def test_missing_checkpoint_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing model checkpoint"):
        load_model_artifacts(tmp_path / "absent.pth", tmp_path / "scaler.joblib")


def test_driver_state_cnn_and_face_preprocessing_shapes():
    landmarks = [type("Point", (), {"x": 0.5, "y": 0.5})() for _ in range(468)]
    for idx, (x, y) in zip((10, 152, 234, 454), ((.5, .2), (.5, .85), (.25, .5), (.75, .5))):
        landmarks[idx].x, landmarks[idx].y = x, y
    landmarks[61].x, landmarks[61].y = .4, .6
    landmarks[291].x, landmarks[291].y = .6, .6
    landmarks[13].x, landmarks[13].y = .5, .6
    landmarks[14].x, landmarks[14].y = .5, .6
    tensor = prepare_face_tensor(np.zeros((480, 640, 3), dtype=np.uint8), landmarks)
    assert tensor.shape == (1, 3, 64, 64)
    model = DriverStateCNN()
    with torch.inference_mode():
        logits = model(tensor)
    assert logits.shape == (1, 3)


def test_driver_state_checkpoint_round_trip_and_video_group_split(tmp_path):
    model = DriverStateCNN()
    checkpoint_path = tmp_path / "driver_state.pth"
    torch.save({"format": CHECKPOINT_FORMAT, "state_dict": model.state_dict(),
                "image_size": 64, "class_to_id": CLASS_TO_ID,
                "normalization_mean": [0.5] * 3, "normalization_std": [0.5] * 3,
                "mouth_gate_version": MOUTH_GATE_VERSION,
                "mouth_open_ratio_threshold": MOUTH_OPEN_RATIO_THRESHOLD,
                "eye_gate_version": EYE_GATE_VERSION,
                "eye_closed_ratio_threshold": EYE_CLOSED_RATIO_THRESHOLD}, checkpoint_path)
    loaded, _ = load_driver_state_model(checkpoint_path)
    landmarks = [type("Point", (), {"x": 0.5, "y": 0.5})() for _ in range(468)]
    for idx, (x, y) in zip((10, 152, 234, 454), ((.5, .2), (.5, .85), (.25, .5), (.75, .5))):
        landmarks[idx].x, landmarks[idx].y = x, y
    landmarks[61].x, landmarks[61].y = .4, .6
    landmarks[291].x, landmarks[291].y = .6, .6
    landmarks[13].x, landmarks[13].y = .5, .6
    landmarks[14].x, landmarks[14].y = .5, .6
    for left, right in ((33, 133), (362, 263)):
        landmarks[left].x, landmarks[left].y = .4, .5
        landmarks[right].x, landmarks[right].y = .6, .5
    for top, bottom in ((160, 144), (158, 153), (385, 380), (387, 373)):
        landmarks[top].x, landmarks[top].y = .5, .5
        landmarks[bottom].x, landmarks[bottom].y = .5, .5
    state, probabilities = predict_driver_state(
        loaded, np.zeros((480, 640, 3), dtype=np.uint8), landmarks)
    assert state in {"alert", "microsleep", "yawning", "uncertain"}
    assert probabilities.shape == (3,)
    assert probabilities.sum() == pytest.approx(1.0)

    records = [(Path(f"{video}-{label}.jpg"), label, video)
               for video in range(15) for label in range(3)]
    train, validation, test = split_records(records)
    groups = np.array([row[2] for row in records])
    assert set(groups[train]).isdisjoint(groups[validation])
    assert set(groups[train]).isdisjoint(groups[test])
    assert set(groups[validation]).isdisjoint(groups[test])


def test_mouth_opening_gate_abstains_on_closed_mouth_yawning_prediction():
    landmarks = [type("Point", (), {"x": 0.5, "y": 0.5})() for _ in range(468)]
    # 20 px mouth width and 1 px opening: 0.05, well below validation threshold.
    landmarks[61].x, landmarks[61].y = .4, .5
    landmarks[291].x, landmarks[291].y = .6, .5
    landmarks[13].x, landmarks[13].y = .5, .495
    landmarks[14].x, landmarks[14].y = .5, .505
    ratio = mouth_aperture_ratio(landmarks, 100, 100)
    assert ratio == pytest.approx(.05)
    assert apply_mouth_consistency_gate("yawning", ratio) == "uncertain"
    assert apply_mouth_consistency_gate("yawning", .8) == "yawning"
    assert apply_mouth_consistency_gate("alert", ratio) == "alert"


def test_eye_closure_gate_abstains_on_alert_with_closed_eyes():
    landmarks = [type("Point", (), {"x": 0.5, "y": 0.5})() for _ in range(468)]
    for left, right in ((33, 133), (362, 263)):
        landmarks[left].x, landmarks[left].y = .4, .5
        landmarks[right].x, landmarks[right].y = .6, .5
    for top, bottom in ((160, 144), (158, 153), (385, 380), (387, 373)):
        landmarks[top].x, landmarks[top].y = .5, .495
        landmarks[bottom].x, landmarks[bottom].y = .5, .505
    ratio = eye_aspect_ratio(landmarks, 100, 100)
    assert ratio == pytest.approx(.05)
    assert apply_eye_consistency_gate("alert", ratio) == "uncertain"
    assert apply_eye_consistency_gate("microsleep", ratio) == "microsleep"
    assert apply_eye_consistency_gate("alert", .3) == "alert"
    assert apply_eye_consistency_gate("microsleep", .3) == "microsleep"
