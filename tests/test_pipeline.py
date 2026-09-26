import numpy as np
import pytest
import torch
import math
from sklearn.preprocessing import StandardScaler

from data.preprocess_activity import contiguous_windows, grouped_split_indices, label_for_video
from live_inference import load_model_artifacts
from models.temporal_gru import TemporalGRU, CHECKPOINT_VERSION
from vision.extractor import VisionExtractor


@pytest.mark.parametrize(("name", "expected"), [
    ("s01_o01_a02_rgb.mp4", 0),
    ("clip_a10_rgb.mp4", 1),
    ("clip_a020_rgb.mp4", None),
    ("s01_o01_a11_rgb.mp4", None),
    ("not_a_video.txt", None),
])
def test_activity_labels_use_exact_token(name, expected):
    assert label_for_video(name) == expected


def test_windows_do_not_cross_face_detection_gaps():
    frames = [0, 1, 2, None, 3, 4, 5, 6]
    assert contiguous_windows(frames, sequence_length=3, stride=1) == [[0, 1, 2], [3, 4, 5], [4, 5, 6]]


def test_window_parameters_must_be_positive():
    with pytest.raises(ValueError):
        contiguous_windows([1, 2], 0)


def test_group_split_keeps_videos_disjoint_and_classes_in_both_sides():
    groups = np.repeat(["safe-a", "safe-b", "risk-a", "risk-b"], 2)
    labels = np.repeat([0, 0, 1, 1], 2)
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
