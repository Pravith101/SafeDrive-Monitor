"""Create contiguous, participant-grouped sequences and the training-only scaler."""
from __future__ import annotations

import json
import argparse
from itertools import chain
import sys
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.extractor import VisionExtractor

FEATURES = VisionExtractor.FEATURES
# UTA-RLDD participant labels are encoded in video basenames: 0, 5, and 10.
# This model deliberately uses only the alert and drowsy endpoints.
LABELS = {0: "alert (UTA-RLDD label 0)", 1: "drowsy (UTA-RLDD label 10)"}
SPLIT_SEED = 42
VALIDATION_FRACTION = 0.2


def label_for_video(path: str | Path) -> int | None:
    """Map verified UTA-RLDD filename labels; omit low vigilance (label 5)."""
    return {"0": 0, "10": 1}.get(Path(path).stem)


def contiguous_windows(frames: list, sequence_length: int, stride: int = 30) -> list:
    """Window only uninterrupted runs; None marks a frame without a detected face."""
    if sequence_length < 1 or stride < 1:
        raise ValueError("sequence_length and stride must be positive")
    windows = []
    run = []
    for frame in chain(frames, (None,)):
        if frame is None:
            windows.extend(run[i:i + sequence_length]
                           for i in range(0, len(run) - sequence_length + 1, stride))
            run = []
        else:
            run.append(frame)
    return windows


def grouped_split_indices(groups: np.ndarray, labels: np.ndarray,
                          test_size: float = VALIDATION_FRACTION, seed: int = SPLIT_SEED):
    if len(groups) < 2 or len(groups) != len(labels):
        raise ValueError("At least two grouped samples with matching labels are required")
    train, validation = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                                random_state=seed).split(np.zeros(len(groups)), labels, groups))
    if set(np.unique(labels[train])) != {0, 1} or set(np.unique(labels[validation])) != {0, 1}:
        raise ValueError("Participant-level split must contain both classes in train and validation; "
                         "provide more participants")
    return train, validation


def process_activity_videos(dataset_dir: str | Path, sequence_length: int = 30, stride: int = 30) -> None:
    if sequence_length < 1 or stride < 1:
        raise ValueError("sequence_length and stride must be positive")
    videos = sorted(p for p in Path(dataset_dir).rglob("*") if p.is_file()
                    and p.suffix.lower() in {".mp4", ".mov"} and label_for_video(p) is not None)
    if not videos:
        raise RuntimeError("No UTA-RLDD endpoint videos found; expected files named 0 and 10 "
                           "under participant folders. Label 5 (low vigilance) is excluded.")
    print(f"Found {len(videos)} UTA-RLDD endpoint videos (labels 0=alert and 10=drowsy); "
          "label 5 (low vigilance) is excluded.")
    sequences, labels, groups = [], [], []
    mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                           min_detection_confidence=0.5)
    extractor = VisionExtractor()
    try:
        for path in videos:
            cap = cv2.VideoCapture(str(path))
            runs = []
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if result.multi_face_landmarks:
                        h, w = frame.shape[:2]
                        try:
                            metrics = extractor.extract_metrics(result.multi_face_landmarks[0].landmark, w, h)
                            runs.append([getattr(metrics, feature) for feature in FEATURES])
                        except (ValueError, cv2.error, FloatingPointError):
                            runs.append(None)
                    else:
                        runs.append(None)
            finally:
                cap.release()
            windows = contiguous_windows(runs, sequence_length, stride)
            sequences.extend(windows)
            labels.extend([label_for_video(path)] * len(windows))
            # In UTA-RLDD the immediate parent is the participant ID (01..60).
            groups.extend([path.parent.name] * len(windows))
            print(f"Processed {path.name}: {len(windows)} contiguous sequences")
    finally:
        mesh.close()

    if not sequences:
        raise RuntimeError("No labeled sequences found; check the UTA-RLDD path and videos named 0/10.")
    x = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    group_ids = np.asarray(groups, dtype=str)
    train_idx, val_idx = grouped_split_indices(group_ids, y)
    train_groups = sorted(np.unique(group_ids[train_idx]).tolist())
    scaler = StandardScaler().fit(x[train_idx].reshape(-1, len(FEATURES)))
    x = scaler.transform(x.reshape(-1, len(FEATURES))).reshape(x.shape).astype(np.float32)
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "sequences.npy", x)
    np.save(out / "labels.npy", y)
    np.save(out / "groups.npy", group_ids)
    joblib.dump(scaler, out / "feature_scaler.joblib")
    metadata = {"feature_order": list(FEATURES), "sequence_length": sequence_length,
                "stride": stride, "labels": {str(k): v for k, v in LABELS.items()},
                "split_seed": SPLIT_SEED, "validation_fraction": VALIDATION_FRACTION,
                "scaler_fit_groups": train_groups}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(y)} sequences to {out}; train={len(train_idx)}, validation={len(val_idx)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "raw_dataset",
                        help="UTA-RLDD directory returned by KaggleHub (default: data/raw_dataset)")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--stride", type=int, default=30)
    args = parser.parse_args()
    process_activity_videos(args.dataset_dir, args.sequence_length, args.stride)
