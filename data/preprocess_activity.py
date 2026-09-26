"""Extract labeled facial-metric sequences and fit the live-use scaler."""
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.extractor import VisionExtractor

FEATURES = ("ear", "mar", "pitch", "yaw", "roll")


def process_activity_videos(dataset_dir: str | Path, sequence_length: int = 30,
                            stride: int = 1) -> None:
    if sequence_length < 1 or stride < 1:
        raise ValueError("sequence_length and stride must be positive")
    dataset_dir = Path(dataset_dir)
    extractor = VisionExtractor()
    sequences, labels, groups = [], [], []
    videos = sorted(p for p in dataset_dir.rglob("*_rgb.mp4") if p.is_file())
    print(f"Found {len(videos)} RGB videos; selecting a02 (safe) and a10 (nodding).")

    mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                           min_detection_confidence=0.5)
    try:
        for video_path in videos:
            name = video_path.name.lower()
            label = 0 if "a02" in name else 1 if "a10" in name else None
            if label is None:
                continue
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if result.multi_face_landmarks:
                        h, w = frame.shape[:2]
                        m = extractor.extract_metrics(result.multi_face_landmarks[0].landmark, w, h)
                        frames.append([getattr(m, f) for f in FEATURES])
            finally:
                cap.release()
            # Sliding windows are formed from successfully detected face frames.
            for start in range(0, len(frames) - sequence_length + 1, stride):
                sequences.append(frames[start:start + sequence_length])
                labels.append(label)
                groups.append(video_path.stem)
            print(f"Processed {video_path.name}: {len(frames)} usable frames")
    finally:
        mesh.close()

    if not sequences:
        raise RuntimeError("No labeled sequences found; check dataset path and a02/a10 files.")
    x = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=str)
    # Fit one per-feature scaler over the training corpus and serialize the exact object
    # used by live_inference.py. Training script consumes these already-scaled arrays.
    scaler = StandardScaler()
    train_idx, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                        .split(x, y, groups))
    scaler.fit(x[train_idx].reshape(-1, x.shape[-1]))
    x_scaled = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "sequences.npy", x_scaled)
    np.save(out / "labels.npy", y)
    np.save(out / "groups.npy", groups)
    joblib.dump(scaler, out / "feature_scaler.joblib")
    print(f"Saved {len(y)} sequences to {out}; classes={np.bincount(y, minlength=2).tolist()}")


if __name__ == "__main__":
    process_activity_videos(ROOT / "data" / "raw_dataset")
