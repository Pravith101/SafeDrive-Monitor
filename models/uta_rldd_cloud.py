"""Five-fold, participant-disjoint UTA-RLDD experiment for Kaggle GPU."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FEATURES = ("ear", "mar", "pitch", "yaw", "roll")
CLASS_NAMES = ("alert", "low_vigilance", "drowsy")
LABELS = {"0": 0, "5": 1, "10": 2}
SEED = 42
SEQUENCE_SECONDS = 30
SAMPLE_FPS = 1.0


class UtaGRU(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(32, num_classes))

    def forward(self, x):
        output, _ = self.gru(x)
        return self.head(output[:, -1])


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def discover_videos(root: Path):
    rows = []
    for path in sorted(root.rglob("*")):
        match = re.fullmatch(r"(0|5|10)(?:[_-].*)?", path.stem) if path.is_file() else None
        label = LABELS.get(match.group(1)) if match else None
        if label is None or path.suffix.lower() not in {".mp4", ".mov"}:
            continue
        # Official UTA archive names groups Fold1_part1, Fold1_part2, etc.;
        # part archives belong to the same official fold.
        fold = next((int(m.group(1)) for part in path.parts
                     if (m := re.fullmatch(r"Fold([1-5])_part\d+", part, re.I))), None)
        participant = path.parent.name
        if fold is None:
            raise ValueError(f"Cannot find official Fold1..Fold5 directory for {path}")
        if not participant.isdigit():
            raise ValueError(f"Expected numeric participant folder directly above video: {path}")
        rows.append({"path": path, "label": label, "participant": participant.zfill(2), "fold": fold})
    if not rows:
        raise FileNotFoundError(f"No UTA-RLDD videos named 0, 5, and 10 under {root}")
    by_person = defaultdict(set)
    person_fold = {}
    for row in rows:
        by_person[row["participant"]].add(row["label"])
        previous = person_fold.setdefault(row["participant"], row["fold"])
        if previous != row["fold"]:
            raise ValueError(f"Participant {row['participant']} spans official folds")
    incomplete = {person: sorted(set(LABELS.values()) - labels)
                  for person, labels in by_person.items() if labels != set(LABELS.values())}
    if incomplete:
        raise ValueError(f"Incomplete participant videos/classes in attached Kaggle dataset: {incomplete}")
    official_folds = {row["fold"] for row in rows}
    if official_folds != set(range(1, 6)):
        raise ValueError(f"Expected all five official participant folds, found {sorted(official_folds)}")
    if len(by_person) != 60:
        raise ValueError(f"Expected 60 participant IDs across the five official folds; found {len(by_person)}")
    per_fold = Counter(person_fold.values())
    if per_fold != Counter({fold: 12 for fold in range(1, 6)}):
        raise ValueError(f"Expected 12 participants in every official fold; found {dict(per_fold)}")
    print(f"Discovered videos={len(rows)} participants={len(by_person)} official_folds={sorted(official_folds)}")
    return rows


def extract_sequences(videos, dataset_root, sequence_seconds=SEQUENCE_SECONDS, sample_fps=SAMPLE_FPS):
    mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                            min_detection_confidence=0.5)
    from vision.extractor import VisionExtractor
    extractor = VisionExtractor()
    xs, ys, participants, video_ids, starts, durations = [], [], [], [], [], []
    try:
        for i, row in enumerate(videos, 1):
            cap = cv2.VideoCapture(str(row["path"]))
            if not cap.isOpened():
                raise RuntimeError(f"OpenCV cannot decode video: {row['path']}")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_step = max(1, round(fps / sample_fps)) if np.isfinite(fps) and fps > 0 else 1
            duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps if fps > 0 else 0.0
            sampled, times = [], []
            frame_index = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_index % frame_step == 0:
                        timestamp = frame_index / fps if fps > 0 else len(times) / sample_fps
                        result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        values = None
                        if result.multi_face_landmarks:
                            h, w = frame.shape[:2]
                            try:
                                m = extractor.extract_metrics(result.multi_face_landmarks[0].landmark, w, h)
                                values = [getattr(m, key) for key in FEATURES]
                                if not np.isfinite(values).all():
                                    values = None
                            except (ValueError, cv2.error, FloatingPointError):
                                pass
                        sampled.append(values)
                        times.append(timestamp)
                    frame_index += 1
            finally:
                cap.release()
            run_x, run_t, count = [], [], 0
            for value, timestamp in zip(sampled + [None], times + [0.0]):
                if value is None:
                    for start in range(0, len(run_x) - sequence_seconds + 1, sequence_seconds):
                        xs.append(run_x[start:start + sequence_seconds])
                        ys.append(row["label"])
                        participants.append(row["participant"])
                        video_ids.append(str(row["path"]))
                        starts.append(run_t[start])
                        durations.append(duration)
                        count += 1
                    run_x, run_t = [], []
                else:
                    run_x.append(value)
                    run_t.append(timestamp)
            print(f"[{i}/{len(videos)}] participant={row['participant']} label={CLASS_NAMES[row['label']]} "
                  f"windows={count} duration_s={duration:.1f}")
    finally:
        mesh.close()
    if not xs:
        raise RuntimeError("No contiguous face-feature windows extracted; verify attached video codecs/content.")
    return {"x": np.asarray(xs, np.float32), "y": np.asarray(ys, np.int64),
            "groups": np.asarray(participants), "video_ids": np.asarray(video_ids),
            "starts": np.asarray(starts, np.float32), "durations": np.asarray(durations, np.float32),
            "manifest": np.asarray(json.dumps(sorted(
                f"{row['path'].relative_to(dataset_root).as_posix()}:{row['path'].stat().st_size}"
                for row in videos)))}


def fit_model(x, y, epochs, device, seed, val_x=None, val_y=None):
    set_seed(seed)
    model = UtaGRU().to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_f1, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(epochs):
        model.train()
        loss_sum = count = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(by)
            count += len(by)
        print(f"epoch={epoch+1:02d}/{epochs} train_loss={loss_sum/max(count,1):.4f}")
        if val_x is not None:
            val_prob = predict(model, val_x, device)
            val_f1 = f1_score(val_y, val_prob.argmax(axis=1), average="macro", zero_division=0)
            print(f"validation_macro_f1={val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1, best_epoch, stale = val_f1, epoch + 1, 0
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= 3:
                    print("Early stopping by validation macro-F1")
                    break
    if val_x is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model, (best_epoch if val_x is not None else epochs)


@torch.inference_mode()
def predict(model, x, device):
    model.eval()
    chunks = []
    for start in range(0, len(x), 512):
        logits = model(torch.from_numpy(x[start:start + 512]).to(device))
        chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(chunks)


def video_rows(data, indices, probabilities):
    by_video = defaultdict(list)
    for local, idx in enumerate(indices):
        by_video[data["video_ids"][idx]].append(local)
    rows = []
    for video, local_idx in sorted(by_video.items()):
        first = local_idx[0]
        source = indices[first]
        rows.append({"video": video, "label": int(data["y"][source]),
                     "participant": str(data["groups"][source]),
                     "duration_s": float(data["durations"][source]),
                     "starts_s": [float(data["starts"][indices[j]]) for j in local_idx],
                     "probabilities": probabilities[local_idx].tolist(),
                     "mean_probabilities": probabilities[local_idx].mean(axis=0).tolist(),
                     "max_drowsy_score": float(probabilities[local_idx, 2].max()),
                     "first_threshold_window_s": None})
    return rows


def choose_threshold(validation_rows, max_false_video_rate=0.10):
    negatives = [r["max_drowsy_score"] for r in validation_rows if r["label"] != 2]
    positives = [r["max_drowsy_score"] for r in validation_rows if r["label"] == 2]
    if not positives or not negatives:
        raise ValueError("Validation split must include drowsy and non-drowsy videos")
    candidates = sorted({0.0, 1.0, *negatives, *positives})
    valid = []
    for threshold in candidates:
        false_rate = float(np.mean(np.asarray(negatives) >= threshold))
        recall = float(np.mean(np.asarray(positives) >= threshold))
        if false_rate <= max_false_video_rate:
            valid.append((recall, threshold))
    return float(max(valid)[1])


def summarize(data, indices, rows, threshold):
    y_true = np.asarray([r["label"] for r in rows], dtype=np.int64)
    probs = np.asarray([r["mean_probabilities"] for r in rows], dtype=np.float32)
    pred = probs.argmax(axis=1)
    y_binary = (y_true == 2)
    warnings = np.asarray([r["max_drowsy_score"] >= threshold for r in rows])
    drowsy_rows = [r for r in rows if r["label"] == 2]
    first_warnings = []
    for r in rows:
        crossing = [start for start, score in zip(r["starts_s"], r["probabilities"])
                    if score[2] >= threshold]
        r["first_threshold_window_s"] = min(crossing) if crossing else None
        if r["label"] == 2 and crossing:
            first_warnings.append(min(crossing))
    return {
        "video_count": len(rows), "participant_count": len({r["participant"] for r in rows}),
        "three_class_video_accuracy": float(accuracy_score(y_true, pred)),
        "three_class_video_macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "three_class_video_confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1, 2]).tolist(),
        "three_class_video_report": classification_report(y_true, pred, labels=[0, 1, 2],
            target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "drowsy_video_recall": float(np.mean(warnings[y_binary])) if y_binary.any() else None,
        "non_drowsy_video_false_warning_rate": float(np.mean(warnings[~y_binary])) if (~y_binary).any() else None,
        "drowsy_videos_missed": int(np.sum(~warnings[y_binary])),
        "drowsy_videos_total": int(np.sum(y_binary)),
        "non_drowsy_videos_warned": int(np.sum(warnings[~y_binary])),
        "non_drowsy_videos_total": int(np.sum(~y_binary)),
        "median_first_warning_seconds_after_clip_start_on_detected_drowsy_videos":
            float(np.median(first_warnings)) if first_warnings else None,
        "warning_threshold": threshold,
        "warning_timing_note": "Labels are predominant video-level states; this is time from clip start to first threshold window, not delay from a known drowsiness onset.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/uta-rldd-results"))
    parser.add_argument("--cache", type=Path, default=Path("/kaggle/working/uta-rldd-features.npz"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-false-warning-video-rate", type=float, default=0.10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator before running this workflow")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed()
    videos = discover_videos(args.dataset_dir)
    manifest = json.dumps(sorted(
        f"{row['path'].relative_to(args.dataset_dir).as_posix()}:{row['path'].stat().st_size}"
        for row in videos))
    if args.cache.is_file():
        cached = np.load(args.cache, allow_pickle=False)
        data = {key: cached[key] for key in cached.files}
        if "manifest" not in data or str(data["manifest"].item()) != manifest:
            raise ValueError("Feature cache does not match the attached UTA-RLDD files; remove the cache and rerun")
        print(f"Loaded feature cache {args.cache}: windows={len(data['y'])}")
    else:
        data = extract_sequences(videos, args.dataset_dir)
        np.savez_compressed(args.cache, **data)
        print(f"Saved reusable feature cache {args.cache}")

    participant_to_fold = {}
    for row in videos:
        participant_to_fold[row["participant"]] = row["fold"]
    groups = data["groups"]
    participant_names = sorted(set(groups.tolist()))
    fold_ids = np.asarray([participant_to_fold[p] for p in groups])
    device = torch.device("cuda")
    fold_reports, fold_best_epochs = [], []
    all_oof = []
    for fold in range(1, 6):
        test_idx = np.flatnonzero(fold_ids == fold)
        outer_train = np.flatnonzero(fold_ids != fold)
        test_people = sorted(set(groups[test_idx].tolist()))
        if not len(test_idx) or not test_people:
            raise ValueError(f"Official fold {fold} has no sequences. Check feature extraction/cache.")
        validation_fold = fold % 5 + 1
        val_idx = np.flatnonzero(fold_ids == validation_fold)
        fit_idx = np.flatnonzero((fold_ids != fold) & (fold_ids != validation_fold))
        if set(groups[fit_idx]) & set(groups[val_idx]) or set(groups[outer_train]) & set(groups[test_idx]):
            raise RuntimeError("Participant leakage across train/validation/test split")
        scaler = StandardScaler().fit(data["x"][fit_idx].reshape(-1, len(FEATURES)))
        x_fit = scaler.transform(data["x"][fit_idx].reshape(-1, len(FEATURES))).reshape(-1, SEQUENCE_SECONDS, len(FEATURES)).astype(np.float32)
        x_val = scaler.transform(data["x"][val_idx].reshape(-1, len(FEATURES))).reshape(-1, SEQUENCE_SECONDS, len(FEATURES)).astype(np.float32)
        inner_model, best_epoch = fit_model(x_fit, data["y"][fit_idx], args.epochs, device,
                                             SEED + fold, x_val, data["y"][val_idx])
        val_prob = predict(inner_model, x_val, device)
        val_rows = video_rows(data, val_idx, val_prob)
        threshold = choose_threshold(val_rows, args.max_false_warning_video_rate)
        fold_best_epochs.append(best_epoch)

        # The official test fold is untouched. Refit on all four non-test folds,
        # retaining the epoch count selected/configured before test evaluation.
        scaler = StandardScaler().fit(data["x"][outer_train].reshape(-1, len(FEATURES)))
        x_train = scaler.transform(data["x"][outer_train].reshape(-1, len(FEATURES))).reshape(-1, SEQUENCE_SECONDS, len(FEATURES)).astype(np.float32)
        x_test = scaler.transform(data["x"][test_idx].reshape(-1, len(FEATURES))).reshape(-1, SEQUENCE_SECONDS, len(FEATURES)).astype(np.float32)
        model, _ = fit_model(x_train, data["y"][outer_train], best_epoch, device, SEED + 100 + fold)
        test_prob = predict(model, x_test, device)
        test_rows = video_rows(data, test_idx, test_prob)
        summary = summarize(data, test_idx, test_rows, threshold)
        report = {"fold": fold, "test_participants": test_people,
                  "training_participant_count": len(set(groups[outer_train])),
                  "validation_participants": sorted(set(groups[val_idx].tolist())),
                  "threshold_source": "inner validation participants only",
                  "threshold_calibration_video_summary": summarize(data, val_idx, val_rows, threshold),
                  "test": summary}
        fold_reports.append(report)
        all_oof.extend([{**r, "fold": fold} for r in val_rows])
        torch.save({"format": "safedrive-uta-rldd-gru-v1", "state_dict": model.cpu().state_dict(),
                    "input_dim": len(FEATURES), "hidden_dim": 64, "num_layers": 2,
                    "num_classes": 3, "sequence_length": SEQUENCE_SECONDS,
                    "sample_fps": SAMPLE_FPS, "feature_order": list(FEATURES),
                    "labels": {str(i): name for i, name in enumerate(CLASS_NAMES)},
                    "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                    "drowsy_warning_threshold": threshold, "fold": fold,
                    "training_participants": sorted(set(groups[outer_train].tolist()))},
                   args.output_dir / f"uta_rldd_fold_{fold}.pth")
        print(f"FOLD_{fold}_TEST " + json.dumps(summary))

    deployment_threshold = choose_threshold(all_oof, args.max_false_warning_video_rate)
    final_scaler = StandardScaler().fit(data["x"].reshape(-1, len(FEATURES)))
    x_all = final_scaler.transform(data["x"].reshape(-1, len(FEATURES))).reshape(
        -1, SEQUENCE_SECONDS, len(FEATURES)).astype(np.float32)
    deployment_epochs = max(1, int(round(float(np.median(fold_best_epochs)))))
    deployment_model, _ = fit_model(x_all, data["y"], deployment_epochs, device, SEED + 999)
    deployment_path = args.output_dir / "uta_rldd_final.pth"
    torch.save({"format": "safedrive-uta-rldd-gru-v1", "state_dict": deployment_model.cpu().state_dict(),
                "input_dim": len(FEATURES), "hidden_dim": 64, "num_layers": 2,
                "num_classes": 3, "sequence_length": SEQUENCE_SECONDS,
                "sample_fps": SAMPLE_FPS, "feature_order": list(FEATURES),
                "labels": {str(i): name for i, name in enumerate(CLASS_NAMES)},
                "scaler_mean": final_scaler.mean_.tolist(), "scaler_scale": final_scaler.scale_.tolist(),
                "drowsy_warning_threshold": deployment_threshold,
                "threshold_source": "pooled participant-held-out inner validation predictions",
                "training_participants": participant_names}, deployment_path)

    video_ids_with_features = set(data["video_ids"].tolist())
    videos_without_windows = sorted(str(row["path"]) for row in videos
                                    if str(row["path"]) not in video_ids_with_features)
    final_summary = {
        "dataset": "UTA-RLDD",
        "source_url": "https://sites.google.com/view/utarldd/home",
        "kaggle_mirror": "rishab260/uta-reallife-drowsiness-dataset (third-party mirror; provenance/rights not independently verified)",
        "citation": "Ghoddoosian, Galib, Athitsos (2019), A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection, CVPR Workshops.",
        "data_labels": {"0": "alert", "5": "low vigilance", "10": "drowsy"},
        "label_granularity": "Participant-provided predominant state for each approximately ten-minute video; not frame/event timestamps.",
        "participant_count": len(participant_names), "video_count": len(videos),
        "videos_without_extractable_windows": videos_without_windows,
        "folds": fold_reports,
        "mean_test_video_accuracy": float(np.mean([r["test"]["three_class_video_accuracy"] for r in fold_reports])),
        "mean_test_video_macro_f1": float(np.mean([r["test"]["three_class_video_macro_f1"] for r in fold_reports])),
        "mean_drowsy_video_recall": float(np.mean([r["test"]["drowsy_video_recall"] for r in fold_reports])),
        "mean_non_drowsy_video_false_warning_rate": float(np.mean([r["test"]["non_drowsy_video_false_warning_rate"] for r in fold_reports])),
        "mean_missed_drowsy_videos": float(np.mean([r["test"]["drowsy_videos_missed"] for r in fold_reports])),
        "mean_median_first_warning_seconds_after_clip_start": float(np.mean([
            r["test"]["median_first_warning_seconds_after_clip_start_on_detected_drowsy_videos"]
            for r in fold_reports if r["test"]["median_first_warning_seconds_after_clip_start_on_detected_drowsy_videos"] is not None])),
        "event_level_misses_and_true_warning_delay": None,
        "event_metric_limitation": "Cannot measure event-level misses or delay to onset from predominant video-level labels; report video-level warning and clip-start timing only.",
        "threshold_selection": f"For each test fold, select the lowest drowsy-score threshold achieving at most {args.max_false_warning_video_rate:.0%} validation non-drowsy videos with any threshold-crossing window.",
        "deployment_checkpoint": str(deployment_path),
        "deployment_threshold": deployment_threshold,
        "deployment_threshold_note": "Calibrated on pooled out-of-fold inner-validation participant videos only; per-fold test metrics remain the independent generalization estimate.",
        "deployment_training_epochs": deployment_epochs,
        "seed": SEED, "sample_fps": SAMPLE_FPS, "sequence_seconds": SEQUENCE_SECONDS,
        "feature_order": list(FEATURES), "model": "two-layer GRU over MediaPipe face features",
        "caution": "Research evaluation only; not a certified or reliable driving safety system.",
    }
    path = args.output_dir / "uta_rldd_cv_report.json"
    path.write_text(json.dumps(final_summary, indent=2) + "\n", encoding="utf-8")
    print("FINAL_UTA_RLDD_CV " + json.dumps({k: final_summary[k] for k in (
        "participant_count", "video_count", "mean_test_video_accuracy", "mean_test_video_macro_f1",
        "mean_drowsy_video_recall", "mean_non_drowsy_video_false_warning_rate",
        "mean_missed_drowsy_videos", "mean_median_first_warning_seconds_after_clip_start")}))
    print(f"Report={path}; deployment checkpoint={deployment_path}")


if __name__ == "__main__":
    main()
