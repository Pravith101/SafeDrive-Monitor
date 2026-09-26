"""Calibrate visual consistency gates from FL3D validation videos, then score held-out videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live_inference import eye_aspect_ratio, mouth_aperture_ratio
from models.driver_state_cnn import (CHECKPOINT_FORMAT, EYE_GATE_VERSION, MOUTH_GATE_VERSION,
                                     collect_records, split_records)

SAMPLES_PER_CLASS = 240
MIN_VALIDATION_RECALL = 0.5


def sample_apertures(records, indices, face_mesh, seed=42):
    rng = np.random.default_rng(seed)
    output = []
    missing = 0
    for label in (0, 1, 2):
        candidates = np.asarray([int(i) for i in indices if records[int(i)][1] == label])
        selected = rng.choice(candidates, min(SAMPLES_PER_CLASS, len(candidates)), replace=False)
        for index in selected:
            image_path, true_label, video = records[int(index)]
            frame = cv2.imread(str(image_path))
            if frame is None:
                missing += 1
                continue
            result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:
                missing += 1
                continue
            landmarks = result.multi_face_landmarks[0].landmark
            try:
                ratio = mouth_aperture_ratio(landmarks, frame.shape[1], frame.shape[0])
                eye_ratio = eye_aspect_ratio(landmarks, frame.shape[1], frame.shape[0])
            except ValueError:
                missing += 1
                continue
            output.append({"label": int(true_label), "ratio": ratio,
                           "eye_ratio": eye_ratio, "video": video})
    return output, missing


def gate_metrics(rows, threshold, *, feature="ratio", positive_label=2, closed=False):
    truth = np.asarray([row["label"] == positive_label for row in rows], dtype=bool)
    values = np.asarray([row[feature] for row in rows], dtype=np.float64)
    predicted = values <= threshold if closed else values >= threshold
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    positives = int(np.sum(truth))
    return {
        "samples_with_face": len(rows),
        "positive_class_id": positive_label,
        "positive_class_samples": positives,
        "cue_positive_predictions": int(np.sum(predicted)),
        "true_positive": tp,
        "false_positive": fp,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(positives, 1)),
    }


def choose_threshold(validation_rows, *, feature="ratio", positive_label=2, closed=False):
    ratios = sorted({row[feature] for row in validation_rows})
    choices = []
    for threshold in ratios:
        metrics = gate_metrics(validation_rows, threshold, feature=feature,
                                positive_label=positive_label, closed=closed)
        if metrics["recall"] >= MIN_VALIDATION_RECALL:
            choices.append((metrics["precision"], metrics["recall"], threshold))
    if not choices:
        raise RuntimeError("Could not find a mouth-opening threshold with at least 50% validation recall")
    # Maximize validation precision, then recall; deterministic lowest threshold breaks any remaining tie.
    best_precision = max(row[0] for row in choices)
    best_recall = max(row[1] for row in choices if row[0] == best_precision)
    return min(row[2] for row in choices if row[0] == best_precision and row[1] == best_recall)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="FL3D dataset directory containing classification_frames/")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights" / "driver_state_cnn.pth")
    args = parser.parse_args()

    records, _ = collect_records(args.dataset_dir)
    _, validation_indices, test_indices = split_records(records)
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                refine_landmarks=True, min_detection_confidence=0.3)
    try:
        validation_rows, validation_missing = sample_apertures(records, validation_indices, face_mesh)
        test_rows, test_missing = sample_apertures(records, test_indices, face_mesh)
    finally:
        face_mesh.close()
    if not validation_rows or not test_rows:
        raise RuntimeError("MediaPipe could not measure mouth landmarks for the selected FL3D samples")

    threshold = choose_threshold(validation_rows)
    eye_validation_rows = [row for row in validation_rows if row["label"] in (0, 1)]
    eye_test_rows = [row for row in test_rows if row["label"] in (0, 1)]
    eye_threshold = choose_threshold(eye_validation_rows, feature="eye_ratio",
                                     positive_label=1, closed=True)
    report = {
        "dataset": "FL3D / Kaggle matjazmuc/frame-level-driver-drowsiness-detection-fl3d",
        "gate": MOUTH_GATE_VERSION,
        "definition": "distance(inner lip landmarks 13,14) / distance(mouth corners 61,291), in frame pixels",
        "eye_gate": EYE_GATE_VERSION,
        "eye_definition": "mean of each eye's standard six-point aspect ratio, landmarks 33/133 and 362/263",
        "sample_per_class_requested": SAMPLES_PER_CLASS,
        "sample_seed": 42,
        "validation_video_groups": sorted({row["video"] for row in validation_rows}),
        "test_video_groups": sorted({row["video"] for row in test_rows}),
        "validation_missing_face_or_image": validation_missing,
        "test_missing_face_or_image": test_missing,
        "minimum_validation_recall": MIN_VALIDATION_RECALL,
        "threshold": threshold,
        "validation": gate_metrics(validation_rows, threshold),
        "test": gate_metrics(test_rows, threshold),
        "eye_closed_ratio_threshold": eye_threshold,
        "eye_validation": gate_metrics(eye_validation_rows, eye_threshold, feature="eye_ratio",
                                       positive_label=1, closed=True),
        "eye_test": gate_metrics(eye_test_rows, eye_threshold, feature="eye_ratio",
                                 positive_label=1, closed=True),
        "interpretation": "The mouth cue can abstain on unsupported yawning predictions. The eye cue can abstain "
                         "when alert/microsleep predictions conflict with measured eye openness. Neither cue is "
                         "a continuous drowsiness detector; the system is not safety-rated.",
    }

    args.checkpoint = args.checkpoint.resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing CNN checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Checkpoint is not a compatible SafeDrive FL3D CNN checkpoint")
    checkpoint["mouth_gate_version"] = MOUTH_GATE_VERSION
    checkpoint["mouth_open_ratio_threshold"] = float(threshold)
    checkpoint["eye_gate_version"] = EYE_GATE_VERSION
    checkpoint["eye_closed_ratio_threshold"] = float(eye_threshold)
    torch.save(checkpoint, args.checkpoint)
    report_path = args.checkpoint.with_name("driver_state_mouth_gate_evaluation.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mouth_threshold": threshold, "validation": report["validation"],
                      "test": report["test"], "eye_threshold": eye_threshold,
                      "eye_validation": report["eye_validation"], "eye_test": report["eye_test"],
                      "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
