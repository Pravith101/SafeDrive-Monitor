"""Calibrate the microsleep logit using validation videos, then re-evaluate test videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.driver_state_cnn import (ID_TO_CLASS, collect_records,
                                     make_loader, metrics, split_records)
from live_inference import load_driver_state_model


@torch.inference_mode()
def collect_logits(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for images, labels in loader:
        logits_all.append(model(images.to(device)).cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights" / "driver_state_cnn.pth")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_driver_state_model(args.checkpoint, device)
    records, skipped = collect_records(args.dataset_dir)
    train_idx, val_idx, test_idx = split_records(records, int(checkpoint.get("seed", 42)))
    val_loader = make_loader(records, val_idx, args.batch_size)
    test_loader = make_loader(records, test_idx, args.batch_size)
    val_logits, y_val = collect_logits(model, val_loader, device)
    candidates = np.arange(-0.5, 2.01, 0.25, dtype=np.float32)
    scored = []
    for bias in candidates:
        p_val = (val_logits + np.asarray([0.0, bias, 0.0], dtype=np.float32)).argmax(axis=1)
        scored.append((metrics(y_val, p_val)["macro_f1"], float(bias)))
    best_f1 = max(row[0] for row in scored)
    best_bias = min(row[1] for row in scored if row[0] == best_f1)

    test_logits, y_test = collect_logits(model, test_loader, device)
    raw_val = metrics(y_val, val_logits.argmax(axis=1))
    raw_test = metrics(y_test, test_logits.argmax(axis=1))
    calibrated_val = metrics(y_val, (val_logits + [0.0, best_bias, 0.0]).argmax(axis=1))
    calibrated_test = metrics(y_test, (test_logits + [0.0, best_bias, 0.0]).argmax(axis=1))

    checkpoint["decision_bias"] = [0.0, best_bias, 0.0]
    torch.save(checkpoint, args.checkpoint)
    report_path = args.checkpoint.with_name("driver_state_evaluation.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["validation_argmax"] = report["validation"]
    report["test_argmax"] = report["test"]
    report["calibration"] = {"method": "microsleep logit bias selected by validation macro-F1",
                             "candidate_biases": candidates.tolist(), "selected_bias": best_bias,
                             "skipped_annotations": skipped}
    report["validation"] = calibrated_val
    report["test"] = calibrated_test
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("validation argmax:", {k: raw_val[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")})
    print("validation calibrated:", {k: calibrated_val[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")})
    print("selected microsleep bias:", best_bias)
    print("test argmax:", {k: raw_test[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")})
    print("test calibrated:", {k: calibrated_test[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")})
    print("test per-class:", {name: calibrated_test["classification_report"][name]
                               for name in ID_TO_CLASS.values()})
    print("test confusion matrix:", calibrated_test["confusion_matrix"])


if __name__ == "__main__":
    main()
