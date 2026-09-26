"""Train and evaluate a compact CNN on the frame-labeled FL3D dataset."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = 64
CLASS_TO_ID = {"alert": 0, "microsleep": 1, "yawning": 2}
ID_TO_CLASS = {str(v): k for k, v in CLASS_TO_ID.items()}
CHECKPOINT_FORMAT = "safedrive-fl3d-cnn-v1"
SEED = 42
MOUTH_GATE_VERSION = "mediapipe-468-inner-lip-ratio-v1"
MOUTH_OPEN_RATIO_THRESHOLD = 0.2684476375579834
EYE_GATE_VERSION = "mediapipe-six-point-ear-v1"
EYE_CLOSED_RATIO_THRESHOLD = 0.12484410527134987


class DriverStateCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_records(dataset_dir: str | Path):
    root = Path(dataset_dir)
    annotations = sorted(root.rglob("annotations_final.json"))
    if not annotations:
        raise FileNotFoundError(f"No annotations_final.json files under {root}")
    records = []
    skipped = Counter()
    for annotation in annotations:
        rows = json.loads(annotation.read_text(encoding="utf-8"))
        for filename, row in rows.items():
            label = row.get("driver_state")
            image = annotation.parent / filename
            if label not in CLASS_TO_ID or not image.is_file():
                skipped[str(label)] += 1
                continue
            records.append((image, CLASS_TO_ID[label], annotation.parent.name))
    if not records:
        raise RuntimeError("No labeled FL3D images found (expected alert, microsleep, and yawning annotations).")
    return records, dict(skipped)


def split_records(records, seed: int = SEED):
    labels = np.asarray([row[1] for row in records], dtype=np.int64)
    groups = np.asarray([row[2] for row in records], dtype=str)
    if len(np.unique(groups)) < 12:
        raise ValueError("At least 12 labeled source-video groups are required for train/validation/test splits")
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(outer.split(np.zeros(len(labels)), labels, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_rel_idx, val_rel_idx = next(inner.split(np.zeros(len(train_val_idx)),
                                                  labels[train_val_idx], groups[train_val_idx]))
    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]
    sets = [set(groups[idx]) for idx in (train_idx, val_idx, test_idx)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("Source-video leakage detected in the group split")
    for name, idx in zip(("train", "validation", "test"), (train_idx, val_idx, test_idx)):
        if set(np.unique(labels[idx])) != set(CLASS_TO_ID.values()):
            raise ValueError(f"{name} split is missing one or more classes; add more labeled videos")
    return train_idx, val_idx, test_idx


class FrameDataset(Dataset):
    def __init__(self, records, indices, augment: bool = False):
        self.records = records
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image_path, label, _ = self.records[int(self.indices[position])]
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            if self.augment:
                if random.random() < 0.5:
                    image = ImageOps.mirror(image)
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        tensor = (tensor - 0.5) / 0.5
        return tensor, int(label)


def make_loader(records, indices, batch_size: int, shuffle: bool = False, augment: bool = False):
    dataset = FrameDataset(records, indices, augment=augment)
    sampler = None
    if shuffle:
        group_class_counts = Counter((records[int(i)][2], records[int(i)][1]) for i in indices)
        class_group_counts = Counter(label for _, label in group_class_counts)
        weights = torch.as_tensor([
            1.0 / (class_group_counts[records[int(i)][1]] *
                   group_class_counts[(records[int(i)][2], records[int(i)][1])])
            for i in indices
        ], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    workers = 2 if torch.cuda.is_available() else 0
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle and sampler is None,
                      sampler=sampler, num_workers=workers, pin_memory=torch.cuda.is_available())


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1, 2], target_names=[ID_TO_CLASS[str(i)] for i in range(3)],
            output_dict=True, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="FL3D dataset directory containing classification_frames/")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "weights" / "driver_state_cnn.pth")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size, and patience must be positive")

    set_seed()
    records, skipped = collect_records(args.dataset_dir)
    train_idx, val_idx, test_idx = split_records(records)
    print(f"Records={len(records)} skipped={skipped}; source videos="
          f"{len(set(row[2] for row in records))}; train/val/test="
          f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)} frames")
    for name, idx in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        print(f"{name} classes={dict(Counter(ID_TO_CLASS[str(records[int(i)][1])] for i in idx))}; "
              f"videos={sorted({records[int(i)][2] for i in idx})}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DriverStateCNN().to(device)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    train_loader = make_loader(records, train_idx, args.batch_size, shuffle=True, augment=True)
    val_loader = make_loader(records, val_idx, args.batch_size)
    test_loader = make_loader(records, test_idx, args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_f1, best_state, stale, best_epoch = -1.0, None, 0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total = correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
        y_val, p_val = predict(model, val_loader, device)
        val_metrics = metrics(y_val, p_val)
        print(f"epoch={epoch:02d} train_loss={total_loss/max(total,1):.4f} "
              f"train_acc={correct/max(total,1):.4f} val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping on validation macro-F1.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    y_val, p_val = predict(model, val_loader, device)
    y_test, p_test = predict(model, test_loader, device)
    report = {
        "dataset": "FL3D / Kaggle matjazmuc/frame-level-driver-drowsiness-detection-fl3d",
        "dataset_dir": str(args.dataset_dir),
        "license": "CC BY-SA 4.0",
        "seed": SEED,
        "device": str(device),
        "image_size": IMAGE_SIZE,
        "labels": ID_TO_CLASS,
        "skipped_annotations": skipped,
        "best_epoch": best_epoch,
        "train_frames": len(train_idx), "validation_frames": len(val_idx), "test_frames": len(test_idx),
        "train_videos": sorted({records[int(i)][2] for i in train_idx}),
        "validation_videos": sorted({records[int(i)][2] for i in val_idx}),
        "test_videos": sorted({records[int(i)][2] for i in test_idx}),
        "validation": metrics(y_val, p_val),
        "test": metrics(y_test, p_test),
    }
    checkpoint = {"format": CHECKPOINT_FORMAT, "state_dict": best_state,
                  "image_size": IMAGE_SIZE, "class_to_id": CLASS_TO_ID,
                  "normalization_mean": [0.5] * 3, "normalization_std": [0.5] * 3,
                  "mouth_gate_version": MOUTH_GATE_VERSION,
                  "mouth_open_ratio_threshold": MOUTH_OPEN_RATIO_THRESHOLD,
                  "eye_gate_version": EYE_GATE_VERSION,
                  "eye_closed_ratio_threshold": EYE_CLOSED_RATIO_THRESHOLD,
                  "best_epoch": best_epoch, "seed": SEED}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report_path = args.output.with_name("driver_state_evaluation.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("FINAL_TEST " + json.dumps({key: report["test"][key]
                                     for key in ("accuracy", "balanced_accuracy", "macro_f1", "confusion_matrix")}))
    print(f"Checkpoint={args.output}; report={report_path}")


if __name__ == "__main__":
    main()
