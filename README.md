# SafeDrive Monitor

A research prototype that estimates driver state from a face crop using a small convolutional neural network. Its three labels are **alert**, **microsleep**, and **yawning**. This is not a certified driver-safety system and must not be used as the sole basis for driving decisions.

## Dataset, labels, and license

The project trains on [FL3D (Frame Level Driver Drowsiness Detection)](https://www.kaggle.com/datasets/matjazmuc/frame-level-driver-drowsiness-detection-fl3d), a frame-labeled derivative of the [NITYMED night-time driver dataset](https://datasets.esdalab.ece.uop.gr/). The FL3D Kaggle dataset is about **645 MB** and declares **CC BY-SA 4.0**. Attribute the FL3D dataset author and the NITYMED authors, preserve the license for distributed adaptations, and cite the referenced dataset work before reuse or redistribution.

FL3D defines three frame labels: `alert`, `microsleep`, and `yawning`. Its annotations mark closed-eye frames in microsleep sessions as `microsleep`, and wide-open-mouth frames in yawning sessions as `yawning`; blink frames are omitted. These are observable event labels, not medical diagnoses or a continuous drowsiness rating. A `microsleep` or `yawning` prediction is a warning cue; `alert` means only that the current frame resembles the dataset's alert class.

The Kaggle release contains 53,331 face-cropped JPEG frames in 44 source-video folders with `annotations_final.json`. Two annotation entries without a usable class/image are skipped. The archived Kaggle split is not used: this workflow assigns complete source-video folders to about 60% train, 21% validation, and 19% test partitions with `StratifiedGroupKFold` (seed 42). No source video appears in more than one partition. The downloaded archive does not provide a reliable participant ID for every frame, so person-level separation across different videos cannot be independently guaranteed. Reported test metrics are held-out-video results, not a verified cross-subject benchmark.

## Setup and download

Use Python 3.10 or 3.11 for MediaPipe compatibility.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dataset download is optional but requires explicit opt-in. Files are stored under the ignored `data/downloaded/` directory, separate from any existing raw data.

```powershell
$env:SAFEDRIVE_CONFIRM_DATASET_DOWNLOAD = "1"
python data/cloud_downloader.py
```

KaggleHub prints the returned dataset path. The default cache is `data/downloaded/kaggle_cache/`.

## Preprocess, train, evaluate, and run the camera demo

Pass the KaggleHub path ending in `versions/1` to the trainer. The trainer reads labels and frame-to-video grouping from each `classification_frames/*/annotations_final.json` file, creates disjoint video-group splits, trains with balanced sampling across class/video groups, selects the checkpoint by validation macro-F1, then evaluates the held-out test videos once.

```powershell
python models/driver_state_cnn.py --dataset-dir "data/downloaded/kaggle_cache/datasets/matjazmuc/frame-level-driver-drowsiness-detection-fl3d/versions/1"
python models/calibrate_driver_state.py --dataset-dir "data/downloaded/kaggle_cache/datasets/matjazmuc/frame-level-driver-drowsiness-detection-fl3d/versions/1/classification_frames"
python models/calibrate_mouth_gate.py --dataset-dir "data/downloaded/kaggle_cache/datasets/matjazmuc/frame-level-driver-drowsiness-detection-fl3d/versions/1"
python live_inference.py
```

Press `q` in the camera window to quit. The live demo uses MediaPipe to crop a face and smooths predictions over nine frames. It no longer presents softmax as a confidence percentage. A CNN `yawning` prediction is shown as `UNCERTAIN` unless MediaPipe also measures an open-mouth cue above the threshold calibrated on the FL3D validation videos. A CNN `alert` prediction is shown as `UNCERTAIN` when both eyes look strongly closed. Check one still image without a webcam using `python live_inference.py --image path\to\frame.jpg`; this prints the raw CNN class, measured cues, and gated result.

These visual cues are consistency checks, not diagnoses or proof of a state. The eye gate only withholds an `alert` result on strong eye-closure evidence; it does not reliably detect every microsleep. The model was trained on night-time, face-cropped footage; daylight webcam performance and person-level generalization require separate validation.

Training writes ignored artifacts: `weights/driver_state_cnn.pth` and `weights/driver_state_evaluation.json`. The report contains exact split video IDs, class counts, validation metrics, test metrics, confusion matrix, and training hardware. No model weights or dataset files are tracked in Git.

## Verified training run

Run completed on the downloaded FL3D snapshot with seed `42`, 64×64 RGB input, class/video-balanced sampling, batch size `256`, and eight epochs. Hardware was CPU-only (`torch 2.13.0+cpu`; CUDA unavailable). The best validation checkpoint was epoch 7. The participant ID limitation above applies to these measurements.

| Held-out split | Frames | Accuracy | Balanced accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Validation (8 source videos) | 11,368 | 95.63% | 94.09% | 93.83% |
| Test (8 source videos) | 10,216 | 91.44% | 83.41% | 86.66% |

Test per-class precision / recall / F1: alert **90.71 / 98.55 / 94.47%**; microsleep **92.54 / 55.51 / 69.40%**; yawning **96.07 / 96.16 / 96.12%**. Confusion matrix (rows = true class, columns = predicted class; order alert, microsleep, yawning): `[[7433, 73, 36], [721, 906, 5], [40, 0, 1002]]`. The lower microsleep recall means many microsleep frames were predicted alert; this model is not suitable as a safety-critical detector.

### Yawning consistency check

`models/calibrate_mouth_gate.py` samples up to 240 frames per class from deterministic validation and held-out test video splits (seed 42). For yawning it measures MediaPipe inner-lip distance (landmarks 13–14) divided by mouth-corner distance (61–291), picks a validation threshold for maximum precision with at least 50% recall, then scores that fixed threshold on test. The threshold was **0.26845**. Of 719 validation images with detected faces, precision was **100.00%** and recall **92.08%**. On 711 test images, precision was **99.09%** and recall **90.42%** (2 false open-mouth cues among 471 non-yawning frames).

The same script calibrates an eye-closure threshold from alert and microsleep frames only. It uses the mean six-point eye aspect ratio for the two eyes. The threshold was **0.12484**. Of 479 validation images with detected faces, precision for the closed-eye cue was **99.32%** with **60.83%** recall. On 471 test images, precision was **100.00%** with only **22.32%** recall. This low test recall means many microsleep frames are missed; the cue only withholds an `alert` result when eyes are strongly closed. These sampled cue measurements are not the CNN’s three-class metrics.

The supplied screenshots were run through the same still-image inference path. On the closed-eye frame, the raw CNN said `alert`, but its eye ratio was **0.0897**, below the calibrated threshold, so the result is `uncertain`. On the open-eye, closed-mouth frame, it said `yawning` with a mouth ratio of **0.0055**, so that result is also `uncertain`. These two examples are regression checks, not independent accuracy estimates.

## Tests

```powershell
python -m pytest -q
```

Tests use synthetic inputs and do not need the dataset, network, camera, or GPU. Model-quality claims must come from the evaluation report produced by training, not from synthetic tests.
