# SafeDrive Monitor

A research prototype that estimates driver state from a face crop. Its three labels are **alert**, **microsleep**, and **yawning**. This is not a certified driver-safety system and must not be used as the sole basis for driving decisions.

## Cloud training with pretrained weights

Use [`models/cloud_training.ipynb`](models/cloud_training.ipynb) in Kaggle to train without storing the dataset on your computer. Add the hosted FL3D dataset as a notebook input, enable a GPU and internet access, then run the cells. The training command fine-tunes ImageNet-pretrained MobileNetV3-Small and saves a video-grouped evaluation report alongside the checkpoint. Internet access is needed to fetch the pretrained weights. Kaggle GPU availability depends on the account and current quotas.

Only the trained checkpoint and report need to be downloaded from Kaggle. Put `driver_state_cnn.pth` in `weights/` and run `python live_inference.py`. The earlier CPU run documented below used the project’s scratch CNN; it is a baseline and does not report results for pretrained fine-tuning.

## UTA-RLDD participant-held-out evaluation

[`models/uta_rldd_cloud.ipynb`](models/uta_rldd_cloud.ipynb) and [`models/uta_rldd_cloud.py`](models/uta_rldd_cloud.py) provide a separate Kaggle workflow for the roughly 111 GB [UTA-RLDD dataset](https://sites.google.com/view/utarldd/home). Attach both `rishab260/uta-reallife-drowsiness-dataset` (participant folds 1–4) and `mathiasviborg/uta-rldd-fold5` (fold 5) as notebook inputs, enable a GPU, and run the notebook. The notebook uses MediaPipe 0.10.35's maintained Tasks Face Landmarker API and downloads only its small landmark model into Kaggle working storage. Videos and extracted features stay in Kaggle; the output contains five held-out-fold reports and checkpoints. Both Kaggle uploaders list CC0, but the mirrors are unofficial and their provenance/right to relicense are not independently verified. Prefer the official UTA source and cite Ghoddoosian, Galib, and Athitsos (CVPR Workshops 2019). Do not publicly share identifiable participant imagery.

The workflow keeps each participant in one published fold, trains a three-class GRU over 30-second MediaPipe feature sequences, and tunes a drowsiness threshold on inner validation participants only. It evaluates the held-out participants at the video level and reports missed drowsy videos, non-drowsy videos with warnings, and time from clip start to the first threshold-crossing window. Since UTA-RLDD labels only each whole video's predominant state, it cannot provide true event-level misses or warning delay from the actual onset of drowsiness. FL3D's frame-level report remains separate; the label sets and evaluation units are not merged.

The completed Kaggle run outputs are saved locally as `weights/uta_rldd_final.pth` and `weights/uta_rldd_evaluation.json`; the five fold checkpoints and the raw report remain available from the notebook output. The final checkpoint can be loaded experimentally with `python live_inference.py --uta-rldd`; threshold crossings are displayed as research flags without a driver warning or sound. Its measured recall is too low for driver-alert use. Do not substitute it for the existing FL3D model or rely on either prototype while driving.

### UTA-RLDD results (Kaggle, five participant-disjoint folds)

The run processed 182 videos from 60 participants. Mean held-out video accuracy was **44.7%**, macro-F1 **42.7%**, and drowsy-video recall **35.0%**. It missed an average of **8 drowsy-labeled videos per fold** (the folds contained 12–13 such videos); the mean non-drowsy video false-warning rate was **14.2%**. The average median time to the first threshold-crossing window was **204.5 seconds after clip start** among detected drowsy-labeled videos only. Because labels describe each entire video’s predominant state and provide no onset time, this is not event-warning delay and event misses cannot be measured. These results do not support using this model for driver alerts. Thresholds were selected using validation participants only; test-fold results were not used for calibration. UTA metrics are video-level and are not directly comparable with FL3D’s frame-level metrics. Per-fold held-out results:

| Official test fold | Video accuracy | Macro-F1 | Drowsy-video recall | Non-drowsy false-warning rate | Drowsy videos missed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 33.3% | 32.8% | 41.7% | 29.2% | 7 / 12 |
| 2 | 42.9% | 39.5% | 25.0% | 4.3% | 9 / 12 |
| 3 | 45.9% | 45.2% | 46.2% | 29.2% | 7 / 13 |
| 4 | 52.8% | 53.1% | 8.3% | 0.0% | 11 / 12 |
| 5 | 48.6% | 43.1% | 53.8% | 8.3% | 6 / 13 |

This research prototype is not safety-certified and cannot establish that a real driving warning is correct or make driving safe.

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

Press `q` in the camera window to quit. The live demo smooths CNN scores across nine frames and uses MediaPipe mouth and eye measurements as consistency checks. It shows possible microsleep evidence while checking, then plays a short system sound and displays **DROWSINESS WARNING — PULL OVER SAFELY** only when the microsleep score is at least `0.90` and the eye-closure cue agrees continuously for one second with no observation gap longer than half a second. The urgent three-tone sound repeats every four seconds while both microsleep cues remain present. A validated `yawning` state gets a softer two-note cue on state entry; `uncertain` gets a single neutral tone on entry; ordinary `alert` is silent. Unconfirmed microsleep predictions stay silent. Non-urgent state cues have a three-second cooldown. A yawning prediction requires an open-mouth cue; an `alert` prediction is withheld as `UNCERTAIN` when the eyes look strongly closed. Check one still image without a webcam using `python live_inference.py --image path\to\frame.jpg`.

The `0.90` softmax score is a conservative trigger setting, not a calibrated probability that a prediction is correct. Requiring sustained agreement from the CNN and eye-closure measurement reduces unsupported warnings, but cannot make the system certain: on held-out FL3D videos the CNN missed about 29% of microsleep frames, and the eye-closure cue also misses many events. Treat warnings as a prompt to check yourself and stop somewhere safe if drowsy; never rely on this prototype as a safety system or use its lack of a warning as evidence that driving is safe. The model was trained on night-time, face-cropped footage; daylight webcam performance and person-level generalization require separate validation.

Training writes ignored artifacts: `weights/driver_state_cnn.pth` and `weights/driver_state_evaluation.json`. The report contains exact split video IDs, class counts, validation metrics, test metrics, confusion matrix, and training hardware. No model weights or dataset files are tracked in Git.

## Current pretrained FL3D checkpoint

The current local `weights/driver_state_cnn.pth` was trained on Kaggle with pretrained MobileNetV3-Small (best validation checkpoint: epoch 6). On its held-out-video FL3D test split it scored **90.47% accuracy**, **88.51% balanced accuracy**, and **90.24% macro-F1**. Microsleep recall was **71.37%** (1,635 of 2,291 microsleep frames detected), so 656 microsleep frames were missed. This is still not a participant-held-out or safety validation.

## Earlier scratch-CNN baseline

This earlier run completed on the downloaded FL3D snapshot with seed `42`, 64×64 RGB input, class/video-balanced sampling, batch size `256`, and eight epochs. Hardware was CPU-only (`torch 2.13.0+cpu`; CUDA unavailable). Its checkpoint/report have been superseded for webcam inference by the pretrained FL3D checkpoint above. The participant ID limitation above applies to these measurements.

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
