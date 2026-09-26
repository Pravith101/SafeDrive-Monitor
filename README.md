# SafeDrive Monitor

A research prototype that extracts facial geometry from video, scales frame features, and classifies 30-frame sequences with a PyTorch GRU. This is not a certified driver-safety system and should not be used as the sole basis for driving decisions.

## Data and class mapping

The preprocessing script currently selects filenames containing the exact activity token `a02` or `a10` and ending in `_rgb.mp4`:

| Class ID | Selected activity | Project interpretation |
| --- | --- | --- |
| `0` | `a02` | Safe / normal driving |
| `1` | `a10` | Drowsiness-related / nodding |

The token is matched as a delimited filename component, so names such as `a020` are excluded. The mapping is a project convention and must match the labels in the dataset copy used for training. UTA-RLDD itself is described by its maintainers as three participant-rated states: alert, low vigilant, and drowsy. It does not define an `a02` “safe” and `a10` “nodding” taxonomy. Do not use the mapping above for UTA-RLDD unless you have verified that your files' names/labels actually encode those classes. This prototype's binary mapping also discards any intermediate state.

UTA-RLDD is about 111 GB; do not download it to a normal local drive. The downloader below requires a deliberate environment-variable opt-in and is intended for a cloud VM with sufficient storage. The project uses the Kaggle mirror slug `rishab260/uta-reallife-drowsiness-dataset`; access and layout may change. See the [official UTA-RLDD page](https://sites.google.com/view/utarldd/home) and [Kaggle mirror](https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset).

## Requirements and setup

Use Python 3.10 or 3.11 for MediaPipe compatibility. Training can run on CPU; a supported CUDA/PyTorch setup is optional. Webcam inference needs a working camera and GUI display.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux, create and activate `.venv` with the platform's normal `python3.11 -m venv .venv` and `source .venv/bin/activate` commands.

## Dataset acquisition and pipeline

Only download the full corpus on a cloud machine with enough storage. Configure Kaggle credentials using Kaggle's supported local credential file or environment variables; never commit credentials. For an intentional download:

```powershell
$env:SAFEDRIVE_CONFIRM_LARGE_DOWNLOAD = "1"
python data/cloud_downloader.py
```

The default KaggleHub cache is `data/raw_dataset/`. You can set `KAGGLEHUB_CACHE` to a cloud volume before running the command. The download script does not unpack or process the dataset. Arrange compatible `*_rgb.mp4` files under `data/raw_dataset/` before preprocessing.

```powershell
python data/preprocess_activity.py
python models/temporal_gru.py
python live_inference.py
```

Press `q` in the webcam window to quit. `python test_demo.py` remains a compatibility alias for the same live command. Each missed or invalid face breaks the frame run, so no training sequence crosses a detection gap; live inference also clears its rolling window when the face is lost. Preprocessing uses a 30-frame stride by default to limit memory use. Sequences from a source video stay in one deterministic train/validation partition (seed `42`, 20% validation); the StandardScaler is fitted only on training videos. Both partitions must contain both classes, which generally requires multiple videos in each class.

## Features and artifacts

Each frame has five ordered features: **EAR, MAR, pitch, yaw, roll**. Pose pitch and yaw come from OpenCV `solvePnP` using six canonical FaceMesh points and a documented approximate face model; roll is the eye-line angle. Angles are in degrees. The same extractor and feature order are used for preprocessing and webcam inference.

Preprocessing writes ignored local files to `data/processed/`: `sequences.npy`, `labels.npy`, `groups.npy`, `feature_scaler.joblib`, and `metadata.json`. Training saves `weights/temporal_gru.pth` with the architecture, window length, class map, feature order, and validation accuracy. Live inference checks these fields and the scaler feature count before opening the camera. Re-run preprocessing and training together after changing features, labels, sequence length, or split settings.

## Tests

```powershell
python -m pytest -q
```

The test suite uses small synthetic values and temporary artifacts; it does not need the full dataset, network, webcam, or GPU. Preprocessing the real corpus, model quality, and webcam performance depend on dataset labels, storage, lighting, camera placement, and hardware. No pretrained checkpoint is included.
