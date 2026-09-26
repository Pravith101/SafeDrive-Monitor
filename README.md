# SafeDrive Monitor

A research prototype that extracts facial geometry from video, scales frame features, and classifies 30-frame sequences with a PyTorch GRU. This is not a certified driver-safety system and should not be used as the sole basis for driving decisions.

## Dataset and verified class mapping

The intended dataset is the [UTA Real-Life Drowsiness Dataset (UTA-RLDD)](https://sites.google.com/view/utarldd/home), distributed through the [Kaggle mirror](https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset). UTA's maintainers describe 180 RGB videos from 60 people, one participant-reported video per person for each state. The official labels are **0 = alert**, **5 = low vigilance**, and **10 = drowsy**. The Kaggle mirror's file manifest confirms files named `0.mov`/`0.mp4`, `5.*`, and `10.*` under participant-ID folders. Kaggle metadata reports **CC0: Public Domain** for the mirror. Cite Ghoddoosian et al., *A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection* (CVPR Workshops 2019), as requested by the dataset maintainers.

This GRU is a binary endpoint experiment: model class `0` is alert (dataset label `0`), and model class `1` is drowsy (dataset label `10`). Dataset label `5` (low vigilance) is excluded, not relabeled. This is not a three-class model and its output is not a complete vigilance assessment.

The corpus is large: UTA's page reports 111.3 GB and Kaggle metadata reports about 96.6 GB. Do not download it to a normal local drive. Use a cloud VM with sufficient storage. The downloader requires explicit opt-in. Preprocessing supports the Kaggle layout where the immediate parent of each video is its participant ID; it selects only videos whose basename is exactly `0` or `10`. The pre-existing development-workspace folder `data/raw_dataset/DriverActivityDataset` does not match this layout and its provenance is unverified, so it must not be used as UTA-RLDD.

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

Only download the full corpus on a cloud machine with enough storage. Configure Kaggle credentials using Kaggle's supported local credential file or environment variables; never commit credentials. Set `KAGGLEHUB_CACHE` to a cloud volume if needed. For an intentional download:

```powershell
$env:SAFEDRIVE_CONFIRM_LARGE_DOWNLOAD = "1"
python data/cloud_downloader.py
```

The default KaggleHub cache is `data/raw_dataset/`. The downloader does not unpack or process the dataset. Preprocess the directory returned by KaggleHub (use `data/raw_dataset/` if that is where it was cached):

```powershell
python data/preprocess_activity.py --dataset-dir "<KaggleHub returned directory>"
python models/temporal_gru.py
python live_inference.py
```

Press `q` in the webcam window to quit. `python test_demo.py` remains a compatibility alias for the same live command. Each missed or invalid face breaks the frame run, so no training sequence crosses a detection gap; live inference also clears its rolling window when the face is lost. Preprocessing uses a 30-frame stride by default. All videos from a participant stay in one deterministic train/validation partition (seed `42`, 20% validation); the StandardScaler is fitted only on training participants. Both partitions must contain both endpoint labels.

## Features and artifacts

Each frame has five ordered features: **EAR, MAR, pitch, yaw, roll**. Pose pitch and yaw come from OpenCV `solvePnP` using six canonical FaceMesh points and a documented approximate face model; roll is the eye-line angle. Angles are in degrees. The same extractor and feature order are used for preprocessing and webcam inference.

Preprocessing writes ignored local files to `data/processed/`: `sequences.npy`, `labels.npy`, `groups.npy`, `feature_scaler.joblib`, and `metadata.json`. Training saves `weights/temporal_gru.pth` with the architecture, window length, class map, feature order, scaler parameters, and participant-level validation accuracy. Live inference checks these fields and the scaler feature count before opening the camera. The validation partition is not an independent test set; UTA-RLDD maintainers recommend five-fold evaluation by participant for comparable research results. Re-run preprocessing and training together after changing features, labels, sequence length, or split settings.

## Tests

```powershell
python -m pytest -q
```

The test suite uses small synthetic values and temporary artifacts; it does not need the full dataset, network, webcam, or GPU. No real-data training results are reported until UTA-RLDD has been processed and evaluated; synthetic tests are not model-quality evidence. No pretrained checkpoint is included.
