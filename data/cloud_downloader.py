"""Download the compact, frame-labeled FL3D driver-state dataset from Kaggle."""
import os
from pathlib import Path


DATASET = "matjazmuc/frame-level-driver-drowsiness-detection-fl3d"


def download_dataset() -> str:
    # Import only after honoring KaggleHub's cache override.
    cache_dir = Path(os.environ.get("KAGGLEHUB_CACHE", Path(__file__).resolve().parent / "downloaded" / "kaggle_cache"))
    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)
    if os.environ.get("SAFEDRIVE_CONFIRM_DATASET_DOWNLOAD") != "1":
        raise RuntimeError("FL3D is a ~645 MB dataset. Set SAFEDRIVE_CONFIRM_DATASET_DOWNLOAD=1 "
                           "to confirm the download and local storage use.")
    import kagglehub

    try:
        path = kagglehub.dataset_download(DATASET)
    except Exception as exc:
        raise RuntimeError(f"Could not download Kaggle dataset {DATASET}: {exc}") from exc
    print(f"Dataset downloaded to: {path}")
    return path


if __name__ == "__main__":
    download_dataset()
