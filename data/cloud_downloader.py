"""Download the UTA-RLDD Kaggle mirror into the selected KaggleHub cache."""
import os
from pathlib import Path


DATASET = "rishab260/uta-reallife-drowsiness-dataset"


def download_dataset() -> str:
    # Import only after honoring KaggleHub's cache override.
    cache_dir = Path(os.environ.get("KAGGLEHUB_CACHE", Path(__file__).resolve().parent / "raw_dataset"))
    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)
    if os.environ.get("SAFEDRIVE_CONFIRM_LARGE_DOWNLOAD") != "1":
        raise RuntimeError("UTA-RLDD is a very large download. Set SAFEDRIVE_CONFIRM_LARGE_DOWNLOAD=1 "
                           "only when running on a machine with sufficient storage, or run this in a cloud VM.")
    import kagglehub

    try:
        path = kagglehub.dataset_download(DATASET)
    except Exception as exc:
        raise RuntimeError(f"Could not download Kaggle dataset {DATASET}: {exc}") from exc
    print(f"Dataset downloaded to: {path}")
    return path


if __name__ == "__main__":
    download_dataset()
