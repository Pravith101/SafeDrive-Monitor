import os
import sys

# 1. Override the cache location BEFORE importing kagglehub
custom_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "raw_dataset"))
os.environ["KAGGLEHUB_CACHE_DIR"] = custom_cache_dir

import kagglehub

def download_dataset() -> str:
    # 2. Validate Credentials
    kaggle_user = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")

    if not kaggle_user or not kaggle_key:
        print("[ERROR] Kaggle credentials missing.")
        print("Please export KAGGLE_USERNAME and KAGGLE_KEY before executing.")
        sys.exit(1)

    dataset_identifier = "guanhualee/driver-activity-dataset"
    print(f"Authenticated as: {kaggle_user}")
    print(f"Routing download to D drive: {custom_cache_dir}...")

    try:
        download_path = kagglehub.dataset_download(dataset_identifier)
        print(f"[SUCCESS] Dataset stored at: {download_path}")
        return download_path
    except Exception as exc:
        print(f"[ERROR] Failed to fetch dataset: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    download_dataset()