import os
import sys
import kagglehub

def download_dataset() -> str:
    """
    Downloads the MRL Eye Dataset via KaggleHub using environment variables.
    Requires KAGGLE_USERNAME and KAGGLE_KEY to be set in the system environment.
    """
    kaggle_user = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")

    if not kaggle_user or not kaggle_key:
        print("[ERROR] Kaggle credentials missing.")
        print("Please export KAGGLE_USERNAME and KAGGLE_KEY before executing.")
        sys.exit(1)

    dataset_identifier = "mrl-dataset/mrl-eye-dataset"
    print(f"Authenticated as: {kaggle_user}")
    print(f"Downloading dataset: {dataset_identifier}...")

    try:
        download_path = kagglehub.dataset_download(dataset_identifier)
        print(f"[SUCCESS] Dataset stored at: {download_path}")
        return download_path
    except Exception as exc:
        print(f"[ERROR] Failed to fetch dataset: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    download_dataset()