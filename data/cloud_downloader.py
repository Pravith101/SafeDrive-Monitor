import kagglehub

def download_uta_dataset():
    print("Initiating cloud download of UTA-RLDD...")
    # This downloads the dataset directly to the Colab/Cloud environment's virtual storage
    path = kagglehub.dataset_download("rishab260/uta-reallife-drowsiness-dataset")
    print(f"Dataset successfully downloaded to cloud path: {path}")

if __name__ == "__main__":
    download_uta_dataset()