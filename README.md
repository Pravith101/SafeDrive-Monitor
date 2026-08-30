# SafeDrive Monitor

## Dataset Instructions 
We are utilizing the 111GB UTA-RLDD dataset for temporal GRU training. **Do not download this manually to your local hard drive.**

To access the data directly in Google Colab for model training:
1. Install the Kaggle library: `pip install kagglehub`
2. Run the automated script: `python data/cloud_downloader.py`
3. When prompted in the terminal, enter your Kaggle username and API token (found in your Kaggle account settings). The script will pull the 90GB zip directly into Google's cloud servers.
