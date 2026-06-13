import os
from huggingface_hub import HfApi, snapshot_download

class StorageManager:
    """
    Handles persistent storage by syncing local files with a Hugging Face Dataset.
    Requires HF_TOKEN (secret) and DATASET_ID (environment variable).
    """
    def __init__(self, repo_id=None, token=None):
        self.repo_id = repo_id or os.environ.get("DATASET_ID")
        self.token = token or os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.token) if self.token else None

    def download_data(self, local_dir):
        """Downloads latest files from the dataset to local storage."""
        if not self.repo_id:
            print("STORAGE_DEBUG: No DATASET_ID provided, skipping sync-down.")
            return
        try:
            print(f"STORAGE_DEBUG: Downloading data from {self.repo_id}...")
            snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                token=self.token,
                allow_patterns=["found_items/*", "metadata.json", "found_items.index"],
                ignore_metadata_files=True
            )
            print("STORAGE_DEBUG: Initial sync-down complete.")
        except Exception as e:
            print(f"STORAGE_DEBUG: Sync-down failed or dataset empty: {e}")

    def upload_data(self, local_dir):
        """Uploads local files to the dataset."""
        if not self.repo_id or not self.token:
            print("STORAGE_DEBUG: DATASET_ID or HF_TOKEN missing, skipping sync-up.")
            return
        try:
            print(f"STORAGE_DEBUG: Syncing data to {self.repo_id}...")
            self.api.upload_folder(
                folder_path=local_dir,
                repo_id=self.repo_id,
                repo_type="dataset",
                path_in_repo=".",
                allow_patterns=["found_items/*", "metadata.json", "found_items.index"]
            )
            print("STORAGE_DEBUG: Sync-up complete.")
        except Exception as e:
            print(f"STORAGE_DEBUG: Sync-up failed: {e}")
