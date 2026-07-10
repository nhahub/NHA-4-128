import os
import sys

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSIFICATION_URL = os.environ.get(
    "CLASSIFICATION_MODEL_URL",
    ""
)
SEGMENTATION_URL = os.environ.get(
    "SEGMENTATION_MODEL_URL",
    ""
)


def download_model(url: str, dest: str):
    if not url:
        print(f"Skipping {dest}: no URL provided")
        return
    import requests
    print(f"Downloading {dest}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    dest_path = os.path.join(MODEL_DIR, dest)
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {dest}")


if __name__ == "__main__":
    if CLASSIFICATION_URL:
        download_model(CLASSIFICATION_URL, "skin_cancer_classifier")
    if SEGMENTATION_URL:
        download_model(SEGMENTATION_URL, "skin_cancer_segmenter")
