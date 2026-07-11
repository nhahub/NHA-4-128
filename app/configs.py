import os
import zipfile
import shutil
import threading
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from huggingface_hub import hf_hub_download

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "omarelrayes/mlflow-artifacts")
CLASSIFIER_MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH", "models/classifier_savedmodel.zip")
SEGMENTER_MODEL_PATH = os.getenv("SEGMENTER_MODEL_PATH", "models/segmenter_savedmodel.zip")
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", tempfile.gettempdir())) / "savedmodels"

_classification_model = None
_segmentation_model = None
_model_lock = threading.Lock()

CLASSIFIER_THRESHOLD = float(os.getenv("CLASSIFIER_THRESHOLD", "0.5"))

request_history: List[Dict[str, Any]] = []


def sigmoid_to_class(confidence: float) -> str:
    return "malicious" if confidence >= CLASSIFIER_THRESHOLD else "benign"


def _download_and_extract(zip_path_in_repo: str, extract_subdir: str):
    extract_dir = MODEL_CACHE_DIR / extract_subdir
    if extract_dir.exists():
        return str(extract_dir)

    print(f"Downloading {zip_path_in_repo} from {HF_MODEL_REPO}...")
    zip_file = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=zip_path_in_repo,
        repo_type="model",
        token=HF_TOKEN or None,
    )

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(extract_dir)

    print(f"Extracted model to {extract_dir}")
    return str(extract_dir)


def _load_tf_model(zip_path: str, subdir: str):
    import tensorflow as tf

    model_dir = _download_and_extract(zip_path, subdir)
    loaded = tf.saved_model.load(model_dir)
    return loaded.signatures["serving_default"]


def get_classification_model():
    global _classification_model
    if _classification_model is None:
        with _model_lock:
            if _classification_model is None:
                print("Loading Classification Model...")
                _classification_model = _load_tf_model(CLASSIFIER_MODEL_PATH, "classifier")
    return _classification_model


def get_segmentation_model():
    global _segmentation_model
    if _segmentation_model is None:
        with _model_lock:
            if _segmentation_model is None:
                print("Loading Segmentation Model...")
                _segmentation_model = _load_tf_model(SEGMENTER_MODEL_PATH, "segmenter")
    return _segmentation_model


def get_loaded_versions() -> List[str]:
    return ["hf_savedmodel"]


CLASSIFIER_URI = os.getenv("CLASSIFIER_MODEL_URI", "hf_savedmodel")
SEGMENTER_URI = os.getenv("SEGMENTER_MODEL_URI", "hf_savedmodel")

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "storage"))
IMAGES_DIR = STORAGE_DIR / "images"
SEGMENTS_DIR = STORAGE_DIR / "segments"
RESULTS_DIR = STORAGE_DIR / "results"

for dir_path in [STORAGE_DIR, IMAGES_DIR, SEGMENTS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
