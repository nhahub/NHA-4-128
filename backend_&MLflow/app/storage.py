import json
import threading
from pathlib import Path
from typing import Optional
from app.configs import IMAGES_DIR, SEGMENTS_DIR, RESULTS_DIR

_results: dict = {}
_results_lock = threading.Lock()
_RESULTS_FILE = RESULTS_DIR / "results.json"


def _load_results():
    global _results
    if _RESULTS_FILE.exists():
        with open(_RESULTS_FILE, "r") as f:
            _results = json.load(f)


def _persist_results():
    with open(_RESULTS_FILE, "w") as f:
        json.dump(_results, f, indent=2)


def save_image(image_id: str, file_bytes: bytes, extension: str = ".jpg") -> Path:
    image_path = IMAGES_DIR / f"{image_id}{extension}"
    with open(image_path, "wb") as f:
        f.write(file_bytes)
    return image_path


def get_image_path(image_id: str) -> Optional[Path]:
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        path = IMAGES_DIR / f"{image_id}{ext}"
        if path.exists():
            return path
    return None


def get_image_bytes(image_id: str) -> Optional[bytes]:
    path = get_image_path(image_id)
    if path is None:
        return None
    with open(path, "rb") as f:
        return f.read()


def save_classification_result(image_id: str, result: dict) -> None:
    with _results_lock:
        if image_id not in _results:
            _results[image_id] = {}
        _results[image_id]["classification"] = result
        _persist_results()


def get_classification_result(image_id: str) -> Optional[dict]:
    with _results_lock:
        if not _results:
            _load_results()
        return _results.get(image_id, {}).get("classification")


def save_segmentation_result(image_id: str, seg_result: dict) -> str:
    seg_filename = f"{image_id}_segment.json"
    seg_path = SEGMENTS_DIR / seg_filename
    with open(seg_path, "w") as f:
        json.dump(seg_result, f, indent=2)

    with _results_lock:
        if image_id not in _results:
            _results[image_id] = {}
        _results[image_id]["segmentation"] = {
            "result_path": str(seg_path),
            "status": seg_result.get("status", "completed"),
            "masks_shape": seg_result.get("masks_shape"),
            "max_confidence": seg_result.get("max_confidence"),
            "error": seg_result.get("error"),
        }
        _persist_results()

    return str(seg_path)


def get_segmentation_result(image_id: str) -> Optional[dict]:
    with _results_lock:
        if not _results:
            _load_results()
        return _results.get(image_id, {}).get("segmentation")


def get_segmentation_result_path(image_id: str) -> Optional[str]:
    for f in SEGMENTS_DIR.glob(f"{image_id}_segment.json"):
        return str(f)
    return None
