import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional
from PIL import Image
import io

logger = logging.getLogger(__name__)

STORAGE_ROOT = Path(__file__).parent / "storage" / "patient_images"


def _ensure_thread_dir(thread_id: str) -> Path:
    thread_dir = STORAGE_ROOT / thread_id
    thread_dir.mkdir(parents=True, exist_ok=True)
    return thread_dir


def _metadata_path(thread_id: str) -> Path:
    return _ensure_thread_dir(thread_id) / "_metadata.json"


def _load_metadata(thread_id: str) -> list:
    mpath = _metadata_path(thread_id)
    if mpath.exists():
        try:
            return json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_metadata(thread_id: str, entries: list) -> None:
    mpath = _metadata_path(thread_id)
    mpath.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def save_uploaded_image(thread_id: str, file_bytes: bytes, original_filename: str) -> Optional[Path]:
    """Save uploaded image to persistent storage. Returns saved file path or None."""
    try:
        ext = Path(original_filename).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            ext = ".jpg"
        timestamp = int(time.time() * 1000)
        unique_name = f"{timestamp}_{hashlib.md5(file_bytes).hexdigest()[:8]}{ext}"
        thread_dir = _ensure_thread_dir(thread_id)
        dest = thread_dir / unique_name
        dest.write_bytes(file_bytes)
        pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        metadata = _load_metadata(thread_id)
        metadata.append({
            "filename": unique_name,
            "original_name": original_filename,
            "path": str(dest),
            "timestamp": timestamp,
            "analysis": None,
            "width": pil_image.width,
            "height": pil_image.height,
        })
        _save_metadata(thread_id, metadata)
        logger.info(f"Saved uploaded image: {dest}")
        return dest
    except Exception as e:
        logger.error(f"Failed to save uploaded image: {e}")
        return None


def get_latest_image(thread_id: str) -> Optional[dict]:
    """Return metadata dict for the most recent image uploaded by this thread."""
    entries = _load_metadata(thread_id)
    if not entries:
        return None
    return entries[-1]


def get_all_images(thread_id: str) -> list:
    """Return all image metadata for this thread."""
    return _load_metadata(thread_id)


def update_analysis(thread_id: str, image_filename: str, analysis: dict) -> None:
    """Associate analysis result with a stored image."""
    entries = _load_metadata(thread_id)
    for entry in entries:
        if entry["filename"] == image_filename:
            entry["analysis"] = analysis
            _save_metadata(thread_id, entries)
            return
    logger.warning(f"Image {image_filename} not found in metadata for {thread_id}")
