from pathlib import Path
import json
from typing import List, Dict, Any
from app.configs import IMAGES_DIR, SEGMENTS_DIR, request_history

def save_image(image_bytes: bytes, filename: str) -> str:
    """Save image to storage"""
    image_path = IMAGES_DIR / filename
    with open(image_path, "wb") as f:
        f.write(image_bytes)
    return str(image_path)

def save_segmentation_result(seg_result: Dict[str, Any], request_id: str, image_filename: str) -> str:
    """Save segmentation JSON result"""
    seg_filename = f"{request_id}_{image_filename}_segment.json"
    seg_path = SEGMENTS_DIR / seg_filename
    with open(seg_path, "w") as f:
        json.dump(seg_result, f, indent=2)
    return str(seg_path)

def update_history(request_id: str, **updates: Any) -> None:
    """Update history item by request_id"""
    for item in request_history:
        if item["request_id"] == request_id:
            item.update(updates)
            break