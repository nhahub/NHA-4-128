"""
extract_global_features.py
===========================
One-time batch script. Run this once (not part of the live app).

Reads images from a LOCAL folder, where each image file is named after the
patient's patient_id (e.g. P0151.jpg, P0028.png ...). For every matching
patient row in the Excel "Patients" sheet, this computes a global feature
vector (MobileNetV2 + global average pooling, 1280 numbers) and writes it
into a "global_features" column as a JSON string.

Run:
    python extract_global_features.py --input data/updated_file_2.xlsx --images data/patient_images

If --output is omitted, a new file "<input>_with_features.xlsx" is created
instead of overwriting the input.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SHEET_NAME = "Patients"
FEATURE_COLUMN = "global_features"
ID_COLUMN = "patient_id"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

_model = None  # lazy singleton, shared with image_registry.py logic


def get_feature_model():
    """Load MobileNetV2 (ImageNet weights, no top layer, global average pooling).
    Produces a 1280-dim vector per image. Loaded once and reused."""
    global _model
    if _model is None:
        from tensorflow.keras.applications import MobileNetV2
        logger.info("Loading MobileNetV2 feature extractor (first call only)...")
        _model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    return _model


def compute_global_features(image: Image.Image) -> list:
    """Return a global feature vector (list of floats) for a PIL image."""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    model = get_feature_model()
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    vector = model.predict(arr, verbose=0)[0]
    return vector.tolist()


def build_image_index(images_dir: Path) -> dict:
    """Map patient_id (uppercased, no extension) -> file path, scanning the folder once."""
    index = {}
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            pid = path.stem.strip().upper()
            index[pid] = path
    return index


def main():
    parser = argparse.ArgumentParser(description="Extract global feature vectors from local patient images.")
    parser.add_argument("--input", required=True, help="Path to the source Excel file.")
    parser.add_argument("--images", required=True, help="Path to the folder containing images named <patient_id>.<ext>.")
    parser.add_argument("--output", default=None, help="Path to write the result (default: <input>_with_features.xlsx).")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                         help="Skip rows that already have a global_features value (default: on).")
    args = parser.parse_args()

    input_path = Path(args.input)
    images_dir = Path(args.images)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    if not images_dir.is_dir():
        logger.error(f"Images folder not found: {images_dir}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_with_features" + input_path.suffix
    )

    logger.info(f"Scanning images folder: {images_dir}")
    image_index = build_image_index(images_dir)
    logger.info(f"Found {len(image_index)} image files.")

    logger.info(f"Reading {input_path} ...")
    all_sheets = pd.read_excel(input_path, sheet_name=None)
    if SHEET_NAME not in all_sheets:
        logger.error(f"Sheet '{SHEET_NAME}' not found. Sheets available: {list(all_sheets.keys())}")
        sys.exit(1)

    df = all_sheets[SHEET_NAME]
    if FEATURE_COLUMN not in df.columns:
        df[FEATURE_COLUMN] = None

    total = len(df)
    done, skipped, missing_file, failed = 0, 0, 0, 0

    for idx, row in df.iterrows():
        pid_raw = row.get(ID_COLUMN, "")
        pid = str(pid_raw).strip().upper()

        existing = row.get(FEATURE_COLUMN)
        if args.skip_existing and isinstance(existing, str) and existing.strip():
            skipped += 1
            continue

        image_path = image_index.get(pid)
        if image_path is None:
            logger.warning(f"[{pid}] No local image file found, skipping.")
            missing_file += 1
            continue

        logger.info(f"[{pid}] ({idx + 1}/{total}) Reading {image_path.name}")
        try:
            image = Image.open(image_path)
            vector = compute_global_features(image)
            df.at[idx, FEATURE_COLUMN] = json.dumps(vector)
            done += 1
        except Exception as e:
            logger.error(f"[{pid}] Feature extraction failed: {e}")
            failed += 1

    all_sheets[SHEET_NAME] = df
    logger.info(f"Writing result to {output_path} ...")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, sheet_df in all_sheets.items():
            sheet_df.to_excel(writer, sheet_name=name, index=False)

    logger.info(
        f"Done. Extracted: {done}, skipped (already had features): {skipped}, "
        f"missing image file: {missing_file}, failed: {failed}"
    )


if __name__ == "__main__":
    main()
