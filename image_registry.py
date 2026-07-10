import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Excel file path
EXCEL_PATH = Path(__file__).parent / "data" / "five_sample_patients_with_features.xlsx"

# Sheet and column names
SHEET_NAME = "Patients"
ID_COLUMN = "patient_id"
FEATURE_COLUMN = "global_features"

# Similarity threshold
MATCH_THRESHOLD = 0.85

# Load model once
_model = None


def get_feature_model():
    # Load MobileNetV2 model

    global _model

    if _model is None:
        from tensorflow.keras.applications import MobileNetV2

        print("Loading MobileNetV2 feature extractor...")

        _model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg"
        )

    return _model


def compute_global_features(image: Image.Image):
    # Extract image features

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    model = get_feature_model()

    img = image.convert("RGB").resize((224, 224))

    arr = np.array(img).astype("float32")

    arr = preprocess_input(arr)

    arr = np.expand_dims(arr, axis=0)

    vector = model.predict(arr, verbose=0)[0]

    return vector.astype(np.float32)


def load_patients_dataframe():
    # Read patient data from Excel

    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Excel file not found: {EXCEL_PATH}"
        )

    return pd.read_excel(
        EXCEL_PATH,
        sheet_name=SHEET_NAME
    )


def find_matching_patient(
    image: Image.Image
) -> Tuple[Optional[str], Optional[float]]:
    # Find matching patient

    df = load_patients_dataframe()

    query_feature = compute_global_features(image)

    best_patient_id = None
    best_similarity = -1.0

    for _, row in df.iterrows():

        feature_string = row.get(FEATURE_COLUMN)

        if pd.isna(feature_string):
            continue

        try:
            stored_feature = np.array(
                json.loads(feature_string),
                dtype=np.float32
            )

        except Exception:
            continue

        similarity = cosine_similarity(
            [query_feature],
            [stored_feature]
        )[0][0]

        if similarity > best_similarity:
            best_similarity = similarity
            best_patient_id = str(
                row.get(ID_COLUMN, "")
            ).strip()

    print("=" * 50)
    print("Best Patient:", best_patient_id)
    print("Best Similarity:", best_similarity)
    print("=" * 50)

    if (
        best_patient_id
        and best_similarity >= MATCH_THRESHOLD
    ):
        return best_patient_id, float(best_similarity)

    return None, float(best_similarity)


def register_image(
    patient_id: str,
    image: Image.Image
) -> None:
    # Kept for compatibility

    pass


def is_registered(patient_id: str) -> bool:
    # Check if patient exists

    df = load_patients_dataframe()

    patient_id = patient_id.strip().upper()

    ids = (
        df[ID_COLUMN]
        .astype(str)
        .str.upper()
        .tolist()
    )

    return patient_id in ids