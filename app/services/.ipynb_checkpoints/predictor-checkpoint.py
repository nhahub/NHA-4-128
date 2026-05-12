import numpy as np
from typing import Tuple
from app.configs import get_classification_model, model_classes
from app.storage import save_image
from app.core.preprocessing import preprocess_image
from app.core.validation import is_valid_image

def predict_image(file_bytes: bytes, filename: str) -> Tuple[str, float, str]:
     # checking if the image is good or no 
    if not is_valid_image(file_bytes):
        raise ValueError("Invalid image")

    # model loading or ( vonnecting with mlflow )
    model = get_classification_model()

    if model is None:
        raise ValueError("Classification model not loaded")

    image_filename = f"{hash(filename)}.jpg"
    image_path = save_image(file_bytes, image_filename)

    img_array = preprocess_image(file_bytes)

    # using the model 
    predictions = model.predict(img_array)

    confidence = float(np.max(predictions))
    predicted_class = int(np.argmax(predictions))

    prediction = model_classes.get(predicted_class, "unknown")

    print(f"CLASSIFICATION: {prediction} ({confidence:.3f})")

    return prediction, confidence, image_path