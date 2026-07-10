import time
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
from app.configs import get_classification_model, sigmoid_to_class
from app.core.preprocessing import preprocess_image
from app.core.validation import is_valid_image
from app.monitoring.metrics import inference_total, inference_latency_seconds


def classify_image(
    file_bytes: bytes,
    model_version: Optional[str] = None,
) -> Tuple[str, float, str]:
    start_time = time.time()

    if not is_valid_image(file_bytes):
        raise ValueError("Invalid image")

    model = get_classification_model()

    if model is None:
        raise ValueError("Classification model not loaded")

    img_array = preprocess_image(file_bytes)

    result = model(tf.constant(img_array))
    if isinstance(result, dict):
        predictions = list(result.values())[0].numpy()
    else:
        predictions = result.numpy()

    confidence = float(predictions[0][0])
    prediction = sigmoid_to_class(confidence)

    latency = time.time() - start_time
    actual_version = model_version or "hf_savedmodel"

    print(f"CLASSIFICATION: {prediction} ({confidence:.3f}) in {latency:.2f}s [{actual_version}]")

    inference_total.labels(model_type="classifier", prediction=prediction).inc()
    inference_latency_seconds.labels(model_type="classifier").observe(latency)

    return prediction, confidence, actual_version
