import time
import numpy as np
import tensorflow as tf
from typing import Dict, Any
from app.configs import get_segmentation_model
from app.core.preprocessing import preprocess_image
from app.monitoring.metrics import inference_total, inference_latency_seconds


def segment_image(file_bytes: bytes) -> Dict[str, Any]:
    start_time = time.time()

    result: Dict[str, Any] = {"detections": []}

    model = get_segmentation_model()

    if model is None:
        result["error"] = "Segmentation model not loaded"
        result["status"] = "failed"
        return result

    try:
        img_array = preprocess_image(file_bytes, target_size=(128, 128))

        pred_result = model(tf.constant(img_array))
        if isinstance(pred_result, dict):
            masks = list(pred_result.values())[0].numpy()
        else:
            masks = pred_result.numpy()

        max_conf = float(np.max(masks))

        result.update({
            "status": "completed",
            "masks_shape": list(masks.shape),
            "max_confidence": max_conf,
        })

        latency = time.time() - start_time

        print(f"SEGMENTATION: {result['masks_shape']} max_conf={max_conf:.3f} in {latency:.2f}s")

        inference_total.labels(model_type="segmenter", prediction="completed").inc()
        inference_latency_seconds.labels(model_type="segmenter").observe(latency)

    except Exception as e:
        result["error"] = str(e)
        result["status"] = "failed"

    return result
