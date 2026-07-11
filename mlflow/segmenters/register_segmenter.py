import mlflow
import mlflow.tensorflow
import tensorflow as tf
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "mlflow" / "models" / "UNet_model.keras"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("segmenter_experiment")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

input_shape = model.input_shape if hasattr(model, "input_shape") else "unknown"
output_shape = model.output_shape if hasattr(model, "output_shape") else "unknown"

# Infer expected spatial dims from model input
expected_h = input_shape[1] if input_shape != "unknown" and len(input_shape) > 1 else 128
expected_w = input_shape[2] if input_shape != "unknown" and len(input_shape) > 2 else 128
expected_c = input_shape[3] if input_shape != "unknown" and len(input_shape) > 3 else 3

params = {
    "model_type": "skin_cancer_segmenter",
    "input_shape": str(input_shape),
    "output_shape": str(output_shape),
    "framework": "tensorflow",
    "architecture": "unet",
}

metrics = {}

# Dummy forward pass with correct input shape
dummy = np.random.randn(1, expected_h, expected_w, expected_c).astype(np.float32)
result = model(dummy, training=False)
metrics["sanity_check"] = 1.0
metrics["output_min"] = round(float(np.min(result.numpy())), 6)
metrics["output_max"] = round(float(np.max(result.numpy())), 6)
metrics["output_mean"] = round(float(np.mean(result.numpy())), 6)
metrics["output_std"] = round(float(np.std(result.numpy())), 6)
metrics["model_loaded"] = 1.0

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(str(BASE_DIR / "data.xlsx"), artifact_path="dataset")

    mlflow.tensorflow.log_model(
        model=model,
        name="model",
        registered_model_name="skin_cancer_segmenter",
    )

    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/model",
        "skin_cancer_segmenter"
    )

    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Logged params: {params}")
    print(f"Logged metrics: {metrics}")
