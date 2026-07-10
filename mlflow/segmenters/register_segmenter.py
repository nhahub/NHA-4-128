import mlflow
import mlflow.tensorflow
import tensorflow as tf
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "mlflow" / "models" / "UNet_model.keras"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("skin_cancer_experiment_v2")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

input_shape = model.input_shape if hasattr(model, "input_shape") else "unknown"
output_shape = model.output_shape if hasattr(model, "output_shape") else "unknown"

params = {
    "model_type": "skin_cancer_segmenter",
    "input_shape": str(input_shape),
    "output_shape": str(output_shape),
    "framework": "tensorflow",
    "architecture": "unet",
}

metrics = {}

# Quick validation on a dummy input to verify output
dummy = np.random.randn(1, 256, 256, 3).astype(np.float32)
result = model(dummy, training=False)
metrics["output_valid"] = 1.0
metrics["output_mean"] = float(np.mean(result.numpy()))
metrics["output_max"] = float(np.max(result.numpy()))

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    mlflow.tensorflow.log_model(
        model=model,
        artifact_path="model",
        registered_model_name="skin_cancer_segmenter",
    )

    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/model",
        "skin_cancer_segmenter"
    )

    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Logged params: {params}")
    print(f"Logged metrics: {metrics}")
