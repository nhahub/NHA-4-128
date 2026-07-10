import mlflow
import mlflow.tensorflow
import tensorflow as tf
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "mlflow" / "models" / "skin_cancer_detection_final.keras"
HOLDOUT_DIR = BASE_DIR / "holdout_data"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("skin_cancer_experiment_v2")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

input_shape = model.input_shape if hasattr(model, "input_shape") else "unknown"
params = {
    "model_type": "skin_cancer_classifier",
    "input_shape": str(input_shape),
    "framework": "tensorflow",
    "output": "sigmoid (binary)",
}

metrics = {}

# Quick sanity check — run a dummy forward pass
dummy = np.random.randn(1, 224, 224, 3).astype(np.float32)
pred = model(dummy, training=False)
metrics["sanity_check"] = 1.0
metrics["dummy_prediction"] = round(float(pred.numpy().flatten()[0]), 6)
metrics["model_loaded"] = 1.0

# Evaluate on holdout data if available
if HOLDOUT_DIR.exists():
    benign_dir = HOLDOUT_DIR / "benign"
    malicious_dir = HOLDOUT_DIR / "malicious"

    if benign_dir.exists() and malicious_dir.exists():
        from PIL import Image

        def load_images_from_dir(directory, label):
            images = []
            for path in directory.iterdir():
                if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    try:
                        img = Image.open(path).resize((224, 224))
                        img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)
                        images.append((img_array, label))
                    except Exception:
                        continue
            return images

        data = []
        data.extend(load_images_from_dir(benign_dir, 0))
        data.extend(load_images_from_dir(malicious_dir, 1))

        if data:
            y_true = []
            y_pred = []
            for img_array, label in data:
                pred = model(img_array, training=False)
                pred_val = float(pred.numpy().flatten()[0])
                y_true.append(label)
                y_pred.append(pred_val)

            y_true_arr = np.array(y_true)
            y_pred_bin = (np.array(y_pred) >= 0.5).astype(int)

            accuracy = np.mean(y_true_arr == y_pred_bin)
            tp = np.sum((y_true_arr == 1) & (y_pred_bin == 1))
            fp = np.sum((y_true_arr == 0) & (y_pred_bin == 1))
            fn = np.sum((y_true_arr == 1) & (y_pred_bin == 0))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics["holdout_accuracy"] = round(accuracy, 4)
            metrics["holdout_precision"] = round(precision, 4)
            metrics["holdout_recall"] = round(recall, 4)
            metrics["holdout_f1"] = round(f1, 4)
            metrics["holdout_samples"] = len(data)
        else:
            metrics["holdout_found"] = 0.0
    else:
        metrics["holdout_found"] = 0.0
else:
    metrics["holdout_found"] = 0.0

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    mlflow.tensorflow.log_model(
        model=model,
        name="model",
        registered_model_name="skin_cancer_classifier",
    )

    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/model",
        "skin_cancer_classifier"
    )

    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Logged params: {params}")
    print(f"Logged metrics: {metrics}")
