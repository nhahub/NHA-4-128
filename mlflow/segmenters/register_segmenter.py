import mlflow
import mlflow.tensorflow
import tensorflow as tf
from pathlib import Path

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("skin_cancer_experiment_v2")

model_path = Path(r"C:\Users\Mega Store\backend\mlflow\models\UNet_model.keras")

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

model = tf.keras.models.load_model(model_path,compile=False)

with mlflow.start_run():
    mlflow.tensorflow.log_model(
        model=model,
        name="model"
    )

    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/model",
        "skin_cancer_segmenter"
    )