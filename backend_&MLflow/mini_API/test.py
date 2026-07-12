import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model = mlflow.pyfunc.load_model("models:/skin_cancer_classifier/1")

print("Model loaded successfully")