import uuid
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from typing import List

from . import configs as config
from .models import PredictionResponse, HistoryItem
from .services.predictor import predict_image
from .services.segmenter import run_segmentation

app = FastAPI(
    title="AI Image Classification API",
    version="2.0.0"
)

# =========================
# Models Status
# =========================
@app.get("/models/status")
async def models_status():
    classification_model = config.get_classification_model()
    segmentation_model = config.get_segmentation_model()

    return {
        "classification_loaded": classification_model is not None,
        "segmentation_loaded": segmentation_model is not None,
        "storage_paths": {
            "images": str(config.IMAGES_DIR),
            "segments": str(config.SEGMENTS_DIR)
        }
    }

# =========================
# Prediction Endpoint
# =========================
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    file_bytes = await file.read()

    #  validation
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (10MB max)")

    try:
        prediction, confidence, image_path = predict_image(
            file_bytes,
            file.filename or "image.jpg"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # get model 
    model = config.get_classification_model()
    model_version = getattr(model, "name", "mlflow-production")

    #create history
    request_id = str(uuid.uuid4())

    history_item = {
        "request_id": request_id,
        "filename": file.filename or "image.jpg",
        "image_path": image_path,
        "prediction": prediction,
        "confidence": confidence,
        "model_version": model_version,
        "timestamp": datetime.now().isoformat(),
        "status": "classified",
        "segmentation_path": None
    }

    config.request_history.append(history_item)

    # run segmentation in background if needed
    if prediction == "malicious":
        background_tasks.add_task(
            run_segmentation,
            request_id,
            Path(image_path).name,
            image_path
        )

    return PredictionResponse(**history_item)

# =========================
# History
# =========================
@app.get("/history", response_model=List[HistoryItem])
async def get_history():
    return [HistoryItem(**item) for item in config.request_history]

@app.get("/history/{request_id}", response_model=HistoryItem)
async def get_prediction(request_id: str):
    for item in config.request_history:
        if item["request_id"] == request_id:
            return HistoryItem(**item)

    raise HTTPException(status_code=404, detail="Prediction not found")

# =========================
# Root
# =========================
@app.get("/")
async def root():
    class_ready = "READY" if config.get_classification_model() else "REQUIRED !!!!"
    seg_ready = " READY" if config.get_segmentation_model() else "PTIONAL !"

    return {
        "service": "AI Image Classification API",
        "classification": class_ready,
        "segmentation": seg_ready,
        "endpoint": "POST /predict",
        "history": f"GET /history ({len(config.request_history)} records)",
        "docs": "/docs"
    }

# =========================
# Health Check
# =========================
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "predict_ready": config.get_classification_model() is not None,
        "total_predictions": len(config.request_history)
    }

# =========================
# Run Server
# =========================
if __name__ == "main":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )