import os
import uuid
import asyncio
import random
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
import io
import tensorflow as tf
from tensorflow import keras

# ========================================
# CONFIGURATION
# ========================================
app = FastAPI(title="AI Image Classification + Segmentation (Keras)", version="2.0.0")

# Storage directories
STORAGE_DIR = Path("storage")
CLASSIFICATION_MODELS_DIR = STORAGE_DIR / "models" / "classification"
SEGMENTATION_MODELS_DIR = STORAGE_DIR / "models" / "segmentation"
IMAGES_DIR = STORAGE_DIR / "images"
SEGMENTS_DIR = STORAGE_DIR / "segments"

# Create directories
for dir_path in [STORAGE_DIR, CLASSIFICATION_MODELS_DIR, SEGMENTATION_MODELS_DIR, IMAGES_DIR, SEGMENTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Global Keras models
classification_model = None
segmentation_model = None
model_classes = {0: "benign", 1: "malicious"}

print("🤖 TensorFlow/Keras ready. Upload .keras models!")

# In-memory history
request_history: List[Dict[str, Any]] = []

# ========================================
# Pydantic Models
# ========================================
class PredictionResponse(BaseModel):
    request_id: str
    filename: str
    image_path: str
    prediction: str
    confidence: float
    model_version: str
    status: str = "completed"

class HistoryItem(PredictionResponse):
    timestamp: str
    segmentation_path: Optional[str] = None

class ModelLoadResponse(BaseModel):
    status: str
    type: str
    path: str
    message: str

# ========================================
# IMAGE PREPROCESSING
# ========================================
def preprocess_image(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    """Preprocess for Keras models"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# ========================================
# MODEL LOADING
# ========================================
def load_keras_model(model_path: str, model_type: str) -> bool:
    """Load .keras model"""
    global classification_model, segmentation_model
    
    try:
        model = keras.models.load_model(model_path)
        if model_type == "classification":
            classification_model = model
        elif model_type == "segmentation":
            segmentation_model = model
        
        print(f"✅ {model_type.upper()} model loaded: {model_path}")
        print(f"   Input: {model.input_shape}, Output: {model.output_shape}")
        return True
    except Exception as e:
        print(f"❌ {model_type} model failed: {e}")
        return False

# ========================================
# MODEL STORAGE
# ========================================
def save_classification_model(model_bytes: bytes, filename: str) -> str:
    model_path = CLASSIFICATION_MODELS_DIR / filename
    with open(model_path, "wb") as f:
        f.write(model_bytes)
    return str(model_path)

def save_segmentation_model(model_bytes: bytes, filename: str) -> str:
    model_path = SEGMENTATION_MODELS_DIR / filename
    with open(model_path, "wb") as f:
        f.write(model_bytes)
    return str(model_path)

# ========================================
# CORE PREDICTION FUNCTIONS
# ========================================
def predict_with_keras(image_bytes: bytes) -> Tuple[str, float]:
    """Classification - BLOCKS if no model"""
    global classification_model
    
    if classification_model is None:
        raise ValueError(
            "No CLASSIFICATION model! Upload .keras: POST /models/classification/load"
        )
    
    img_array = preprocess_image(image_bytes)
    predictions = classification_model.predict(img_array, verbose=0)
    confidence = float(np.max(predictions))
    predicted_class = np.argmax(predictions)
    prediction = model_classes.get(predicted_class, "unknown")
    
    print(f"🔮 CLASSIFICATION: {prediction} ({confidence:.3f})")
    return prediction, confidence

async def run_segmentation(request_id: str, image_filename: str, image_path: str):
    """Segmentation using Keras model or dummy"""
    global segmentation_model
    
    print(f"🔍 SEGMENTING {request_id}...")
    await asyncio.sleep(random.uniform(2, 5))
    
    seg_filename = f"{request_id}_{image_filename}_segment.json"
    seg_path = SEGMENTS_DIR / seg_filename
    
    seg_result = {"request_id": request_id, "image_filename": image_filename}
    
    if segmentation_model:
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            img_array = preprocess_image(img_bytes, target_size=(256, 256))
            masks = segmentation_model.predict(img_array, verbose=0)
            seg_result.update({
                "keras_model_used": True,
                "masks_shape": str(masks.shape),
                "max_confidence": float(np.max(masks))
            })
        except Exception as e:
            seg_result["error"] = str(e)
    else:
        seg_result["dummy_mode"] = True
    
    seg_result.update({
        "timestamp": datetime.now().isoformat(),
        "detections": [{"type": "malware", "confidence": 0.92}]
    })
    
    with open(seg_path, "w") as f:
        json.dump(seg_result, f, indent=2)
    
    # Update history
    for item in request_history:
        if item["request_id"] == request_id:
            item["segmentation_path"] = str(seg_path)
            item["status"] = "segmented"
            break

# ========================================
# HELPER FUNCTIONS
# ========================================
def is_valid_image(file_bytes: bytes) -> bool:
    try:
        Image.open(io.BytesIO(file_bytes)).verify()
        return True
    except:
        return False

def save_image(file_bytes: bytes, filename: str) -> str:
    image_path = IMAGES_DIR / filename
    with open(image_path, "wb") as f:
        f.write(file_bytes)
    return str(image_path)

# ========================================
# API ROUTES
# ========================================
@app.post("/models/classification/load", response_model=ModelLoadResponse)
async def load_classification_model(model_file: UploadFile = File(...)):
    """📤 UPLOAD CLASSIFICATION MODEL (.keras ONLY)"""
    if not model_file.filename.endswith('.keras'):
        raise HTTPException(400, "❌ Classification model must be .keras")
    
    model_bytes = await model_file.read()
    model_filename = f"class_{uuid.uuid4()}.keras"
    model_path = save_classification_model(model_bytes, model_filename)
    
    if load_keras_model(model_path, "classification"):
        return ModelLoadResponse(
            status="success", type="classification", 
            path=model_path,
            message="✅ Classification model ready! Use /predict"
        )
    raise HTTPException(500, "Failed to load classification model")

@app.post("/models/segmentation/load", response_model=ModelLoadResponse)
async def load_segmentation_model(model_file: UploadFile = File(...)):
    """🔍 UPLOAD SEGMENTATION MODEL (.keras ONLY)"""
    if not model_file.filename.endswith('.keras'):
        raise HTTPException(400, "❌ Segmentation model must be .keras")
    
    model_bytes = await model_file.read()
    model_filename = f"seg_{uuid.uuid4()}.keras"
    model_path = save_segmentation_model(model_bytes, model_filename)
    
    if load_keras_model(model_path, "segmentation"):
        return ModelLoadResponse(
            status="success", type="segmentation",
            path=model_path,
            message="✅ Segmentation model ready for malicious images"
        )
    raise HTTPException(500, "Failed to load segmentation model")

@app.get("/models/status")
async def models_status():
    """📊 MODEL STATUS"""
    return {
        "classification_loaded": classification_model is not None,
        "segmentation_loaded": segmentation_model is not None,
        "paths": {
            "classification": str(CLASSIFICATION_MODELS_DIR),
            "segmentation": str(SEGMENTATION_MODELS_DIR),
            "images": str(IMAGES_DIR),
            "segments": str(SEGMENTS_DIR)
        },
        "predict_ready": classification_model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = "Untitled"
):
    """🎯 CLASSIFY IMAGE (Requires classification model)"""
    
    # Validation
    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large (10MB max)")
    if not is_valid_image(file_bytes):
        raise HTTPException(400, "Invalid image")
    
    # Process
    request_id = str(uuid.uuid4())
    image_filename = f"{uuid.uuid4()}.jpg"
    image_path = save_image(file_bytes, image_filename)
    
    # Keras prediction
    prediction, confidence = predict_with_keras(file_bytes)
    model_version = getattr(classification_model, 'name', 'unknown')
    
    # Save to history
    history_item = {
        "request_id": request_id, "filename": file.filename or image_filename,
        "image_path": image_path, "prediction": prediction,
        "confidence": confidence, "model_version": model_version,
        "timestamp": datetime.now().isoformat(), "status": "classified",
        "segmentation_path": None
    }
    request_history.append(history_item)
    
    # Async segmentation for malicious
    if prediction == "malicious":
        background_tasks.add_task(run_segmentation, request_id, image_filename, image_path)
    
    return PredictionResponse(**history_item)

@app.get("/history", response_model=List[HistoryItem])
async def get_history():
    return [HistoryItem(**item) for item in request_history]

@app.get("/history/{request_id}")
async def get_prediction(request_id: str):
    for item in request_history:
        if item["request_id"] == request_id:
            return HistoryItem(**item)
    raise HTTPException(404, "Not found")

@app.get("/")
async def root():
    class_ready = "✅ READY" if classification_model else "❌ REQUIRED"
    seg_ready = "✅ READY" if segmentation_model else "⚠️ OPTIONAL"
    return {
        "🚀 Dual Keras Model API",
        f"📤 Classification: {class_ready}",
        f"🔍 Segmentation: {seg_ready}",
        "📁 Upload .keras models to:",
        f"   → POST /models/classification/load",
        f"   → POST /models/segmentation/load", 
        "📊 Total predictions:", len(request_history)
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "predict_ready": classification_model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)