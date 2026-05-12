from pydantic import BaseModel
from typing import Optional
from datetime import datetime

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