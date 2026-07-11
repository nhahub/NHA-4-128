from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    image_id: str
    filename: str
    size_bytes: int
    content_type: str
    uploaded_at: str
    url: str


class ClassifyRequest(BaseModel):
    image_id: str


class ClassifyResponse(BaseModel):
    image_id: str
    prediction: str
    confidence: float
    model_version: str
    status: str = "completed"


class SegmentRequest(BaseModel):
    image_id: str


class SegmentResponse(BaseModel):
    image_id: str
    status: str
    masks_shape: Optional[list] = None
    max_confidence: Optional[float] = None
    result_url: Optional[str] = None
    error: Optional[str] = None
