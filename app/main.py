import uuid
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse

from . import configs as config
from .models import UploadResponse, ClassifyRequest, ClassifyResponse, SegmentRequest, SegmentResponse
from .services.predictor import classify_image
from .services.segmenter import segment_image
from .services.routing import router as model_router
from .storage import (
    save_image,
    get_image_path,
    get_image_bytes,
    save_classification_result,
    get_classification_result,
    save_segmentation_result,
    get_segmentation_result,
)
from .monitoring.metrics import MetricsMiddleware, metrics_endpoint

app = FastAPI(
    title="AI Image Classification API",
    version="3.1.0"
)

app.add_middleware(MetricsMiddleware)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_FILE_SIZE = 10 * 1024 * 1024


@app.get("/metrics")
async def prometheus_metrics(request: Request):
    return await metrics_endpoint(request)


@app.get("/models/status")
async def models_status():
    classification_model = config.get_classification_model()
    segmentation_model = config.get_segmentation_model()

    return {
        "classification_loaded": classification_model is not None,
        "segmentation_loaded": segmentation_model is not None,
        "ab_testing": model_router.get_ab_status(),
        "loaded_versions": config.get_loaded_versions(),
        "storage_paths": {
            "images": str(config.IMAGES_DIR),
            "segments": str(config.SEGMENTS_DIR),
        },
    }


@app.get("/api/ab-status")
async def ab_testing_status():
    return model_router.get_ab_status()


@app.post("/api/upload", status_code=201, response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}. Allowed: {ALLOWED_CONTENT_TYPES}",
        )

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (10MB max)")

    ext = Path(file.filename or "image.jpg").suffix or ".jpg"
    image_id = str(uuid.uuid4())

    save_image(image_id, file_bytes, ext)

    return UploadResponse(
        image_id=image_id,
        filename=file.filename or "image.jpg",
        size_bytes=len(file_bytes),
        content_type=file.content_type,
        uploaded_at=datetime.now().isoformat(),
        url=f"/api/images/{image_id}",
    )


@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    image_path = get_image_path(image_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Image not found")

    media_type = "image/jpeg"
    if image_path.suffix.lower() in {".png"}:
        media_type = "image/png"

    return FileResponse(path=image_path, media_type=media_type)


@app.post("/api/classify", response_model=ClassifyResponse)
async def classify(body: ClassifyRequest):
    image_bytes = get_image_bytes(body.image_id)
    if image_bytes is None:
        raise HTTPException(status_code=404, detail="Image not found")

    existing = get_classification_result(body.image_id)
    if existing and not body.model_version:
        return ClassifyResponse(
            image_id=body.image_id,
            prediction=existing["prediction"],
            confidence=existing["confidence"],
            model_version=existing["model_version"],
            status=existing.get("status", "completed"),
        )

    resolved_uri = model_router.resolve_classifier_version(body.model_version)

    try:
        prediction, confidence, actual_version = classify_image(
            image_bytes,
            model_version=resolved_uri,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result = {
        "prediction": prediction,
        "confidence": confidence,
        "model_version": actual_version,
        "status": "completed",
    }
    save_classification_result(body.image_id, result)

    return ClassifyResponse(
        image_id=body.image_id,
        **result,
    )


@app.get("/api/classify/{image_id}", response_model=ClassifyResponse)
async def get_classify_result(image_id: str):
    result = get_classification_result(image_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Classification result not found")

    return ClassifyResponse(
        image_id=image_id,
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_version=result["model_version"],
        status=result.get("status", "completed"),
    )


@app.post("/api/segment", response_model=SegmentResponse)
async def segment(body: SegmentRequest):
    image_bytes = get_image_bytes(body.image_id)
    if image_bytes is None:
        raise HTTPException(status_code=404, detail="Image not found")

    existing = get_segmentation_result(body.image_id)
    if existing:
        return SegmentResponse(
            image_id=body.image_id,
            status=existing.get("status", "completed"),
            masks_shape=existing.get("masks_shape"),
            max_confidence=existing.get("max_confidence"),
            result_url=existing.get("result_path"),
            error=existing.get("error"),
        )

    seg_result = segment_image(image_bytes)
    seg_path = save_segmentation_result(body.image_id, seg_result)

    return SegmentResponse(
        image_id=body.image_id,
        status=seg_result.get("status", "completed"),
        masks_shape=seg_result.get("masks_shape"),
        max_confidence=seg_result.get("max_confidence"),
        result_url=seg_path,
        error=seg_result.get("error"),
    )


@app.get("/api/segment/{image_id}", response_model=SegmentResponse)
async def get_segment_result(image_id: str):
    result = get_segmentation_result(image_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Segmentation result not found")

    return SegmentResponse(
        image_id=image_id,
        status=result.get("status", "completed"),
        masks_shape=result.get("masks_shape"),
        max_confidence=result.get("max_confidence"),
        result_url=result.get("result_path"),
        error=result.get("error"),
    )


@app.get("/")
async def root():
    class_ready = "READY" if config.get_classification_model() else "REQUIRED"
    seg_ready = "READY" if config.get_segmentation_model() else "OPTIONAL"

    return {
        "service": "AI Image Classification API",
        "version": "3.1.0",
        "classification": class_ready,
        "segmentation": seg_ready,
        "endpoints": {
            "upload": "POST /api/upload",
            "classify": "POST /api/classify",
            "segment": "POST /api/segment",
            "health": "GET /health",
            "metrics": "GET /metrics",
            "ab_status": "GET /api/ab-status",
            "docs": "/docs",
        },
        "models_status": "/models/status",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "predict_ready": config.get_classification_model() is not None,
    }
