# AI Image Classification & Segmentation API

A FastAPI-based REST API for skin cancer image analysis, featuring classification and segmentation models.

## Features

- **Image Classification** — Classify skin lesion images as benign or malignant (binary sigmoid)
- **Image Segmentation** — Run segmentation on skin lesion images
- **Decoupled Endpoints** — Upload, classify, and segment are independent operations
- **Lazy Model Loading** — Models loaded on first request, cached in memory
- **A/B Testing** — Route traffic between classifier model versions
- **Prometheus Monitoring** — HTTP and inference metrics exposed at `/metrics`
- **Health Monitoring** — Model status and service health endpoints

## Folder Structure

```
localback/
├── app/                          # Application package
│   ├── __init__.py               # Package exports
│   ├── main.py                   # FastAPI app, routes, server entry point
│   ├── configs.py                # Model loading, storage paths
│   ├── models.py                 # Pydantic request/response schemas
│   ├── storage.py                # File I/O for images and results
│   ├── core/
│   │   ├── __init__.py           # Core subpackage
│   │   ├── preprocessing.py      # Image resize, normalization for model input
│   │   └── validation.py         # Image integrity validation (PIL.verify)
│   ├── monitoring/
│   │   ├── __init__.py           # Monitoring subpackage
│   │   └── metrics.py            # Prometheus metrics (HTTP & inference)
│   └── services/
│       ├── __init__.py           # Services subpackage
│       ├── predictor.py          # Classification inference
│       ├── segmenter.py          # Segmentation inference
│       └── routing.py            # A/B testing model router
├── mlflow/                       # MLflow tracking data & scripts
├── storage/                      # Runtime file storage
│   ├── images/                   # Uploaded images
│   ├── segments/                 # Segmentation JSON results
│   └── results/                  # API result index (results.json)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## API Overview

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service root — model status & info |
| GET | `/health` | Health check — service & model availability |
| GET | `/metrics` | Prometheus metrics endpoint |
| GET | `/models/status` | Detailed model load status |
| GET | `/api/ab-status` | A/B testing configuration status |
| POST | `/api/upload` | Upload an image, returns `image_id` |
| POST | `/api/classify` | Classify by `image_id` |
| GET | `/api/classify/{image_id}` | Get stored classification result |
| POST | `/api/segment` | Segment by `image_id` |
| GET | `/api/segment/{image_id}` | Get stored segmentation result |
| GET | `/api/images/{image_id}` | Retrieve uploaded image |

## Installation

### Prerequisites

- Python 3.11+
- TensorFlow-compatible hardware (CPU is sufficient for development)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd localback

# Create and activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Start the API Server

```bash
# From the project root
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API is now available at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CLASSIFIER_THRESHOLD` | `0.5` | Sigmoid decision threshold (>= malicious) |
| `AB_TESTING_ENABLED` | `false` | Enable A/B testing between model versions |
| `AB_CLASSIFIER_B` | — | Alternative model URI for A/B version B |
| `STORAGE_DIR` | `./storage` | File storage root directory |

## API Endpoint Documentation

### POST `/api/upload`

Upload an image file. Returns an `image_id` used for subsequent classification and segmentation requests.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | UploadFile | Yes | Image file (JPEG, PNG) — max 10 MB |

**Response 201: `UploadResponse`**

```json
{
  "image_id": "a1b2c3d4-e5f6-...",
  "filename": "lesion.jpg",
  "size_bytes": 123456,
  "content_type": "image/jpeg",
  "uploaded_at": "2026-07-09T03:50:00",
  "url": "/api/images/a1b2c3d4-e5f6-..."
}
```

**Errors:**

| Status | Condition |
|---|---|
| 400 | File too large (>10 MB) |
| 400 | Unsupported content type |

---

### GET `/api/images/{image_id}`

Retrieve an uploaded image by its ID.

| Parameter | Type | Description |
|---|---|---|
| `image_id` | Path (string) | UUID from upload response |

**Response 200:** Raw image bytes with correct `Content-Type` header

**Response 404:** Image not found

---

### POST `/api/classify`

Run classification on a previously uploaded image.

**Request body: `ClassifyRequest`**

```json
{
  "image_id": "a1b2c3d4-e5f6-..."
}
```

**Response 200: `ClassifyResponse`**

```json
{
  "image_id": "a1b2c3d4-e5f6-...",
  "prediction": "benign",
  "confidence": 0.987,
  "model_version": "default",
  "status": "completed"
}
```

> **Prediction logic:** The model outputs a single sigmoid value (0-1). The class is determined by `CLASSIFIER_THRESHOLD` (default 0.5):
> `malicious` if confidence >= 0.5, else `benign`.

**Response 404:** Image not found

**Errors:**

| Status | Condition |
|---|---|
| 400 | Invalid image format |
| 500 | Model prediction failure |

Results are cached — re-classifying the same `image_id` returns the stored result.

---

### GET `/api/classify/{image_id}`

Retrieve a previously computed classification result.

| Parameter | Type | Description |
|---|---|---|
| `image_id` | Path (string) | UUID from upload response |

**Response 200:** `ClassifyResponse` (same schema as POST)

**Response 404:** Classification result not found

---

### POST `/api/segment`

Run segmentation on a previously uploaded image.

**Request body: `SegmentRequest`**

```json
{
  "image_id": "a1b2c3d4-e5f6-..."
}
```

**Response 200: `SegmentResponse`**

```json
{
  "image_id": "a1b2c3d4-e5f6-...",
  "status": "completed",
  "masks_shape": [1, 256, 256, 1],
  "max_confidence": 0.95,
  "result_url": "storage/segments/a1b2c3d4-e5f6-_segment.json",
  "error": null
}
```

Results are cached — re-segmenting the same `image_id` returns the stored result.

---

### GET `/api/segment/{image_id}`

Retrieve a previously computed segmentation result.

**Response 200:** `SegmentResponse` (same schema as POST)

**Response 404:** Segmentation result not found

---

### GET `/health`

**Response 200:**

```json
{
  "status": "healthy",
  "predict_ready": true
}
```

---

### GET `/models/status`

**Response 200:**

```json
{
  "classification_loaded": true,
  "segmentation_loaded": true,
  "storage_paths": {
    "images": "storage/images",
    "segments": "storage/segments"
  }
}
```

---

### GET `/`

**Response 200:**

```json
{
  "service": "AI Image Classification API",
  "version": "3.1.0",
  "classification": "READY",
  "segmentation": "READY",
    "endpoints": {
      "upload": "POST /api/upload",
      "classify": "POST /api/classify",
      "segment": "POST /api/segment",
      "health": "GET /health",
      "metrics": "GET /metrics",
      "ab_status": "GET /api/ab-status",
      "docs": "/docs"
    },
    "models_status": "/models/status"
}
```

## Storage Architecture

All data is stored on the local filesystem:

```
storage/
├── images/           # Uploaded images
│   └── <image_id>.jpg
├── segments/         # Segmentation results (JSON)
│   └── <image_id>_segment.json
└── results/          # API result index
    └── results.json
```

`results.json` persists classification and segmentation metadata across restarts using a JSON file with thread-safe locking.

## MLflow Tracking

The project uses **MLflow** for experiment tracking and model registry.

### Start the MLflow Tracking Server (from project root)

```bash
mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts --workers 1
```

Or run from inside `mlflow/`:

```bash
cd mlflow
mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --workers 1
```

Open `http://127.0.0.1:5000` in your browser.

### Register a Model

Run from the **project root**:

```bash
# Register classifier
python mlflow/classifiers/register_classifier.py

# Register segmenter
python mlflow/segmenters/register_segmenter.py
```

Each script:
1. Loads the `.keras` model from `mlflow/models/`
2. Logs model parameters (architecture, input shape, framework)
3. Evaluates against holdout data and logs metrics (accuracy, precision, recall, F1)
4. Logs the model via `mlflow.tensorflow.log_model()`
5. Registers the model in the Model Registry

### View Registered Models

After registering, models appear in the MLflow UI under the **Models** tab or query via CLI:

```bash
mlflow models list
```

### Tracking URI

The scripts connect to `http://127.0.0.1:5000` by default. To use a different tracking server:

```bash
export MLFLOW_TRACKING_URI=http://your-server:5000
```

## Prometheus Monitoring

The API exposes real-time metrics at `GET /metrics` for Prometheus scraping.

### Metrics Exported

| Metric | Type | Labels | Description |
|---|---|---|---|
| `http_requests_total` | Counter | `method`, `endpoint`, `status` | Total HTTP requests |
| `http_request_duration_seconds` | Histogram | `method`, `endpoint` | Request latency distribution |
| `inference_total` | Counter | `model_type`, `prediction` | Total inference calls |
| `inference_latency_seconds` | Histogram | `model_type` | Inference latency distribution |

### Scraping Configuration

Add this to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: "image_api"
    scrape_interval: 15s
    static_configs:
      - targets: ["localhost:8000"]
```

### Grafana Dashboard

A pre-built dashboard is available at `grafana/dashboard.json`. Import it into Grafana:

1. Open Grafana -> **+** -> **Import**
2. Upload `grafana/dashboard.json`
3. Select the Prometheus data source
4. Click **Import**

The dashboard includes panels for:
- HTTP request rate and latency (p50, p95, p99)
- Inference rate by model type
- Inference latency percentiles

## Model A/B Testing

The API supports A/B testing between multiple model versions.

### Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `AB_TESTING_ENABLED` | `false` | Enable A/B testing |
| `AB_CLASSIFIER_B` | — | Alternative model URI for version B |

### How It Works

1. **Explicit version selection** — Pass `model_version` in the classify request body to use a specific version
2. **Automatic routing** — When no version is specified and A/B testing is enabled, traffic is randomly split between the default classifier and `AB_CLASSIFIER_B`
3. **Result tracking** — The `model_version` field in the response shows which version handled each request

### Example

```bash
# A/B testing enabled — random split between default and version B
set AB_TESTING_ENABLED=true
set AB_CLASSIFIER_B=some_other_model

# Explicitly use a specific version
curl -X POST http://127.0.0.1:8000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"image_id": "IMAGE_ID", "model_version": "default"}'
```

### Check A/B Status

```bash
curl http://127.0.0.1:8000/api/ab-status
```

Response:
```json
{
  "enabled": true,
  "classifier_a": "default",
  "classifier_b": "some_other_model",
  "segmenter": "default"
}
```

## Testing

### Prerequisites for Testing

- API server running on `127.0.0.1:8000`

### Test with curl

```bash
# Health check
curl http://127.0.0.1:8000/health

# Model status
curl http://127.0.0.1:8000/models/status

# Upload an image
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@/path/to/test_image.jpg"

# Classify (replace IMAGE_ID with the upload response)
curl -X POST http://127.0.0.1:8000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"image_id": "IMAGE_ID"}'

# Segment
curl -X POST http://127.0.0.1:8000/api/segment \
  -H "Content-Type: application/json" \
  -d '{"image_id": "IMAGE_ID"}'
```

### Test with Python

```python
import requests

# Upload
with open("test_image.jpg", "rb") as f:
    resp = requests.post(
        "http://127.0.0.1:8000/api/upload",
        files={"file": ("test.jpg", f, "image/jpeg")}
    )
    data = resp.json()
    image_id = data["image_id"]
    print(f"Uploaded: {image_id}")

# Classify
resp = requests.post(
    "http://127.0.0.1:8000/api/classify",
    json={"image_id": image_id}
)
print(resp.json())

# Segment
resp = requests.post(
    "http://127.0.0.1:8000/api/segment",
    json={"image_id": image_id}
)
print(resp.json())
```

## Troubleshooting

### "ImportError: attempted relative import with no known parent package"

Run uvicorn from the project root, not from `app/`:
```bash
# Correct
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Wrong (will fail)
cd app && uvicorn main:app ...
```

### "Form data requires python-multipart"

```bash
pip install python-multipart
```

### "Storage paths are wrong"

Ensure you start the server from the project root directory so that `storage/` resolves correctly.

## License

MIT
