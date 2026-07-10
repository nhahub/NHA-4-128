---
title: Skin Cancer Detection API
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
license: mit
short_description: FastAPI for skin cancer classification & segmentation
---

# AI Image Classification & Segmentation API

A FastAPI-based REST API for skin cancer image analysis, featuring classification and segmentation models served via MLflow.

## Features

- **Image Classification** — Classify skin lesion images as benign or malignant
- **Image Segmentation** — Run UNet-based segmentation on classified malignant images
- **MLflow Integration** — Models are versioned, registered, and served through MLflow
- **Decoupled Endpoints** — Upload, classify, and segment are independent operations
- **Health Monitoring** — Model status and service health endpoints
- **MLflow Tracing** — Inference calls are traced with span-level observability

## Project Architecture

```
Client ──HTTP──→ FastAPI ──huggingface_hub──→ HF Hub Model Repo
                     │                              │
                     │                      ┌───────┴───────┐
                     │                 Classifier Model  Segmenter Model
                     │                 (SavedModel .zip) (SavedModel .zip)
                     │
               Local Storage
            (storage/images/)
            (storage/segments/)
```

**Key Components:**

| Layer | Technology | Purpose |
|---|---|---|
| API Server | FastAPI + Uvicorn | HTTP request handling, routing |
| Model Source | Hugging Face Hub | Model download and caching |
| ML Runtime | TensorFlow / Keras | Model inference (classifier + UNet) |
| Storage | Local filesystem | Image and segmentation result persistence |
| Validation | Pillow | Image integrity checking |

## Folder Structure

```
localback/
├── app/                          # Application package
│   ├── __init__.py               # Package exports
│   ├── main.py                   # FastAPI app, routes, server entry point
│   ├── configs.py                # HF Hub model loading, storage paths
│   ├── models.py                 # Pydantic request/response schemas
│   ├── storage.py                # File I/O for images and results
│   ├── core/
│   │   ├── preprocessing.py      # Image resize, normalization for model input
│   │   └── validation.py         # Image integrity validation (PIL.verify)
│   ├── monitoring/
│   │   ├── metrics.py            # Prometheus metrics (HTTP, inference, GPU)
│   │   └── gpu.py                # GPU metrics via pynvml
│   └── services/
│       ├── predictor.py          # Classification inference
│       ├── segmenter.py          # Segmentation inference
│       └── routing.py            # A/B testing model router
├── storage/                      # Runtime file storage
│   ├── images/                   # Uploaded images
│   ├── segments/                 # Segmentation JSON results
│   └── results/                  # API result index
├── scripts/                      # Utility scripts
│   ├── dataset_metadata.py       # Dataset logging utility
│   ├── log_dataset.py            # Log Kaggle ISIC dataset from Excel
│   └── validation/
│       ├── holdout_loader.py     # Load holdout images for validation
│       ├── validate_classifier.py   # Accuracy/precision/recall/F1 checks
│       └── validate_segmenter.py    # Segmentation confidence checks
├── forproduction/                # Hugging Face Spaces deployment config
│   └── api/
│       └── README.md             # HF Spaces metadata (YAML frontmatter)
├── grafana/                      # Pre-built Grafana dashboard
│   └── dashboard.json
├── Dockerfile                    # HF Spaces container build
├── requirements.txt              # Python dependencies
├── .gitattributes                # Git LFS configuration for HF
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
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
- Hugging Face token with access to the model repository

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

# Set your Hugging Face token (required for model download)
export HF_TOKEN="your_hf_token_here"
```

### Start the API Server

```bash
# From the project root
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API is now available at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

Models are downloaded automatically from Hugging Face Hub on first request and cached locally under `/tmp/savedmodels/`.

## Environment Variables

| Variable | Default | Purpose | Location |
|---|---|---|---|---|
| `HF_TOKEN` | — | **Required.** Hugging Face token for model access | `app/configs.py:9` |
| `HF_MODEL_REPO` | `omarelrayes/mlflow-artifacts` | HF Hub repo containing model zips | `app/configs.py:10` |
| `CLASSIFIER_MODEL_PATH` | `models/classifier_savedmodel.zip` | Path to classifier zip in HF repo | `app/configs.py:11` |
| `SEGMENTER_MODEL_PATH` | `models/segmenter_savedmodel.zip` | Path to segmenter zip in HF repo | `app/configs.py:12` |
| `STORAGE_DIR` | `./storage` | File storage root directory | `app/configs.py:86` |
| `CLASSIFIER_THRESHOLD` | `0.5` | Sigmoid decision threshold (≥ → malicious) | `app/configs.py:24` |
| `AB_TESTING_ENABLED` | `false` | Enable A/B testing between model versions | `app/services/routing.py:14` |
| `AB_TESTING_PERCENT_B` | `50` | % of traffic to route to version B | `app/services/routing.py:15` |
| `AB_TESTING_MODEL_VERSION_B` | `""` | Model URI for version B | `app/services/routing.py:16` |
| `HOLDOUT_DATA_DIR` | `holdout_data` | Validation holdout dataset path | `scripts/validation/validate_classifier.py:18` |
| `CLASSIFIER_THRESHOLD` | `0.5` | Sigmoid decision threshold | `app/configs.py:24` |

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
  "model_version": "models:/skin_cancer_classifier/5",
  "status": "completed"
}
```

> **Prediction logic:** The model outputs a single sigmoid value (0–1). The class is determined by `CLASSIFIER_THRESHOLD` (default 0.5):
> `malicious` if confidence ≥ 0.5, else `benign`.

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
  "result_url": "storage\\segments\\a1b2c3d4-e5f6-_segment.json",
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
    "images": "storage\\images",
    "segments": "storage\\segments"
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

**Production recommendations:**
- Move to S3/GCS/MinIO for durable, scalable storage
- Implement a retention/TTL policy
- Add database persistence for result metadata

## Model Loading Process

Models are loaded lazily on first request via MLflow:

```
Request → get_classification_model() / get_segmentation_model()
          │
          ├── Is model cached in memory? → Return cached model
          │
          └── No → mlflow.pyfunc.load_model("models:/<name>/<version>")
                   → Cache in global variable
                   → Return model
```

**Model sources:**

| Model | HF Hub Path |
|---|---|
| Classifier | `omarelrayes/mlflow-artifacts/models/classifier_savedmodel.zip` |
| Segmenter | `omarelrayes/mlflow-artifacts/models/segmenter_savedmodel.zip` |

## Model Loading from Hugging Face Hub

### Overview

Models are stored as TensorFlow SavedModel zip archives on Hugging Face Hub and downloaded at runtime. No MLflow server is required.

**Download flow:**

```
Request → get_classification_model() / get_segmentation_model()
          │
          ├── Is model cached on disk? → Return cached model
          │
          └── No → hf_hub_download(repo_id, filename)
                   → Extract zip to /tmp/savedmodels/<name>/
                   → tf.saved_model.load()
                   → Return model signature
```

### Configuration (`app/configs.py`)

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | **Required.** Hugging Face access token |
| `HF_MODEL_REPO` | `omarelrayes/mlflow-artifacts` | HF Hub repository with model zips |
| `CLASSIFIER_MODEL_PATH` | `models/classifier_savedmodel.zip` | Path to classifier zip |
| `SEGMENTER_MODEL_PATH` | `models/segmenter_savedmodel.zip` | Path to segmenter zip |

### Model Loading

Models are loaded lazily on first request (`app/configs.py`):

1. First call to `get_classification_model()` or `get_segmentation_model()` checks in-memory cache
2. If `None`, acquires a **threading.Lock** (prevents race conditions)
3. Downloads model zip from Hugging Face Hub via `hf_hub_download()`
4. Extracts to `/tmp/savedmodels/<name>/`
5. Loads via `tf.saved_model.load()` and returns the `serving_default` signature
6. Caches the loaded model for all subsequent requests

### Model Validation

A validation script evaluates the model against a holdout dataset:

```bash
export HOLDOUT_DATA_DIR="./holdout_data"
export VALIDATION_ACCURACY_THRESHOLD="0.80"
python scripts/validation/validate_classifier.py
```

| Metric | Description |
|---|---|
| `holdout_accuracy` | Fraction of correct predictions (threshold 0.5) |
| `holdout_precision` | Precision score |
| `holdout_recall` | Recall score |
| `holdout_f1` | F1 score |

> **Note:** The classifier uses a **single sigmoid output** (not 2-class softmax). Predictions use a 0.5 threshold: `malicious` if confidence ≥ 0.5, else `benign`.

**Holdout data directory structure:**
```
holdout_data/
├── benign/
│   ├── image001.jpg
│   └── ...
└── malicious/
    ├── image100.jpg
    └── ...
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
| `gpu_utilization_percent` | Gauge | — | GPU utilization percentage |
| `gpu_memory_used_bytes` | Gauge | — | GPU memory used |
| `gpu_temperature_celsius` | Gauge | — | GPU temperature |

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

1. Open Grafana → **+** → **Import**
2. Upload `grafana/dashboard.json`
3. Select the Prometheus data source
4. Click **Import**

The dashboard includes panels for:
- HTTP request rate and latency (p50, p95, p99)
- Inference rate by model type
- Inference latency percentiles
- GPU utilization and memory usage

## GPU Metrics

If an NVIDIA GPU is available, the API automatically captures GPU metrics using `pynvml`. Metrics are:

- Collected on each `/metrics` scrape
- Exported as Prometheus gauges
- Gracefully degrade to `gpu_available: 0` when no GPU is present

No configuration is needed. The GPU metrics module (`app/monitoring/gpu.py`) uses a lazy initialization pattern — it attempts to connect to the NVIDIA Management Library (`nvml`) on first access and caches the result.

## Model A/B Testing

The API supports A/B testing between multiple model versions. This allows you to test new model versions against the production version with controlled traffic splitting.

### Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `AB_TESTING_ENABLED` | `false` | Enable A/B testing |
| `AB_TESTING_PERCENT_B` | `50` | Percentage of traffic routed to version B |
| `AB_TESTING_MODEL_VERSION_B` | `""` | Model URI for version B (e.g., `models:/skin_cancer_classifier/2`) |

### How It Works

1. **Explicit version selection** — Pass `model_version` in the classify request body to use a specific version
2. **Automatic routing** — When no version is specified and A/B testing is enabled, traffic is split based on `AB_TESTING_PERCENT_B`
3. **Result tracking** — The `model_version` field in the response shows which version handled each request

### Example

```bash
# A/B testing enabled — 50% of traffic goes to version 2
export AB_TESTING_ENABLED=true
export AB_TESTING_PERCENT_B=50
export AB_TESTING_MODEL_VERSION_B="models:/skin_cancer_classifier/2"

# Explicitly use version 2 for a specific request
curl -X POST http://127.0.0.1:8000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"image_id": "IMAGE_ID", "model_version": "models:/skin_cancer_classifier/2"}'
```

### Check A/B Status

```bash
curl http://127.0.0.1:8000/api/ab-status
```

Response:
```json
{
  "enabled": true,
  "version_a": "models:/skin_cancer_classifier/1",
  "version_b": "models:/skin_cancer_classifier/2",
  "percent_b": 50
}
```

## Testing

### Prerequisites for Testing

- MLflow server running on `127.0.0.1:5000`
- Both models registered (`skin_cancer_classifier`, `skin_cancer_segmenter`)
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

## Running with Docker

```bash
# Build the image
docker build -t skin-cancer-api .

# Run (HF Spaces uses port 7860)
docker run -p 7860:7860 -e HF_TOKEN=your_token_here skin-cancer-api
```

**Note:** Storage directories are ephemeral inside the container. For local development, mount a volume:
```bash
docker run -p 7860:7860 -v ./storage:/app/storage -e HF_TOKEN=your_token_here skin-cancer-api
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

### "HF_TOKEN not set"

```bash
export HF_TOKEN="your_token_here"
```
On Hugging Face Spaces, set this in the Space **Settings → Repository Secrets**.

### "Model download failed"

Ensure `HF_TOKEN` has access to the model repository. Verify the repo ID and model paths:
```bash
export HF_MODEL_REPO="omarelrayes/mlflow-artifacts"
export CLASSIFIER_MODEL_PATH="models/classifier_savedmodel.zip"
```

### "Storage paths are wrong"

Ensure you start the server from the project root directory so that `storage/` resolves correctly.

## Hugging Face Spaces Deployment

### Architecture

The API is deployed as a **Docker Space** on Hugging Face Spaces. Models are stored on HF Hub as zip archives and downloaded at runtime.

```
User → HF Spaces (Docker) → FastAPI → HF Hub (model zips)
                                     → /tmp/savedmodels/ (cached)
                                     → /app/storage/ (ephemeral)
```

### Required Secrets

| Secret | Description |
|---|---|
| `HF_TOKEN` | Hugging Face token with read access to the model repository |

Set these in your Space: **Settings → Repository Secrets → New secret**.

### Build Process

1. HF Spaces builds the Docker image using the `Dockerfile`
2. `pip install -r requirements.txt` installs dependencies
3. On first request, models download from HF Hub and cache in `/tmp/savedmodels/`

### Persistent vs Ephemeral Storage

| Path | Type | Behavior |
|---|---|---|
| `/app/storage/` | Ephemeral | Lost on Space restart |
| `/tmp/savedmodels/` | Ephemeral | Lost on Space restart; re-downloaded |
| `/tmp/hf_cache/` | Ephemeral | HF Hub cache |

**Do not rely on uploaded images persisting** across Space restarts. The space restarts after 48 hours of inactivity (free tier) or on redeployment.

### Startup Command

Defined in `Dockerfile`:
```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

HF Spaces uses port `7860` by default.

### Updating the Deployed Application

1. Push changes to the HF Space git repository
2. HF automatically rebuilds the Docker image
3. The Space restarts with the new code
4. Models re-download on first request (cached in `/tmp/`)

### Files for HF Spaces Repository

**Upload:**
- `app/` — Application source code
- `Dockerfile` — Container build instructions
- `requirements.txt` — Python dependencies
- `.gitattributes` — Git LFS configuration
- `.gitignore` — Ignore rules
- `README.md` — Space metadata + documentation
- `storage/` — Directory structure (with `.gitkeep` files)

**Do NOT upload:**
- `venv/` — Local virtual environment
- `mlflow/` — Local MLflow database and artifacts
- `__pycache__/` — Python cache
- `.vscode/` — IDE configuration
- `*.keras`, `*.h5`, `*.pkl` — Model files (not needed, models are on HF Hub)
- `data.xlsx` — Local dataset file
- `.ipynb_checkpoints/` — Jupyter notebook checkpoints
- `grafana/` — Grafana dashboard (for local monitoring only)

### Known Limitations

| Limitation | Impact |
|---|---|
| 16 GB disk (free tier) | Sufficient for model caches + images |
| 2 vCPU + 16 GB RAM (free) | Adequate for inference |
| 48h inactivity timeout | Space sleeps; restarts on next request (cold start ≈ 30s) |
| No GPU (free tier) | CPU inference only; ~200-500ms per classification |
| Ephemeral storage | Uploaded images lost on restart |
| No background tasks | Long segmentation blocks the request |

## Future Improvements

- [x] Hugging Face Spaces deployment
- [x] Prometheus metrics + monitoring
- [x] GPU utilization monitoring (falls back gracefully)
- [x] Model A/B testing framework
- [x] Model download from HF Hub with caching
- [ ] Replace JSON result store with SQLite/PostgreSQL
- [ ] Async job queue for long-running segmentation
- [ ] OAuth2 authentication
- [ ] Rate limiting per endpoint
- [ ] Structured logging (JSON format)
- [ ] Response caching with Redis

## License

MIT
