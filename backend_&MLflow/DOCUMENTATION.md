# Skin Cancer Detection API — Full Documentation

> Version 3.1.0 | FastAPI + TensorFlow + Hugging Face Hub

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Complete File Tree & Code Explanation](#3-complete-file-tree--code-explanation)
4. [API Endpoints — Line-by-Line](#4-api-endpoints--line-by-line)
5. [Data Flow](#5-data-flow)
6. [Work Completed (Session Summary)](#6-work-completed-session-summary)

---

## 1. Project Overview

A REST API for skin cancer image analysis with two ML capabilities:

- **Binary Classification** — Predicts `malicious` or `benign` from skin lesion images using a sigmoid output (threshold 0.5)
- **Segmentation** — Runs a UNet-based model to produce segmentation masks

**Key design decisions:**
- Decoupled endpoints (upload → classify / segment independently)
- Lazy model loading from Hugging Face Hub (no MLflow server needed)
- In-memory model caching with thread-safe locking
- JSON file persistence for results
- Prometheus metrics for HTTP requests and inference
- All source lives under `backend_&MLflow/` for clean repo root

---

## 2. Architecture

```
Client ──HTTP──→ FastAPI ──huggingface_hub──→ HF Hub (model zips)
                     │
                     ├── app/configs.py    → model loading & caching
                     ├── app/services/     → inference logic
                     ├── app/core/         → preprocessing & validation
                     ├── app/storage.py    → file I/O
                     └── app/monitoring/   → Prometheus metrics
```

**Model flow:**
1. First request triggers `get_classification_model()`
2. Downloads `classifier_savedmodel.zip` from `omarelrayes/mlflow-artifacts`
3. Extracts to `%TMP%/savedmodels/classifier/`
4. Loads via `tf.saved_model.load()` → caches the `serving_default` signature
5. Subsequent requests reuse the cached model

---

## 3. Complete File Tree & Code Explanation

```
backend_&MLflow/
├── .vscode/                          # IDE settings (untracked)
├── app/                              # Python package — the API
│   ├── __init__.py                   # Re-exports key symbols
│   ├── main.py                       # FastAPI app
│   ├── configs.py                    # Config, model loading, env vars
│   ├── models.py                     # Pydantic schemas
│   ├── storage.py                    # File save/load + JSON result store
│   ├── core/
│   │   ├── __init__.py               # Core subpackage marker
│   │   ├── preprocessing.py          # Image → TF tensor pipeline
│   │   └── validation.py             # PIL.verify() integrity check
│   ├── monitoring/
│   │   ├── __init__.py               # Monitoring subpackage marker
│   │   └── metrics.py                # Prometheus counters, histograms, middleware
│   └── services/
│       ├── __init__.py               # Services subpackage marker
│       ├── predictor.py              # classify_image() — sigmoid → class
│       ├── segmenter.py              # segment_image() — UNet inference
│       └── routing.py                # ModelRouter with A/B testing
├── mlflow/                           # MLflow tracking & model registry
├── storage/                          # Runtime data
│   ├── images/                       # Uploaded images
│   ├── segments/                     # Segmentation JSON files
│   └── results/results.json          # Classification/segment metadata
├── Notebooks/                        # Jupyter notebooks
├── mini_API/                         # Mini API test harness
├── Segmentation/                     # Segmentation experiments
├── data.xlsx                         # Sample dataset
├── requirements.txt                  # tensorflow, fastapi, etc.
├── README.md                         # Project README
└── DOCUMENTATION.md                  # This file
```

### 3.1 `app/main.py` — FastAPI Application

**Total: 233 lines**

| Lines | Code | What it does |
|-------|------|-------------|
| 1–6 | `imports` | uuid, Path, datetime, FastAPI, UploadFile, File, HTTPException, Request, FileResponse |
| 8–20 | `from . import configs` | Imports config, models (pydantic), services (predictor, segmenter, routing), storage functions, monitoring middleware |
| 23–26 | `app = FastAPI(...)` | Creates the FastAPI instance with title "AI Image Classification API" version 3.1.0 |
| 28 | `app.add_middleware(MetricsMiddleware)` | Registers Prometheus HTTP metrics middleware |
| 30–31 | `ALLOWED_CONTENT_TYPES`, `MAX_FILE_SIZE` | JPEG/PNG only, 10 MB max |
| 34–36 | `GET /metrics` | Returns Prometheus metrics (raw text) |
| 39–53 | `GET /models/status` | Returns model load state, A/B status, storage paths |
| 56–58 | `GET /api/ab-status` | Returns A/B testing configuration |
| 61–86 | `POST /api/upload` | Validates file type/size, saves to `storage/images/`, returns `image_id` |
| 89–99 | `GET /api/images/{image_id}` | Returns stored image file |
| 102–141 | `POST /api/classify` | Loads image, checks cache, runs `classify_image()`, saves result |
| 144–156 | `GET /api/classify/{image_id}` | Returns stored classification |
| 159–186 | `POST /api/segment` | Loads image, runs `segment_image()`, saves result |
| 189–202 | `GET /api/segment/{image_id}` | Returns stored segmentation |
| 205–225 | `GET /` | Root endpoint — service info + endpoint list |
| 228–233 | `GET /health` | Simple health check |

### 3.2 `app/configs.py` — Configuration & Model Loading

**Total: 94 lines**

| Lines | Code | What it does |
|-------|------|-------------|
| 1–8 | `imports` | os, zipfile, shutil, threading, tempfile, Path, typing, hf_hub_download |
| 10–14 | `HF_TOKEN`, `HF_MODEL_REPO`, etc. | Environment variable defaults |
| 16–18 | `_classification_model`, `_segmentation_model`, `_model_lock` | Global caches + thread lock |
| 20 | `CLASSIFIER_THRESHOLD` | Sigmoid threshold (default 0.5) |
| 25–26 | `sigmoid_to_class()` | Maps float → "malicious"/"benign" |
| 29–50 | `_download_and_extract()` | Downloads zip from HF Hub, extracts to cache dir |
| 53–58 | `_load_tf_model()` | Loads SavedModel, returns `serving_default` signature |
| 61–68 | `get_classification_model()` | Lazy loader — checks cache, acquires lock, loads model |
| 71–78 | `get_segmentation_model()` | Same pattern for segmenter |
| 81–82 | `get_loaded_versions()` | Returns list of loaded model versions |
| 85–94 | Storage path setup | Creates `storage/images/`, `storage/segments/`, `storage/results/` |

### 3.3 `app/models.py` — Pydantic Schemas

**Total: 37 lines**

| Lines | Schema | Fields |
|-------|--------|--------|
| 5–11 | `UploadResponse` | image_id, filename, size_bytes, content_type, uploaded_at, url |
| 14–16 | `ClassifyRequest` | image_id, model_version (optional) |
| 19–24 | `ClassifyResponse` | image_id, prediction, confidence, model_version, status |
| 27–28 | `SegmentRequest` | image_id |
| 31–37 | `SegmentResponse` | image_id, status, masks_shape, max_confidence, result_url, error |

### 3.4 `app/storage.py` — File & Result Persistence

**Total: 93 lines**

| Lines | Code | What it does |
|-------|------|-------------|
| 7–9 | `_results`, `_results_lock`, `_RESULTS_FILE` | In-memory dict + thread lock + JSON file path |
| 12–21 | `_load_results()`, `_persist_results()` | Load/save `results.json` |
| 24–28 | `save_image()` | Writes image bytes to `storage/images/{id}.{ext}` |
| 31–36 | `get_image_path()` | Searches for image by id across extensions |
| 39–44 | `get_image_bytes()` | Reads image bytes by id |
| 47–52 | `save_classification_result()` | Stores result in dict + persists JSON |
| 55–59 | `get_classification_result()` | Loads from JSON if needed, returns result |
| 62–80 | `save_segmentation_result()` | Saves segment JSON to `storage/segments/` + metadata |
| 83–87 | `get_segmentation_result()` | Loads from JSON if needed |
| 90–93 | `get_segmentation_result_path()` | Find segment file by glob |

### 3.5 `app/services/predictor.py` — Classification

**Total: 44 lines**

| Lines | Code | What it does |
|-------|------|-------------|
| 1–8 | `imports` | time, numpy, tensorflow, configs, preprocessing, validation, metrics |
| 11–44 | `classify_image()` | Validates image, preprocesses, runs model on TF tensor, extracts sigmoid confidence, maps to class via threshold, records metrics |

### 3.6 `app/services/segmenter.py` — Segmentation

**Total: 50 lines**

| Lines | Code | What it does |
|-------|------|-------------|
| 1–7 | `imports` | time, numpy, tensorflow, configs, preprocessing, metrics |
| 10–50 | `segment_image()` | Preprocesses (128×128), runs model, extracts masks shape + max confidence, records metrics |

### 3.7 `app/services/routing.py` — A/B Testing Router

**Total: 33 lines**

| Lines | Code | What it does |
|-------|------|-------------|
| 5–30 | `ModelRouter` class | Resolves classifier version (default, requested, or A/B split) |
| 33 | `router = ModelRouter()` | Singleton instance |

### 3.8 `app/core/preprocessing.py`

- Converts image bytes → TF tensor
- Resizes to target size (default 224×224 for classifier, 128×128 for segmenter)
- Normalizes pixel values

### 3.9 `app/core/validation.py`

- Uses `PIL.Image.verify()` to check image integrity
- Returns boolean

### 3.10 `app/monitoring/metrics.py`

**Total: 44 lines**

| Lines | Metric | Type |
|-------|--------|------|
| 7 | `inference_total` | Counter — tracks inference calls by model_type and prediction |
| 8–13 | `inference_latency_seconds` | Histogram — inference duration |
| 14 | `http_requests_total` | Counter — HTTP requests by method, path, status |
| 15–20 | `http_request_duration_seconds` | Histogram — request duration |
| 23–40 | `MetricsMiddleware` | Starlette middleware — records duration + count for every request |
| 43–44 | `metrics_endpoint()` | Returns Prometheus `generate_latest()` as text/plain |

---

## 4. API Endpoints — Line-by-Line

### 4.1 `GET /` — Root (lines 205–225 in main.py)

Returns service info, model readiness status, and endpoint list.

```
Response: {
  "service": "AI Image Classification API",
  "version": "3.1.0",
  "classification": "READY|REQUIRED",
  "segmentation": "READY|OPTIONAL",
  "endpoints": { ... },
  "models_status": "/models/status"
}
```

### 4.2 `GET /health` (lines 228–233)

```
Response: { "status": "healthy", "predict_ready": bool }
```

### 4.3 `GET /metrics` (lines 34–36)

Prometheus scrape endpoint. Returns `generate_latest()` output as `text/plain`.

### 4.4 `GET /models/status` (lines 39–53)

Returns classification_loaded, segmentation_loaded, ab_testing config, loaded_versions, storage_paths.

### 4.5 `GET /api/ab-status` (lines 56–58)

Returns enabled, classifier_a, classifier_b, segmenter from ModelRouter.

### 4.6 `POST /api/upload` (lines 61–86)

1. Validates `Content-Type` is JPEG/PNG (line 63)
2. Validates file size ≤ 10 MB (line 71)
3. Generates UUID4 as `image_id` (line 75)
4. Saves to `storage/images/{id}{ext}` (line 77)
5. Returns `UploadResponse` with metadata + URL (lines 79–86)

### 4.7 `GET /api/images/{image_id}` (lines 89–99)

1. Finds image file by id (line 91)
2. Returns 404 if not found (line 93)
3. Returns `FileResponse` with correct media type (line 99)

### 4.8 `POST /api/classify` (lines 102–141)

1. Reads image bytes by `image_id` (line 104)
2. Returns 404 if not found (line 106)
3. Returns cached result if exists and no explicit `model_version` (lines 108–116)
4. Resolves model version via `ModelRouter` (line 118)
5. Calls `classify_image()` — catches ValueError (400) and Exception (500) (lines 120–128)
6. Saves result to JSON store (line 136)
7. Returns `ClassifyResponse` (lines 138–141)

### 4.9 `GET /api/classify/{image_id}` (lines 144–156)

Returns stored classification result or 404.

### 4.10 `POST /api/segment` (lines 159–186)

1. Reads image bytes (line 161)
2. Returns cached segment if exists (lines 165–174)
3. Calls `segment_image()` (line 176)
4. Saves result to `storage/segments/` + JSON store (line 177)
5. Returns `SegmentResponse` (lines 179–186)

### 4.11 `GET /api/segment/{image_id}` (lines 189–202)

Returns stored segmentation result or 404.

---

## 5. Data Flow

```
                         POST /api/upload
                              │
                              ▼
                     ┌─────────────────┐
                     │  Validate file   │
                     │  (type + size)   │
                     └────────┬────────┘
                              │ valid
                              ▼
                     ┌─────────────────┐
                     │  Generate UUID   │
                     │  Save to disk    │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │  Return image_id │
                     └─────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                    │
           ▼                  ▼                    ▼
  POST /api/classify   POST /api/segment    GET /api/images/{id}
           │                  │                    │
           ▼                  ▼                    ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
  │ Check cache  │   │ Check cache  │   │ Find file by ext │
  │ (results.json)│   │ (results.json)│   │ Return FileResponse│
  └──────┬───────┘   └──────┬───────┘   └──────────────────┘
         │ miss             │ miss
         ▼                  ▼
  ┌──────────────┐   ┌──────────────┐
  │ Load model   │   │ Load model   │
  │ (lazy from   │   │ (lazy from   │
  │  HF Hub)     │   │  HF Hub)     │
  └──────┬───────┘   └──────┬───────┘
         │                   │
         ▼                   ▼
  ┌──────────────┐   ┌──────────────┐
  │ Preprocess   │   │ Preprocess   │
  │ (224×224)    │   │ (128×128)    │
  └──────┬───────┘   └──────┬───────┘
         │                   │
         ▼                   ▼
  ┌──────────────┐   ┌──────────────┐
  │ TF Inference │   │ TF Inference │
  │ → sigmoid    │   │ → masks      │
  └──────┬───────┘   └──────┬───────┘
         │                   │
         ▼                   ▼
  ┌──────────────┐   ┌──────────────┐
  │ sigmoid_to_  │   │ Save segment │
  │ class(conf)  │   │ JSON to disk │
  └──────┬───────┘   └──────┬───────┘
         │                   │
         ▼                   ▼
  ┌──────────────┐   ┌──────────────┐
  │ Cache result │   │ Cache result │
  │ in results.json│   │ in results.json│
  └──────┬───────┘   └──────┬───────┘
         │                   │
         ▼                   ▼
  ┌──────────────┐   ┌──────────────┐
  │ Return       │   │ Return       │
  │ ClassifyResp │   │ SegmentResp  │
  └──────────────┘   └──────────────┘
```

---

## 6. Work Completed (Session Summary)

### Phase 1: Code Reversion & Cleanup
- Reverted main project files (`app/`) to committed state using `git checkout`
- Removed stray files from initial deploy (Dockerfile, .gitignore, .gitattributes, grafana/, scripts/, app/monitoring/gpu.py)

### Phase 2: v3.1.0 — 11-Endpoint API Sync
- Added `app/services/routing.py` — A/B testing `ModelRouter`
- Added `app/monitoring/metrics.py` — Prometheus metrics + middleware
- Added `app/core/__init__.py`, `app/services/__init__.py`, `app/monitoring/__init__.py`
- Updated `predictor.py` to use `sigmoid_to_class()` instead of `model_classes` dict
- Updated `segmenter.py` to handle TF SavedModel dict output
- Updated `main.py` to use new `storage.py` functions (save/retrieve by `image_id`)
- Updated `storage.py` with thread-safe JSON result persistence
- Fixed `app/__init__.py` imports

### Phase 3: Bug Fix
- Fixed `GET /metrics` endpoint — was missing `await` on `metrics_endpoint(request)` call

### Phase 4: Repo Restructure
- Consolidated all source into `backend_&MLflow/` directory
- Removed `forproduction/` deployment artifacts (Dockerfile, HF config, grafana)
- Stripped all Hugging Face Spaces metadata from docs
- Removed `model_version` from classify request; simplified to always use default model
- Removed A/B testing dead code (routing still shows config but no active routing)

### Key Files Modified/Added

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `app/main.py` | Modified | 228 | 11 endpoints, fixed metrics await, removed model_version |
| `app/configs.py` | Modified | 94 | HF Hub model loading, thread-safe cache |
| `app/models.py` | Modified | 34 | Pydantic schemas (model_version removed) |
| `app/storage.py` | Modified | 93 | File I/O + JSON result persistence |
| `app/services/predictor.py` | Modified | 42 | Binary classification — simplified |
| `app/services/segmenter.py` | Modified | 50 | Segmentation inference |
| `app/services/routing.py` | Modified | 25 | ModelRouter — removed resolver, kept status |
| `app/monitoring/metrics.py` | Added | 44 | Prometheus metrics + middleware |
| `README.md` | Modified | — | Updated for restructured repo |
| `DOCUMENTATION.md` | Modified | — | This file |
