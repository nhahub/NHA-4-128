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

# Skin Cancer Detection API

FastAPI application for skin cancer classification and segmentation, deployed on Hugging Face Spaces.

## API Endpoints

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

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | - | Hugging Face token for model downloads |
| `HF_MODEL_REPO` | `omarelrayes/mlflow-artifacts` | HF repo with model zips |
| `CLASSIFIER_MODEL_PATH` | `models/classifier_savedmodel.zip` | Path in HF repo |
| `SEGMENTER_MODEL_PATH` | `models/segmenter_savedmodel.zip` | Path in HF repo |
| `CLASSIFIER_THRESHOLD` | `0.5` | Confidence threshold |
| `AB_TESTING_ENABLED` | `false` | Enable A/B testing |
| `AB_CLASSIFIER_B` | - | Alternative model URI |
