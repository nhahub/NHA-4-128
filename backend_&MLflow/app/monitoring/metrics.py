import time
from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

inference_total = Counter("inference_total", "Total inferences", ["model_type", "prediction"])
inference_latency_seconds = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
    ["model_type"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
http_requests_total = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        http_requests_total.labels(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
        ).inc()

        http_request_duration_seconds.labels(
            method=request.method,
            path=request.url.path,
        ).observe(duration)

        return response


async def metrics_endpoint(request: Request) -> Response:
    return Response(content=generate_latest(), media_type="text/plain")
