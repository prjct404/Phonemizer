# monitoring_metrics.py
import os
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    start_http_server,
)

# Use a private registry so we don't collide with other libs
REGISTRY = CollectorRegistry()

# Define metrics ONCE (at module import time)
INFER_REQUESTS = Counter(
    "infer_requests",                 # no "_total" suffix here
    "Total inference requests",
    registry=REGISTRY,
)

INFER_LATENCY = Histogram(
    "infer_latency_seconds",
    "Inference latency (seconds)",
    registry=REGISTRY,
)

# Start the /metrics server once per process
_METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
start_http_server(_METRICS_PORT, registry=REGISTRY)
