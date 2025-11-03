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

# --------------------------------------------------------------------
# 1) GENERIC METRICS FOR INFERENCE SERVICES
# --------------------------------------------------------------------

INFER_REQUESTS = Counter(
    "infer_requests",
    "Total inference requests",
    registry=REGISTRY,
)

INFER_LATENCY = Histogram(
    "infer_latency_seconds",
    "Inference latency (seconds)",
    # you can tweak buckets if you like
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10),
    registry=REGISTRY,
)

# --------------------------------------------------------------------
# 2) RICHER METRICS FOR THE PHONEMIZER APP
# --------------------------------------------------------------------

# Traffic / usage
PHONEMIZER_REQUESTS = Counter(
    "phonemizer_requests_total",
    "Total phonemizer requests",
    labelnames=["outcome", "voice"],  # outcome = success|error, voice = TTS voice
    registry=REGISTRY,
)

# Latency per stage (GE2PE, LLM, TTS, end-to-end)
PHONEMIZER_STAGE_LATENCY = Histogram(
    "phonemizer_stage_latency_seconds",
    "Latency per processing stage",
    labelnames=["stage"],  # stage = ge2pe | llm | tts_input | tts_output | end_to_end
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
    registry=REGISTRY,
)

# Errors by stage and reason
PHONEMIZER_ERRORS = Counter(
    "phonemizer_errors_total",
    "Total errors by stage and reason",
    labelnames=["stage", "reason"],  # e.g. stage="llm", reason="AuthenticationError"
    registry=REGISTRY,
)

# TTS-specific failures (by voice)
PHONEMIZER_TTS_FAILURES = Counter(
    "phonemizer_tts_failures_total",
    "TTS failures by voice and reason",
    labelnames=["voice", "reason"],
    registry=REGISTRY,
)

# Input text length distribution
PHONEMIZER_INPUT_CHARS = Histogram(
    "phonemizer_input_chars",
    "Distribution of input text length (characters)",
    buckets=(20, 50, 100, 200, 400, 800, 1200),
    registry=REGISTRY,
)

# LLM output length distribution
PHONEMIZER_OUTPUT_CHARS = Histogram(
    "phonemizer_output_chars",
    "Distribution of LLM output text length (characters)",
    buckets=(20, 50, 100, 200, 400, 800, 1200),
    registry=REGISTRY,
)

# --------------------------------------------------------------------
# 3) Start the /metrics server once per process
# --------------------------------------------------------------------

_METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
start_http_server(_METRICS_PORT, registry=REGISTRY)
