# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (cacheable)
COPY requirements.txt .
ENV PIP_DEFAULT_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    pip install --retries 8 --timeout 180 -r requirements.txt -i https://pypi.org/simple

# Copy ALL source code at repo root (includes GE2PE.py, app.py, config.py, prompt_base.txt, etc.)
# We'll exclude big stuff via .dockerignore.
COPY . .

# Ensure Python can import from /app (for GE2PE.py)
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true PYTHONUNBUFFERED=1
EXPOSE 8501
CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]
