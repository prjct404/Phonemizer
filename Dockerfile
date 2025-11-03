FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# Copy deps first for better caching
COPY requirements.txt .

# Keep pip at a known-good version during build
RUN python -m pip install --upgrade "pip==24.2"

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0

# Rest of  Python deps
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.org/simple

# Copy app code
COPY . .

# Streamlit runs on 8501
EXPOSE 8501

# Optional Streamlit envs (nice defaults)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
