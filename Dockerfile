# ============================================================
# MLOps Production Pipeline - Docker Image
# Multi-stage build for optimized production deployment
# ============================================================

FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# ---- Dependencies Stage ----
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Production Stage ----
FROM dependencies AS production

# Create non-root user
RUN groupadd -r mlops && useradd -r -g mlops -d /app -s /sbin/nologin mlops

# Copy application code
COPY src/ ./src/
COPY setup.py .
COPY requirements.txt .
COPY data/ ./data/

# Set ownership
RUN chown -R mlops:mlops /app

USER mlops

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run with uvicorn for production
CMD ["python", "-m", "uvicorn", "src.models.pipeline:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- Test Stage ----
FROM dependencies AS test

COPY . .
RUN pip install pytest pytest-cov
CMD ["pytest", "--cov=src", "--cov-report=term-missing", "-v"]
