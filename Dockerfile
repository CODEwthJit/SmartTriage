# ==============================================================================
# Production Dockerfile for SmartTriage
# Multi-stage CPU-optimized container with non-root security execution
# ==============================================================================

FROM python:3.11-slim AS runtime

# Set environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Install minimal OS dependencies for network & certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies with CPU-optimized PyTorch
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application source code and serialized model registry
COPY src/ /app/src/
COPY models/registry/ /app/models/registry/
COPY data/processed/ /app/data/processed/
COPY ui/ /app/ui/

# Create non-root system user for security compliance (Principle of Least Privilege)
RUN useradd -u 10001 -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose API and UI ports
EXPOSE 8000 8501

# Healthcheck probe verifying API liveness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Default command: Launch FastAPI production gateway
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
