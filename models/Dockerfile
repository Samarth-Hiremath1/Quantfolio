# Multi-stage build for ML Models Service
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY models/requirements.txt ./models/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r models/requirements.txt

# Development stage
FROM base as development
COPY . .
CMD ["python", "-m", "models.main"]

# Production stage
FROM base as production
COPY models/ ./models/
COPY shared/ ./shared/
COPY config/ ./config/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8002/health')"

CMD ["python", "-m", "models.main"]