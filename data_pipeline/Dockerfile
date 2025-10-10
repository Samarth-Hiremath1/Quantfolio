# Multi-stage build for Data Pipeline Service
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY data_pipeline/requirements.txt ./data_pipeline/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r data_pipeline/requirements.txt

# Development stage
FROM base as development
COPY . .
CMD ["python", "-m", "data_pipeline.main"]

# Production stage
FROM base as production
COPY data_pipeline/ ./data_pipeline/
COPY shared/ ./shared/
COPY config/ ./config/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')"

CMD ["python", "-m", "data_pipeline.main"]