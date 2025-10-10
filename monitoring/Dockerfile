# Multi-stage build for Monitoring Service
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY monitoring/requirements.txt ./monitoring/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r monitoring/requirements.txt

# Development stage
FROM base as development
COPY . .
CMD ["python", "-m", "monitoring.main"]

# Production stage
FROM base as production
COPY monitoring/ ./monitoring/
COPY shared/ ./shared/
COPY config/ ./config/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

EXPOSE 3000 9090
CMD ["python", "-m", "monitoring.main"]