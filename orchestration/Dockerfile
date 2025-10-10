# Multi-stage build for Orchestration Service (Airflow)
FROM apache/airflow:2.7.0-python3.11 as base

# Switch to root to install system dependencies
USER root
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

WORKDIR /opt/airflow

# Copy requirements
COPY requirements.txt .
COPY orchestration/requirements.txt ./orchestration/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r orchestration/requirements.txt

# Development stage
FROM base as development
COPY . .
CMD ["airflow", "webserver"]

# Production stage
FROM base as production
COPY orchestration/ ./orchestration/
COPY shared/ ./shared/
COPY config/ ./config/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

EXPOSE 8080
CMD ["airflow", "webserver"]