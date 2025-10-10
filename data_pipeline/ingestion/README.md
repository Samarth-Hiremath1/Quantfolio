# Data Ingestion Service

A cloud-native data ingestion service for the QuantFolio ML pipeline that fetches financial data from external APIs, validates it, stores it with cloud-native partitioning patterns, and publishes events for downstream processing.

## Features

### 🚀 Cloud-Native Architecture
- **Microservices Design**: Independently deployable and scalable components
- **Event-Driven Architecture**: Loose coupling through event publishing
- **Circuit Breaker Pattern**: Fault tolerance for external API calls
- **Horizontal Scalability**: Designed for cloud auto-scaling
- **Observability**: Comprehensive logging and metrics

### 📊 Data Sources
- **Yahoo Finance API**: Free access to stock data (no API key required)
- **Alpha Vantage API**: Professional financial data (API key required)
- **Extensible**: Easy to add new data providers

### 🔍 Data Validation
- **Completeness Checks**: Ensure minimum data requirements
- **Price Validity**: Validate OHLC relationships and detect negative prices
- **Freshness Validation**: Check data age and staleness
- **Duplicate Detection**: Identify and handle duplicate records
- **Outlier Detection**: Statistical analysis for anomaly detection

### 💾 Storage System
- **Cloud Object Storage Simulation**: Local filesystem that mimics S3/GCS patterns
- **Date/Symbol Partitioning**: Efficient data organization
- **Parquet Format**: Columnar storage for optimal performance
- **Metadata Tracking**: Comprehensive ingestion logs

### 🔄 Event Publishing
- **Multiple Publishers**: In-memory, file-based, and composite publishers
- **Event History**: Track all data availability events
- **Subscriber Management**: Flexible event handling

### ⚡ Resilience Features
- **Circuit Breaker**: Configurable failure thresholds and recovery
- **Retry Logic**: Exponential backoff for transient failures
- **Timeout Handling**: Prevent hanging requests
- **Error Recovery**: Graceful degradation and fallback mechanisms

## Quick Start

### Basic Usage

```python
import asyncio
from data_pipeline.ingestion.service import DataIngestionService, IngestionConfig

async def main():
    # Create configuration
    config = IngestionConfig(
        api_provider="yahoo",
        symbols=["AAPL", "GOOGL", "MSFT"],
        days_to_fetch=5,
        storage_base_path="data"
    )
    
    # Run ingestion
    async with DataIngestionService(config) as service:
        result = await service.run_ingestion_pipeline()
        print(f"Success: {result['success']}")
        print(f"Records stored: {result['records_stored']}")

asyncio.run(main())
```

### Command Line Interface

```bash
# Fetch data for specific symbols
python -m data_pipeline.ingestion.cli ingest --symbols AAPL,GOOGL,MSFT --days 5

# Use Alpha Vantage with API key
python -m data_pipeline.ingestion.cli ingest --provider alpha_vantage --api-key YOUR_KEY --symbols AAPL

# List available data
python -m data_pipeline.ingestion.cli list --verbose

# Custom circuit breaker settings
python -m data_pipeline.ingestion.cli ingest --symbols AAPL --cb-failure-threshold 2 --cb-recovery-timeout 30
```

### Advanced Configuration

```python
from data_pipeline.ingestion.circuit_breaker import CircuitBreakerConfig

# Custom circuit breaker
cb_config = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=60.0,
    success_threshold=2,
    timeout=30.0
)

config = IngestionConfig(
    api_provider="yahoo",
    symbols=["AAPL", "TSLA", "NVDA"],
    days_to_fetch=10,
    storage_base_path="data",
    circuit_breaker_config=cb_config,
    enable_validation=True,
    enable_events=True
)
```

## Architecture

### Component Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Clients   │    │   Data Validator │    │ Storage System  │
│                 │    │                  │    │                 │
│ • Yahoo Finance │    │ • Completeness   │    │ • Partitioning  │
│ • Alpha Vantage │    │ • Price Validity │    │ • Parquet Files │
│ • Circuit Breaker│   │ • Freshness      │    │ • Metadata      │
└─────────────────┘    │ • Outliers       │    └─────────────────┘
                       │ • Duplicates     │
                       └──────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Event System   │    │ Ingestion Service│    │ Configuration   │
│                 │    │                  │    │                 │
│ • Publishers    │◄───┤ • Orchestration  │    │ • Environment   │
│ • Subscribers   │    │ • Error Handling │    │ • Validation    │
│ • Event History │    │ • Monitoring     │    │ • Defaults      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Data Flow

1. **Fetch**: API clients retrieve data with circuit breaker protection
2. **Validate**: Multiple validation rules ensure data quality
3. **Store**: Data is partitioned and stored in Parquet format
4. **Publish**: Events notify downstream systems of data availability

### Storage Structure

```
data/
├── raw/daily/
│   ├── 2024/01/01/
│   │   ├── AAPL.parquet
│   │   ├── GOOGL.parquet
│   │   └── MSFT.parquet
│   └── 2024/01/02/
│       └── ...
├── raw/metadata/
│   └── ingestion_logs.json
└── events/
    └── data_available_*.json
```

## Configuration

### Environment Variables

```bash
# API Configuration
export INGESTION_API_PROVIDER=yahoo
export INGESTION_API_KEY=your_api_key
export INGESTION_SYMBOLS=AAPL,GOOGL,MSFT,AMZN,TSLA

# Storage Configuration
export INGESTION_STORAGE_PATH=/path/to/data
export INGESTION_DAYS_TO_FETCH=5

# Circuit Breaker Configuration
export CB_FAILURE_THRESHOLD=3
export CB_RECOVERY_TIMEOUT=60.0
export CB_SUCCESS_THRESHOLD=2
export CB_TIMEOUT=30.0

# Feature Flags
export INGESTION_ENABLE_VALIDATION=true
export INGESTION_ENABLE_EVENTS=true

# Logging
export INGESTION_LOG_LEVEL=INFO
```

### Configuration Class

```python
from data_pipeline.ingestion.config import IngestionServiceConfig

config = IngestionServiceConfig()
print(config.to_dict())  # View all configuration
```

## Validation Rules

### Built-in Rules

1. **Completeness Rule**: Ensures minimum records per symbol
2. **Price Validity Rule**: Validates OHLC relationships and positive prices
3. **Freshness Rule**: Checks data age (default: 48 hours)
4. **Duplicate Detection Rule**: Identifies duplicate symbol/date combinations
5. **Outlier Detection Rule**: Statistical analysis using z-scores

### Custom Validation Rules

```python
from data_pipeline.ingestion.validator import ValidationRule, DataValidator

class CustomRule(ValidationRule):
    def __init__(self):
        super().__init__("custom_rule", severity="warning")
    
    def validate(self, data):
        # Custom validation logic
        return True, "Custom validation passed"

validator = DataValidator()
validator.add_rule(CustomRule())
```

## Event System

### Event Publishers

```python
from data_pipeline.ingestion.events import (
    InMemoryEventPublisher, 
    FileEventPublisher, 
    CompositeEventPublisher
)

# In-memory publisher (default)
memory_publisher = InMemoryEventPublisher()

# File-based publisher
file_publisher = FileEventPublisher("events/")

# Composite publisher (multiple targets)
composite_publisher = CompositeEventPublisher([
    memory_publisher, 
    file_publisher
])
```

### Event Subscribers

```python
def handle_data_event(event):
    print(f"New data available: {event.symbols}")
    print(f"Records: {event.record_count}")
    print(f"Files: {event.file_paths}")

# Subscribe to events
publisher.subscribe(handle_data_event)
```

## Monitoring and Observability

### Circuit Breaker Stats

```python
async with DataIngestionService(config) as service:
    stats = service.get_circuit_breaker_stats()
    print(f"State: {stats['state']}")
    print(f"Failures: {stats['failure_count']}")
```

### Storage Statistics

```python
stats = service.get_storage_stats()
print(f"Total files: {stats['total_files']}")
print(f"Total size: {stats['total_size_mb']} MB")
print(f"Symbols: {stats['symbols']}")
print(f"Date range: {stats['date_range']}")
```

### Available Data

```python
available = service.list_available_data()
for symbol, dates in available.items():
    print(f"{symbol}: {len(dates)} days")
```

## Error Handling

### Circuit Breaker States

- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Failing, requests are blocked
- **HALF_OPEN**: Testing recovery, limited requests allowed

### Error Types

- **APIClientError**: General API communication errors
- **RateLimitError**: API rate limit exceeded
- **DataNotFoundError**: Requested data not available
- **CircuitBreakerError**: Circuit breaker is open
- **ValidationError**: Data validation failures

### Retry Logic

- Exponential backoff for transient failures
- Configurable retry limits
- Dead letter queue simulation

## Testing

### Unit Tests

```bash
# Run all ingestion tests
python -m pytest tests/unit/test_ingestion_*.py -v

# Run specific test file
python -m pytest tests/unit/test_circuit_breaker.py -v

# Run with coverage
python -m pytest tests/unit/ --cov=data_pipeline.ingestion
```

### Integration Tests

```bash
# Run integration tests
python -m pytest tests/integration/test_ingestion_integration.py -v
```

### Example Scripts

```bash
# Run basic example
python examples/ingestion_example.py

# Test CLI
python -m data_pipeline.ingestion.cli ingest --symbols AAPL --days 1
```

## Performance Considerations

### Optimization Tips

1. **Batch Size**: Fetch multiple days at once to reduce API calls
2. **Parallel Processing**: Use multiple symbols concurrently
3. **Caching**: Enable validation caching for repeated checks
4. **Storage Format**: Parquet provides excellent compression and query performance
5. **Circuit Breaker**: Tune thresholds based on API reliability

### Scalability

- **Horizontal Scaling**: Service is stateless and can be scaled horizontally
- **Resource Isolation**: Circuit breakers prevent resource exhaustion
- **Event-Driven**: Loose coupling enables independent scaling of components
- **Storage Partitioning**: Efficient data organization for large datasets

## Deployment

### Local Development

```bash
# Install dependencies
pip install -r data_pipeline/requirements.txt

# Run ingestion
python -m data_pipeline.ingestion.cli ingest --symbols AAPL,GOOGL
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY data_pipeline/requirements.txt .
RUN pip install -r requirements.txt

COPY data_pipeline/ ./data_pipeline/
CMD ["python", "-m", "data_pipeline.ingestion.cli", "ingest"]
```

### Cloud Deployment

The service is designed for cloud deployment with:
- Container orchestration (Kubernetes, ECS)
- Serverless functions (Lambda, Cloud Functions)
- Managed services integration (S3, GCS, SQS, Pub/Sub)

## Contributing

### Adding New Data Providers

1. Inherit from `BaseAPIClient`
2. Implement `fetch_daily_data` method
3. Add to `APIClientFactory`
4. Write unit tests

### Adding Validation Rules

1. Inherit from `ValidationRule`
2. Implement `validate` method
3. Add to `DataValidator` defaults or use `add_rule`
4. Write unit tests

### Adding Event Publishers

1. Inherit from `EventPublisher`
2. Implement `publish` and `close` methods
3. Add to `CompositeEventPublisher` if needed
4. Write unit tests

## License

This project is part of the QuantFolio ML pipeline and follows the same licensing terms.