# QuantFolio

**QuantFolio** is a full-stack quantitative research platform intended for quantitative software engineering and ML engineer workflows.

## Features (Phase 1)
- **Data Ingestion**: Automated OHLCV data scraping from yfinance and Alpha Vantage.
- **Data Transforms**: Forward-filling missing gaps, outlier handling, and engineering moving averages, MACD, and RSI without lookahead bias.
- **Orchestration**: End-to-end Airflow ETL pipeline.
- **Storage**: Raw dumps stored locally via S3-emulator LocalStack, structured features stored in PostgreSQL.
- **Monitoring**: Basic Prometheus scraping of pipelines logic.

## Architecture

| Layer | Component |
| ----- | --------- |
| **Pipeline Core** | Python 3.11, pandas, boto3 |
| **Database** | PostgreSQL |
| **Cloud/S3** | LocalStack |
| **Orchestrator** | Apache Airflow |
| **Monitoring** | Prometheus + Pushgateway + Grafana |
| **API Backend** | FastAPI (stub) |

## Quickstart

1. Copy `.env.example` to `.env` and fill in any API keys (e.g., Alpha Vantage).
2. Boot up the local infrastructure:
   ```bash
   docker-compose up -d
   ```
3. Initialize LocalStack S3 buckets (the script `init-aws.sh` will auto-run on boot).
4. Run the data ingestion DAG in Airflow manually or wait for the daily schedule.

Note: Local development requires `apache-airflow` installed.
