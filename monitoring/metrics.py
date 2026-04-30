from prometheus_client import Gauge, start_http_server
import logging

logger = logging.getLogger(__name__)

# Define custom Prometheus metrics
MODEL_MAE_ROLLING_30D = Gauge(
    'model_mae_rolling_30d',
    '30-day Rolling Mean Absolute Error of the active forecasting model to detect drift',
    ['ticker', 'model_type']
)

def push_model_drift_metric(ticker: str, model_type: str, mae_value: float):
    """
    Updates the Prometheus gauge for model drift monitoring.
    """
    MODEL_MAE_ROLLING_30D.labels(ticker=ticker, model_type=model_type).set(mae_value)
    logger.info(f"Pushed metric: model_mae_rolling_30d={mae_value} for {ticker} ({model_type})")

def start_metrics_server(port: int = 8001):
    """
    Starts an isolated HTTP server to expose metrics for Prometheus to scrape.
    """
    start_http_server(port)
    logger.info(f"Custom metrics exporter started on port {port}")
