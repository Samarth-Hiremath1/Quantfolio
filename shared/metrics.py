"""
Metrics collection and observability infrastructure for QuantFolio ML Pipeline.
Provides Prometheus-compatible metrics with cloud-native patterns.
"""

import time
import functools
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from threading import Lock
import json

from config.base import get_config
from shared.logging import get_logger


@dataclass
class MetricValue:
    """Represents a metric value with metadata"""
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsRegistry:
    """Thread-safe metrics registry for collecting application metrics"""
    
    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = defaultdict(list)
        self._summaries: Dict[str, list] = defaultdict(list)
        self._labels: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._lock = Lock()
        self._logger = get_logger(__name__)
    
    def counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment a counter metric"""
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value
            if labels:
                self._labels[key] = labels
            
            self._logger.debug(
                f"Counter {name} incremented",
                extra_fields={
                    "metric_name": name,
                    "metric_type": "counter",
                    "value": value,
                    "labels": labels or {},
                }
            )
    
    def gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge metric value"""
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value
            if labels:
                self._labels[key] = labels
            
            self._logger.debug(
                f"Gauge {name} set",
                extra_fields={
                    "metric_name": name,
                    "metric_type": "gauge",
                    "value": value,
                    "labels": labels or {},
                }
            )
    
    def histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a value in a histogram"""
        with self._lock:
            key = self._make_key(name, labels)
            self._histograms[key].append(value)
            if labels:
                self._labels[key] = labels
            
            self._logger.debug(
                f"Histogram {name} recorded",
                extra_fields={
                    "metric_name": name,
                    "metric_type": "histogram",
                    "value": value,
                    "labels": labels or {},
                }
            )
    
    def summary(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a value in a summary"""
        with self._lock:
            key = self._make_key(name, labels)
            self._summaries[key].append(value)
            if labels:
                self._labels[key] = labels
            
            self._logger.debug(
                f"Summary {name} recorded",
                extra_fields={
                    "metric_name": name,
                    "metric_type": "summary",
                    "value": value,
                    "labels": labels or {},
                }
            )
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create a unique key for metric with labels"""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics in Prometheus format"""
        with self._lock:
            metrics = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: self._calculate_histogram_stats(v) for k, v in self._histograms.items()},
                "summaries": {k: self._calculate_summary_stats(v) for k, v in self._summaries.items()},
                "labels": dict(self._labels),
                "timestamp": time.time(),
            }
            return metrics
    
    def _calculate_histogram_stats(self, values: list) -> Dict[str, float]:
        """Calculate histogram statistics"""
        if not values:
            return {"count": 0, "sum": 0, "min": 0, "max": 0, "avg": 0}
        
        return {
            "count": len(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "p50": self._percentile(values, 0.5),
            "p90": self._percentile(values, 0.9),
            "p95": self._percentile(values, 0.95),
            "p99": self._percentile(values, 0.99),
        }
    
    def _calculate_summary_stats(self, values: list) -> Dict[str, float]:
        """Calculate summary statistics"""
        return self._calculate_histogram_stats(values)
    
    def _percentile(self, values: list, percentile: float) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(percentile * (len(sorted_values) - 1))
        return sorted_values[index]
    
    def reset(self) -> None:
        """Reset all metrics"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._summaries.clear()
            self._labels.clear()
            
            self._logger.info("Metrics registry reset")


# Global metrics registry
metrics_registry = MetricsRegistry()


class MetricsCollector:
    """High-level metrics collection interface"""
    
    def __init__(self, service_name: str = None):
        self.service_name = service_name or get_config().service_name
        self.registry = metrics_registry
        self.logger = get_logger(__name__, service=self.service_name)
    
    def increment_counter(self, name: str, value: float = 1.0, **labels) -> None:
        """Increment a counter with service labels"""
        labels["service"] = self.service_name
        self.registry.counter(name, value, labels)
    
    def set_gauge(self, name: str, value: float, **labels) -> None:
        """Set a gauge value with service labels"""
        labels["service"] = self.service_name
        self.registry.gauge(name, value, labels)
    
    def record_histogram(self, name: str, value: float, **labels) -> None:
        """Record a histogram value with service labels"""
        labels["service"] = self.service_name
        self.registry.histogram(name, value, labels)
    
    def record_summary(self, name: str, value: float, **labels) -> None:
        """Record a summary value with service labels"""
        labels["service"] = self.service_name
        self.registry.summary(name, value, labels)
    
    def time_function(self, metric_name: str, **labels):
        """Decorator to time function execution"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Record successful execution time
                    self.record_histogram(
                        f"{metric_name}_duration_seconds",
                        execution_time,
                        status="success",
                        function=func.__name__,
                        **labels
                    )
                    
                    # Increment success counter
                    self.increment_counter(
                        f"{metric_name}_total",
                        status="success",
                        function=func.__name__,
                        **labels
                    )
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    # Record failed execution time
                    self.record_histogram(
                        f"{metric_name}_duration_seconds",
                        execution_time,
                        status="error",
                        function=func.__name__,
                        error_type=type(e).__name__,
                        **labels
                    )
                    
                    # Increment error counter
                    self.increment_counter(
                        f"{metric_name}_total",
                        status="error",
                        function=func.__name__,
                        error_type=type(e).__name__,
                        **labels
                    )
                    
                    raise
            
            return wrapper
        return decorator
    
    def track_api_request(self, endpoint: str, method: str):
        """Decorator to track API request metrics"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Record request metrics
                    self.record_histogram(
                        "api_request_duration_seconds",
                        execution_time,
                        endpoint=endpoint,
                        method=method,
                        status="success"
                    )
                    
                    self.increment_counter(
                        "api_requests_total",
                        endpoint=endpoint,
                        method=method,
                        status="success"
                    )
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    self.record_histogram(
                        "api_request_duration_seconds",
                        execution_time,
                        endpoint=endpoint,
                        method=method,
                        status="error",
                        error_type=type(e).__name__
                    )
                    
                    self.increment_counter(
                        "api_requests_total",
                        endpoint=endpoint,
                        method=method,
                        status="error",
                        error_type=type(e).__name__
                    )
                    
                    raise
            
            return wrapper
        return decorator


# Application-specific metrics collectors
class FinancialMetrics:
    """Metrics specific to financial operations"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def record_portfolio_performance(self, sharpe_ratio: float, volatility: float, returns: float):
        """Record portfolio performance metrics"""
        self.collector.set_gauge("portfolio_sharpe_ratio", sharpe_ratio)
        self.collector.set_gauge("portfolio_volatility", volatility)
        self.collector.set_gauge("portfolio_returns", returns)
    
    def record_model_accuracy(self, model_name: str, accuracy: float, framework: str):
        """Record model accuracy metrics"""
        self.collector.set_gauge(
            "model_accuracy",
            accuracy,
            model_name=model_name,
            framework=framework
        )
    
    def record_data_ingestion(self, symbol: str, records_count: int, success: bool):
        """Record data ingestion metrics"""
        status = "success" if success else "error"
        
        self.collector.increment_counter(
            "data_ingestion_total",
            symbol=symbol,
            status=status
        )
        
        if success:
            self.collector.set_gauge(
                "data_records_ingested",
                records_count,
                symbol=symbol
            )
    
    def record_optimization_result(self, optimization_time: float, num_assets: int, success: bool):
        """Record portfolio optimization metrics"""
        status = "success" if success else "error"
        
        self.collector.record_histogram(
            "optimization_duration_seconds",
            optimization_time,
            status=status
        )
        
        self.collector.increment_counter(
            "optimization_total",
            status=status
        )
        
        if success:
            self.collector.set_gauge("portfolio_assets_count", num_assets)


def get_metrics_collector(service_name: str = None) -> MetricsCollector:
    """Get a metrics collector instance"""
    return MetricsCollector(service_name)


def get_financial_metrics(service_name: str = None) -> FinancialMetrics:
    """Get financial metrics collector"""
    collector = get_metrics_collector(service_name)
    return FinancialMetrics(collector)


def export_prometheus_metrics() -> str:
    """Export metrics in Prometheus format"""
    metrics = metrics_registry.get_metrics()
    
    prometheus_output = []
    
    # Export counters
    for name, value in metrics["counters"].items():
        labels = metrics["labels"].get(name, {})
        label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
        if label_str:
            prometheus_output.append(f'{name.split("{")[0]}{{{label_str}}} {value}')
        else:
            prometheus_output.append(f'{name} {value}')
    
    # Export gauges
    for name, value in metrics["gauges"].items():
        labels = metrics["labels"].get(name, {})
        label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
        if label_str:
            prometheus_output.append(f'{name.split("{")[0]}{{{label_str}}} {value}')
        else:
            prometheus_output.append(f'{name} {value}')
    
    # Export histograms
    for name, stats in metrics["histograms"].items():
        labels = metrics["labels"].get(name, {})
        base_name = name.split("{")[0]
        
        for stat_name, stat_value in stats.items():
            metric_name = f"{base_name}_{stat_name}"
            label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
            if label_str:
                prometheus_output.append(f'{metric_name}{{{label_str}}} {stat_value}')
            else:
                prometheus_output.append(f'{metric_name} {stat_value}')
    
    return "\n".join(prometheus_output)