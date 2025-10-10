"""
Centralized logging configuration for QuantFolio ML Pipeline.
Provides structured logging with cloud-native observability features.
"""

import logging
import logging.config
import json
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import os

from config.base import get_config


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def __init__(self, service_name: str = "quantfolio"):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": self.service_name,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add correlation ID if present (for distributed tracing)
        if hasattr(record, 'correlation_id'):
            log_entry["correlation_id"] = record.correlation_id
        
        # Add request ID if present (for API requests)
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        
        return json.dumps(log_entry)


class CloudNativeLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds cloud-native context to log records"""
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any] = None):
        super().__init__(logger, extra or {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message and add extra context"""
        # Add extra fields to the log record
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        # Add adapter's extra fields
        kwargs['extra'].update(self.extra)
        
        # Create a custom attribute for structured logging
        if 'extra_fields' not in kwargs['extra']:
            kwargs['extra']['extra_fields'] = {}
        
        kwargs['extra']['extra_fields'].update(self.extra)
        
        return msg, kwargs
    
    def with_context(self, **context) -> 'CloudNativeLoggerAdapter':
        """Create a new adapter with additional context"""
        new_extra = self.extra.copy()
        new_extra.update(context)
        return CloudNativeLoggerAdapter(self.logger, new_extra)


def setup_logging(service_name: str = None, log_level: str = None) -> None:
    """Setup centralized logging configuration"""
    config = get_config()
    
    if service_name is None:
        service_name = config.service_name
    
    if log_level is None:
        log_level = config.monitoring.log_level
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Logging configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "()": StructuredFormatter,
                "service_name": service_name,
            },
            "simple": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "structured" if config.environment.value != "local" else "simple",
                "stream": sys.stdout,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "structured",
                "filename": f"logs/{service_name}.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "structured",
                "filename": f"logs/{service_name}_errors.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "": {  # Root logger
                "level": log_level,
                "handlers": ["console", "file", "error_file"],
                "propagate": False,
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "sqlalchemy": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "boto3": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "botocore": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
    }
    
    # Apply logging configuration
    logging.config.dictConfig(logging_config)
    
    # Set up exception handling
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger = get_logger("uncaught_exception")
        logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = handle_exception


def get_logger(name: str, **context) -> CloudNativeLoggerAdapter:
    """Get a logger with cloud-native context"""
    logger = logging.getLogger(name)
    return CloudNativeLoggerAdapter(logger, context)


def log_function_call(func_name: str, **kwargs):
    """Decorator to log function calls with parameters"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            logger.info(
                f"Calling function {func_name}",
                extra_fields={
                    "function": func_name,
                    "args_count": len(args),
                    "kwargs": list(kwargs.keys()),
                }
            )
            
            try:
                result = func(*args, **kwargs)
                logger.info(
                    f"Function {func_name} completed successfully",
                    extra_fields={"function": func_name}
                )
                return result
            except Exception as e:
                logger.error(
                    f"Function {func_name} failed",
                    exc_info=True,
                    extra_fields={
                        "function": func_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                )
                raise
        
        return wrapper
    return decorator


# Performance monitoring decorator
def log_performance(func):
    """Decorator to log function performance metrics"""
    def wrapper(*args, **kwargs):
        import time
        
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.info(
                f"Performance metrics for {func.__name__}",
                extra_fields={
                    "function": func.__name__,
                    "execution_time_seconds": execution_time,
                    "status": "success",
                }
            )
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(
                f"Performance metrics for {func.__name__} (failed)",
                exc_info=True,
                extra_fields={
                    "function": func.__name__,
                    "execution_time_seconds": execution_time,
                    "status": "error",
                    "error_type": type(e).__name__,
                }
            )
            raise
    
    return wrapper