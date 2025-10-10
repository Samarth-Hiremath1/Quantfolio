"""Configuration management for data ingestion service."""

import os
from typing import List, Optional
from dataclasses import dataclass

from .circuit_breaker import CircuitBreakerConfig


@dataclass
class IngestionServiceConfig:
    """Configuration for the data ingestion service."""
    
    # API Configuration
    api_provider: str = "yahoo"  # "yahoo" or "alpha_vantage"
    api_key: str = ""
    
    # Data Configuration
    symbols: List[str] = None
    days_to_fetch: int = 1
    
    # Storage Configuration
    storage_base_path: str = "data"
    
    # Circuit Breaker Configuration
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_recovery_timeout: float = 60.0
    circuit_breaker_success_threshold: int = 2
    circuit_breaker_timeout: float = 30.0
    
    # Feature Flags
    enable_validation: bool = True
    enable_events: bool = True
    
    # Logging Configuration
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.symbols is None:
            self.symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        
        # Load from environment variables if available
        self.api_provider = os.getenv("INGESTION_API_PROVIDER", self.api_provider)
        self.api_key = os.getenv("INGESTION_API_KEY", self.api_key)
        self.storage_base_path = os.getenv("INGESTION_STORAGE_PATH", self.storage_base_path)
        
        # Parse symbols from environment
        env_symbols = os.getenv("INGESTION_SYMBOLS")
        if env_symbols:
            self.symbols = [s.strip() for s in env_symbols.split(",")]
        
        # Parse other numeric configs
        self.days_to_fetch = int(os.getenv("INGESTION_DAYS_TO_FETCH", str(self.days_to_fetch)))
        self.circuit_breaker_failure_threshold = int(os.getenv("CB_FAILURE_THRESHOLD", str(self.circuit_breaker_failure_threshold)))
        self.circuit_breaker_recovery_timeout = float(os.getenv("CB_RECOVERY_TIMEOUT", str(self.circuit_breaker_recovery_timeout)))
        self.circuit_breaker_success_threshold = int(os.getenv("CB_SUCCESS_THRESHOLD", str(self.circuit_breaker_success_threshold)))
        self.circuit_breaker_timeout = float(os.getenv("CB_TIMEOUT", str(self.circuit_breaker_timeout)))
        
        # Parse boolean configs
        self.enable_validation = os.getenv("INGESTION_ENABLE_VALIDATION", "true").lower() == "true"
        self.enable_events = os.getenv("INGESTION_ENABLE_EVENTS", "true").lower() == "true"
        
        self.log_level = os.getenv("INGESTION_LOG_LEVEL", self.log_level)
    
    def get_circuit_breaker_config(self) -> CircuitBreakerConfig:
        """Get circuit breaker configuration."""
        return CircuitBreakerConfig(
            failure_threshold=self.circuit_breaker_failure_threshold,
            recovery_timeout=self.circuit_breaker_recovery_timeout,
            success_threshold=self.circuit_breaker_success_threshold,
            timeout=self.circuit_breaker_timeout
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'api_provider': self.api_provider,
            'api_key': '***' if self.api_key else '',  # Mask API key
            'symbols': self.symbols,
            'days_to_fetch': self.days_to_fetch,
            'storage_base_path': self.storage_base_path,
            'circuit_breaker': {
                'failure_threshold': self.circuit_breaker_failure_threshold,
                'recovery_timeout': self.circuit_breaker_recovery_timeout,
                'success_threshold': self.circuit_breaker_success_threshold,
                'timeout': self.circuit_breaker_timeout
            },
            'enable_validation': self.enable_validation,
            'enable_events': self.enable_events,
            'log_level': self.log_level
        }


def load_config() -> IngestionServiceConfig:
    """Load configuration from environment variables and defaults."""
    return IngestionServiceConfig()