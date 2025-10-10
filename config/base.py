"""
Base configuration management system for QuantFolio ML Pipeline.
Supports environment-specific configurations with validation.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "quantfolio"
    username: str = "postgres"
    password: str = "password"
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class APIConfig:
    """External API configuration"""
    yahoo_finance_base_url: str = "https://query1.finance.yahoo.com"
    alpha_vantage_base_url: str = "https://www.alphavantage.co"
    alpha_vantage_api_key: str = ""
    rate_limit_requests_per_minute: int = 60
    timeout_seconds: int = 30
    retry_attempts: int = 3
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60


@dataclass
class MLConfig:
    """Machine Learning configuration"""
    pytorch_device: str = "cpu"
    tensorflow_device: str = "/CPU:0"
    model_cache_size: int = 100
    training_batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    max_epochs: int = 100
    learning_rate: float = 0.001


@dataclass
class MLflowConfig:
    """MLflow configuration"""
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "quantfolio-experiments"
    model_registry_uri: str = "http://localhost:5000"
    artifact_root: str = "./mlruns"
    default_artifact_root: str = "./artifacts"


@dataclass
class StorageConfig:
    """Storage configuration"""
    data_root: str = "./data"
    raw_data_path: str = "./data/raw"
    processed_data_path: str = "./data/processed"
    features_path: str = "./data/features"
    models_path: str = "./data/models"
    portfolios_path: str = "./data/portfolios"
    backup_enabled: bool = True
    backup_retention_days: int = 30


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    prometheus_port: int = 9090
    grafana_port: int = 3000
    metrics_collection_interval: int = 30
    log_level: str = "INFO"
    enable_distributed_tracing: bool = True
    alert_webhook_url: str = ""


@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_authentication: bool = False
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    api_key_header: str = "X-API-Key"
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class BaseConfig:
    """Base configuration class containing all service configurations"""
    environment: Environment = Environment.LOCAL
    debug: bool = True
    service_name: str = "quantfolio"
    version: str = "1.0.0"
    
    # Service configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)


class ConfigManager:
    """Configuration manager with environment-specific loading"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._config: Optional[BaseConfig] = None
        self._environment = self._get_environment()
    
    def _get_environment(self) -> Environment:
        """Get current environment from environment variable"""
        env_name = os.getenv("ENVIRONMENT", "local").lower()
        try:
            return Environment(env_name)
        except ValueError:
            return Environment.LOCAL
    
    def load_config(self) -> BaseConfig:
        """Load configuration based on current environment"""
        if self._config is None:
            self._config = self._load_environment_config()
        return self._config
    
    def _load_environment_config(self) -> BaseConfig:
        """Load configuration for specific environment"""
        # Start with base configuration
        config = BaseConfig()
        config.environment = self._environment
        
        # Load base configuration file
        base_config_path = self.config_dir / "base.yaml"
        if base_config_path.exists():
            base_data = self._load_yaml_file(base_config_path)
            config = self._merge_config(config, base_data)
        
        # Load environment-specific configuration
        env_config_path = self.config_dir / f"{self._environment.value}.yaml"
        if env_config_path.exists():
            env_data = self._load_yaml_file(env_config_path)
            config = self._merge_config(config, env_data)
        
        # Override with environment variables
        config = self._apply_env_overrides(config)
        
        return config
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load config file {file_path}: {e}")
            return {}
    
    def _merge_config(self, config: BaseConfig, data: Dict[str, Any]) -> BaseConfig:
        """Merge configuration data into config object"""
        for section_name, section_data in data.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section_obj = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
            elif hasattr(config, section_name):
                setattr(config, section_name, section_data)
        
        return config
    
    def _apply_env_overrides(self, config: BaseConfig) -> BaseConfig:
        """Apply environment variable overrides"""
        # Common environment variable mappings
        env_mappings = {
            "LOG_LEVEL": ("monitoring", "log_level"),
            "DEBUG": ("debug",),
            "SERVICE_NAME": ("service_name",),
            "DATABASE_HOST": ("database", "host"),
            "DATABASE_PORT": ("database", "port"),
            "DATABASE_NAME": ("database", "database"),
            "DATABASE_USER": ("database", "username"),
            "DATABASE_PASSWORD": ("database", "password"),
            "ALPHA_VANTAGE_API_KEY": ("api", "alpha_vantage_api_key"),
            "MLFLOW_TRACKING_URI": ("mlflow", "tracking_uri"),
            "PYTORCH_DEVICE": ("ml", "pytorch_device"),
            "TENSORFLOW_DEVICE": ("ml", "tensorflow_device"),
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_config(config, config_path, env_value)
        
        return config
    
    def _set_nested_config(self, config: BaseConfig, path: tuple, value: str):
        """Set nested configuration value"""
        obj = config
        for key in path[:-1]:
            obj = getattr(obj, key)
        
        # Convert string values to appropriate types
        final_key = path[-1]
        if hasattr(obj, final_key):
            current_value = getattr(obj, final_key)
            if isinstance(current_value, bool):
                value = value.lower() in ('true', '1', 'yes', 'on')
            elif isinstance(current_value, int):
                value = int(value)
            elif isinstance(current_value, float):
                value = float(value)
            
            setattr(obj, final_key, value)


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> BaseConfig:
    """Get the current configuration"""
    return config_manager.load_config()


def reload_config() -> BaseConfig:
    """Reload configuration from files"""
    config_manager._config = None
    return config_manager.load_config()