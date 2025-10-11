"""Data models for feature engineering."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np


class FeatureValidationStatus(Enum):
    """Status of feature validation."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class TechnicalFeatures:
    """Technical indicators computed from price data."""
    symbol: str
    timestamp: datetime
    
    # Returns
    returns: Dict[str, float] = field(default_factory=dict)  # simple, log, adjusted
    
    # Moving averages
    moving_averages: Dict[str, float] = field(default_factory=dict)  # sma_5, sma_20, ema_12, etc.
    
    # Momentum indicators
    momentum: Dict[str, float] = field(default_factory=dict)  # rsi, macd, stochastic
    
    # Volatility measures
    volatility: Dict[str, float] = field(default_factory=dict)  # realized_vol, garch_vol
    
    # Price-based features
    price_features: Dict[str, float] = field(default_factory=dict)  # high_low_ratio, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'returns': self.returns,
            'moving_averages': self.moving_averages,
            'momentum': self.momentum,
            'volatility': self.volatility,
            'price_features': self.price_features
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TechnicalFeatures':
        """Create from dictionary."""
        return cls(
            symbol=data['symbol'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            returns=data.get('returns', {}),
            moving_averages=data.get('moving_averages', {}),
            momentum=data.get('momentum', {}),
            volatility=data.get('volatility', {}),
            price_features=data.get('price_features', {})
        )


@dataclass
class TimeSeriesFeatures:
    """Time series features for ML training."""
    symbol: str
    target_date: datetime
    window_size: int
    
    # Feature sequences (time series)
    feature_sequences: Dict[str, List[float]] = field(default_factory=dict)
    
    # Target variable (next period return)
    target_return: Optional[float] = None
    
    # Metadata
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    missing_days: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'symbol': self.symbol,
            'target_date': self.target_date.isoformat(),
            'window_size': self.window_size,
            'feature_sequences': self.feature_sequences,
            'target_return': self.target_return,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'missing_days': self.missing_days
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeSeriesFeatures':
        """Create from dictionary."""
        return cls(
            symbol=data['symbol'],
            target_date=datetime.fromisoformat(data['target_date']),
            window_size=data['window_size'],
            feature_sequences=data.get('feature_sequences', {}),
            target_return=data.get('target_return'),
            start_date=datetime.fromisoformat(data['start_date']) if data.get('start_date') else None,
            end_date=datetime.fromisoformat(data['end_date']) if data.get('end_date') else None,
            missing_days=data.get('missing_days', 0)
        )


@dataclass
class FeatureValidationResult:
    """Result of feature validation."""
    status: FeatureValidationStatus
    passed_checks: int
    total_checks: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.status == FeatureValidationStatus.PASSED


@dataclass
class CacheResult:
    """Result of cache operation."""
    success: bool
    cache_key: str
    hit: bool = False
    error_message: Optional[str] = None
    cached_at: Optional[datetime] = None


@dataclass
class FeatureComputationConfig:
    """Configuration for feature computation."""
    
    # Moving average periods
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26])
    
    # RSI configuration
    rsi_period: int = 14
    
    # MACD configuration
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Volatility configuration
    volatility_window: int = 20
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Stochastic oscillator
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    
    # Time windows for ML features
    time_windows: List[int] = field(default_factory=lambda: [30, 60])
    
    # Validation thresholds
    max_missing_ratio: float = 0.1
    outlier_threshold: float = 5.0  # standard deviations
    min_data_points: int = 50