"""Data models for the ingestion service."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class ValidationStatus(Enum):
    """Status of data validation."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class PriceData:
    """Raw price data from financial APIs."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'adjusted_close': self.adjusted_close
        }


@dataclass
class DataBatch:
    """Batch of price data for multiple symbols."""
    data: List[PriceData]
    fetch_timestamp: datetime
    source: str
    
    def get_symbols(self) -> List[str]:
        """Get unique symbols in the batch."""
        return list(set(item.symbol for item in self.data))


@dataclass
class ValidationResult:
    """Result of data validation."""
    status: ValidationStatus
    passed_checks: int
    total_checks: int
    errors: List[str]
    warnings: List[str]
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.status == ValidationStatus.PASSED


@dataclass
class StorageResult:
    """Result of data storage operation."""
    success: bool
    files_written: List[str]
    total_records: int
    error_message: Optional[str] = None


@dataclass
class DataAvailableEvent:
    """Event published when new data is available."""
    symbols: List[str]
    date: datetime
    source: str
    file_paths: List[str]
    record_count: int