"""Data validation pipeline with quality checks."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Callable
from statistics import mean, stdev
import math

from .models import DataBatch, PriceData, ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


class ValidationRule:
    """Base class for validation rules."""
    
    def __init__(self, name: str, severity: str = "error"):
        self.name = name
        self.severity = severity  # "error" or "warning"
    
    def validate(self, data: List[PriceData]) -> tuple[bool, str]:
        """Validate data and return (is_valid, message)."""
        raise NotImplementedError


class CompletenessRule(ValidationRule):
    """Check data completeness."""
    
    def __init__(self, min_records_per_symbol: int = 1):
        super().__init__("completeness_check")
        self.min_records_per_symbol = min_records_per_symbol
    
    def validate(self, data: List[PriceData]) -> tuple[bool, str]:
        """Check if we have minimum required records per symbol."""
        if not data:
            return False, "No data provided"
        
        symbol_counts = {}
        for record in data:
            symbol_counts[record.symbol] = symbol_counts.get(record.symbol, 0) + 1
        
        insufficient_symbols = [
            symbol for symbol, count in symbol_counts.items() 
            if count < self.min_records_per_symbol
        ]
        
        if insufficient_symbols:
            return False, f"Insufficient data for symbols: {insufficient_symbols}"
        
        return True, f"All {len(symbol_counts)} symbols have sufficient data"


class PriceValidityRule(ValidationRule):
    """Check price data validity."""
    
    def validate(self, data: List[PriceData]) -> tuple[bool, str]:
        """Check if price data is valid."""
        invalid_records = []
        
        for record in data:
            # Check for negative prices
            if any(price < 0 for price in [record.open, record.high, record.low, record.close, record.adjusted_close]):
                invalid_records.append(f"{record.symbol}@{record.timestamp}: negative price")
                continue
            
            # Check OHLC relationships
            if not (record.low <= record.open <= record.high and 
                   record.low <= record.close <= record.high):
                invalid_records.append(f"{record.symbol}@{record.timestamp}: invalid OHLC relationship")
                continue
            
            # Check for zero volume (warning only)
            if record.volume == 0:
                invalid_records.append(f"{record.symbol}@{record.timestamp}: zero volume")
        
        if invalid_records:
            return False, f"Invalid price data: {invalid_records[:5]}"  # Show first 5
        
        return True, "All price data is valid"


class OutlierDetectionRule(ValidationRule):
    """Detect price outliers using statistical methods."""
    
    def __init__(self, z_score_threshold: float = 3.0):
        super().__init__("outlier_detection", severity="warning")
        self.z_score_threshold = z_score_threshold
    
    def validate(self, data: List[PriceData]) -> tuple[bool, str]:
        """Detect outliers in price data."""
        if len(data) < 3:
            return True, "Insufficient data for outlier detection"
        
        outliers = []
        
        # Group by symbol for outlier detection
        symbol_data = {}
        for record in data:
            if record.symbol not in symbol_data:
                symbol_data[record.symbol] = []
            symbol_data[record.symbol].append(record)
        
        for symbol, records in symbol_data.items():
            if len(records) < 3:
                continue
            
            # Calculate daily returns
            sorted_records = sorted(records, key=lambda x: x.timestamp)
            returns = []
            for i in range(1, len(sorted_records)):
                prev_close = sorted_records[i-1].close
                curr_close = sorted_records[i].close
                if prev_close > 0:
                    returns.append((curr_close - prev_close) / prev_close)
            
            if len(returns) < 2:
                continue
            
            # Calculate z-scores
            mean_return = mean(returns)
            std_return = stdev(returns) if len(returns) > 1 else 0
            
            if std_return > 0:
                for i, ret in enumerate(returns):
                    z_score = abs((ret - mean_return) / std_return)
                    if z_score > self.z_score_threshold:
                        outliers.append(f"{symbol}: return {ret:.4f} (z-score: {z_score:.2f})")
        
        if outliers:
            return False, f"Potential outliers detected: {outliers[:3]}"
        
        return True, "No significant outliers detected"


class FreshnessRule(ValidationRule):
    """Check data freshness."""
    
    def __init__(self, max_age_hours: int = 48):
        super().__init__("freshness_check")
        self.max_age_hours = max_age_hours
    
    def validate(self, data: List[PriceData]) -> tuple[bool, str]:
        """Check if data is fresh enough."""
        if not data:
            return False, "No data to check freshness"
        
        now = datetime.now()
        cutoff_time = now - timedelta(hours=self.max_age_hours)
        
        stale_records = [
            record for record in data 
            if record.timestamp < cutoff_time
        ]
        
        if stale_records:
            oldest = min(stale_records, key=lambda x: x.timestamp)
            age_hours = (now - oldest.timestamp).total_seconds() / 3600
            return False, f"Stale data detected. Oldest record is {age_hours:.1f} hours old"
        
        return True, f"All data is fresh (within {self.max_age_hours} hours)"


class DuplicateDetectionRule(ValidationRule):
    """Detect duplicate records."""
    
    def validate(self, data: List[PriceData]) -> tuple[bool, str]:
        """Check for duplicate records."""
        seen = set()
        duplicates = []
        
        for record in data:
            key = (record.symbol, record.timestamp.date())
            if key in seen:
                duplicates.append(f"{record.symbol}@{record.timestamp.date()}")
            else:
                seen.add(key)
        
        if duplicates:
            return False, f"Duplicate records found: {duplicates[:5]}"
        
        return True, "No duplicate records found"


class DataValidator:
    """Data validation pipeline with configurable rules."""
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default validation rules."""
        self.rules = [
            CompletenessRule(min_records_per_symbol=1),
            PriceValidityRule(),
            FreshnessRule(max_age_hours=48),
            DuplicateDetectionRule(),
            OutlierDetectionRule(z_score_threshold=3.0)
        ]
    
    def add_rule(self, rule: ValidationRule):
        """Add custom validation rule."""
        self.rules.append(rule)
    
    def validate_batch(self, batch: DataBatch) -> ValidationResult:
        """Validate a data batch."""
        logger.info(f"Validating batch with {len(batch.data)} records from {batch.source}")
        
        errors = []
        warnings = []
        passed_checks = 0
        total_checks = len(self.rules)
        
        for rule in self.rules:
            try:
                is_valid, message = rule.validate(batch.data)
                
                if is_valid:
                    passed_checks += 1
                    logger.debug(f"✓ {rule.name}: {message}")
                else:
                    if rule.severity == "error":
                        errors.append(f"{rule.name}: {message}")
                        logger.error(f"✗ {rule.name}: {message}")
                    else:
                        warnings.append(f"{rule.name}: {message}")
                        logger.warning(f"⚠ {rule.name}: {message}")
                        passed_checks += 1  # Warnings don't fail validation
                        
            except Exception as e:
                errors.append(f"{rule.name}: Validation failed with error: {e}")
                logger.error(f"✗ {rule.name}: Validation failed with error: {e}")
        
        # Determine overall status
        if errors:
            status = ValidationStatus.FAILED
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASSED
        
        result = ValidationResult(
            status=status,
            passed_checks=passed_checks,
            total_checks=total_checks,
            errors=errors,
            warnings=warnings
        )
        
        logger.info(f"Validation complete: {status.value} ({passed_checks}/{total_checks} checks passed)")
        return result
    
    def validate_records(self, records: List[PriceData]) -> ValidationResult:
        """Validate individual records."""
        # Create a temporary batch for validation
        batch = DataBatch(
            data=records,
            fetch_timestamp=datetime.now(),
            source="validation"
        )
        return self.validate_batch(batch)