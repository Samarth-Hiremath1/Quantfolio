"""Unit tests for data validator."""

import pytest
from datetime import datetime, timedelta
from data_pipeline.ingestion.validator import (
    DataValidator, CompletenessRule, PriceValidityRule, OutlierDetectionRule,
    FreshnessRule, DuplicateDetectionRule, ValidationRule
)
from data_pipeline.ingestion.models import PriceData, DataBatch, ValidationStatus


class TestValidationRules:
    """Test individual validation rules."""
    
    def test_completeness_rule_pass(self):
        """Test completeness rule with sufficient data."""
        rule = CompletenessRule(min_records_per_symbol=2)
        
        data = [
            PriceData("AAPL", datetime(2024, 1, 1), 150.0, 155.0, 149.0, 154.0, 1000000, 154.0),
            PriceData("AAPL", datetime(2024, 1, 2), 154.0, 158.0, 153.0, 157.0, 1100000, 157.0),
            PriceData("GOOGL", datetime(2024, 1, 1), 2800.0, 2850.0, 2790.0, 2840.0, 500000, 2840.0),
            PriceData("GOOGL", datetime(2024, 1, 2), 2840.0, 2880.0, 2830.0, 2870.0, 550000, 2870.0)
        ]
        
        is_valid, message = rule.validate(data)
        
        assert is_valid is True
        assert "All 2 symbols have sufficient data" in message
    
    def test_completeness_rule_fail(self):
        """Test completeness rule with insufficient data."""
        rule = CompletenessRule(min_records_per_symbol=2)
        
        data = [
            PriceData("AAPL", datetime(2024, 1, 1), 150.0, 155.0, 149.0, 154.0, 1000000, 154.0),
            PriceData("GOOGL", datetime(2024, 1, 1), 2800.0, 2850.0, 2790.0, 2840.0, 500000, 2840.0)
        ]
        
        is_valid, message = rule.validate(data)
        
        assert is_valid is False
        assert "Insufficient data for symbols" in message
    
    def test_completeness_rule_empty_data(self):
        """Test completeness rule with empty data."""
        rule = CompletenessRule()
        
        is_valid, message = rule.validate([])
        
        assert is_valid is False
        assert message == "No data provided"
    
    def test_price_validity_rule_pass(self):
        """Test price validity rule with valid data."""
        rule = PriceValidityRule()
        
        data = [
            PriceData("AAPL", datetime(2024, 1, 1), 150.0, 155.0, 149.0, 154.0, 1000000, 154.0),
            PriceData("GOOGL", datetime(2024, 1, 1), 2800.0, 2850.0, 2790.0, 2840.0, 500000, 2840.0)
        ]
        
        is_valid, message = rule.validate(data)
        
        assert is_valid is True
        assert message == "All price data is valid"
    
    def test_price_validity_rule_negative_price(self):
        """Test price validity rule with negative prices."""
        rule = PriceValidityRule()
        
        data = [
            PriceData("AAPL", datetime(2024, 1, 1), -150.0, 155.0, 149.0, 154.0, 1000000, 154.0)
        ]
        
        is_valid, message = rule.validate(data)
        
        assert is_valid is False
        assert "negative price" in message
    
    def test_price_validity_rule_invalid_ohlc(self):
        """Test price validity rule with invalid OHLC relationship."""
        rule = PriceValidityRule()
        
        data = [
            PriceData("AAPL", datetime(2024, 1, 1), 150.0, 155.0, 160.0, 154.0, 1000000, 154.0)  # Low > High
        ]
        
        is_valid, message = rule.validate(data)
        
        assert is_valid is False
        assert "invalid OHLC relationship" in message
    
    def test_freshness_rule_pass(self):
        """Test freshness rule with fresh data."""
        rule = FreshnessRule(max_age_hours=24)
        
        recent_time = datetime.now() - timedelta(hours=1)
        data = [
            PriceData("AAPL", recent_time, 150.0, 155.0, 149.0, 154.0, 1000000, 154.0)
        ]
        
        is_valid, message = rule.validate(data)
        
        assert is_valid is True
        assert "All data is fresh" in message
    
    def test_freshness_rule_fail(self):
        """Test freshness rule with stale data."""
        rule = FreshnessRule(max_age_hours=24)
        
        old_time = datetime.now() - timedelta(hours=48)
        data = [
            PriceData("AAPL", old_time, 150.0, 155.0, 149.0, 154.0, 1000000, 154.0)
        ]
        
        is_valid, message = rule.validate(data)
        
        assert is_valid is False
        assert "Stale data detected" in message
    
    def test_duplicate_detection_rule_pass(self):
        """Test duplicate detection rule with no duplicates."""
        rule = DuplicateDetectionRule()
        
        data = [
            PriceData("AAPL", datetime(2024, 1, 1), 150.0, 155.0, 149.0, 154.0, 1000000, 154.0),
            PriceData("AAPL", datetime(2024, 1, 2), 154.0, 158.0, 153.0, 157.0, 1100000, 157.0),
            PriceData("GOOGL", datetime(2024, 1, 1), 2800.0, 2850.0, 2790.0, 2840.0, 500000, 2840.0)
        ]
        
        is_valid, message = rule.validate(data)
        
        assert is_valid is True
        assert message == "No duplicate records found"
    
    def test_duplicate_detection_rule_fail(self):
        """Test duplicate detection rule with duplicates."""
        rule = DuplicateDetectionRule()
        
        data = [
            PriceData("AAPL", datetime(2024, 1, 1), 150.0, 155.0, 149.0, 154.0, 1000000, 154.0),
            PriceData("AAPL", datetime(2024, 1, 1), 151.0, 156.0, 150.0, 155.0, 1100000, 155.0)  # Same date
        ]
        
        is_valid, message = rule.validate(data)
        
        assert is_valid is False
        assert "Duplicate records found" in message
    
    def test_outlier_detection_rule_insufficient_data(self):
        """Test outlier detection with insufficient data."""
        rule = OutlierDetectionRule()
        
        data = [
            PriceData("AAPL", datetime(2024, 1, 1), 150.0, 155.0, 149.0, 154.0, 1000000, 154.0)
        ]
        
        is_valid, message = rule.validate(data)
        
        assert is_valid is True
        assert message == "Insufficient data for outlier detection"
    
    def test_outlier_detection_rule_normal_data(self):
        """Test outlier detection with normal data."""
        rule = OutlierDetectionRule(z_score_threshold=2.0)
        
        # Create data with normal price movements
        base_price = 100.0
        data = []
        for i in range(10):
            price = base_price + i * 0.5  # Small incremental changes
            data.append(PriceData(
                "AAPL", 
                datetime(2024, 1, i + 1), 
                price, price + 1, price - 1, price + 0.5, 
                1000000, price + 0.5
            ))
        
        is_valid, message = rule.validate(data)
        
        assert is_valid is True
        assert "No significant outliers detected" in message


class TestDataValidator:
    """Test DataValidator class."""
    
    def test_validator_initialization(self):
        """Test validator initialization with default rules."""
        validator = DataValidator()
        
        assert len(validator.rules) == 5  # Default rules
        rule_names = [rule.name for rule in validator.rules]
        expected_names = [
            "completeness_check", "price_validity", "freshness_check",
            "duplicate_detection", "outlier_detection"
        ]
        
        for name in expected_names:
            assert name in rule_names
    
    def test_add_custom_rule(self):
        """Test adding custom validation rule."""
        validator = DataValidator()
        initial_count = len(validator.rules)
        
        class CustomRule(ValidationRule):
            def __init__(self):
                super().__init__("custom_rule")
            
            def validate(self, data):
                return True, "Custom validation passed"
        
        custom_rule = CustomRule()
        validator.add_rule(custom_rule)
        
        assert len(validator.rules) == initial_count + 1
        assert validator.rules[-1].name == "custom_rule"
    
    def test_validate_batch_success(self):
        """Test successful batch validation."""
        validator = DataValidator()
        
        recent_time = datetime.now() - timedelta(hours=1)
        data = [
            PriceData("AAPL", recent_time, 150.0, 155.0, 149.0, 154.0, 1000000, 154.0),
            PriceData("GOOGL", recent_time, 2800.0, 2850.0, 2790.0, 2840.0, 500000, 2840.0)
        ]
        
        batch = DataBatch(
            data=data,
            fetch_timestamp=datetime.now(),
            source="test"
        )
        
        result = validator.validate_batch(batch)
        
        assert result.status == ValidationStatus.PASSED
        assert result.is_valid is True
        assert result.passed_checks == len(validator.rules)
        assert result.total_checks == len(validator.rules)
        assert len(result.errors) == 0
    
    def test_validate_batch_with_errors(self):
        """Test batch validation with errors."""
        validator = DataValidator()
        
        # Create data with validation errors
        data = [
            PriceData("AAPL", datetime(2024, 1, 1), -150.0, 155.0, 149.0, 154.0, 1000000, 154.0)  # Negative price
        ]
        
        batch = DataBatch(
            data=data,
            fetch_timestamp=datetime.now(),
            source="test"
        )
        
        result = validator.validate_batch(batch)
        
        assert result.status == ValidationStatus.FAILED
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert result.passed_checks < result.total_checks
    
    def test_validate_batch_with_warnings(self):
        """Test batch validation with warnings only."""
        validator = DataValidator()
        
        # Create data that might trigger warnings but not errors
        recent_time = datetime.now() - timedelta(hours=1)
        data = [
            PriceData("AAPL", recent_time, 150.0, 155.0, 149.0, 154.0, 0, 154.0)  # Zero volume
        ]
        
        batch = DataBatch(
            data=data,
            fetch_timestamp=datetime.now(),
            source="test"
        )
        
        result = validator.validate_batch(batch)
        
        # This might pass or have warnings depending on the specific validation logic
        assert result.status in [ValidationStatus.PASSED, ValidationStatus.WARNING]
        assert result.is_valid is True  # Warnings don't make data invalid
    
    def test_validate_records(self):
        """Test validate_records method."""
        validator = DataValidator()
        
        recent_time = datetime.now() - timedelta(hours=1)
        records = [
            PriceData("AAPL", recent_time, 150.0, 155.0, 149.0, 154.0, 1000000, 154.0)
        ]
        
        result = validator.validate_records(records)
        
        assert isinstance(result.status, ValidationStatus)
        assert isinstance(result.passed_checks, int)
        assert isinstance(result.total_checks, int)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)