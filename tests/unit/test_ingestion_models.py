"""Unit tests for ingestion models."""

import pytest
from datetime import datetime
from data_pipeline.ingestion.models import (
    PriceData, DataBatch, ValidationResult, ValidationStatus,
    StorageResult, DataAvailableEvent
)


class TestPriceData:
    """Test PriceData model."""
    
    def test_price_data_creation(self):
        """Test PriceData creation."""
        timestamp = datetime(2024, 1, 1, 9, 30)
        price_data = PriceData(
            symbol="AAPL",
            timestamp=timestamp,
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
            adjusted_close=154.0
        )
        
        assert price_data.symbol == "AAPL"
        assert price_data.timestamp == timestamp
        assert price_data.open == 150.0
        assert price_data.high == 155.0
        assert price_data.low == 149.0
        assert price_data.close == 154.0
        assert price_data.volume == 1000000
        assert price_data.adjusted_close == 154.0
    
    def test_price_data_to_dict(self):
        """Test PriceData to_dict conversion."""
        timestamp = datetime(2024, 1, 1, 9, 30)
        price_data = PriceData(
            symbol="AAPL",
            timestamp=timestamp,
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
            adjusted_close=154.0
        )
        
        data_dict = price_data.to_dict()
        
        assert data_dict['symbol'] == "AAPL"
        assert data_dict['timestamp'] == timestamp.isoformat()
        assert data_dict['open'] == 150.0
        assert data_dict['high'] == 155.0
        assert data_dict['low'] == 149.0
        assert data_dict['close'] == 154.0
        assert data_dict['volume'] == 1000000
        assert data_dict['adjusted_close'] == 154.0


class TestDataBatch:
    """Test DataBatch model."""
    
    def test_data_batch_creation(self):
        """Test DataBatch creation."""
        timestamp = datetime(2024, 1, 1, 9, 30)
        fetch_timestamp = datetime.now()
        
        price_data = [
            PriceData("AAPL", timestamp, 150.0, 155.0, 149.0, 154.0, 1000000, 154.0),
            PriceData("GOOGL", timestamp, 2800.0, 2850.0, 2790.0, 2840.0, 500000, 2840.0)
        ]
        
        batch = DataBatch(
            data=price_data,
            fetch_timestamp=fetch_timestamp,
            source="yahoo_finance"
        )
        
        assert len(batch.data) == 2
        assert batch.fetch_timestamp == fetch_timestamp
        assert batch.source == "yahoo_finance"
    
    def test_get_symbols(self):
        """Test get_symbols method."""
        timestamp = datetime(2024, 1, 1, 9, 30)
        fetch_timestamp = datetime.now()
        
        price_data = [
            PriceData("AAPL", timestamp, 150.0, 155.0, 149.0, 154.0, 1000000, 154.0),
            PriceData("GOOGL", timestamp, 2800.0, 2850.0, 2790.0, 2840.0, 500000, 2840.0),
            PriceData("AAPL", timestamp, 151.0, 156.0, 150.0, 155.0, 1100000, 155.0)  # Duplicate symbol
        ]
        
        batch = DataBatch(
            data=price_data,
            fetch_timestamp=fetch_timestamp,
            source="yahoo_finance"
        )
        
        symbols = batch.get_symbols()
        assert set(symbols) == {"AAPL", "GOOGL"}
        assert len(symbols) == 2


class TestValidationResult:
    """Test ValidationResult model."""
    
    def test_validation_result_passed(self):
        """Test ValidationResult with passed status."""
        result = ValidationResult(
            status=ValidationStatus.PASSED,
            passed_checks=5,
            total_checks=5,
            errors=[],
            warnings=[]
        )
        
        assert result.status == ValidationStatus.PASSED
        assert result.is_valid is True
        assert result.passed_checks == 5
        assert result.total_checks == 5
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_validation_result_failed(self):
        """Test ValidationResult with failed status."""
        result = ValidationResult(
            status=ValidationStatus.FAILED,
            passed_checks=3,
            total_checks=5,
            errors=["Price validation failed", "Completeness check failed"],
            warnings=[]
        )
        
        assert result.status == ValidationStatus.FAILED
        assert result.is_valid is False
        assert result.passed_checks == 3
        assert result.total_checks == 5
        assert len(result.errors) == 2
        assert len(result.warnings) == 0
    
    def test_validation_result_warning(self):
        """Test ValidationResult with warning status."""
        result = ValidationResult(
            status=ValidationStatus.WARNING,
            passed_checks=4,
            total_checks=5,
            errors=[],
            warnings=["Potential outliers detected"]
        )
        
        assert result.status == ValidationStatus.WARNING
        assert result.is_valid is True  # Warnings still count as valid
        assert result.passed_checks == 4
        assert result.total_checks == 5
        assert len(result.errors) == 0
        assert len(result.warnings) == 1


class TestStorageResult:
    """Test StorageResult model."""
    
    def test_storage_result_success(self):
        """Test successful StorageResult."""
        result = StorageResult(
            success=True,
            files_written=["data/raw/daily/2024/01/01/AAPL.parquet"],
            total_records=100
        )
        
        assert result.success is True
        assert len(result.files_written) == 1
        assert result.total_records == 100
        assert result.error_message is None
    
    def test_storage_result_failure(self):
        """Test failed StorageResult."""
        result = StorageResult(
            success=False,
            files_written=[],
            total_records=0,
            error_message="Disk full"
        )
        
        assert result.success is False
        assert len(result.files_written) == 0
        assert result.total_records == 0
        assert result.error_message == "Disk full"


class TestDataAvailableEvent:
    """Test DataAvailableEvent model."""
    
    def test_data_available_event_creation(self):
        """Test DataAvailableEvent creation."""
        date = datetime(2024, 1, 1)
        event = DataAvailableEvent(
            symbols=["AAPL", "GOOGL"],
            date=date,
            source="yahoo_finance",
            file_paths=["data/raw/daily/2024/01/01/AAPL.parquet"],
            record_count=100
        )
        
        assert event.symbols == ["AAPL", "GOOGL"]
        assert event.date == date
        assert event.source == "yahoo_finance"
        assert len(event.file_paths) == 1
        assert event.record_count == 100