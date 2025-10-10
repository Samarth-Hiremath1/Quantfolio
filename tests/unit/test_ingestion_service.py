"""Unit tests for data ingestion service."""

import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from data_pipeline.ingestion.service import DataIngestionService, IngestionConfig
from data_pipeline.ingestion.models import DataBatch, PriceData, ValidationResult, ValidationStatus, StorageResult
from data_pipeline.ingestion.circuit_breaker import CircuitBreakerConfig


class TestIngestionConfig:
    """Test IngestionConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = IngestionConfig()
        
        assert config.api_provider == "yahoo"
        assert config.api_key == ""
        assert config.symbols == ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        assert config.days_to_fetch == 1
        assert config.storage_base_path == "data"
        assert config.enable_validation is True
        assert config.enable_events is True
        assert isinstance(config.circuit_breaker_config, CircuitBreakerConfig)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        cb_config = CircuitBreakerConfig(failure_threshold=2)
        config = IngestionConfig(
            api_provider="alpha_vantage",
            api_key="test_key",
            symbols=["AAPL", "GOOGL"],
            days_to_fetch=5,
            storage_base_path="/tmp/data",
            circuit_breaker_config=cb_config,
            enable_validation=False,
            enable_events=False
        )
        
        assert config.api_provider == "alpha_vantage"
        assert config.api_key == "test_key"
        assert config.symbols == ["AAPL", "GOOGL"]
        assert config.days_to_fetch == 5
        assert config.storage_base_path == "/tmp/data"
        assert config.circuit_breaker_config is cb_config
        assert config.enable_validation is False
        assert config.enable_events is False


class TestDataIngestionService:
    """Test DataIngestionService."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return IngestionConfig(
            api_provider="yahoo",
            api_key="test_key",
            symbols=["AAPL", "GOOGL"],
            days_to_fetch=1,
            storage_base_path=temp_dir
        )
    
    @pytest.fixture
    def service(self, config):
        """Create data ingestion service."""
        return DataIngestionService(config)
    
    @pytest.fixture
    def sample_batch(self):
        """Create sample data batch."""
        data = [
            PriceData("AAPL", datetime(2024, 1, 1), 150.0, 155.0, 149.0, 154.0, 1000000, 154.0),
            PriceData("GOOGL", datetime(2024, 1, 1), 2800.0, 2850.0, 2790.0, 2840.0, 500000, 2840.0)
        ]
        return DataBatch(
            data=data,
            fetch_timestamp=datetime.now(),
            source="test_source"
        )
    
    def test_service_initialization(self, service, config):
        """Test service initialization."""
        assert service.config is config
        assert service.api_client is None
        assert service.validator is not None  # Validation enabled by default
        assert service.storage is not None
        assert service.event_bus is not None  # Events enabled by default
        assert service._running is False
    
    def test_service_initialization_disabled_features(self, temp_dir):
        """Test service initialization with disabled features."""
        config = IngestionConfig(
            storage_base_path=temp_dir,
            enable_validation=False,
            enable_events=False
        )
        service = DataIngestionService(config)
        
        assert service.validator is None
        assert service.event_bus is None
    
    @pytest.mark.asyncio
    async def test_context_manager(self, service):
        """Test service as async context manager."""
        with patch.object(service, 'initialize', new_callable=AsyncMock) as mock_init:
            with patch.object(service, 'cleanup', new_callable=AsyncMock) as mock_cleanup:
                async with service:
                    mock_init.assert_called_once()
                    assert service._running is True
                
                mock_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize(self, service):
        """Test service initialization."""
        with patch('data_pipeline.ingestion.service.APIClientFactory.create_client') as mock_factory:
            mock_client = AsyncMock()
            mock_factory.return_value = mock_client
            
            await service.initialize()
            
            assert service._running is True
            assert service.api_client is mock_client
            mock_client.__aenter__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup(self, service):
        """Test service cleanup."""
        # Setup service as if initialized
        service._running = True
        service.api_client = AsyncMock()
        service.event_bus = AsyncMock()
        
        await service.cleanup()
        
        assert service._running is False
        service.api_client.__aexit__.assert_called_once()
        service.event_bus.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fetch_daily_data(self, service, sample_batch):
        """Test fetching daily data."""
        service._running = True
        service.api_client = AsyncMock()
        service.api_client.fetch_daily_data.return_value = sample_batch
        
        result = await service.fetch_daily_data(["AAPL"], 1)
        
        assert result is sample_batch
        service.api_client.fetch_daily_data.assert_called_once_with(["AAPL"], 1)
    
    @pytest.mark.asyncio
    async def test_fetch_daily_data_not_initialized(self, service):
        """Test fetching data when service not initialized."""
        with pytest.raises(RuntimeError, match="Service not initialized"):
            await service.fetch_daily_data()
    
    @pytest.mark.asyncio
    async def test_fetch_daily_data_default_params(self, service, sample_batch):
        """Test fetching data with default parameters."""
        service._running = True
        service.api_client = AsyncMock()
        service.api_client.fetch_daily_data.return_value = sample_batch
        
        await service.fetch_daily_data()
        
        # Should use config defaults
        service.api_client.fetch_daily_data.assert_called_once_with(
            service.config.symbols, service.config.days_to_fetch
        )
    
    @pytest.mark.asyncio
    async def test_validate_data_quality(self, service, sample_batch):
        """Test data quality validation."""
        mock_result = ValidationResult(
            status=ValidationStatus.PASSED,
            passed_checks=5,
            total_checks=5,
            errors=[],
            warnings=[]
        )
        service.validator = MagicMock()
        service.validator.validate_batch.return_value = mock_result
        
        result = await service.validate_data_quality(sample_batch)
        
        assert result is mock_result
        service.validator.validate_batch.assert_called_once_with(sample_batch)
    
    @pytest.mark.asyncio
    async def test_validate_data_quality_disabled(self, temp_dir):
        """Test validation when disabled."""
        config = IngestionConfig(storage_base_path=temp_dir, enable_validation=False)
        service = DataIngestionService(config)
        
        result = await service.validate_data_quality(sample_batch)
        
        assert result.status == "passed"
        assert result.passed_checks == 0
        assert result.total_checks == 0
    
    @pytest.mark.asyncio
    async def test_store_data(self, service, sample_batch):
        """Test data storage."""
        mock_result = StorageResult(
            success=True,
            files_written=["test_file.parquet"],
            total_records=2
        )
        service.storage = MagicMock()
        service.storage.store_batch.return_value = mock_result
        
        result = await service.store_data(sample_batch)
        
        assert result is mock_result
        service.storage.store_batch.assert_called_once_with(sample_batch)
    
    @pytest.mark.asyncio
    async def test_store_data_exception(self, service, sample_batch):
        """Test storage with exception."""
        service.storage = MagicMock()
        service.storage.store_batch.side_effect = Exception("Storage error")
        
        result = await service.store_data(sample_batch)
        
        assert result.success is False
        assert result.error_message == "Storage error"
        assert result.total_records == 0
    
    @pytest.mark.asyncio
    async def test_publish_data_event(self, service, sample_batch):
        """Test publishing data event."""
        storage_result = StorageResult(
            success=True,
            files_written=["test_file.parquet"],
            total_records=2
        )
        
        service.event_bus = AsyncMock()
        service.event_bus.publish_data_available.return_value = True
        
        result = await service.publish_data_event(sample_batch, storage_result)
        
        assert result is True
        service.event_bus.publish_data_available.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_publish_data_event_disabled(self, temp_dir, sample_batch):
        """Test publishing when events disabled."""
        config = IngestionConfig(storage_base_path=temp_dir, enable_events=False)
        service = DataIngestionService(config)
        
        storage_result = StorageResult(success=True, files_written=[], total_records=0)
        result = await service.publish_data_event(sample_batch, storage_result)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_publish_data_event_storage_failed(self, service, sample_batch):
        """Test publishing when storage failed."""
        storage_result = StorageResult(success=False, files_written=[], total_records=0)
        
        result = await service.publish_data_event(sample_batch, storage_result)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_run_ingestion_pipeline_success(self, service, sample_batch):
        """Test successful ingestion pipeline."""
        # Mock all dependencies
        service._running = True
        service.api_client = AsyncMock()
        service.api_client.fetch_daily_data.return_value = sample_batch
        
        service.validator = MagicMock()
        service.validator.validate_batch.return_value = ValidationResult(
            status=ValidationStatus.PASSED,
            passed_checks=5,
            total_checks=5,
            errors=[],
            warnings=[]
        )
        
        service.storage = MagicMock()
        service.storage.store_batch.return_value = StorageResult(
            success=True,
            files_written=["test_file.parquet"],
            total_records=2
        )
        
        service.event_bus = AsyncMock()
        service.event_bus.publish_data_available.return_value = True
        
        result = await service.run_ingestion_pipeline()
        
        assert result['success'] is True
        assert result['records_fetched'] == 2
        assert result['records_stored'] == 2
        assert result['event_published'] is True
        assert len(result['errors']) == 0
        assert 'start_time' in result
        assert 'end_time' in result
        assert 'duration_seconds' in result
    
    @pytest.mark.asyncio
    async def test_run_ingestion_pipeline_no_data(self, service):
        """Test pipeline when no data is fetched."""
        service._running = True
        service.api_client = AsyncMock()
        
        empty_batch = DataBatch(data=[], fetch_timestamp=datetime.now(), source="test")
        service.api_client.fetch_daily_data.return_value = empty_batch
        
        result = await service.run_ingestion_pipeline()
        
        assert result['success'] is False
        assert result['records_fetched'] == 0
        assert "No data fetched" in result['errors']
    
    @pytest.mark.asyncio
    async def test_run_ingestion_pipeline_validation_error(self, service, sample_batch):
        """Test pipeline with validation errors."""
        service._running = True
        service.api_client = AsyncMock()
        service.api_client.fetch_daily_data.return_value = sample_batch
        
        service.validator = MagicMock()
        service.validator.validate_batch.return_value = ValidationResult(
            status=ValidationStatus.FAILED,
            passed_checks=3,
            total_checks=5,
            errors=["Validation failed"],
            warnings=[]
        )
        
        result = await service.run_ingestion_pipeline()
        
        assert result['success'] is False
        assert "Validation failed" in result['errors']
    
    @pytest.mark.asyncio
    async def test_run_ingestion_pipeline_storage_error(self, service, sample_batch):
        """Test pipeline with storage errors."""
        service._running = True
        service.api_client = AsyncMock()
        service.api_client.fetch_daily_data.return_value = sample_batch
        
        service.validator = MagicMock()
        service.validator.validate_batch.return_value = ValidationResult(
            status=ValidationStatus.PASSED,
            passed_checks=5,
            total_checks=5,
            errors=[],
            warnings=[]
        )
        
        service.storage = MagicMock()
        service.storage.store_batch.return_value = StorageResult(
            success=False,
            files_written=[],
            total_records=0,
            error_message="Storage failed"
        )
        
        result = await service.run_ingestion_pipeline()
        
        assert result['success'] is False
        assert "Storage failed: Storage failed" in result['errors']
    
    @pytest.mark.asyncio
    async def test_run_ingestion_pipeline_exception(self, service):
        """Test pipeline with unexpected exception."""
        service._running = True
        service.api_client = AsyncMock()
        service.api_client.fetch_daily_data.side_effect = Exception("Unexpected error")
        
        result = await service.run_ingestion_pipeline()
        
        assert result['success'] is False
        assert "Unexpected error" in result['errors']
    
    def test_get_circuit_breaker_stats(self, service):
        """Test getting circuit breaker stats."""
        mock_client = MagicMock()
        mock_client.circuit_breaker.get_stats.return_value = {"state": "closed"}
        service.api_client = mock_client
        
        stats = service.get_circuit_breaker_stats()
        
        assert stats == {"state": "closed"}
    
    def test_get_circuit_breaker_stats_no_client(self, service):
        """Test getting circuit breaker stats without client."""
        stats = service.get_circuit_breaker_stats()
        assert stats == {}
    
    def test_get_storage_stats(self, service):
        """Test getting storage stats."""
        service.storage = MagicMock()
        service.storage.get_storage_stats.return_value = {"total_files": 5}
        
        stats = service.get_storage_stats()
        
        assert stats == {"total_files": 5}
    
    def test_list_available_data(self, service):
        """Test listing available data."""
        service.storage = MagicMock()
        service.storage.list_available_data.return_value = {"AAPL": ["2024-01-01"]}
        
        data = service.list_available_data()
        
        assert data == {"AAPL": ["2024-01-01"]}


@pytest.mark.asyncio
async def test_run_data_ingestion_convenience_function(temp_dir):
    """Test convenience function for running ingestion."""
    from data_pipeline.ingestion.service import run_data_ingestion
    
    config = IngestionConfig(storage_base_path=temp_dir)
    
    with patch('data_pipeline.ingestion.service.DataIngestionService') as mock_service_class:
        mock_service = AsyncMock()
        mock_service.run_ingestion_pipeline.return_value = {"success": True}
        mock_service_class.return_value = mock_service
        
        result = await run_data_ingestion(config)
        
        assert result == {"success": True}
        mock_service.run_ingestion_pipeline.assert_called_once()