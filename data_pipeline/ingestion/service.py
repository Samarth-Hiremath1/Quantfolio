"""Main data ingestion service with cloud-native patterns."""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .models import DataBatch, ValidationResult, StorageResult
from .api_clients import BaseAPIClient, APIClientFactory, CircuitBreakerConfig
from .validator import DataValidator
from .storage import CloudStorageSimulator
from .events import EventBus, get_event_bus

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for data ingestion service."""
    api_provider: str = "yahoo"  # "yahoo" or "alpha_vantage"
    api_key: str = ""
    symbols: List[str] = None
    days_to_fetch: int = 1
    storage_base_path: str = "data"
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    enable_validation: bool = True
    enable_events: bool = True
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        
        if self.circuit_breaker_config is None:
            self.circuit_breaker_config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60.0,
                success_threshold=2,
                timeout=30.0
            )


class DataIngestionService:
    """Cloud-native data ingestion service."""
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.api_client: Optional[BaseAPIClient] = None
        self.validator = DataValidator() if config.enable_validation else None
        self.storage = CloudStorageSimulator(config.storage_base_path)
        self.event_bus = get_event_bus() if config.enable_events else None
        self._running = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize the service."""
        logger.info("Initializing data ingestion service")
        
        # Create API client
        self.api_client = APIClientFactory.create_client(
            provider=self.config.api_provider,
            api_key=self.config.api_key,
            circuit_breaker_config=self.config.circuit_breaker_config
        )
        
        # Initialize API client
        await self.api_client.__aenter__()
        
        self._running = True
        logger.info(f"Data ingestion service initialized with {self.config.api_provider} provider")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._running:
            logger.info("Cleaning up data ingestion service")
            
            if self.api_client:
                await self.api_client.__aexit__(None, None, None)
            
            if self.event_bus:
                await self.event_bus.close()
            
            self._running = False
            logger.info("Data ingestion service cleaned up")
    
    async def fetch_daily_data(self, symbols: Optional[List[str]] = None, 
                              days: Optional[int] = None) -> DataBatch:
        """Fetch daily data for specified symbols."""
        if not self._running:
            raise RuntimeError("Service not initialized")
        
        symbols = symbols or self.config.symbols
        days = days or self.config.days_to_fetch
        
        logger.info(f"Fetching {days} days of data for {len(symbols)} symbols")
        
        try:
            batch = await self.api_client.fetch_daily_data(symbols, days)
            logger.info(f"Successfully fetched {len(batch.data)} records")
            return batch
            
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise
    
    async def validate_data_quality(self, batch: DataBatch) -> ValidationResult:
        """Validate data quality."""
        if not self.validator:
            logger.warning("Validation disabled, skipping quality checks")
            return ValidationResult(
                status="passed",
                passed_checks=0,
                total_checks=0,
                errors=[],
                warnings=[]
            )
        
        logger.info("Validating data quality")
        result = self.validator.validate_batch(batch)
        
        if result.is_valid:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation failed: {result.errors}")
        
        return result
    
    async def store_data(self, batch: DataBatch) -> StorageResult:
        """Store data with cloud-native partitioning."""
        logger.info("Storing data batch")
        
        try:
            result = self.storage.store_batch(batch)
            
            if result.success:
                logger.info(f"Successfully stored {result.total_records} records in {len(result.files_written)} files")
            else:
                logger.error(f"Failed to store data: {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Storage operation failed: {e}")
            return StorageResult(
                success=False,
                files_written=[],
                total_records=0,
                error_message=str(e)
            )
    
    async def publish_data_event(self, batch: DataBatch, storage_result: StorageResult) -> bool:
        """Publish data availability event."""
        if not self.event_bus or not storage_result.success:
            return False
        
        logger.info("Publishing data availability event")
        
        try:
            # Get the most recent date from the batch
            latest_date = max(record.timestamp for record in batch.data) if batch.data else datetime.now()
            
            success = await self.event_bus.publish_data_available(
                symbols=batch.get_symbols(),
                date=latest_date,
                source=batch.source,
                file_paths=storage_result.files_written,
                record_count=storage_result.total_records
            )
            
            if success:
                logger.info("Data availability event published successfully")
            else:
                logger.warning("Failed to publish data availability event")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False
    
    async def run_ingestion_pipeline(self, symbols: Optional[List[str]] = None, 
                                   days: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete ingestion pipeline."""
        logger.info("Starting data ingestion pipeline")
        
        pipeline_start = datetime.now()
        results = {
            'success': False,
            'start_time': pipeline_start.isoformat(),
            'end_time': None,
            'duration_seconds': None,
            'records_fetched': 0,
            'records_stored': 0,
            'validation_result': None,
            'storage_result': None,
            'event_published': False,
            'errors': []
        }
        
        try:
            # Step 1: Fetch data
            logger.info("Pipeline step 1: Fetching data")
            batch = await self.fetch_daily_data(symbols, days)
            results['records_fetched'] = len(batch.data)
            
            if not batch.data:
                results['errors'].append("No data fetched")
                return results
            
            # Step 2: Validate data
            logger.info("Pipeline step 2: Validating data")
            validation_result = await self.validate_data_quality(batch)
            results['validation_result'] = {
                'status': validation_result.status.value,
                'passed_checks': validation_result.passed_checks,
                'total_checks': validation_result.total_checks,
                'errors': validation_result.errors,
                'warnings': validation_result.warnings
            }
            
            # Continue even with validation warnings, but stop on errors
            if not validation_result.is_valid and validation_result.errors:
                results['errors'].extend(validation_result.errors)
                return results
            
            # Step 3: Store data
            logger.info("Pipeline step 3: Storing data")
            storage_result = await self.store_data(batch)
            results['storage_result'] = {
                'success': storage_result.success,
                'files_written': len(storage_result.files_written),
                'total_records': storage_result.total_records,
                'error_message': storage_result.error_message
            }
            results['records_stored'] = storage_result.total_records
            
            if not storage_result.success:
                results['errors'].append(f"Storage failed: {storage_result.error_message}")
                return results
            
            # Step 4: Publish event
            logger.info("Pipeline step 4: Publishing event")
            event_published = await self.publish_data_event(batch, storage_result)
            results['event_published'] = event_published
            
            results['success'] = True
            logger.info("Data ingestion pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['errors'].append(str(e))
        
        finally:
            pipeline_end = datetime.now()
            results['end_time'] = pipeline_end.isoformat()
            results['duration_seconds'] = (pipeline_end - pipeline_start).total_seconds()
        
        return results
    
    def get_circuit_breaker_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        if self.api_client and hasattr(self.api_client, 'circuit_breaker'):
            return self.api_client.circuit_breaker.get_stats()
        return {}
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return self.storage.get_storage_stats()
    
    def list_available_data(self) -> Dict[str, List[str]]:
        """List available data."""
        return self.storage.list_available_data()


# Convenience function for running ingestion
async def run_data_ingestion(config: IngestionConfig) -> Dict[str, Any]:
    """Run data ingestion with the specified configuration."""
    async with DataIngestionService(config) as service:
        return await service.run_ingestion_pipeline()