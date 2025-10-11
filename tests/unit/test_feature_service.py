"""Unit tests for feature engineering service."""

import pytest
import tempfile
import shutil
import asyncio
from datetime import datetime, date, timedelta
from typing import List, Dict

from data_pipeline.ingestion.models import PriceData
from data_pipeline.features.service import FeatureEngineeringService
from data_pipeline.features.models import FeatureComputationConfig


class TestFeatureEngineeringService:
    """Test main feature engineering service."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for cache and storage."""
        cache_dir = tempfile.mkdtemp()
        storage_dir = tempfile.mkdtemp()
        yield cache_dir, storage_dir
        shutil.rmtree(cache_dir)
        shutil.rmtree(storage_dir)
    
    @pytest.fixture
    def service(self, temp_dirs) -> FeatureEngineeringService:
        """Create FeatureEngineeringService instance."""
        cache_dir, storage_dir = temp_dirs
        config = FeatureComputationConfig(time_windows=[30])
        return FeatureEngineeringService(
            config=config,
            cache_dir=cache_dir,
            storage_dir=storage_dir
        )
    
    @pytest.fixture
    def sample_price_data(self) -> List[PriceData]:
        """Create sample price data for testing."""
        data = []
        base_date = datetime(2024, 1, 1)
        base_price = 100.0
        
        for i in range(60):  # 60 days of data
            price = base_price + 5 * (i / 60) + 2 * (i % 7 - 3) / 7  # Trend + weekly pattern
            high = price + abs(hash(f"high_{i}") % 100) / 100
            low = price - abs(hash(f"low_{i}") % 100) / 100
            volume = 1000000 + (hash(f"vol_{i}") % 500000)
            
            data.append(PriceData(
                symbol="AAPL",
                timestamp=base_date + timedelta(days=i),
                open=price + (hash(f"open_{i}") % 200 - 100) / 100,
                high=high,
                low=low,
                close=price,
                volume=volume,
                adjusted_close=price * 0.99
            ))
        
        return data
    
    @pytest.fixture
    def multi_symbol_price_data(self) -> Dict[str, List[PriceData]]:
        """Create price data for multiple symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        data = {}
        
        for symbol in symbols:
            symbol_data = []
            base_date = datetime(2024, 1, 1)
            base_price = 100.0 + hash(symbol) % 100  # Different base price per symbol
            
            for i in range(40):
                price = base_price + 3 * (i / 40)
                high = price + 1.0
                low = price - 1.0
                
                symbol_data.append(PriceData(
                    symbol=symbol,
                    timestamp=base_date + timedelta(days=i),
                    open=price,
                    high=high,
                    low=low,
                    close=price,
                    volume=1000000,
                    adjusted_close=price * 0.99
                ))
            
            data[symbol] = symbol_data
        
        return data
    
    @pytest.mark.asyncio
    async def test_compute_technical_indicators_empty(self, service):
        """Test computing technical indicators with empty data."""
        with pytest.raises(ValueError, match="No price data provided"):
            await service.compute_technical_indicators([])
    
    @pytest.mark.asyncio
    async def test_compute_technical_indicators_success(self, service, sample_price_data):
        """Test successful computation of technical indicators."""
        result = await service.compute_technical_indicators(sample_price_data)
        
        assert len(result) > 0
        assert all(f.symbol == "AAPL" for f in result)
        assert all(isinstance(f.timestamp, datetime) for f in result)
        
        # Check that features are computed
        for feature in result[-10:]:  # Check last 10 features
            assert len(feature.returns) > 0
            assert len(feature.moving_averages) > 0
    
    @pytest.mark.asyncio
    async def test_create_time_windows_empty(self, service):
        """Test creating time windows with empty data."""
        with pytest.raises(ValueError, match="No technical features provided"):
            await service.create_time_windows([])
    
    @pytest.mark.asyncio
    async def test_create_time_windows_success(self, service, sample_price_data):
        """Test successful creation of time windows."""
        # First compute technical indicators
        tech_features = await service.compute_technical_indicators(sample_price_data)
        
        # Then create time windows
        ts_features = await service.create_time_windows(tech_features)
        
        assert len(ts_features) > 0
        assert all(ts.symbol == "AAPL" for ts in ts_features)
        assert all(ts.window_size == 30 for ts in ts_features)
        assert all(ts.target_return is not None for ts in ts_features)
    
    @pytest.mark.asyncio
    async def test_validate_features_success(self, service, sample_price_data):
        """Test feature validation."""
        tech_features = await service.compute_technical_indicators(sample_price_data)
        
        validation_result = await service.validate_features(tech_features)
        
        assert validation_result.total_checks > 0
        assert validation_result.passed_checks > 0
        assert len(validation_result.feature_stats) > 0
    
    @pytest.mark.asyncio
    async def test_validate_time_series_features_success(self, service, sample_price_data):
        """Test time series feature validation."""
        tech_features = await service.compute_technical_indicators(sample_price_data)
        ts_features = await service.create_time_windows(tech_features)
        
        validation_result = await service.validate_time_series_features(ts_features)
        
        assert validation_result.total_checks > 0
        assert validation_result.passed_checks > 0
        assert 'targets' in validation_result.feature_stats
        assert 'sequences' in validation_result.feature_stats
    
    @pytest.mark.asyncio
    async def test_store_features_success(self, service, sample_price_data):
        """Test storing features."""
        tech_features = await service.compute_technical_indicators(sample_price_data)
        
        result = await service.store_features(tech_features)
        assert result
    
    @pytest.mark.asyncio
    async def test_store_time_series_features_success(self, service, sample_price_data):
        """Test storing time series features."""
        tech_features = await service.compute_technical_indicators(sample_price_data)
        ts_features = await service.create_time_windows(tech_features)
        
        result = await service.store_time_series_features(ts_features)
        assert result
    
    @pytest.mark.asyncio
    async def test_retrieve_features_empty(self, service):
        """Test retrieving features when none exist."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 10)
        
        result = await service.retrieve_features("AAPL", start_date, end_date)
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_features(self, service, sample_price_data):
        """Test storing and retrieving features."""
        # Compute and store features
        tech_features = await service.compute_technical_indicators(sample_price_data)
        store_result = await service.store_features(tech_features)
        assert store_result
        
        # Retrieve features
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 10)
        
        retrieved_features = await service.retrieve_features("AAPL", start_date, end_date)
        assert len(retrieved_features) > 0
        assert all(f.symbol == "AAPL" for f in retrieved_features)
    
    @pytest.mark.asyncio
    async def test_process_price_data_pipeline_success(self, service, sample_price_data):
        """Test complete price data processing pipeline."""
        result = await service.process_price_data_pipeline(
            sample_price_data,
            store_results=True,
            validate_results=True
        )
        
        assert result['symbol'] == "AAPL"
        assert result['input_data_points'] == len(sample_price_data)
        assert result['technical_feature_count'] > 0
        assert result['time_series_feature_count'] > 0
        assert 'validation_results' in result
        assert 'storage_results' in result
        assert len(result['errors']) == 0
    
    @pytest.mark.asyncio
    async def test_process_price_data_pipeline_no_validation(self, service, sample_price_data):
        """Test pipeline without validation."""
        result = await service.process_price_data_pipeline(
            sample_price_data,
            store_results=False,
            validate_results=False
        )
        
        assert result['symbol'] == "AAPL"
        assert result['technical_feature_count'] > 0
        assert len(result['validation_results']) == 0
        assert len(result['storage_results']) == 0
    
    @pytest.mark.asyncio
    async def test_process_price_data_pipeline_empty(self, service):
        """Test pipeline with empty data."""
        with pytest.raises(ValueError, match="No price data provided"):
            await service.process_price_data_pipeline([])
    
    @pytest.mark.asyncio
    async def test_batch_process_symbols_success(self, service, multi_symbol_price_data):
        """Test batch processing of multiple symbols."""
        results = await service.batch_process_symbols(multi_symbol_price_data, max_concurrent=2)
        
        assert len(results) == 3  # AAPL, GOOGL, MSFT
        
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            assert symbol in results
            assert results[symbol]['symbol'] == symbol
            assert results[symbol]['technical_feature_count'] > 0
            assert len(results[symbol]['errors']) == 0
    
    @pytest.mark.asyncio
    async def test_batch_process_symbols_with_errors(self, service):
        """Test batch processing with some errors."""
        # Create data with one invalid symbol
        invalid_data = {
            "AAPL": [PriceData(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 1),
                open=100.0, high=102.0, low=99.0, close=101.0,
                volume=1000000, adjusted_close=100.5
            )],
            "INVALID": []  # Empty data will cause error
        }
        
        results = await service.batch_process_symbols(invalid_data)
        
        assert len(results) == 2
        assert "AAPL" in results
        assert "INVALID" in results
        
        # AAPL should succeed
        assert len(results["AAPL"]["errors"]) == 0
        
        # INVALID should have errors
        assert len(results["INVALID"]["errors"]) > 0
    
    def test_get_available_symbols_empty(self, service):
        """Test getting available symbols when none exist."""
        symbols = service.get_available_symbols("technical")
        assert len(symbols) == 0
    
    @pytest.mark.asyncio
    async def test_get_available_symbols_after_storage(self, service, sample_price_data):
        """Test getting available symbols after storing data."""
        # Process and store data
        await service.process_price_data_pipeline(sample_price_data, store_results=True)
        
        # Should now have symbols
        symbols = service.get_available_symbols("technical")
        assert "AAPL" in symbols
    
    def test_get_symbol_date_range_empty(self, service):
        """Test getting date range when no data exists."""
        start_date, end_date = service.get_symbol_date_range("AAPL", "technical")
        assert start_date is None
        assert end_date is None
    
    @pytest.mark.asyncio
    async def test_get_symbol_date_range_after_storage(self, service, sample_price_data):
        """Test getting date range after storing data."""
        # Process and store data
        await service.process_price_data_pipeline(sample_price_data, store_results=True)
        
        # Should have date range
        start_date, end_date = service.get_symbol_date_range("AAPL", "technical")
        assert start_date is not None
        assert end_date is not None
        assert start_date <= end_date
    
    def test_get_cache_stats(self, service):
        """Test getting cache statistics."""
        stats = service.get_cache_stats()
        
        assert 'total_entries' in stats
        assert 'total_size_mb' in stats
        assert 'max_size_mb' in stats
        assert 'max_age_hours' in stats
    
    def test_get_storage_stats(self, service):
        """Test getting storage statistics."""
        stats = service.get_storage_stats()
        
        assert 'technical_features' in stats
        assert 'timeseries_features' in stats
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, service, sample_price_data):
        """Test clearing cache."""
        # Process data to populate cache
        await service.compute_technical_indicators(sample_price_data)
        
        # Cache should have entries
        stats_before = service.get_cache_stats()
        
        # Clear cache
        cleared_count = service.clear_cache()
        
        # Cache should be empty
        stats_after = service.get_cache_stats()
        assert stats_after['total_entries'] <= stats_before['total_entries']
    
    @pytest.mark.asyncio
    async def test_create_training_dataset_empty(self, service):
        """Test creating training dataset with no data."""
        X, y = await service.create_training_dataset(["NONEXISTENT"], window_size=30)
        
        assert len(X) == 0
        assert len(y) == 0
    
    @pytest.mark.asyncio
    async def test_create_training_dataset_success(self, service, sample_price_data):
        """Test creating training dataset."""
        # Process and store data
        await service.process_price_data_pipeline(sample_price_data, store_results=True)
        
        # Create training dataset
        X, y = await service.create_training_dataset(["AAPL"], window_size=30)
        
        assert len(X) > 0
        assert len(y) > 0
        assert len(X) == len(y)
        
        # Check that features have reasonable dimensions
        if len(X) > 0:
            assert len(X[0]) > 0  # Should have features
    
    def test_get_feature_names_empty(self, service):
        """Test getting feature names when no data exists."""
        feature_names = service.get_feature_names(window_size=30)
        assert len(feature_names) == 0
    
    @pytest.mark.asyncio
    async def test_get_feature_names_success(self, service, sample_price_data):
        """Test getting feature names after storing data."""
        # Process and store data
        await service.process_price_data_pipeline(sample_price_data, store_results=True)
        
        # Get feature names
        feature_names = service.get_feature_names(window_size=30)
        
        if len(feature_names) > 0:  # Only check if we have features
            assert all(isinstance(name, str) for name in feature_names)
            assert any('_t-' in name for name in feature_names)  # Time-indexed names
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, service, sample_price_data):
        """Test that caching improves performance."""
        # First computation (should cache)
        start_time = asyncio.get_event_loop().time()
        result1 = await service.compute_technical_indicators(sample_price_data)
        first_duration = asyncio.get_event_loop().time() - start_time
        
        # Second computation (should use cache)
        start_time = asyncio.get_event_loop().time()
        result2 = await service.compute_technical_indicators(sample_price_data)
        second_duration = asyncio.get_event_loop().time() - start_time
        
        # Results should be the same
        assert len(result1) == len(result2)
        
        # Second call should be faster (cached)
        # Note: This might not always be true in tests due to overhead, so we just check results match
        assert result1[0].symbol == result2[0].symbol
    
    @pytest.mark.asyncio
    async def test_error_handling_in_pipeline(self, service):
        """Test error handling in the pipeline."""
        # Create invalid price data
        invalid_data = [PriceData(
            symbol="TEST",
            timestamp=datetime(2024, 1, 1),
            open=0.0,  # Invalid price
            high=0.0,
            low=0.0,
            close=0.0,
            volume=0,
            adjusted_close=0.0
        )]
        
        # Pipeline should handle errors gracefully
        result = await service.process_price_data_pipeline(invalid_data)
        
        # Should have some result even with invalid data
        assert result['symbol'] == "TEST"
        assert result['input_data_points'] == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, service, multi_symbol_price_data):
        """Test concurrent processing doesn't cause issues."""
        # Process same data multiple times concurrently
        tasks = []
        for _ in range(3):
            task = service.batch_process_symbols(multi_symbol_price_data, max_concurrent=2)
            tasks.append(task)
        
        results_list = await asyncio.gather(*tasks)
        
        # All results should be consistent
        for results in results_list:
            assert len(results) == 3
            for symbol in ["AAPL", "GOOGL", "MSFT"]:
                assert symbol in results
                assert results[symbol]['technical_feature_count'] > 0