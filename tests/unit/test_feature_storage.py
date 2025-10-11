"""Unit tests for feature storage system."""

import pytest
import tempfile
import shutil
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List

from data_pipeline.features.models import TechnicalFeatures, TimeSeriesFeatures
from data_pipeline.features.storage import FeatureStorage


class TestFeatureStorage:
    """Test feature storage system."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def storage(self, temp_storage_dir) -> FeatureStorage:
        """Create FeatureStorage instance with temporary directory."""
        return FeatureStorage(temp_storage_dir)
    
    @pytest.fixture
    def sample_technical_features(self) -> List[TechnicalFeatures]:
        """Create sample technical features for testing."""
        features = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(10):
            feature = TechnicalFeatures(
                symbol="AAPL",
                timestamp=base_date + timedelta(days=i)
            )
            feature.returns = {
                'simple_return': 0.01 * i,
                'log_return': 0.009 * i
            }
            feature.moving_averages = {
                'sma_5': 100.0 + i,
                'sma_20': 99.0 + i
            }
            feature.momentum = {
                'rsi': 50.0 + i,
                'macd_line': 0.1 * i
            }
            feature.volatility = {
                'realized_volatility': 0.2 + 0.01 * i
            }
            feature.price_features = {
                'high_low_ratio': 1.02 + 0.001 * i
            }
            features.append(feature)
        
        return features
    
    @pytest.fixture
    def sample_timeseries_features(self) -> List[TimeSeriesFeatures]:
        """Create sample time series features for testing."""
        features = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(5):
            ts_feature = TimeSeriesFeatures(
                symbol="AAPL",
                target_date=base_date + timedelta(days=30 + i),
                window_size=30,
                start_date=base_date + timedelta(days=i),
                end_date=base_date + timedelta(days=29 + i),
                missing_days=0
            )
            ts_feature.feature_sequences = {
                'simple_return': [0.01 + 0.001 * j for j in range(30)],
                'ma_sma_5': [100.0 + j for j in range(30)],
                'momentum_rsi': [50.0 + j for j in range(30)]
            }
            ts_feature.target_return = 0.01 * i
            features.append(ts_feature)
        
        return features
    
    def test_storage_initialization(self, temp_storage_dir):
        """Test storage initialization."""
        storage = FeatureStorage(temp_storage_dir)
        
        # Check directory structure
        storage_path = Path(temp_storage_dir)
        assert (storage_path / "technical").exists()
        assert (storage_path / "time_series").exists()
        assert (storage_path / "metadata").exists()
        assert (storage_path / "time_series" / "windows_30d").exists()
        assert (storage_path / "time_series" / "windows_60d").exists()
    
    def test_store_technical_features_empty(self, storage):
        """Test storing empty technical features."""
        result = storage.store_technical_features([])
        assert not result
    
    def test_store_technical_features_success(self, storage, sample_technical_features):
        """Test successful storage of technical features."""
        result = storage.store_technical_features(sample_technical_features)
        assert result
        
        # Check that files were created
        storage_path = Path(storage.base_path)
        parquet_files = list(storage_path.rglob("*.parquet"))
        assert len(parquet_files) > 0
    
    def test_retrieve_technical_features_empty(self, storage):
        """Test retrieving technical features when none exist."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 10)
        
        result = storage.retrieve_technical_features("AAPL", start_date, end_date)
        assert len(result) == 0
    
    def test_store_and_retrieve_technical_features(self, storage, sample_technical_features):
        """Test storing and retrieving technical features."""
        # Store features
        store_result = storage.store_technical_features(sample_technical_features)
        assert store_result
        
        # Retrieve features
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 9)
        
        retrieved_features = storage.retrieve_technical_features("AAPL", start_date, end_date)
        
        assert len(retrieved_features) == len(sample_technical_features)
        assert retrieved_features[0].symbol == "AAPL"
        
        # Check that data is preserved
        original_first = sample_technical_features[0]
        retrieved_first = retrieved_features[0]
        
        assert retrieved_first.timestamp == original_first.timestamp
        assert retrieved_first.returns['simple_return'] == original_first.returns['simple_return']
        assert retrieved_first.moving_averages['sma_5'] == original_first.moving_averages['sma_5']
    
    def test_store_timeseries_features_success(self, storage, sample_timeseries_features):
        """Test successful storage of time series features."""
        result = storage.store_timeseries_features(sample_timeseries_features)
        assert result
        
        # Check that files were created
        storage_path = Path(storage.base_path)
        parquet_files = list(storage_path.rglob("*_timeseries.parquet"))
        assert len(parquet_files) > 0
    
    def test_store_and_retrieve_timeseries_features(self, storage, sample_timeseries_features):
        """Test storing and retrieving time series features."""
        # Store features
        store_result = storage.store_timeseries_features(sample_timeseries_features)
        assert store_result
        
        # Retrieve features
        retrieved_features = storage.retrieve_timeseries_features("AAPL", 30)
        
        assert len(retrieved_features) == len(sample_timeseries_features)
        assert retrieved_features[0].symbol == "AAPL"
        assert retrieved_features[0].window_size == 30
        
        # Check that data is preserved
        original_first = sample_timeseries_features[0]
        retrieved_first = retrieved_features[0]
        
        assert retrieved_first.target_date == original_first.target_date
        assert retrieved_first.target_return == original_first.target_return
        assert retrieved_first.feature_sequences == original_first.feature_sequences
    
    def test_list_available_symbols_technical(self, storage, sample_technical_features):
        """Test listing available symbols for technical features."""
        # Initially empty
        symbols = storage.list_available_symbols("technical")
        assert len(symbols) == 0
        
        # Store features
        storage.store_technical_features(sample_technical_features)
        
        # Should now have symbols
        symbols = storage.list_available_symbols("technical")
        assert "AAPL" in symbols
    
    def test_list_available_symbols_timeseries(self, storage, sample_timeseries_features):
        """Test listing available symbols for time series features."""
        # Initially empty
        symbols = storage.list_available_symbols("timeseries")
        assert len(symbols) == 0
        
        # Store features
        storage.store_timeseries_features(sample_timeseries_features)
        
        # Should now have symbols
        symbols = storage.list_available_symbols("timeseries")
        assert "AAPL" in symbols
    
    def test_get_date_range_technical(self, storage, sample_technical_features):
        """Test getting date range for technical features."""
        # Initially no data
        start_date, end_date = storage.get_date_range("AAPL", "technical")
        assert start_date is None
        assert end_date is None
        
        # Store features
        storage.store_technical_features(sample_technical_features)
        
        # Should now have date range
        start_date, end_date = storage.get_date_range("AAPL", "technical")
        assert start_date is not None
        assert end_date is not None
        assert start_date <= end_date
    
    def test_get_date_range_timeseries(self, storage, sample_timeseries_features):
        """Test getting date range for time series features."""
        # Store features
        storage.store_timeseries_features(sample_timeseries_features)
        
        # Should have date range
        start_date, end_date = storage.get_date_range("AAPL", "timeseries")
        assert start_date is not None
        assert end_date is not None
        assert start_date <= end_date
    
    def test_multiple_symbols_storage(self, storage):
        """Test storing features for multiple symbols."""
        # Create features for multiple symbols
        all_features = []
        symbols = ["AAPL", "GOOGL", "MSFT"]
        base_date = datetime(2024, 1, 1)
        
        for symbol in symbols:
            for i in range(5):
                feature = TechnicalFeatures(
                    symbol=symbol,
                    timestamp=base_date + timedelta(days=i)
                )
                feature.returns = {'simple_return': 0.01}
                all_features.append(feature)
        
        # Store all features
        result = storage.store_technical_features(all_features)
        assert result
        
        # Should be able to list all symbols
        stored_symbols = storage.list_available_symbols("technical")
        for symbol in symbols:
            assert symbol in stored_symbols
        
        # Should be able to retrieve each symbol separately
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)
        
        for symbol in symbols:
            features = storage.retrieve_technical_features(symbol, start_date, end_date)
            assert len(features) == 5
            assert all(f.symbol == symbol for f in features)
    
    def test_date_partitioning(self, storage, sample_technical_features):
        """Test date-based partitioning of technical features."""
        # Store features
        storage.store_technical_features(sample_technical_features)
        
        # Check that files are created in date-based directories
        storage_path = Path(storage.base_path)
        
        # Should have year/month/day structure
        year_dirs = list((storage_path / "technical").glob("2024"))
        assert len(year_dirs) > 0
        
        month_dirs = list((storage_path / "technical" / "2024").glob("01"))
        assert len(month_dirs) > 0
        
        day_dirs = list((storage_path / "technical" / "2024" / "01").glob("*"))
        assert len(day_dirs) > 0
    
    def test_window_partitioning(self, storage, sample_timeseries_features):
        """Test window-based partitioning of time series features."""
        # Store features
        storage.store_timeseries_features(sample_timeseries_features)
        
        # Check that files are created in window-based directories
        storage_path = Path(storage.base_path)
        
        window_30_files = list((storage_path / "time_series" / "windows_30d").glob("*.parquet"))
        assert len(window_30_files) > 0
    
    def test_metadata_updates(self, storage, sample_technical_features, sample_timeseries_features):
        """Test metadata updates."""
        # Store technical features
        storage.store_technical_features(sample_technical_features)
        
        # Check technical metadata
        tech_metadata_file = storage.metadata_path / "technical_features_metadata.json"
        assert tech_metadata_file.exists()
        
        # Store time series features
        storage.store_timeseries_features(sample_timeseries_features)
        
        # Check time series metadata
        ts_metadata_file = storage.metadata_path / "timeseries_features_metadata.json"
        assert ts_metadata_file.exists()
    
    def test_get_storage_stats(self, storage, sample_technical_features, sample_timeseries_features):
        """Test getting storage statistics."""
        # Initially empty stats
        stats = storage.get_storage_stats()
        assert 'technical_features' in stats
        assert 'timeseries_features' in stats
        
        # Store features
        storage.store_technical_features(sample_technical_features)
        storage.store_timeseries_features(sample_timeseries_features)
        
        # Should have updated stats
        stats = storage.get_storage_stats()
        assert 'storage_size' in stats
        assert stats['storage_size']['total_mb'] > 0
    
    def test_feature_data_integrity(self, storage):
        """Test that feature data maintains integrity through storage/retrieval."""
        # Create feature with all data types
        feature = TechnicalFeatures(
            symbol="TEST",
            timestamp=datetime(2024, 1, 1, 12, 30, 45)  # Include time
        )
        
        # Add various data types
        feature.returns = {
            'simple_return': 0.0123456789,
            'log_return': -0.0098765432
        }
        feature.moving_averages = {
            'sma_5': 123.456789,
            'ema_12': 98.765432
        }
        feature.momentum = {
            'rsi': 67.89,
            'macd_line': -1.23456
        }
        feature.volatility = {
            'realized_volatility': 0.234567
        }
        feature.price_features = {
            'high_low_ratio': 1.0234,
            'close_position': 0.6789
        }
        
        # Store and retrieve
        storage.store_technical_features([feature])
        
        retrieved = storage.retrieve_technical_features(
            "TEST", 
            date(2024, 1, 1), 
            date(2024, 1, 1)
        )
        
        assert len(retrieved) == 1
        retrieved_feature = retrieved[0]
        
        # Check all data is preserved with reasonable precision
        assert abs(retrieved_feature.returns['simple_return'] - 0.0123456789) < 1e-6
        assert abs(retrieved_feature.moving_averages['sma_5'] - 123.456789) < 1e-6
        assert abs(retrieved_feature.momentum['rsi'] - 67.89) < 1e-6
        assert abs(retrieved_feature.volatility['realized_volatility'] - 0.234567) < 1e-6
        assert abs(retrieved_feature.price_features['high_low_ratio'] - 1.0234) < 1e-6
    
    def test_large_dataset_handling(self, storage):
        """Test handling of large datasets."""
        # Create a large number of features
        features = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(1000):  # 1000 features
            feature = TechnicalFeatures(
                symbol="LARGE",
                timestamp=base_date + timedelta(days=i % 365)  # Cycle through year
            )
            feature.returns = {'simple_return': 0.001 * i}
            feature.moving_averages = {f'sma_{j}': 100.0 + j for j in range(10)}
            features.append(feature)
        
        # Should handle large dataset
        result = storage.store_technical_features(features)
        assert result
        
        # Should be able to retrieve subset
        retrieved = storage.retrieve_technical_features(
            "LARGE",
            date(2024, 1, 1),
            date(2024, 1, 31)
        )
        
        assert len(retrieved) > 0
    
    def test_concurrent_access_simulation(self, storage, sample_technical_features):
        """Test simulation of concurrent access patterns."""
        # Store initial features
        storage.store_technical_features(sample_technical_features)
        
        # Simulate multiple retrievals (as if from different processes)
        results = []
        for _ in range(5):
            retrieved = storage.retrieve_technical_features(
                "AAPL",
                date(2024, 1, 1),
                date(2024, 1, 5)
            )
            results.append(len(retrieved))
        
        # All retrievals should return same number of features
        assert all(count == results[0] for count in results)
        assert results[0] > 0