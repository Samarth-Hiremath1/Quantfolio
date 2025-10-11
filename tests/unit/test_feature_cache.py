"""Unit tests for feature caching."""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from data_pipeline.features.models import TechnicalFeatures, TimeSeriesFeatures
from data_pipeline.features.cache import FeatureCache


class TestFeatureCache:
    """Test feature caching functionality."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache(self, temp_cache_dir) -> FeatureCache:
        """Create FeatureCache instance with temporary directory."""
        return FeatureCache(
            cache_dir=temp_cache_dir,
            max_age_hours=24,
            max_cache_size_mb=10
        )
    
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
            feature.returns = {'simple_return': 0.01 * i}
            feature.moving_averages = {'sma_5': 100.0 + i}
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
                end_date=base_date + timedelta(days=29 + i)
            )
            ts_feature.feature_sequences = {
                'simple_return': [0.01] * 30,
                'ma_sma_5': [100.0] * 30
            }
            ts_feature.target_return = 0.01 * i
            features.append(ts_feature)
        
        return features
    
    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initialization."""
        cache = FeatureCache(temp_cache_dir)
        
        # Check directory structure
        cache_path = Path(temp_cache_dir)
        assert (cache_path / "technical").exists()
        assert (cache_path / "timeseries").exists()
        assert (cache_path / "metadata").exists()
    
    def test_cache_technical_features_empty(self, cache):
        """Test caching empty technical features."""
        result = cache.cache_technical_features([])
        
        assert not result.success
        assert "No features to cache" in result.error_message
    
    def test_cache_technical_features_success(self, cache, sample_technical_features):
        """Test successful caching of technical features."""
        result = cache.cache_technical_features(sample_technical_features)
        
        assert result.success
        assert result.cache_key
        assert result.cached_at is not None
    
    def test_get_technical_features_cache_miss(self, cache):
        """Test getting technical features with cache miss."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        result = cache.get_technical_features("AAPL", start_date, end_date)
        
        assert result is None
    
    def test_get_technical_features_cache_hit(self, cache, sample_technical_features):
        """Test getting technical features with cache hit."""
        # Cache the features first
        cache_result = cache.cache_technical_features(sample_technical_features)
        assert cache_result.success
        
        # Retrieve from cache
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 9)
        
        cached_features = cache.get_technical_features("AAPL", start_date, end_date)
        
        assert cached_features is not None
        assert len(cached_features) == len(sample_technical_features)
        assert cached_features[0].symbol == "AAPL"
    
    def test_cache_timeseries_features_success(self, cache, sample_timeseries_features):
        """Test successful caching of time series features."""
        result = cache.cache_timeseries_features(sample_timeseries_features)
        
        assert result.success
        assert result.cache_key
        assert result.cached_at is not None
    
    def test_get_timeseries_features_cache_hit(self, cache, sample_timeseries_features):
        """Test getting time series features with cache hit."""
        # Cache the features first
        cache_result = cache.cache_timeseries_features(sample_timeseries_features)
        assert cache_result.success
        
        # Retrieve from cache
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)
        
        cached_features = cache.get_timeseries_features("AAPL", 30, start_date, end_date)
        
        assert cached_features is not None
        assert len(cached_features) == len(sample_timeseries_features)
        assert cached_features[0].symbol == "AAPL"
        assert cached_features[0].window_size == 30
    
    def test_cache_key_generation(self, cache):
        """Test cache key generation."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        key1 = cache._generate_cache_key("AAPL", start_date, end_date, "technical")
        key2 = cache._generate_cache_key("AAPL", start_date, end_date, "technical")
        key3 = cache._generate_cache_key("GOOGL", start_date, end_date, "technical")
        
        # Same inputs should generate same key
        assert key1 == key2
        
        # Different inputs should generate different keys
        assert key1 != key3
    
    def test_cache_expiration(self, temp_cache_dir):
        """Test cache expiration."""
        # Create cache with very short expiration
        cache = FeatureCache(
            cache_dir=temp_cache_dir,
            max_age_hours=0.001,  # Very short expiration
            max_cache_size_mb=10
        )
        
        # Create and cache features
        features = [TechnicalFeatures(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1)
        )]
        features[0].returns = {'simple_return': 0.01}
        
        cache_result = cache.cache_technical_features(features)
        assert cache_result.success
        
        # Wait a bit and try to retrieve (should be expired)
        import time
        time.sleep(0.1)
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 1)
        
        cached_features = cache.get_technical_features("AAPL", start_date, end_date)
        assert cached_features is None  # Should be expired
    
    def test_cache_size_limit(self, temp_cache_dir):
        """Test cache size limit enforcement."""
        # Create cache with very small size limit
        cache = FeatureCache(
            cache_dir=temp_cache_dir,
            max_age_hours=24,
            max_cache_size_mb=0.001  # Very small limit
        )
        
        # Create multiple sets of features to exceed limit
        for i in range(5):
            features = []
            for j in range(100):  # Large number of features
                feature = TechnicalFeatures(
                    symbol=f"STOCK{i}",
                    timestamp=datetime(2024, 1, 1) + timedelta(days=j)
                )
                feature.returns = {'simple_return': 0.01 * j}
                feature.moving_averages = {f'sma_{k}': 100.0 + k for k in range(10)}
                features.append(feature)
            
            cache.cache_technical_features(features)
        
        # Check that cleanup occurred
        stats = cache.get_cache_stats()
        # Should have cleaned up some entries due to size limit
        assert stats['total_entries'] < 5  # Not all 5 should remain
    
    def test_clear_cache_all(self, cache, sample_technical_features):
        """Test clearing all cache entries."""
        # Cache some features
        cache.cache_technical_features(sample_technical_features)
        
        # Verify cache has entries
        stats_before = cache.get_cache_stats()
        assert stats_before['total_entries'] > 0
        
        # Clear all cache
        cleared_count = cache.clear_cache()
        
        # Verify cache is empty
        stats_after = cache.get_cache_stats()
        assert stats_after['total_entries'] == 0
        assert cleared_count > 0
    
    def test_clear_cache_by_symbol(self, cache, sample_technical_features):
        """Test clearing cache by symbol."""
        # Cache features for AAPL
        cache.cache_technical_features(sample_technical_features)
        
        # Cache features for another symbol
        googl_features = []
        for i in range(5):
            feature = TechnicalFeatures(
                symbol="GOOGL",
                timestamp=datetime(2024, 1, 1) + timedelta(days=i)
            )
            feature.returns = {'simple_return': 0.01}
            googl_features.append(feature)
        
        cache.cache_technical_features(googl_features)
        
        # Clear cache for AAPL only
        cleared_count = cache.clear_cache(symbol="AAPL")
        
        assert cleared_count > 0
        
        # GOOGL features should still be cached
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)
        
        googl_cached = cache.get_technical_features("GOOGL", start_date, end_date)
        assert googl_cached is not None
        
        aapl_cached = cache.get_technical_features("AAPL", start_date, end_date)
        assert aapl_cached is None
    
    def test_clear_cache_by_feature_type(self, cache, sample_technical_features, sample_timeseries_features):
        """Test clearing cache by feature type."""
        # Cache both types
        cache.cache_technical_features(sample_technical_features)
        cache.cache_timeseries_features(sample_timeseries_features)
        
        # Clear only technical features
        cleared_count = cache.clear_cache(feature_type="technical")
        
        assert cleared_count > 0
        
        # Time series should still be cached
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)
        
        ts_cached = cache.get_timeseries_features("AAPL", 30, start_date, end_date)
        assert ts_cached is not None
        
        tech_cached = cache.get_technical_features("AAPL", start_date, end_date)
        assert tech_cached is None
    
    def test_get_cache_stats(self, cache, sample_technical_features, sample_timeseries_features):
        """Test getting cache statistics."""
        # Initially empty
        stats = cache.get_cache_stats()
        assert stats['total_entries'] == 0
        assert stats['total_size_mb'] == 0
        
        # Cache some features
        cache.cache_technical_features(sample_technical_features)
        cache.cache_timeseries_features(sample_timeseries_features)
        
        # Check updated stats
        stats = cache.get_cache_stats()
        assert stats['total_entries'] > 0
        assert stats['total_size_mb'] > 0
        assert 'by_type' in stats
        assert 'technical' in stats['by_type']
        assert 'timeseries' in stats['by_type']
    
    def test_metadata_persistence(self, temp_cache_dir, sample_technical_features):
        """Test that metadata persists across cache instances."""
        # Create cache and add features
        cache1 = FeatureCache(temp_cache_dir)
        cache1.cache_technical_features(sample_technical_features)
        
        stats1 = cache1.get_cache_stats()
        assert stats1['total_entries'] > 0
        
        # Create new cache instance with same directory
        cache2 = FeatureCache(temp_cache_dir)
        
        stats2 = cache2.get_cache_stats()
        assert stats2['total_entries'] == stats1['total_entries']
    
    def test_cache_with_multiple_symbols(self, cache):
        """Test caching with multiple symbols."""
        # Create features for multiple symbols
        all_features = []
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        for symbol in symbols:
            for i in range(5):
                feature = TechnicalFeatures(
                    symbol=symbol,
                    timestamp=datetime(2024, 1, 1) + timedelta(days=i)
                )
                feature.returns = {'simple_return': 0.01}
                all_features.append(feature)
        
        # Cache all features together
        result = cache.cache_technical_features(all_features)
        assert result.success
        
        # Should be able to retrieve each symbol separately
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)
        
        for symbol in symbols:
            cached_features = cache.get_technical_features(symbol, start_date, end_date)
            assert cached_features is not None
            assert len(cached_features) == 5
            assert all(f.symbol == symbol for f in cached_features)
    
    def test_cache_file_corruption_handling(self, cache, sample_technical_features):
        """Test handling of corrupted cache files."""
        # Cache features
        result = cache.cache_technical_features(sample_technical_features)
        assert result.success
        
        # Corrupt the cache file
        cache_key = list(cache.metadata.keys())[0]
        cache_file = cache._get_cache_file_path(cache_key, "technical")
        
        with open(cache_file, 'w') as f:
            f.write("corrupted data")
        
        # Try to retrieve (should handle corruption gracefully)
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        cached_features = cache.get_technical_features("AAPL", start_date, end_date)
        assert cached_features is None  # Should return None for corrupted file