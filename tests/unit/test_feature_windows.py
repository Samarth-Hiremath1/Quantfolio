"""Unit tests for time window generation."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List

from data_pipeline.features.models import TechnicalFeatures, FeatureComputationConfig
from data_pipeline.features.windows import TimeWindowGenerator


class TestTimeWindowGenerator:
    """Test time window generation for ML training data."""
    
    @pytest.fixture
    def sample_technical_features(self) -> List[TechnicalFeatures]:
        """Create sample technical features for testing."""
        features = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(100):  # 100 days of features
            feature = TechnicalFeatures(
                symbol="AAPL",
                timestamp=base_date + timedelta(days=i)
            )
            
            # Add sample returns
            feature.returns = {
                'simple_return': np.random.normal(0.001, 0.02),
                'log_return': np.random.normal(0.001, 0.02)
            }
            
            # Add sample moving averages
            feature.moving_averages = {
                'sma_5': 100 + np.random.normal(0, 5),
                'sma_20': 100 + np.random.normal(0, 3),
                'ema_12': 100 + np.random.normal(0, 4)
            }
            
            # Add sample momentum indicators
            feature.momentum = {
                'rsi': np.random.uniform(20, 80),
                'macd_line': np.random.normal(0, 1)
            }
            
            # Add sample volatility
            feature.volatility = {
                'realized_volatility': np.random.uniform(0.1, 0.5)
            }
            
            # Add sample price features
            feature.price_features = {
                'high_low_ratio': np.random.uniform(1.01, 1.05),
                'close_position': np.random.uniform(0.3, 0.7)
            }
            
            features.append(feature)
        
        return features
    
    @pytest.fixture
    def window_generator(self) -> TimeWindowGenerator:
        """Create TimeWindowGenerator instance."""
        config = FeatureComputationConfig(time_windows=[30, 60])
        return TimeWindowGenerator(config)
    
    def test_create_time_windows_empty_data(self, window_generator):
        """Test creating time windows with empty data."""
        result = window_generator.create_time_windows([])
        assert result == []
    
    def test_create_time_windows_insufficient_data(self, window_generator):
        """Test creating time windows with insufficient data."""
        # Create only 10 features (less than window size of 30)
        features = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(10):
            feature = TechnicalFeatures(
                symbol="AAPL",
                timestamp=base_date + timedelta(days=i)
            )
            feature.returns = {'simple_return': 0.01}
            features.append(feature)
        
        result = window_generator.create_time_windows(features, [30])
        assert len(result) == 0  # Not enough data
    
    def test_create_time_windows_basic(self, window_generator, sample_technical_features):
        """Test basic time window creation."""
        result = window_generator.create_time_windows(sample_technical_features, [30])
        
        # Should create windows for 100 - 30 = 70 target dates
        assert len(result) == 70
        
        # Check window properties
        for ts_feature in result:
            assert ts_feature.symbol == "AAPL"
            assert ts_feature.window_size == 30
            assert isinstance(ts_feature.target_date, datetime)
            assert ts_feature.target_return is not None
    
    def test_create_time_windows_multiple_sizes(self, window_generator, sample_technical_features):
        """Test creating multiple window sizes."""
        result = window_generator.create_time_windows(sample_technical_features, [30, 60])
        
        # Should create windows for both sizes
        window_30_count = sum(1 for ts in result if ts.window_size == 30)
        window_60_count = sum(1 for ts in result if ts.window_size == 60)
        
        assert window_30_count == 70  # 100 - 30
        assert window_60_count == 40  # 100 - 60
        assert len(result) == 110  # Total
    
    def test_feature_sequence_extraction(self, window_generator, sample_technical_features):
        """Test feature sequence extraction."""
        result = window_generator.create_time_windows(sample_technical_features, [30])
        
        # Check first time series feature
        ts_feature = result[0]
        
        # Should have feature sequences
        assert len(ts_feature.feature_sequences) > 0
        
        # Check specific sequence types
        expected_sequences = [
            'simple_return', 'log_return',
            'ma_sma_5', 'ma_sma_20', 'ma_ema_12',
            'momentum_rsi', 'momentum_macd_line',
            'vol_realized_volatility',
            'price_high_low_ratio', 'price_close_position'
        ]
        
        for seq_name in expected_sequences:
            if seq_name in ts_feature.feature_sequences:
                sequence = ts_feature.feature_sequences[seq_name]
                assert len(sequence) == 30  # Window size
                assert all(isinstance(v, (int, float)) for v in sequence)
    
    def test_target_return_assignment(self, window_generator, sample_technical_features):
        """Test target return assignment."""
        result = window_generator.create_time_windows(sample_technical_features, [30])
        
        for ts_feature in result:
            assert ts_feature.target_return is not None
            assert isinstance(ts_feature.target_return, float)
            # Should be reasonable return value
            assert -0.5 < ts_feature.target_return < 0.5
    
    def test_date_range_calculation(self, window_generator, sample_technical_features):
        """Test date range calculation."""
        result = window_generator.create_time_windows(sample_technical_features, [30])
        
        for ts_feature in result:
            assert ts_feature.start_date is not None
            assert ts_feature.end_date is not None
            assert ts_feature.start_date < ts_feature.end_date
            assert ts_feature.end_date < ts_feature.target_date
    
    def test_missing_days_calculation(self, window_generator):
        """Test missing days calculation with gaps in data."""
        # Create features with gaps
        features = []
        base_date = datetime(2024, 1, 1)
        
        # Create 50 features with some gaps
        dates_to_skip = [5, 10, 15, 20]  # Skip these days
        
        day_counter = 0
        for i in range(50):
            if i in dates_to_skip:
                continue
            
            feature = TechnicalFeatures(
                symbol="AAPL",
                timestamp=base_date + timedelta(days=day_counter)
            )
            feature.returns = {'simple_return': 0.01}
            features.append(feature)
            day_counter += 1
        
        result = window_generator.create_time_windows(features, [30])
        
        # Should still create some windows
        assert len(result) > 0
        
        # Check missing days calculation
        for ts_feature in result:
            assert ts_feature.missing_days >= 0
    
    def test_multiple_symbols(self, window_generator):
        """Test time window creation with multiple symbols."""
        features = []
        symbols = ["AAPL", "GOOGL", "MSFT"]
        base_date = datetime(2024, 1, 1)
        
        for symbol in symbols:
            for i in range(50):
                feature = TechnicalFeatures(
                    symbol=symbol,
                    timestamp=base_date + timedelta(days=i)
                )
                feature.returns = {'simple_return': np.random.normal(0.001, 0.02)}
                feature.moving_averages = {'sma_5': 100 + np.random.normal(0, 5)}
                features.append(feature)
        
        result = window_generator.create_time_windows(features, [30])
        
        # Should create windows for all symbols
        symbols_in_result = set(ts.symbol for ts in result)
        assert symbols_in_result == set(symbols)
        
        # Each symbol should have the same number of windows
        for symbol in symbols:
            symbol_windows = [ts for ts in result if ts.symbol == symbol]
            assert len(symbol_windows) == 20  # 50 - 30
    
    def test_create_training_dataset(self, window_generator, sample_technical_features):
        """Test creating training dataset from time series features."""
        ts_features = window_generator.create_time_windows(sample_technical_features, [30])
        
        X, y = window_generator.create_training_dataset(ts_features)
        
        assert len(X) == len(y)
        assert len(X) > 0
        
        # Check dimensions
        if len(X) > 0:
            assert len(X[0]) > 0  # Should have features
            
        # Check target values
        for target in y:
            assert isinstance(target, (int, float))
    
    def test_create_training_dataset_empty(self, window_generator):
        """Test creating training dataset with empty features."""
        X, y = window_generator.create_training_dataset([])
        
        assert len(X) == 0
        assert len(y) == 0
    
    def test_get_feature_names(self, window_generator, sample_technical_features):
        """Test getting feature names."""
        ts_features = window_generator.create_time_windows(sample_technical_features, [30])
        
        if ts_features:
            feature_names = window_generator.get_feature_names(ts_features)
            
            assert len(feature_names) > 0
            assert all(isinstance(name, str) for name in feature_names)
            
            # Should have time-indexed names
            assert any('_t-' in name for name in feature_names)
    
    def test_forward_fill_functionality(self, window_generator):
        """Test forward fill of missing values."""
        # Create sequence with NaN values
        sequence = [1.0, 2.0, np.nan, np.nan, 3.0, np.nan]
        
        filled = window_generator._forward_fill(sequence)
        
        expected = [1.0, 2.0, 2.0, 2.0, 3.0, 3.0]
        assert filled == expected
    
    def test_data_quality_filtering(self, window_generator):
        """Test filtering of low-quality data."""
        # Create features with high missing ratio
        features = []
        base_date = datetime(2024, 1, 1)
        
        # Create sparse data (high missing ratio)
        for i in range(0, 100, 5):  # Only every 5th day
            feature = TechnicalFeatures(
                symbol="AAPL",
                timestamp=base_date + timedelta(days=i)
            )
            feature.returns = {'simple_return': 0.01}
            features.append(feature)
        
        # Use strict config
        config = FeatureComputationConfig(max_missing_ratio=0.05)  # Very strict
        strict_generator = TimeWindowGenerator(config)
        
        result = strict_generator.create_time_windows(features, [30])
        
        # Should filter out high-missing samples
        high_missing_count = sum(1 for ts in result if ts.missing_days > 5)
        assert high_missing_count < len(result)  # Some should be filtered
    
    def test_sequence_length_consistency(self, window_generator, sample_technical_features):
        """Test that all sequences have consistent lengths."""
        result = window_generator.create_time_windows(sample_technical_features, [30])
        
        for ts_feature in result:
            for sequence_name, sequence in ts_feature.feature_sequences.items():
                assert len(sequence) == ts_feature.window_size, \
                    f"Sequence {sequence_name} has length {len(sequence)}, expected {ts_feature.window_size}"