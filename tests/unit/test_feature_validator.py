"""Unit tests for feature validation."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List

from data_pipeline.features.models import (
    TechnicalFeatures, TimeSeriesFeatures, FeatureComputationConfig,
    FeatureValidationStatus
)
from data_pipeline.features.validator import FeatureValidator


class TestFeatureValidator:
    """Test feature validation and quality assurance."""
    
    @pytest.fixture
    def validator(self) -> FeatureValidator:
        """Create FeatureValidator instance."""
        config = FeatureComputationConfig()
        return FeatureValidator(config)
    
    @pytest.fixture
    def valid_technical_features(self) -> List[TechnicalFeatures]:
        """Create valid technical features for testing."""
        features = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(50):
            feature = TechnicalFeatures(
                symbol="AAPL",
                timestamp=base_date + timedelta(days=i)
            )
            
            # Add valid returns
            feature.returns = {
                'simple_return': np.random.normal(0.001, 0.02),
                'log_return': np.random.normal(0.001, 0.02)
            }
            
            # Add valid moving averages
            feature.moving_averages = {
                'sma_5': 100 + np.random.normal(0, 5),
                'sma_20': 100 + np.random.normal(0, 3)
            }
            
            # Add valid momentum indicators
            feature.momentum = {
                'rsi': np.random.uniform(20, 80),
                'macd_line': np.random.normal(0, 1)
            }
            
            # Add valid volatility
            feature.volatility = {
                'realized_volatility': np.random.uniform(0.1, 0.5)
            }
            
            # Add valid price features
            feature.price_features = {
                'high_low_ratio': np.random.uniform(1.01, 1.05)
            }
            
            features.append(feature)
        
        return features
    
    @pytest.fixture
    def valid_timeseries_features(self) -> List[TimeSeriesFeatures]:
        """Create valid time series features for testing."""
        features = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(30):
            ts_feature = TimeSeriesFeatures(
                symbol="AAPL",
                target_date=base_date + timedelta(days=30 + i),
                window_size=30,
                start_date=base_date + timedelta(days=i),
                end_date=base_date + timedelta(days=29 + i),
                missing_days=0
            )
            
            # Add feature sequences
            ts_feature.feature_sequences = {
                'simple_return': [np.random.normal(0.001, 0.02) for _ in range(30)],
                'ma_sma_5': [100 + np.random.normal(0, 5) for _ in range(30)],
                'momentum_rsi': [np.random.uniform(20, 80) for _ in range(30)]
            }
            
            # Add target return
            ts_feature.target_return = np.random.normal(0.001, 0.02)
            
            features.append(ts_feature)
        
        return features
    
    def test_validate_technical_features_empty(self, validator):
        """Test validation with empty features."""
        result = validator.validate_technical_features([])
        
        assert result.status == FeatureValidationStatus.FAILED
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "No features provided" in result.errors[0]
    
    def test_validate_technical_features_valid(self, validator, valid_technical_features):
        """Test validation with valid features."""
        result = validator.validate_technical_features(valid_technical_features)
        
        assert result.status in [FeatureValidationStatus.PASSED, FeatureValidationStatus.WARNING]
        assert result.passed_checks > 0
        assert result.total_checks > 0
        assert len(result.feature_stats) > 0
    
    def test_validate_technical_features_timestamp_issues(self, validator):
        """Test validation with timestamp issues."""
        features = []
        base_date = datetime(2024, 1, 1)
        
        # Create features with duplicate timestamps
        for i in range(5):
            feature = TechnicalFeatures(
                symbol="AAPL",
                timestamp=base_date  # Same timestamp for all
            )
            feature.returns = {'simple_return': 0.01}
            features.append(feature)
        
        result = validator.validate_technical_features(features)
        
        assert result.status == FeatureValidationStatus.FAILED
        assert any("Duplicate timestamps" in error for error in result.errors)
    
    def test_validate_technical_features_extreme_returns(self, validator):
        """Test validation with extreme returns."""
        features = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(10):
            feature = TechnicalFeatures(
                symbol="AAPL",
                timestamp=base_date + timedelta(days=i)
            )
            
            # Add extreme returns
            feature.returns = {
                'simple_return': 0.8 if i % 2 == 0 else -0.8  # 80% daily returns
            }
            
            features.append(feature)
        
        result = validator.validate_technical_features(features)
        
        # Should generate warnings about extreme returns
        assert any("Extreme returns" in warning for warning in result.warnings)
    
    def test_validate_technical_features_missing_data(self, validator):
        """Test validation with missing data."""
        features = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(20):
            feature = TechnicalFeatures(
                symbol="AAPL",
                timestamp=base_date + timedelta(days=i)
            )
            
            # Most features have empty dictionaries (missing data)
            if i < 2:  # Only first 2 have data
                feature.returns = {'simple_return': 0.01}
                feature.moving_averages = {'sma_5': 100.0}
            
            features.append(feature)
        
        result = validator.validate_technical_features(features)
        
        # Should detect missing data
        assert any("missing" in warning.lower() for warning in result.warnings)
    
    def test_validate_technical_features_rsi_range(self, validator):
        """Test validation of RSI range."""
        features = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(10):
            feature = TechnicalFeatures(
                symbol="AAPL",
                timestamp=base_date + timedelta(days=i)
            )
            
            # Add invalid RSI values
            feature.momentum = {
                'rsi': 150.0 if i % 2 == 0 else -50.0  # Invalid RSI range
            }
            
            features.append(feature)
        
        result = validator.validate_technical_features(features)
        
        # Should detect RSI range issues
        assert any("RSI values outside" in warning for warning in result.warnings)
    
    def test_validate_timeseries_features_empty(self, validator):
        """Test validation with empty time series features."""
        result = validator.validate_time_series_features([])
        
        assert result.status == FeatureValidationStatus.FAILED
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_validate_timeseries_features_valid(self, validator, valid_timeseries_features):
        """Test validation with valid time series features."""
        result = validator.validate_time_series_features(valid_timeseries_features)
        
        assert result.status in [FeatureValidationStatus.PASSED, FeatureValidationStatus.WARNING]
        assert result.passed_checks > 0
        assert result.total_checks > 0
        assert 'targets' in result.feature_stats
        assert 'sequences' in result.feature_stats
    
    def test_validate_timeseries_features_sequence_length_mismatch(self, validator):
        """Test validation with sequence length mismatches."""
        features = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(5):
            ts_feature = TimeSeriesFeatures(
                symbol="AAPL",
                target_date=base_date + timedelta(days=30 + i),
                window_size=30
            )
            
            # Add sequences with wrong lengths
            ts_feature.feature_sequences = {
                'simple_return': [0.01] * 25,  # Wrong length (should be 30)
                'ma_sma_5': [100.0] * 35      # Wrong length (should be 30)
            }
            
            ts_feature.target_return = 0.01
            features.append(ts_feature)
        
        result = validator.validate_time_series_features(features)
        
        assert result.status == FeatureValidationStatus.FAILED
        assert any("Sequence length mismatch" in error for error in result.errors)
    
    def test_validate_timeseries_features_missing_targets(self, validator):
        """Test validation with missing target returns."""
        features = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(10):
            ts_feature = TimeSeriesFeatures(
                symbol="AAPL",
                target_date=base_date + timedelta(days=30 + i),
                window_size=30
            )
            
            ts_feature.feature_sequences = {
                'simple_return': [0.01] * 30
            }
            
            # Most features missing target returns
            if i < 2:  # Only first 2 have targets
                ts_feature.target_return = 0.01
            
            features.append(ts_feature)
        
        result = validator.validate_time_series_features(features)
        
        # Should detect missing targets
        assert result.feature_stats['targets']['missing_count'] > 0
    
    def test_validate_timeseries_features_high_missing_days(self, validator):
        """Test validation with high missing days."""
        # Use strict config
        config = FeatureComputationConfig(max_missing_ratio=0.05)
        strict_validator = FeatureValidator(config)
        
        features = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(10):
            ts_feature = TimeSeriesFeatures(
                symbol="AAPL",
                target_date=base_date + timedelta(days=30 + i),
                window_size=30,
                missing_days=10  # High missing days
            )
            
            ts_feature.feature_sequences = {
                'simple_return': [0.01] * 30
            }
            ts_feature.target_return = 0.01
            
            features.append(ts_feature)
        
        result = strict_validator.validate_time_series_features(features)
        
        # Should detect high missing data
        assert any("missing data" in warning.lower() for warning in result.warnings)
    
    def test_validate_timeseries_features_insufficient_features(self, validator):
        """Test validation with insufficient feature sequences."""
        features = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(10):
            ts_feature = TimeSeriesFeatures(
                symbol="AAPL",
                target_date=base_date + timedelta(days=30 + i),
                window_size=30
            )
            
            # Very few feature sequences
            ts_feature.feature_sequences = {
                'simple_return': [0.01] * 30
            }  # Only 1 sequence type
            
            ts_feature.target_return = 0.01
            features.append(ts_feature)
        
        result = validator.validate_time_series_features(features)
        
        # Should detect insufficient features
        assert any("insufficient features" in warning.lower() for warning in result.warnings)
    
    def test_outlier_detection(self, validator):
        """Test outlier detection in returns."""
        features = []
        base_date = datetime(2024, 1, 1)
        
        # Create mostly normal returns with some outliers
        returns = [np.random.normal(0.001, 0.02) for _ in range(45)]
        returns.extend([0.5, -0.5, 0.8, -0.8, 1.0])  # Add outliers
        
        for i, ret in enumerate(returns):
            feature = TechnicalFeatures(
                symbol="AAPL",
                timestamp=base_date + timedelta(days=i)
            )
            feature.returns = {'simple_return': ret}
            features.append(feature)
        
        result = validator.validate_technical_features(features)
        
        # Should detect outliers
        assert any("outliers" in warning.lower() for warning in result.warnings)
    
    def test_feature_statistics_calculation(self, validator, valid_technical_features):
        """Test feature statistics calculation."""
        result = validator.validate_technical_features(valid_technical_features)
        
        # Check that statistics are calculated
        assert 'returns' in result.feature_stats
        
        returns_stats = result.feature_stats['returns']
        assert 'count' in returns_stats
        assert 'mean' in returns_stats
        assert 'std' in returns_stats
        assert 'min' in returns_stats
        assert 'max' in returns_stats
        
        # Statistics should be reasonable
        assert returns_stats['count'] > 0
        assert isinstance(returns_stats['mean'], float)
        assert returns_stats['std'] >= 0
    
    def test_target_statistics_calculation(self, validator, valid_timeseries_features):
        """Test target variable statistics calculation."""
        result = validator.validate_time_series_features(valid_timeseries_features)
        
        # Check target statistics
        target_stats = result.feature_stats['targets']
        
        assert 'total_count' in target_stats
        assert 'valid_count' in target_stats
        assert 'missing_count' in target_stats
        
        if target_stats['valid_count'] > 0:
            assert 'mean' in target_stats
            assert 'std' in target_stats
            assert 'min' in target_stats
            assert 'max' in target_stats
    
    def test_custom_validation_config(self):
        """Test validator with custom configuration."""
        config = FeatureComputationConfig(
            max_missing_ratio=0.05,  # Very strict
            outlier_threshold=3.0,   # More sensitive
            min_data_points=100      # Higher minimum
        )
        
        validator = FeatureValidator(config)
        
        assert validator.config.max_missing_ratio == 0.05
        assert validator.config.outlier_threshold == 3.0
        assert validator.config.min_data_points == 100
    
    def test_validation_with_multiple_symbols(self, validator):
        """Test validation with multiple symbols."""
        features = []
        symbols = ["AAPL", "GOOGL", "MSFT"]
        base_date = datetime(2024, 1, 1)
        
        for symbol in symbols:
            for i in range(20):
                feature = TechnicalFeatures(
                    symbol=symbol,
                    timestamp=base_date + timedelta(days=i)
                )
                feature.returns = {'simple_return': np.random.normal(0.001, 0.02)}
                features.append(feature)
        
        result = validator.validate_technical_features(features)
        
        # Should handle multiple symbols
        assert result.passed_checks > 0
        
        # Should detect reasonable number of symbols
        symbols_in_data = set(f.symbol for f in features)
        assert len(symbols_in_data) == 3