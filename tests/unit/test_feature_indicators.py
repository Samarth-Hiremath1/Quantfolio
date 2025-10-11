"""Unit tests for technical indicators calculation."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List

from data_pipeline.ingestion.models import PriceData
from data_pipeline.features.indicators import TechnicalIndicators
from data_pipeline.features.models import FeatureComputationConfig


class TestTechnicalIndicators:
    """Test technical indicators calculation."""
    
    @pytest.fixture
    def sample_price_data(self) -> List[PriceData]:
        """Create sample price data for testing."""
        base_date = datetime(2024, 1, 1)
        data = []
        
        # Create 50 days of sample data with some patterns
        for i in range(50):
            price = 100 + 5 * np.sin(i * 0.1) + np.random.normal(0, 1)
            high = price + np.random.uniform(0.5, 2.0)
            low = price - np.random.uniform(0.5, 2.0)
            volume = int(1000000 + np.random.uniform(-200000, 200000))
            
            data.append(PriceData(
                symbol="AAPL",
                timestamp=base_date + timedelta(days=i),
                open=price + np.random.uniform(-0.5, 0.5),
                high=high,
                low=low,
                close=price,
                volume=volume,
                adjusted_close=price * 0.98  # Slight adjustment
            ))
        
        return data
    
    @pytest.fixture
    def indicators(self) -> TechnicalIndicators:
        """Create TechnicalIndicators instance."""
        config = FeatureComputationConfig()
        return TechnicalIndicators(config)
    
    def test_compute_all_features_empty_data(self, indicators):
        """Test computing features with empty data."""
        result = indicators.compute_all_features([])
        assert result == []
    
    def test_compute_all_features_single_point(self, indicators):
        """Test computing features with single data point."""
        data = [PriceData(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1),
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=1000000,
            adjusted_close=100.5
        )]
        
        result = indicators.compute_all_features(data)
        assert len(result) == 0  # Need at least 2 points
    
    def test_compute_all_features_basic(self, indicators, sample_price_data):
        """Test basic feature computation."""
        result = indicators.compute_all_features(sample_price_data)
        
        assert len(result) > 0
        assert len(result) == len(sample_price_data) - 1  # First point used for comparison
        
        # Check that all features have the correct symbol
        for feature in result:
            assert feature.symbol == "AAPL"
            assert isinstance(feature.timestamp, datetime)
    
    def test_returns_calculation(self, indicators, sample_price_data):
        """Test returns calculation."""
        result = indicators.compute_all_features(sample_price_data)
        
        # Check that returns are calculated
        for feature in result:
            assert 'simple_return' in feature.returns
            assert isinstance(feature.returns['simple_return'], float)
            
            # Returns should be reasonable (not extreme)
            assert -0.5 < feature.returns['simple_return'] < 0.5
    
    def test_moving_averages_calculation(self, indicators, sample_price_data):
        """Test moving averages calculation."""
        result = indicators.compute_all_features(sample_price_data)
        
        # Check later features (after enough data for MA calculation)
        later_features = result[20:]  # Skip first 20 to ensure enough data
        
        for feature in later_features:
            # Should have some moving averages
            assert len(feature.moving_averages) > 0
            
            # Check specific MAs
            if 'sma_5' in feature.moving_averages:
                assert isinstance(feature.moving_averages['sma_5'], float)
                assert feature.moving_averages['sma_5'] > 0
    
    def test_momentum_indicators_calculation(self, indicators, sample_price_data):
        """Test momentum indicators calculation."""
        result = indicators.compute_all_features(sample_price_data)
        
        # Check later features (after enough data for momentum calculation)
        later_features = result[20:]
        
        for feature in later_features:
            # Should have some momentum indicators
            if 'rsi' in feature.momentum:
                rsi = feature.momentum['rsi']
                assert isinstance(rsi, float)
                assert 0 <= rsi <= 100  # RSI should be between 0 and 100
    
    def test_volatility_calculation(self, indicators, sample_price_data):
        """Test volatility calculation."""
        result = indicators.compute_all_features(sample_price_data)
        
        # Check later features
        later_features = result[25:]  # Need enough data for volatility
        
        for feature in later_features:
            if 'realized_volatility' in feature.volatility:
                vol = feature.volatility['realized_volatility']
                assert isinstance(vol, float)
                assert vol >= 0  # Volatility should be non-negative
    
    def test_price_features_calculation(self, indicators, sample_price_data):
        """Test price features calculation."""
        result = indicators.compute_all_features(sample_price_data)
        
        for feature in result:
            # Should have some price features
            assert len(feature.price_features) > 0
            
            if 'high_low_ratio' in feature.price_features:
                ratio = feature.price_features['high_low_ratio']
                assert isinstance(ratio, float)
                assert ratio >= 1.0  # High should be >= low
    
    def test_rsi_calculation_edge_cases(self, indicators):
        """Test RSI calculation with edge cases."""
        # Create data with only gains
        data = []
        base_price = 100.0
        for i in range(20):
            price = base_price + i  # Only increasing prices
            data.append(PriceData(
                symbol="TEST",
                timestamp=datetime(2024, 1, 1) + timedelta(days=i),
                open=price,
                high=price + 1,
                low=price - 0.5,
                close=price,
                volume=1000000,
                adjusted_close=price
            ))
        
        result = indicators.compute_all_features(data)
        
        # Check RSI for later features
        later_features = result[15:]
        for feature in later_features:
            if 'rsi' in feature.momentum:
                rsi = feature.momentum['rsi']
                assert 50 <= rsi <= 100  # Should be high due to only gains
    
    def test_custom_config(self):
        """Test indicators with custom configuration."""
        config = FeatureComputationConfig(
            sma_periods=[10, 30],
            rsi_period=21,
            volatility_window=15
        )
        
        indicators = TechnicalIndicators(config)
        assert indicators.config.sma_periods == [10, 30]
        assert indicators.config.rsi_period == 21
        assert indicators.config.volatility_window == 15
    
    def test_data_sorting(self, indicators):
        """Test that data is properly sorted by timestamp."""
        # Create unsorted data
        data = []
        dates = [
            datetime(2024, 1, 3),
            datetime(2024, 1, 1),
            datetime(2024, 1, 2)
        ]
        
        for i, date in enumerate(dates):
            data.append(PriceData(
                symbol="TEST",
                timestamp=date,
                open=100.0 + i,
                high=102.0 + i,
                low=99.0 + i,
                close=101.0 + i,
                volume=1000000,
                adjusted_close=100.5 + i
            ))
        
        result = indicators.compute_all_features(data)
        
        # Check that result is sorted by timestamp
        for i in range(1, len(result)):
            assert result[i].timestamp > result[i-1].timestamp
    
    def test_missing_data_handling(self, indicators):
        """Test handling of missing or invalid data."""
        data = [
            PriceData(
                symbol="TEST",
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=1000000,
                adjusted_close=100.5
            ),
            PriceData(
                symbol="TEST",
                timestamp=datetime(2024, 1, 2),
                open=0.0,  # Invalid price
                high=0.0,
                low=0.0,
                close=0.0,
                volume=0,
                adjusted_close=0.0
            )
        ]
        
        # Should not crash with invalid data
        result = indicators.compute_all_features(data)
        assert len(result) >= 0  # Should handle gracefully
    
    def test_feature_consistency(self, indicators, sample_price_data):
        """Test that features are consistent across runs."""
        result1 = indicators.compute_all_features(sample_price_data)
        result2 = indicators.compute_all_features(sample_price_data)
        
        assert len(result1) == len(result2)
        
        # Check that same timestamps produce same results
        for f1, f2 in zip(result1, result2):
            assert f1.timestamp == f2.timestamp
            assert f1.symbol == f2.symbol
            
            # Check returns are identical
            for key in f1.returns:
                if key in f2.returns:
                    assert abs(f1.returns[key] - f2.returns[key]) < 1e-10