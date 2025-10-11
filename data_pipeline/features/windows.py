"""Time window generation for ML training data."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .models import TechnicalFeatures, TimeSeriesFeatures, FeatureComputationConfig

logger = logging.getLogger(__name__)


class TimeWindowGenerator:
    """Generate time windows for ML training data."""
    
    def __init__(self, config: Optional[FeatureComputationConfig] = None):
        """Initialize with configuration."""
        self.config = config or FeatureComputationConfig()
    
    def create_time_windows(self, features: List[TechnicalFeatures], 
                          window_sizes: Optional[List[int]] = None) -> List[TimeSeriesFeatures]:
        """Create time windows from technical features."""
        if not features:
            return []
        
        window_sizes = window_sizes or self.config.time_windows
        
        # Sort features by timestamp
        features_sorted = sorted(features, key=lambda x: x.timestamp)
        
        # Group by symbol
        symbol_features = self._group_by_symbol(features_sorted)
        
        all_windows = []
        
        for symbol, symbol_feature_list in symbol_features.items():
            for window_size in window_sizes:
                windows = self._create_windows_for_symbol(symbol_feature_list, window_size)
                all_windows.extend(windows)
        
        return all_windows
    
    def _group_by_symbol(self, features: List[TechnicalFeatures]) -> Dict[str, List[TechnicalFeatures]]:
        """Group features by symbol."""
        symbol_groups = {}
        
        for feature in features:
            if feature.symbol not in symbol_groups:
                symbol_groups[feature.symbol] = []
            symbol_groups[feature.symbol].append(feature)
        
        return symbol_groups
    
    def _create_windows_for_symbol(self, features: List[TechnicalFeatures], 
                                 window_size: int) -> List[TimeSeriesFeatures]:
        """Create time windows for a single symbol."""
        windows = []
        
        if len(features) < window_size + 1:  # Need window + 1 for target
            logger.warning(f"Not enough data points for symbol {features[0].symbol}. "
                         f"Need {window_size + 1}, have {len(features)}")
            return windows
        
        # Create sliding windows
        for i in range(window_size, len(features)):
            window_features = features[i-window_size:i]
            target_feature = features[i]
            
            # Create time series features
            ts_features = self._create_time_series_features(
                window_features, target_feature, window_size
            )
            
            if ts_features:
                windows.append(ts_features)
        
        return windows
    
    def _create_time_series_features(self, window_features: List[TechnicalFeatures],
                                   target_feature: TechnicalFeatures,
                                   window_size: int) -> Optional[TimeSeriesFeatures]:
        """Create time series features from a window of technical features."""
        if not window_features or not target_feature:
            return None
        
        symbol = window_features[0].symbol
        target_date = target_feature.timestamp
        
        # Initialize time series features
        ts_features = TimeSeriesFeatures(
            symbol=symbol,
            target_date=target_date,
            window_size=window_size,
            start_date=window_features[0].timestamp,
            end_date=window_features[-1].timestamp
        )
        
        # Extract feature sequences
        feature_sequences = {}
        
        # Process each feature type
        self._extract_return_sequences(window_features, feature_sequences)
        self._extract_ma_sequences(window_features, feature_sequences)
        self._extract_momentum_sequences(window_features, feature_sequences)
        self._extract_volatility_sequences(window_features, feature_sequences)
        self._extract_price_sequences(window_features, feature_sequences)
        
        # Set target return
        target_return = target_feature.returns.get('simple_return')
        ts_features.target_return = target_return
        
        # Calculate missing days
        expected_days = (ts_features.end_date - ts_features.start_date).days + 1
        actual_days = len(window_features)
        ts_features.missing_days = max(0, expected_days - actual_days)
        
        # Validate minimum data quality
        if ts_features.missing_days / expected_days > self.config.max_missing_ratio:
            logger.warning(f"Too many missing days for {symbol} at {target_date}: "
                         f"{ts_features.missing_days}/{expected_days}")
            return None
        
        ts_features.feature_sequences = feature_sequences
        
        return ts_features
    
    def _extract_return_sequences(self, features: List[TechnicalFeatures], 
                                sequences: Dict[str, List[float]]) -> None:
        """Extract return sequences."""
        return_types = ['simple_return', 'log_return', 'adjusted_return']
        
        for return_type in return_types:
            sequence = []
            for feature in features:
                value = feature.returns.get(return_type, 0.0)
                sequence.append(value)
            
            if any(v != 0.0 for v in sequence):  # Only add if not all zeros
                sequences[return_type] = sequence
    
    def _extract_ma_sequences(self, features: List[TechnicalFeatures], 
                            sequences: Dict[str, List[float]]) -> None:
        """Extract moving average sequences."""
        # Get all MA keys from first feature that has them
        ma_keys = set()
        for feature in features:
            ma_keys.update(feature.moving_averages.keys())
        
        for ma_key in ma_keys:
            sequence = []
            for feature in features:
                value = feature.moving_averages.get(ma_key, np.nan)
                sequence.append(value)
            
            # Only add if we have enough valid values
            valid_count = sum(1 for v in sequence if not np.isnan(v))
            if valid_count >= len(sequence) * 0.5:  # At least 50% valid
                # Forward fill NaN values
                sequence = self._forward_fill(sequence)
                sequences[f'ma_{ma_key}'] = sequence
    
    def _extract_momentum_sequences(self, features: List[TechnicalFeatures], 
                                  sequences: Dict[str, List[float]]) -> None:
        """Extract momentum indicator sequences."""
        momentum_keys = set()
        for feature in features:
            momentum_keys.update(feature.momentum.keys())
        
        for momentum_key in momentum_keys:
            sequence = []
            for feature in features:
                value = feature.momentum.get(momentum_key, np.nan)
                sequence.append(value)
            
            valid_count = sum(1 for v in sequence if not np.isnan(v))
            if valid_count >= len(sequence) * 0.3:  # At least 30% valid for momentum
                sequence = self._forward_fill(sequence)
                sequences[f'momentum_{momentum_key}'] = sequence
    
    def _extract_volatility_sequences(self, features: List[TechnicalFeatures], 
                                    sequences: Dict[str, List[float]]) -> None:
        """Extract volatility sequences."""
        vol_keys = set()
        for feature in features:
            vol_keys.update(feature.volatility.keys())
        
        for vol_key in vol_keys:
            sequence = []
            for feature in features:
                value = feature.volatility.get(vol_key, np.nan)
                sequence.append(value)
            
            valid_count = sum(1 for v in sequence if not np.isnan(v))
            if valid_count >= len(sequence) * 0.3:
                sequence = self._forward_fill(sequence)
                sequences[f'vol_{vol_key}'] = sequence
    
    def _extract_price_sequences(self, features: List[TechnicalFeatures], 
                               sequences: Dict[str, List[float]]) -> None:
        """Extract price feature sequences."""
        price_keys = set()
        for feature in features:
            price_keys.update(feature.price_features.keys())
        
        for price_key in price_keys:
            sequence = []
            for feature in features:
                value = feature.price_features.get(price_key, np.nan)
                sequence.append(value)
            
            valid_count = sum(1 for v in sequence if not np.isnan(v))
            if valid_count >= len(sequence) * 0.5:
                sequence = self._forward_fill(sequence)
                sequences[f'price_{price_key}'] = sequence
    
    def _forward_fill(self, sequence: List[float]) -> List[float]:
        """Forward fill NaN values in sequence."""
        filled = []
        last_valid = 0.0
        
        for value in sequence:
            if not np.isnan(value):
                last_valid = value
                filled.append(value)
            else:
                filled.append(last_valid)
        
        return filled
    
    def create_training_dataset(self, time_series_features: List[TimeSeriesFeatures]) -> Tuple[np.ndarray, np.ndarray]:
        """Create training dataset from time series features."""
        if not time_series_features:
            return np.array([]), np.array([])
        
        # Filter out samples without target
        valid_samples = [ts for ts in time_series_features if ts.target_return is not None]
        
        if not valid_samples:
            logger.warning("No valid samples with target returns found")
            return np.array([]), np.array([])
        
        # Get feature names from first sample
        feature_names = list(valid_samples[0].feature_sequences.keys())
        
        # Create feature matrix
        X = []
        y = []
        
        for ts in valid_samples:
            # Create feature vector by concatenating all sequences
            feature_vector = []
            
            for feature_name in feature_names:
                sequence = ts.feature_sequences.get(feature_name, [])
                if sequence:
                    feature_vector.extend(sequence)
                else:
                    # Pad with zeros if missing
                    feature_vector.extend([0.0] * ts.window_size)
            
            if feature_vector:  # Only add if we have features
                X.append(feature_vector)
                y.append(ts.target_return)
        
        return np.array(X), np.array(y)
    
    def get_feature_names(self, time_series_features: List[TimeSeriesFeatures]) -> List[str]:
        """Get feature names for the dataset."""
        if not time_series_features:
            return []
        
        feature_names = []
        sample = time_series_features[0]
        
        for feature_type in sample.feature_sequences.keys():
            for i in range(sample.window_size):
                feature_names.append(f"{feature_type}_t-{sample.window_size-i-1}")
        
        return feature_names