"""Feature validation and quality assurance."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging
from scipy import stats

from .models import (
    TechnicalFeatures, TimeSeriesFeatures, FeatureValidationResult, 
    FeatureValidationStatus, FeatureComputationConfig
)

logger = logging.getLogger(__name__)


class FeatureValidator:
    """Validate features and ensure data quality."""
    
    def __init__(self, config: Optional[FeatureComputationConfig] = None):
        """Initialize with configuration."""
        self.config = config or FeatureComputationConfig()
    
    def validate_technical_features(self, features: List[TechnicalFeatures]) -> FeatureValidationResult:
        """Validate technical features."""
        if not features:
            return FeatureValidationResult(
                status=FeatureValidationStatus.FAILED,
                passed_checks=0,
                total_checks=1,
                errors=["No features provided for validation"]
            )
        
        errors = []
        warnings = []
        passed_checks = 0
        total_checks = 0
        feature_stats = {}
        
        # Check 1: Basic data presence
        total_checks += 1
        if features:
            passed_checks += 1
        else:
            errors.append("No features to validate")
        
        # Check 2: Symbol consistency
        total_checks += 1
        symbols = set(f.symbol for f in features)
        if len(symbols) <= 10:  # Reasonable number of symbols
            passed_checks += 1
        else:
            warnings.append(f"Large number of symbols: {len(symbols)}")
        
        # Check 3: Timestamp ordering
        total_checks += 1
        timestamp_issues = self._check_timestamp_ordering(features)
        if not timestamp_issues:
            passed_checks += 1
        else:
            errors.extend(timestamp_issues)
        
        # Check 4: Feature value ranges
        total_checks += 1
        range_issues, stats_dict = self._check_feature_ranges(features)
        feature_stats.update(stats_dict)
        if not range_issues:
            passed_checks += 1
        else:
            warnings.extend(range_issues)
        
        # Check 5: Missing values
        total_checks += 1
        missing_issues = self._check_missing_values(features)
        if not missing_issues:
            passed_checks += 1
        else:
            warnings.extend(missing_issues)
        
        # Check 6: Outlier detection
        total_checks += 1
        outlier_issues = self._detect_outliers(features)
        if not outlier_issues:
            passed_checks += 1
        else:
            warnings.extend(outlier_issues)
        
        # Determine overall status
        if errors:
            status = FeatureValidationStatus.FAILED
        elif warnings:
            status = FeatureValidationStatus.WARNING
        else:
            status = FeatureValidationStatus.PASSED
        
        return FeatureValidationResult(
            status=status,
            passed_checks=passed_checks,
            total_checks=total_checks,
            errors=errors,
            warnings=warnings,
            feature_stats=feature_stats
        )
    
    def validate_time_series_features(self, ts_features: List[TimeSeriesFeatures]) -> FeatureValidationResult:
        """Validate time series features."""
        if not ts_features:
            return FeatureValidationResult(
                status=FeatureValidationStatus.FAILED,
                passed_checks=0,
                total_checks=1,
                errors=["No time series features provided for validation"]
            )
        
        errors = []
        warnings = []
        passed_checks = 0
        total_checks = 0
        feature_stats = {}
        
        # Check 1: Basic data presence
        total_checks += 1
        if ts_features:
            passed_checks += 1
        
        # Check 2: Window size consistency
        total_checks += 1
        window_sizes = set(ts.window_size for ts in ts_features)
        if len(window_sizes) <= 3:  # Reasonable number of window sizes
            passed_checks += 1
            feature_stats['window_sizes'] = {'values': list(window_sizes)}
        else:
            warnings.append(f"Many different window sizes: {window_sizes}")
        
        # Check 3: Feature sequence lengths
        total_checks += 1
        length_issues = self._check_sequence_lengths(ts_features)
        if not length_issues:
            passed_checks += 1
        else:
            errors.extend(length_issues)
        
        # Check 4: Target variable presence
        total_checks += 1
        target_stats = self._check_target_variables(ts_features)
        feature_stats['targets'] = target_stats
        if target_stats['missing_count'] == 0:
            passed_checks += 1
        elif target_stats['missing_count'] < len(ts_features) * 0.1:
            passed_checks += 1
            warnings.append(f"Some missing targets: {target_stats['missing_count']}/{len(ts_features)}")
        else:
            errors.append(f"Too many missing targets: {target_stats['missing_count']}/{len(ts_features)}")
        
        # Check 5: Feature sequence statistics
        total_checks += 1
        sequence_stats = self._analyze_sequence_statistics(ts_features)
        feature_stats['sequences'] = sequence_stats
        if sequence_stats.get('valid_sequences', 0) > 0:
            passed_checks += 1
        else:
            errors.append("No valid feature sequences found")
        
        # Check 6: Data quality metrics
        total_checks += 1
        quality_issues = self._check_data_quality(ts_features)
        if not quality_issues:
            passed_checks += 1
        else:
            warnings.extend(quality_issues)
        
        # Determine overall status
        if errors:
            status = FeatureValidationStatus.FAILED
        elif warnings:
            status = FeatureValidationStatus.WARNING
        else:
            status = FeatureValidationStatus.PASSED
        
        return FeatureValidationResult(
            status=status,
            passed_checks=passed_checks,
            total_checks=total_checks,
            errors=errors,
            warnings=warnings,
            feature_stats=feature_stats
        )
    
    def _check_timestamp_ordering(self, features: List[TechnicalFeatures]) -> List[str]:
        """Check if timestamps are properly ordered."""
        issues = []
        
        # Group by symbol
        symbol_groups = {}
        for feature in features:
            if feature.symbol not in symbol_groups:
                symbol_groups[feature.symbol] = []
            symbol_groups[feature.symbol].append(feature)
        
        for symbol, symbol_features in symbol_groups.items():
            timestamps = [f.timestamp for f in symbol_features]
            sorted_timestamps = sorted(timestamps)
            
            if timestamps != sorted_timestamps:
                issues.append(f"Timestamps not ordered for symbol {symbol}")
            
            # Check for duplicates
            if len(timestamps) != len(set(timestamps)):
                issues.append(f"Duplicate timestamps found for symbol {symbol}")
        
        return issues
    
    def _check_feature_ranges(self, features: List[TechnicalFeatures]) -> Tuple[List[str], Dict[str, Any]]:
        """Check if feature values are within reasonable ranges."""
        issues = []
        stats_dict = {}
        
        # Collect all feature values by type
        feature_collections = {
            'returns': [],
            'moving_averages': [],
            'momentum': [],
            'volatility': [],
            'price_features': []
        }
        
        for feature in features:
            feature_collections['returns'].extend(feature.returns.values())
            feature_collections['moving_averages'].extend(feature.moving_averages.values())
            feature_collections['momentum'].extend(feature.momentum.values())
            feature_collections['volatility'].extend(feature.volatility.values())
            feature_collections['price_features'].extend(feature.price_features.values())
        
        # Analyze each feature type
        for feature_type, values in feature_collections.items():
            if not values:
                continue
            
            values_array = np.array([v for v in values if not np.isnan(v) and np.isfinite(v)])
            
            if len(values_array) == 0:
                continue
            
            stats_dict[feature_type] = {
                'count': len(values_array),
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array))
            }
            
            # Check for extreme values
            if feature_type == 'returns':
                if np.any(np.abs(values_array) > 0.5):  # 50% daily return is extreme
                    issues.append(f"Extreme returns detected in {feature_type}")
            
            elif feature_type == 'momentum':
                # RSI should be 0-100, MACD can vary
                rsi_like = [v for v in values_array if 0 <= v <= 100]
                if len(rsi_like) > len(values_array) * 0.3:  # Likely RSI values
                    outliers = [v for v in values_array if v < 0 or v > 100]
                    if outliers:
                        issues.append(f"RSI values outside 0-100 range: {len(outliers)} values")
        
        return issues, stats_dict
    
    def _check_missing_values(self, features: List[TechnicalFeatures]) -> List[str]:
        """Check for missing values in features."""
        issues = []
        
        total_features = len(features)
        if total_features == 0:
            return issues
        
        # Count features with empty dictionaries
        empty_returns = sum(1 for f in features if not f.returns)
        empty_ma = sum(1 for f in features if not f.moving_averages)
        empty_momentum = sum(1 for f in features if not f.momentum)
        empty_volatility = sum(1 for f in features if not f.volatility)
        
        missing_threshold = total_features * self.config.max_missing_ratio
        
        if empty_returns > missing_threshold:
            issues.append(f"Too many features missing returns: {empty_returns}/{total_features}")
        
        if empty_ma > missing_threshold:
            issues.append(f"Too many features missing moving averages: {empty_ma}/{total_features}")
        
        if empty_momentum > missing_threshold * 2:  # More lenient for momentum
            issues.append(f"Too many features missing momentum indicators: {empty_momentum}/{total_features}")
        
        return issues
    
    def _detect_outliers(self, features: List[TechnicalFeatures]) -> List[str]:
        """Detect outliers in feature values."""
        issues = []
        
        # Collect returns for outlier detection
        returns = []
        for feature in features:
            returns.extend([v for v in feature.returns.values() if not np.isnan(v)])
        
        if len(returns) < 10:  # Need minimum data for outlier detection
            return issues
        
        returns_array = np.array(returns)
        
        # Use z-score for outlier detection
        z_scores = np.abs(stats.zscore(returns_array))
        outliers = np.sum(z_scores > self.config.outlier_threshold)
        
        if outliers > len(returns) * 0.05:  # More than 5% outliers
            issues.append(f"High number of outliers in returns: {outliers}/{len(returns)}")
        
        return issues
    
    def _check_sequence_lengths(self, ts_features: List[TimeSeriesFeatures]) -> List[str]:
        """Check if sequence lengths match window sizes."""
        issues = []
        
        for ts in ts_features:
            for feature_name, sequence in ts.feature_sequences.items():
                if len(sequence) != ts.window_size:
                    issues.append(
                        f"Sequence length mismatch for {ts.symbol} {feature_name}: "
                        f"expected {ts.window_size}, got {len(sequence)}"
                    )
        
        return issues
    
    def _check_target_variables(self, ts_features: List[TimeSeriesFeatures]) -> Dict[str, Any]:
        """Check target variable statistics."""
        targets = [ts.target_return for ts in ts_features if ts.target_return is not None]
        missing_count = len(ts_features) - len(targets)
        
        stats_dict = {
            'total_count': len(ts_features),
            'valid_count': len(targets),
            'missing_count': missing_count
        }
        
        if targets:
            targets_array = np.array(targets)
            stats_dict.update({
                'mean': float(np.mean(targets_array)),
                'std': float(np.std(targets_array)),
                'min': float(np.min(targets_array)),
                'max': float(np.max(targets_array))
            })
        
        return stats_dict
    
    def _analyze_sequence_statistics(self, ts_features: List[TimeSeriesFeatures]) -> Dict[str, Any]:
        """Analyze statistics of feature sequences."""
        if not ts_features:
            return {'valid_sequences': 0}
        
        # Count valid sequences
        valid_sequences = 0
        sequence_types = set()
        
        for ts in ts_features:
            for feature_name, sequence in ts.feature_sequences.items():
                if sequence and len(sequence) > 0:
                    valid_sequences += 1
                    sequence_types.add(feature_name)
        
        return {
            'valid_sequences': valid_sequences,
            'unique_sequence_types': len(sequence_types),
            'sequence_types': list(sequence_types)
        }
    
    def _check_data_quality(self, ts_features: List[TimeSeriesFeatures]) -> List[str]:
        """Check overall data quality metrics."""
        issues = []
        
        if not ts_features:
            return issues
        
        # Check missing days ratio
        high_missing_count = 0
        for ts in ts_features:
            if ts.missing_days > 0:
                expected_days = (ts.end_date - ts.start_date).days + 1 if ts.end_date and ts.start_date else ts.window_size
                missing_ratio = ts.missing_days / expected_days
                if missing_ratio > self.config.max_missing_ratio:
                    high_missing_count += 1
        
        if high_missing_count > len(ts_features) * 0.1:
            issues.append(f"Too many samples with high missing data: {high_missing_count}/{len(ts_features)}")
        
        # Check for minimum data points
        insufficient_data = sum(1 for ts in ts_features if len(ts.feature_sequences) < 3)
        if insufficient_data > len(ts_features) * 0.2:
            issues.append(f"Too many samples with insufficient features: {insufficient_data}/{len(ts_features)}")
        
        return issues