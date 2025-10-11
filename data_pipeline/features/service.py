"""Main feature engineering service."""

import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, timedelta
import logging

from ..ingestion.models import PriceData
from .models import (
    TechnicalFeatures, TimeSeriesFeatures, FeatureValidationResult,
    FeatureComputationConfig, CacheResult
)
from .indicators import TechnicalIndicators
from .windows import TimeWindowGenerator
from .validator import FeatureValidator
from .cache import FeatureCache
from .storage import FeatureStorage

logger = logging.getLogger(__name__)


class FeatureEngineeringService:
    """Main service for feature engineering with caching capabilities."""
    
    def __init__(self, config: Optional[FeatureComputationConfig] = None,
                 cache_dir: str = "data/cache/features",
                 storage_dir: str = "data/features"):
        """Initialize feature engineering service.
        
        Args:
            config: Feature computation configuration
            cache_dir: Directory for feature caching
            storage_dir: Directory for feature storage
        """
        self.config = config or FeatureComputationConfig()
        
        # Initialize components
        self.indicators = TechnicalIndicators(self.config)
        self.window_generator = TimeWindowGenerator(self.config)
        self.validator = FeatureValidator(self.config)
        self.cache = FeatureCache(cache_dir)
        self.storage = FeatureStorage(storage_dir)
        
        logger.info("Feature engineering service initialized")
    
    async def compute_technical_indicators(self, price_data: List[PriceData]) -> TechnicalFeatures:
        """Compute technical indicators from price data."""
        if not price_data:
            raise ValueError("No price data provided")
        
        # Check cache first
        symbol = price_data[0].symbol
        start_date = min(p.timestamp for p in price_data)
        end_date = max(p.timestamp for p in price_data)
        
        cached_features = self.cache.get_technical_features(symbol, start_date, end_date)
        if cached_features:
            logger.info(f"Using cached technical features for {symbol}")
            return cached_features
        
        # Compute features
        logger.info(f"Computing technical indicators for {symbol} ({len(price_data)} data points)")
        features = self.indicators.compute_all_features(price_data)
        
        # Cache the results
        if features:
            cache_result = self.cache.cache_technical_features(features)
            if cache_result.success:
                logger.debug(f"Cached technical features: {cache_result.cache_key}")
        
        return features
    
    async def create_time_windows(self, features: List[TechnicalFeatures], 
                                window_sizes: Optional[List[int]] = None) -> List[TimeSeriesFeatures]:
        """Create time windows for ML training data."""
        if not features:
            raise ValueError("No technical features provided")
        
        window_sizes = window_sizes or self.config.time_windows
        
        # Check cache for each window size
        symbol = features[0].symbol
        start_date = min(f.timestamp for f in features)
        end_date = max(f.timestamp for f in features)
        
        all_cached_features = []
        missing_windows = []
        
        for window_size in window_sizes:
            cached_ts_features = self.cache.get_timeseries_features(
                symbol, window_size, start_date, end_date
            )
            if cached_ts_features:
                all_cached_features.extend(cached_ts_features)
                logger.info(f"Using cached time series features for {symbol} (window: {window_size})")
            else:
                missing_windows.append(window_size)
        
        # Compute missing windows
        if missing_windows:
            logger.info(f"Computing time windows for {symbol} (windows: {missing_windows})")
            new_features = self.window_generator.create_time_windows(features, missing_windows)
            
            # Cache the new features
            if new_features:
                cache_result = self.cache.cache_timeseries_features(new_features)
                if cache_result.success:
                    logger.debug(f"Cached time series features: {cache_result.cache_key}")
            
            all_cached_features.extend(new_features)
        
        return all_cached_features
    
    async def validate_features(self, features: List[TechnicalFeatures]) -> FeatureValidationResult:
        """Validate technical features."""
        logger.info(f"Validating {len(features)} technical features")
        return self.validator.validate_technical_features(features)
    
    async def validate_time_series_features(self, ts_features: List[TimeSeriesFeatures]) -> FeatureValidationResult:
        """Validate time series features."""
        logger.info(f"Validating {len(ts_features)} time series features")
        return self.validator.validate_time_series_features(ts_features)
    
    async def cache_features(self, features: List[TechnicalFeatures]) -> CacheResult:
        """Cache technical features."""
        return self.cache.cache_technical_features(features)
    
    async def cache_time_series_features(self, ts_features: List[TimeSeriesFeatures]) -> CacheResult:
        """Cache time series features."""
        return self.cache.cache_timeseries_features(ts_features)
    
    async def store_features(self, features: List[TechnicalFeatures]) -> bool:
        """Store technical features to persistent storage."""
        logger.info(f"Storing {len(features)} technical features")
        return self.storage.store_technical_features(features)
    
    async def store_time_series_features(self, ts_features: List[TimeSeriesFeatures]) -> bool:
        """Store time series features to persistent storage."""
        logger.info(f"Storing {len(ts_features)} time series features")
        return self.storage.store_timeseries_features(ts_features)
    
    async def retrieve_features(self, symbol: str, start_date: date, 
                              end_date: date) -> List[TechnicalFeatures]:
        """Retrieve technical features from storage."""
        logger.info(f"Retrieving technical features for {symbol} from {start_date} to {end_date}")
        return self.storage.retrieve_technical_features(symbol, start_date, end_date)
    
    async def retrieve_time_series_features(self, symbol: str, 
                                          window_size: int) -> List[TimeSeriesFeatures]:
        """Retrieve time series features from storage."""
        logger.info(f"Retrieving time series features for {symbol} (window: {window_size})")
        return self.storage.retrieve_timeseries_features(symbol, window_size)
    
    async def process_price_data_pipeline(self, price_data: List[PriceData], 
                                        store_results: bool = True,
                                        validate_results: bool = True) -> Dict[str, any]:
        """Complete pipeline: compute features, create windows, validate, and store."""
        if not price_data:
            raise ValueError("No price data provided")
        
        symbol = price_data[0].symbol
        logger.info(f"Starting feature engineering pipeline for {symbol}")
        
        results = {
            'symbol': symbol,
            'input_data_points': len(price_data),
            'technical_features': [],
            'time_series_features': [],
            'validation_results': {},
            'storage_results': {},
            'errors': []
        }
        
        try:
            # Step 1: Compute technical indicators
            logger.info("Step 1: Computing technical indicators")
            technical_features = await self.compute_technical_indicators(price_data)
            results['technical_features'] = technical_features
            results['technical_feature_count'] = len(technical_features)
            
            if not technical_features:
                results['errors'].append("No technical features computed")
                return results
            
            # Step 2: Validate technical features
            if validate_results:
                logger.info("Step 2: Validating technical features")
                tech_validation = await self.validate_features(technical_features)
                results['validation_results']['technical'] = tech_validation
                
                if not tech_validation.is_valid:
                    logger.warning(f"Technical feature validation failed: {tech_validation.errors}")
                    results['errors'].extend(tech_validation.errors)
            
            # Step 3: Create time windows
            logger.info("Step 3: Creating time windows")
            time_series_features = await self.create_time_windows(technical_features)
            results['time_series_features'] = time_series_features
            results['time_series_feature_count'] = len(time_series_features)
            
            # Step 4: Validate time series features
            if validate_results and time_series_features:
                logger.info("Step 4: Validating time series features")
                ts_validation = await self.validate_time_series_features(time_series_features)
                results['validation_results']['time_series'] = ts_validation
                
                if not ts_validation.is_valid:
                    logger.warning(f"Time series feature validation failed: {ts_validation.errors}")
                    results['errors'].extend(ts_validation.errors)
            
            # Step 5: Store results
            if store_results:
                logger.info("Step 5: Storing features")
                
                # Store technical features
                tech_stored = await self.store_features(technical_features)
                results['storage_results']['technical_stored'] = tech_stored
                
                # Store time series features
                if time_series_features:
                    ts_stored = await self.store_time_series_features(time_series_features)
                    results['storage_results']['time_series_stored'] = ts_stored
            
            logger.info(f"Feature engineering pipeline completed for {symbol}")
            
        except Exception as e:
            logger.error(f"Feature engineering pipeline failed for {symbol}: {e}")
            results['errors'].append(str(e))
        
        return results
    
    async def batch_process_symbols(self, symbol_price_data: Dict[str, List[PriceData]],
                                  max_concurrent: int = 3) -> Dict[str, Dict[str, any]]:
        """Process multiple symbols concurrently."""
        logger.info(f"Starting batch processing for {len(symbol_price_data)} symbols")
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_symbol(symbol: str, price_data: List[PriceData]) -> Tuple[str, Dict[str, any]]:
            async with semaphore:
                try:
                    result = await self.process_price_data_pipeline(price_data)
                    return symbol, result
                except Exception as e:
                    logger.error(f"Failed to process {symbol}: {e}")
                    return symbol, {'errors': [str(e)]}
        
        # Process all symbols concurrently
        tasks = [
            process_symbol(symbol, price_data)
            for symbol, price_data in symbol_price_data.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize results
        batch_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
                continue
            
            symbol, symbol_result = result
            batch_results[symbol] = symbol_result
        
        logger.info(f"Batch processing completed for {len(batch_results)} symbols")
        return batch_results
    
    def get_available_symbols(self, feature_type: str = "technical") -> List[str]:
        """Get list of available symbols in storage."""
        return self.storage.list_available_symbols(feature_type)
    
    def get_symbol_date_range(self, symbol: str, feature_type: str = "technical") -> Tuple[Optional[date], Optional[date]]:
        """Get date range of available data for a symbol."""
        return self.storage.get_date_range(symbol, feature_type)
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        return self.cache.get_cache_stats()
    
    def get_storage_stats(self) -> Dict[str, any]:
        """Get storage statistics."""
        return self.storage.get_storage_stats()
    
    def clear_cache(self, symbol: Optional[str] = None, 
                   feature_type: Optional[str] = None) -> int:
        """Clear cache entries."""
        return self.cache.clear_cache(symbol, feature_type)
    
    async def create_training_dataset(self, symbols: List[str], 
                                    window_size: int = 30) -> Tuple[List[any], List[float]]:
        """Create training dataset from stored time series features."""
        logger.info(f"Creating training dataset for {len(symbols)} symbols (window: {window_size})")
        
        all_features = []
        
        # Retrieve time series features for all symbols
        for symbol in symbols:
            ts_features = await self.retrieve_time_series_features(symbol, window_size)
            all_features.extend(ts_features)
        
        if not all_features:
            logger.warning("No time series features found for training dataset")
            return [], []
        
        # Convert to training format
        X, y = self.window_generator.create_training_dataset(all_features)
        
        logger.info(f"Created training dataset: {X.shape[0]} samples, {X.shape[1] if len(X.shape) > 1 else 0} features")
        
        return X.tolist() if len(X) > 0 else [], y.tolist() if len(y) > 0 else []
    
    def get_feature_names(self, window_size: int = 30) -> List[str]:
        """Get feature names for the training dataset."""
        # Get a sample to extract feature names
        symbols = self.get_available_symbols("timeseries")
        if not symbols:
            return []
        
        # Try to get features for the first available symbol
        for symbol in symbols[:3]:  # Try first 3 symbols
            try:
                ts_features = self.storage.retrieve_timeseries_features(symbol, window_size)
                if ts_features:
                    return self.window_generator.get_feature_names(ts_features)
            except Exception:
                continue
        
        return []