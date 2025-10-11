"""Feature caching layer for improved performance."""

import json
import hashlib
import pickle
import os
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path

from .models import TechnicalFeatures, TimeSeriesFeatures, CacheResult

logger = logging.getLogger(__name__)


class FeatureCache:
    """Cache computed features to improve performance."""
    
    def __init__(self, cache_dir: str = "data/cache/features", 
                 max_age_hours: int = 24, 
                 max_cache_size_mb: int = 500):
        """Initialize feature cache.
        
        Args:
            cache_dir: Directory to store cached features
            max_age_hours: Maximum age of cached items in hours
            max_cache_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_age = timedelta(hours=max_age_hours)
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        
        # Create subdirectories for different feature types
        (self.cache_dir / "technical").mkdir(exist_ok=True)
        (self.cache_dir / "timeseries").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)
        
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load cache metadata."""
        metadata_file = self.cache_dir / "metadata" / "cache_metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        metadata_file = self.cache_dir / "metadata" / "cache_metadata.json"
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _generate_cache_key(self, symbol: str, start_date: datetime, 
                          end_date: datetime, feature_type: str = "technical") -> str:
        """Generate cache key for features."""
        key_data = f"{symbol}_{start_date.isoformat()}_{end_date.isoformat()}_{feature_type}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str, feature_type: str = "technical") -> Path:
        """Get cache file path for a given key."""
        return self.cache_dir / feature_type / f"{cache_key}.pkl"
    
    def cache_technical_features(self, features: List[TechnicalFeatures]) -> CacheResult:
        """Cache technical features."""
        if not features:
            return CacheResult(
                success=False,
                cache_key="",
                error_message="No features to cache"
            )
        
        try:
            # Group features by symbol
            symbol_groups = {}
            for feature in features:
                if feature.symbol not in symbol_groups:
                    symbol_groups[feature.symbol] = []
                symbol_groups[feature.symbol].append(feature)
            
            cached_keys = []
            
            for symbol, symbol_features in symbol_groups.items():
                if not symbol_features:
                    continue
                
                # Sort by timestamp
                symbol_features.sort(key=lambda x: x.timestamp)
                
                start_date = symbol_features[0].timestamp
                end_date = symbol_features[-1].timestamp
                
                cache_key = self._generate_cache_key(symbol, start_date, end_date, "technical")
                cache_file = self._get_cache_file_path(cache_key, "technical")
                
                # Serialize features
                serialized_features = [f.to_dict() for f in symbol_features]
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(serialized_features, f)
                
                # Update metadata
                self.metadata[cache_key] = {
                    'symbol': symbol,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'feature_type': 'technical',
                    'cached_at': datetime.now().isoformat(),
                    'file_size': cache_file.stat().st_size,
                    'feature_count': len(symbol_features)
                }
                
                cached_keys.append(cache_key)
            
            self._save_metadata()
            self._cleanup_old_cache()
            
            return CacheResult(
                success=True,
                cache_key=",".join(cached_keys),
                cached_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to cache technical features: {e}")
            return CacheResult(
                success=False,
                cache_key="",
                error_message=str(e)
            )
    
    def get_technical_features(self, symbol: str, start_date: datetime, 
                             end_date: datetime) -> Optional[List[TechnicalFeatures]]:
        """Retrieve cached technical features."""
        cache_key = self._generate_cache_key(symbol, start_date, end_date, "technical")
        
        # Check if cache exists and is valid
        if not self._is_cache_valid(cache_key):
            return None
        
        try:
            cache_file = self._get_cache_file_path(cache_key, "technical")
            
            with open(cache_file, 'rb') as f:
                serialized_features = pickle.load(f)
            
            # Deserialize features
            features = [TechnicalFeatures.from_dict(data) for data in serialized_features]
            
            logger.info(f"Cache hit for technical features: {symbol} ({len(features)} features)")
            return features
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached technical features: {e}")
            return None
    
    def cache_timeseries_features(self, features: List[TimeSeriesFeatures]) -> CacheResult:
        """Cache time series features."""
        if not features:
            return CacheResult(
                success=False,
                cache_key="",
                error_message="No time series features to cache"
            )
        
        try:
            # Group by symbol and window size
            groups = {}
            for feature in features:
                key = f"{feature.symbol}_{feature.window_size}"
                if key not in groups:
                    groups[key] = []
                groups[key].append(feature)
            
            cached_keys = []
            
            for group_key, group_features in groups.items():
                if not group_features:
                    continue
                
                # Sort by target date
                group_features.sort(key=lambda x: x.target_date)
                
                symbol = group_features[0].symbol
                window_size = group_features[0].window_size
                start_date = min(f.start_date for f in group_features if f.start_date)
                end_date = max(f.target_date for f in group_features)
                
                cache_key = self._generate_cache_key(
                    f"{symbol}_w{window_size}", start_date, end_date, "timeseries"
                )
                cache_file = self._get_cache_file_path(cache_key, "timeseries")
                
                # Serialize features
                serialized_features = [f.to_dict() for f in group_features]
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(serialized_features, f)
                
                # Update metadata
                self.metadata[cache_key] = {
                    'symbol': symbol,
                    'window_size': window_size,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'feature_type': 'timeseries',
                    'cached_at': datetime.now().isoformat(),
                    'file_size': cache_file.stat().st_size,
                    'feature_count': len(group_features)
                }
                
                cached_keys.append(cache_key)
            
            self._save_metadata()
            self._cleanup_old_cache()
            
            return CacheResult(
                success=True,
                cache_key=",".join(cached_keys),
                cached_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to cache time series features: {e}")
            return CacheResult(
                success=False,
                cache_key="",
                error_message=str(e)
            )
    
    def get_timeseries_features(self, symbol: str, window_size: int, 
                              start_date: datetime, end_date: datetime) -> Optional[List[TimeSeriesFeatures]]:
        """Retrieve cached time series features."""
        cache_key = self._generate_cache_key(
            f"{symbol}_w{window_size}", start_date, end_date, "timeseries"
        )
        
        # Check if cache exists and is valid
        if not self._is_cache_valid(cache_key):
            return None
        
        try:
            cache_file = self._get_cache_file_path(cache_key, "timeseries")
            
            with open(cache_file, 'rb') as f:
                serialized_features = pickle.load(f)
            
            # Deserialize features
            features = [TimeSeriesFeatures.from_dict(data) for data in serialized_features]
            
            logger.info(f"Cache hit for time series features: {symbol} w{window_size} ({len(features)} features)")
            return features
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached time series features: {e}")
            return None
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid."""
        if cache_key not in self.metadata:
            return False
        
        metadata = self.metadata[cache_key]
        
        # Check if file exists
        feature_type = metadata.get('feature_type', 'technical')
        cache_file = self._get_cache_file_path(cache_key, feature_type)
        
        if not cache_file.exists():
            # Remove from metadata if file doesn't exist
            del self.metadata[cache_key]
            self._save_metadata()
            return False
        
        # Check age
        cached_at = datetime.fromisoformat(metadata['cached_at'])
        if datetime.now() - cached_at > self.max_age:
            logger.info(f"Cache entry expired: {cache_key}")
            self._remove_cache_entry(cache_key)
            return False
        
        return True
    
    def _remove_cache_entry(self, cache_key: str) -> None:
        """Remove a cache entry."""
        if cache_key in self.metadata:
            metadata = self.metadata[cache_key]
            feature_type = metadata.get('feature_type', 'technical')
            cache_file = self._get_cache_file_path(cache_key, feature_type)
            
            try:
                if cache_file.exists():
                    cache_file.unlink()
                del self.metadata[cache_key]
                logger.info(f"Removed cache entry: {cache_key}")
            except Exception as e:
                logger.error(f"Failed to remove cache entry {cache_key}: {e}")
    
    def _cleanup_old_cache(self) -> None:
        """Clean up old cache entries."""
        # Remove expired entries
        expired_keys = []
        for cache_key in list(self.metadata.keys()):
            if not self._is_cache_valid(cache_key):
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            self._remove_cache_entry(key)
        
        # Check total cache size
        total_size = sum(
            metadata.get('file_size', 0) 
            for metadata in self.metadata.values()
        )
        
        if total_size > self.max_cache_size_bytes:
            logger.info(f"Cache size ({total_size / 1024 / 1024:.1f} MB) exceeds limit, cleaning up...")
            
            # Sort by cached_at (oldest first)
            sorted_entries = sorted(
                self.metadata.items(),
                key=lambda x: x[1].get('cached_at', '1970-01-01')
            )
            
            # Remove oldest entries until under limit
            for cache_key, metadata in sorted_entries:
                if total_size <= self.max_cache_size_bytes * 0.8:  # Leave some buffer
                    break
                
                file_size = metadata.get('file_size', 0)
                self._remove_cache_entry(cache_key)
                total_size -= file_size
        
        self._save_metadata()
    
    def clear_cache(self, symbol: Optional[str] = None, 
                   feature_type: Optional[str] = None) -> int:
        """Clear cache entries.
        
        Args:
            symbol: If provided, only clear cache for this symbol
            feature_type: If provided, only clear cache for this feature type
            
        Returns:
            Number of entries cleared
        """
        keys_to_remove = []
        
        for cache_key, metadata in self.metadata.items():
            should_remove = True
            
            if symbol and metadata.get('symbol') != symbol:
                should_remove = False
            
            if feature_type and metadata.get('feature_type') != feature_type:
                should_remove = False
            
            if should_remove:
                keys_to_remove.append(cache_key)
        
        for key in keys_to_remove:
            self._remove_cache_entry(key)
        
        self._save_metadata()
        
        logger.info(f"Cleared {len(keys_to_remove)} cache entries")
        return len(keys_to_remove)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.metadata)
        total_size = sum(metadata.get('file_size', 0) for metadata in self.metadata.values())
        
        # Group by feature type
        type_stats = {}
        for metadata in self.metadata.values():
            feature_type = metadata.get('feature_type', 'unknown')
            if feature_type not in type_stats:
                type_stats[feature_type] = {'count': 0, 'size': 0}
            
            type_stats[feature_type]['count'] += 1
            type_stats[feature_type]['size'] += metadata.get('file_size', 0)
        
        return {
            'total_entries': total_entries,
            'total_size_mb': total_size / 1024 / 1024,
            'max_size_mb': self.max_cache_size_bytes / 1024 / 1024,
            'max_age_hours': self.max_age.total_seconds() / 3600,
            'by_type': type_stats
        }