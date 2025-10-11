"""Feature storage system with efficient retrieval mechanisms."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, date
import logging
import os

from .models import TechnicalFeatures, TimeSeriesFeatures

logger = logging.getLogger(__name__)


class FeatureStorage:
    """Efficient storage and retrieval system for features."""
    
    def __init__(self, base_path: str = "data/features"):
        """Initialize feature storage.
        
        Args:
            base_path: Base directory for feature storage
        """
        self.base_path = Path(base_path)
        
        # Create directory structure
        self.technical_path = self.base_path / "technical"
        self.timeseries_path = self.base_path / "time_series"
        self.metadata_path = self.base_path / "metadata"
        
        for path in [self.technical_path, self.timeseries_path, self.metadata_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for time series windows
        for window_size in [30, 60]:
            (self.timeseries_path / f"windows_{window_size}d").mkdir(exist_ok=True)
    
    def store_technical_features(self, features: List[TechnicalFeatures]) -> bool:
        """Store technical features with date-based partitioning."""
        if not features:
            logger.warning("No technical features to store")
            return False
        
        try:
            # Group features by symbol and date
            grouped_features = self._group_technical_features(features)
            
            stored_files = []
            
            for (symbol, date_str), symbol_features in grouped_features.items():
                # Create date-based directory structure
                date_path = self.technical_path / date_str[:4] / date_str[5:7] / date_str[8:10]
                date_path.mkdir(parents=True, exist_ok=True)
                
                # Convert to DataFrame for efficient storage
                df = self._technical_features_to_dataframe(symbol_features)
                
                # Store as parquet for efficient querying
                file_path = date_path / f"{symbol}_features.parquet"
                df.to_parquet(file_path, index=False)
                
                stored_files.append(str(file_path))
                
                logger.debug(f"Stored {len(symbol_features)} technical features for {symbol} on {date_str}")
            
            # Update metadata
            self._update_technical_metadata(features, stored_files)
            
            logger.info(f"Successfully stored technical features to {len(stored_files)} files")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store technical features: {e}")
            return False
    
    def store_timeseries_features(self, features: List[TimeSeriesFeatures]) -> bool:
        """Store time series features with window-based partitioning."""
        if not features:
            logger.warning("No time series features to store")
            return False
        
        try:
            # Group by window size and symbol
            grouped_features = self._group_timeseries_features(features)
            
            stored_files = []
            
            for (window_size, symbol), symbol_features in grouped_features.items():
                # Create window-based directory
                window_path = self.timeseries_path / f"windows_{window_size}d"
                
                # Convert to DataFrame
                df = self._timeseries_features_to_dataframe(symbol_features)
                
                # Store as parquet
                file_path = window_path / f"{symbol}_timeseries.parquet"
                df.to_parquet(file_path, index=False)
                
                stored_files.append(str(file_path))
                
                logger.debug(f"Stored {len(symbol_features)} time series features for {symbol} (window: {window_size})")
            
            # Update metadata
            self._update_timeseries_metadata(features, stored_files)
            
            logger.info(f"Successfully stored time series features to {len(stored_files)} files")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store time series features: {e}")
            return False
    
    def retrieve_technical_features(self, symbol: str, start_date: date, 
                                  end_date: date) -> List[TechnicalFeatures]:
        """Retrieve technical features for a symbol and date range."""
        features = []
        
        try:
            # Generate date range
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                date_path = self.technical_path / date_str[:4] / date_str[5:7] / date_str[8:10]
                file_path = date_path / f"{symbol}_features.parquet"
                
                if file_path.exists():
                    try:
                        df = pd.read_parquet(file_path)
                        daily_features = self._dataframe_to_technical_features(df)
                        features.extend(daily_features)
                    except Exception as e:
                        logger.warning(f"Failed to read features from {file_path}: {e}")
                
                current_date += pd.Timedelta(days=1)
            
            # Sort by timestamp
            features.sort(key=lambda x: x.timestamp)
            
            logger.info(f"Retrieved {len(features)} technical features for {symbol}")
            return features
            
        except Exception as e:
            logger.error(f"Failed to retrieve technical features: {e}")
            return []
    
    def retrieve_timeseries_features(self, symbol: str, window_size: int) -> List[TimeSeriesFeatures]:
        """Retrieve time series features for a symbol and window size."""
        try:
            window_path = self.timeseries_path / f"windows_{window_size}d"
            file_path = window_path / f"{symbol}_timeseries.parquet"
            
            if not file_path.exists():
                logger.warning(f"No time series features found for {symbol} with window {window_size}")
                return []
            
            df = pd.read_parquet(file_path)
            features = self._dataframe_to_timeseries_features(df)
            
            # Sort by target date
            features.sort(key=lambda x: x.target_date)
            
            logger.info(f"Retrieved {len(features)} time series features for {symbol} (window: {window_size})")
            return features
            
        except Exception as e:
            logger.error(f"Failed to retrieve time series features: {e}")
            return []
    
    def list_available_symbols(self, feature_type: str = "technical") -> List[str]:
        """List all available symbols in storage."""
        symbols = set()
        
        try:
            if feature_type == "technical":
                # Scan technical features directory
                for year_dir in self.technical_path.iterdir():
                    if not year_dir.is_dir():
                        continue
                    for month_dir in year_dir.iterdir():
                        if not month_dir.is_dir():
                            continue
                        for day_dir in month_dir.iterdir():
                            if not day_dir.is_dir():
                                continue
                            for file_path in day_dir.glob("*_features.parquet"):
                                symbol = file_path.stem.replace("_features", "")
                                symbols.add(symbol)
            
            elif feature_type == "timeseries":
                # Scan time series features directory
                for window_dir in self.timeseries_path.iterdir():
                    if not window_dir.is_dir():
                        continue
                    for file_path in window_dir.glob("*_timeseries.parquet"):
                        symbol = file_path.stem.replace("_timeseries", "")
                        symbols.add(symbol)
            
            return sorted(list(symbols))
            
        except Exception as e:
            logger.error(f"Failed to list available symbols: {e}")
            return []
    
    def get_date_range(self, symbol: str, feature_type: str = "technical") -> Tuple[Optional[date], Optional[date]]:
        """Get the date range of available data for a symbol."""
        try:
            if feature_type == "technical":
                dates = []
                
                # Scan for all dates with data for this symbol
                for year_dir in self.technical_path.iterdir():
                    if not year_dir.is_dir():
                        continue
                    for month_dir in year_dir.iterdir():
                        if not month_dir.is_dir():
                            continue
                        for day_dir in month_dir.iterdir():
                            if not day_dir.is_dir():
                                continue
                            
                            file_path = day_dir / f"{symbol}_features.parquet"
                            if file_path.exists():
                                date_str = f"{year_dir.name}-{month_dir.name}-{day_dir.name}"
                                try:
                                    dates.append(datetime.strptime(date_str, "%Y-%m-%d").date())
                                except ValueError:
                                    continue
                
                if dates:
                    return min(dates), max(dates)
            
            elif feature_type == "timeseries":
                # For time series, read metadata from the file
                for window_dir in self.timeseries_path.iterdir():
                    if not window_dir.is_dir():
                        continue
                    
                    file_path = window_dir / f"{symbol}_timeseries.parquet"
                    if file_path.exists():
                        try:
                            df = pd.read_parquet(file_path)
                            if 'target_date' in df.columns:
                                dates = pd.to_datetime(df['target_date']).dt.date
                                return dates.min(), dates.max()
                        except Exception:
                            continue
            
            return None, None
            
        except Exception as e:
            logger.error(f"Failed to get date range for {symbol}: {e}")
            return None, None
    
    def _group_technical_features(self, features: List[TechnicalFeatures]) -> Dict[Tuple[str, str], List[TechnicalFeatures]]:
        """Group technical features by symbol and date."""
        groups = {}
        
        for feature in features:
            date_str = feature.timestamp.strftime("%Y-%m-%d")
            key = (feature.symbol, date_str)
            
            if key not in groups:
                groups[key] = []
            groups[key].append(feature)
        
        return groups
    
    def _group_timeseries_features(self, features: List[TimeSeriesFeatures]) -> Dict[Tuple[int, str], List[TimeSeriesFeatures]]:
        """Group time series features by window size and symbol."""
        groups = {}
        
        for feature in features:
            key = (feature.window_size, feature.symbol)
            
            if key not in groups:
                groups[key] = []
            groups[key].append(feature)
        
        return groups
    
    def _technical_features_to_dataframe(self, features: List[TechnicalFeatures]) -> pd.DataFrame:
        """Convert technical features to DataFrame."""
        data = []
        
        for feature in features:
            row = {
                'symbol': feature.symbol,
                'timestamp': feature.timestamp,
            }
            
            # Flatten nested dictionaries
            for key, value in feature.returns.items():
                row[f'returns_{key}'] = value
            
            for key, value in feature.moving_averages.items():
                row[f'ma_{key}'] = value
            
            for key, value in feature.momentum.items():
                row[f'momentum_{key}'] = value
            
            for key, value in feature.volatility.items():
                row[f'vol_{key}'] = value
            
            for key, value in feature.price_features.items():
                row[f'price_{key}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _timeseries_features_to_dataframe(self, features: List[TimeSeriesFeatures]) -> pd.DataFrame:
        """Convert time series features to DataFrame."""
        data = []
        
        for feature in features:
            row = {
                'symbol': feature.symbol,
                'target_date': feature.target_date,
                'window_size': feature.window_size,
                'target_return': feature.target_return,
                'start_date': feature.start_date,
                'end_date': feature.end_date,
                'missing_days': feature.missing_days,
            }
            
            # Store feature sequences as JSON strings
            row['feature_sequences'] = json.dumps(feature.feature_sequences)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _dataframe_to_technical_features(self, df: pd.DataFrame) -> List[TechnicalFeatures]:
        """Convert DataFrame to technical features."""
        features = []
        
        for _, row in df.iterrows():
            feature = TechnicalFeatures(
                symbol=row['symbol'],
                timestamp=pd.to_datetime(row['timestamp'])
            )
            
            # Extract nested dictionaries
            for col in df.columns:
                if col.startswith('returns_'):
                    key = col.replace('returns_', '')
                    if not pd.isna(row[col]):
                        feature.returns[key] = row[col]
                
                elif col.startswith('ma_'):
                    key = col.replace('ma_', '')
                    if not pd.isna(row[col]):
                        feature.moving_averages[key] = row[col]
                
                elif col.startswith('momentum_'):
                    key = col.replace('momentum_', '')
                    if not pd.isna(row[col]):
                        feature.momentum[key] = row[col]
                
                elif col.startswith('vol_'):
                    key = col.replace('vol_', '')
                    if not pd.isna(row[col]):
                        feature.volatility[key] = row[col]
                
                elif col.startswith('price_'):
                    key = col.replace('price_', '')
                    if not pd.isna(row[col]):
                        feature.price_features[key] = row[col]
            
            features.append(feature)
        
        return features
    
    def _dataframe_to_timeseries_features(self, df: pd.DataFrame) -> List[TimeSeriesFeatures]:
        """Convert DataFrame to time series features."""
        features = []
        
        for _, row in df.iterrows():
            feature = TimeSeriesFeatures(
                symbol=row['symbol'],
                target_date=pd.to_datetime(row['target_date']),
                window_size=row['window_size'],
                target_return=row['target_return'] if not pd.isna(row['target_return']) else None,
                start_date=pd.to_datetime(row['start_date']) if not pd.isna(row['start_date']) else None,
                end_date=pd.to_datetime(row['end_date']) if not pd.isna(row['end_date']) else None,
                missing_days=row['missing_days'] if not pd.isna(row['missing_days']) else 0
            )
            
            # Parse feature sequences from JSON
            if 'feature_sequences' in row and not pd.isna(row['feature_sequences']):
                try:
                    feature.feature_sequences = json.loads(row['feature_sequences'])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse feature sequences for {feature.symbol}")
                    feature.feature_sequences = {}
            
            features.append(feature)
        
        return features
    
    def _update_technical_metadata(self, features: List[TechnicalFeatures], stored_files: List[str]) -> None:
        """Update metadata for technical features."""
        metadata_file = self.metadata_path / "technical_features_metadata.json"
        
        # Load existing metadata
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}
        else:
            metadata = {}
        
        # Update with new information
        symbols = set(f.symbol for f in features)
        dates = set(f.timestamp.strftime("%Y-%m-%d") for f in features)
        
        metadata.update({
            'last_updated': datetime.now().isoformat(),
            'total_features': len(features),
            'symbols': sorted(list(symbols)),
            'date_range': {
                'start': min(dates),
                'end': max(dates)
            },
            'stored_files': stored_files
        })
        
        # Save metadata
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update technical features metadata: {e}")
    
    def _update_timeseries_metadata(self, features: List[TimeSeriesFeatures], stored_files: List[str]) -> None:
        """Update metadata for time series features."""
        metadata_file = self.metadata_path / "timeseries_features_metadata.json"
        
        # Load existing metadata
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}
        else:
            metadata = {}
        
        # Update with new information
        symbols = set(f.symbol for f in features)
        window_sizes = set(f.window_size for f in features)
        target_dates = set(f.target_date.strftime("%Y-%m-%d") for f in features)
        
        metadata.update({
            'last_updated': datetime.now().isoformat(),
            'total_features': len(features),
            'symbols': sorted(list(symbols)),
            'window_sizes': sorted(list(window_sizes)),
            'target_date_range': {
                'start': min(target_dates),
                'end': max(target_dates)
            },
            'stored_files': stored_files
        })
        
        # Save metadata
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update time series features metadata: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            'technical_features': {},
            'timeseries_features': {}
        }
        
        # Technical features stats
        try:
            metadata_file = self.metadata_path / "technical_features_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    stats['technical_features'] = json.load(f)
        except Exception:
            pass
        
        # Time series features stats
        try:
            metadata_file = self.metadata_path / "timeseries_features_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    stats['timeseries_features'] = json.load(f)
        except Exception:
            pass
        
        # Calculate directory sizes
        try:
            stats['storage_size'] = {
                'technical_mb': self._get_directory_size(self.technical_path) / 1024 / 1024,
                'timeseries_mb': self._get_directory_size(self.timeseries_path) / 1024 / 1024,
                'total_mb': self._get_directory_size(self.base_path) / 1024 / 1024
            }
        except Exception:
            stats['storage_size'] = {}
        
        return stats
    
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size