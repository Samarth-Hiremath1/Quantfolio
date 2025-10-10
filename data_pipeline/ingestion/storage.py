"""Local storage system that simulates cloud object storage with partitioning."""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from .models import DataBatch, PriceData, StorageResult

logger = logging.getLogger(__name__)


class CloudStorageSimulator:
    """Simulates cloud object storage (S3/GCS) with local filesystem."""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.raw_data_path = self.base_path / "raw" / "daily"
        self.metadata_path = self.base_path / "raw" / "metadata"
        
        # Create directory structure
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directory structure."""
        directories = [
            self.raw_data_path,
            self.metadata_path,
            self.base_path / "features" / "technical",
            self.base_path / "features" / "time_series" / "windows_30d",
            self.base_path / "features" / "time_series" / "windows_60d",
            self.base_path / "models" / "pytorch",
            self.base_path / "models" / "tensorflow",
            self.base_path / "models" / "metadata",
            self.base_path / "portfolios" / "allocations",
            self.base_path / "portfolios" / "backtests",
            self.base_path / "processed"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def store_batch(self, batch: DataBatch) -> StorageResult:
        """Store data batch with date/symbol partitioning."""
        logger.info(f"Storing batch with {len(batch.data)} records from {batch.source}")
        
        try:
            files_written = []
            total_records = 0
            
            # Group data by date and symbol for partitioning
            partitioned_data = self._partition_data(batch.data)
            
            for (date, symbol), records in partitioned_data.items():
                file_path = self._get_file_path(date, symbol)
                
                # Convert to DataFrame for efficient storage
                df = self._records_to_dataframe(records)
                
                # Store as Parquet for efficiency (simulating cloud columnar storage)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(file_path, index=False, compression='snappy')
                
                files_written.append(str(file_path))
                total_records += len(records)
                logger.debug(f"Stored {len(records)} records for {symbol} on {date} to {file_path}")
            
            # Store metadata
            metadata_file = self._store_metadata(batch, files_written)
            files_written.append(str(metadata_file))
            
            logger.info(f"Successfully stored {total_records} records in {len(files_written)} files")
            
            return StorageResult(
                success=True,
                files_written=files_written,
                total_records=total_records
            )
            
        except Exception as e:
            logger.error(f"Failed to store batch: {e}")
            return StorageResult(
                success=False,
                files_written=[],
                total_records=0,
                error_message=str(e)
            )
    
    def _partition_data(self, data: List[PriceData]) -> Dict[tuple, List[PriceData]]:
        """Partition data by date and symbol."""
        partitioned = {}
        
        for record in data:
            key = (record.timestamp.date(), record.symbol)
            if key not in partitioned:
                partitioned[key] = []
            partitioned[key].append(record)
        
        return partitioned
    
    def _get_file_path(self, date: datetime.date, symbol: str) -> Path:
        """Generate file path with date partitioning."""
        year = date.year
        month = f"{date.month:02d}"
        day = f"{date.day:02d}"
        
        return self.raw_data_path / str(year) / month / day / f"{symbol}.parquet"
    
    def _records_to_dataframe(self, records: List[PriceData]) -> pd.DataFrame:
        """Convert PriceData records to DataFrame."""
        data = []
        for record in records:
            data.append({
                'symbol': record.symbol,
                'timestamp': record.timestamp,
                'open': record.open,
                'high': record.high,
                'low': record.low,
                'close': record.close,
                'volume': record.volume,
                'adjusted_close': record.adjusted_close
            })
        
        return pd.DataFrame(data)
    
    def _store_metadata(self, batch: DataBatch, files_written: List[str]) -> Path:
        """Store batch metadata."""
        metadata = {
            'batch_id': f"{batch.source}_{batch.fetch_timestamp.isoformat()}",
            'source': batch.source,
            'fetch_timestamp': batch.fetch_timestamp.isoformat(),
            'symbols': batch.get_symbols(),
            'record_count': len(batch.data),
            'files_written': files_written,
            'storage_timestamp': datetime.now().isoformat()
        }
        
        metadata_file = self.metadata_path / f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_file
    
    def load_data(self, symbol: str, start_date: datetime.date, 
                  end_date: Optional[datetime.date] = None) -> List[PriceData]:
        """Load data for a symbol within date range."""
        if end_date is None:
            end_date = start_date
        
        logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
        
        all_records = []
        current_date = start_date
        
        while current_date <= end_date:
            file_path = self._get_file_path(current_date, symbol)
            
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    records = self._dataframe_to_records(df)
                    all_records.extend(records)
                    logger.debug(f"Loaded {len(records)} records from {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
            
            current_date = datetime(current_date.year, current_date.month, current_date.day) + timedelta(days=1)
            current_date = current_date.date()
        
        logger.info(f"Loaded {len(all_records)} total records for {symbol}")
        return all_records
    
    def _dataframe_to_records(self, df: pd.DataFrame) -> List[PriceData]:
        """Convert DataFrame to PriceData records."""
        records = []
        for _, row in df.iterrows():
            records.append(PriceData(
                symbol=row['symbol'],
                timestamp=pd.to_datetime(row['timestamp']).to_pydatetime(),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume']),
                adjusted_close=float(row['adjusted_close'])
            ))
        return records
    
    def list_available_data(self) -> Dict[str, List[str]]:
        """List available data by symbol and date."""
        available_data = {}
        
        if not self.raw_data_path.exists():
            return available_data
        
        # Walk through the directory structure
        for year_dir in self.raw_data_path.iterdir():
            if not year_dir.is_dir():
                continue
                
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue
                    
                for day_dir in month_dir.iterdir():
                    if not day_dir.is_dir():
                        continue
                    
                    date_str = f"{year_dir.name}-{month_dir.name}-{day_dir.name}"
                    
                    for file_path in day_dir.glob("*.parquet"):
                        symbol = file_path.stem
                        
                        if symbol not in available_data:
                            available_data[symbol] = []
                        
                        available_data[symbol].append(date_str)
        
        # Sort dates for each symbol
        for symbol in available_data:
            available_data[symbol].sort()
        
        return available_data
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'symbols': set(),
            'date_range': {'earliest': None, 'latest': None}
        }
        
        if not self.raw_data_path.exists():
            return stats
        
        for file_path in self.raw_data_path.rglob("*.parquet"):
            stats['total_files'] += 1
            stats['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
            
            # Extract symbol from filename
            symbol = file_path.stem
            stats['symbols'].add(symbol)
            
            # Extract date from path
            parts = file_path.parts
            if len(parts) >= 6:  # .../year/month/day/symbol.parquet
                try:
                    year, month, day = parts[-4], parts[-3], parts[-2]
                    date_str = f"{year}-{month}-{day}"
                    
                    if stats['date_range']['earliest'] is None or date_str < stats['date_range']['earliest']:
                        stats['date_range']['earliest'] = date_str
                    
                    if stats['date_range']['latest'] is None or date_str > stats['date_range']['latest']:
                        stats['date_range']['latest'] = date_str
                        
                except (ValueError, IndexError):
                    continue
        
        stats['symbols'] = list(stats['symbols'])
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        
        return stats