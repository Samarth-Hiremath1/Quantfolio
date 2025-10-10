"""Unit tests for storage system."""

import pytest
import tempfile
import shutil
from datetime import datetime, date
from pathlib import Path
from data_pipeline.ingestion.storage import CloudStorageSimulator
from data_pipeline.ingestion.models import PriceData, DataBatch


class TestCloudStorageSimulator:
    """Test CloudStorageSimulator functionality."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        temp_dir = tempfile.mkdtemp()
        storage = CloudStorageSimulator(base_path=temp_dir)
        yield storage
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_batch(self):
        """Create sample data batch for testing."""
        data = [
            PriceData("AAPL", datetime(2024, 1, 1, 9, 30), 150.0, 155.0, 149.0, 154.0, 1000000, 154.0),
            PriceData("AAPL", datetime(2024, 1, 2, 9, 30), 154.0, 158.0, 153.0, 157.0, 1100000, 157.0),
            PriceData("GOOGL", datetime(2024, 1, 1, 9, 30), 2800.0, 2850.0, 2790.0, 2840.0, 500000, 2840.0),
            PriceData("GOOGL", datetime(2024, 1, 2, 9, 30), 2840.0, 2880.0, 2830.0, 2870.0, 550000, 2870.0)
        ]
        
        return DataBatch(
            data=data,
            fetch_timestamp=datetime.now(),
            source="test_source"
        )
    
    def test_initialization(self, temp_storage):
        """Test storage initialization."""
        assert temp_storage.base_path.exists()
        assert temp_storage.raw_data_path.exists()
        assert temp_storage.metadata_path.exists()
        
        # Check that all expected directories are created
        expected_dirs = [
            "raw/daily",
            "raw/metadata",
            "features/technical",
            "features/time_series/windows_30d",
            "features/time_series/windows_60d",
            "models/pytorch",
            "models/tensorflow",
            "models/metadata",
            "portfolios/allocations",
            "portfolios/backtests",
            "processed"
        ]
        
        for dir_path in expected_dirs:
            full_path = temp_storage.base_path / dir_path
            assert full_path.exists(), f"Directory {dir_path} should exist"
    
    def test_store_batch_success(self, temp_storage, sample_batch):
        """Test successful batch storage."""
        result = temp_storage.store_batch(sample_batch)
        
        assert result.success is True
        assert result.total_records == 4
        assert len(result.files_written) > 0
        assert result.error_message is None
        
        # Check that files were actually created
        for file_path in result.files_written:
            if file_path.endswith('.parquet'):  # Skip metadata files
                assert Path(file_path).exists()
    
    def test_partition_data(self, temp_storage, sample_batch):
        """Test data partitioning by date and symbol."""
        partitioned = temp_storage._partition_data(sample_batch.data)
        
        # Should have 4 partitions: (2024-01-01, AAPL), (2024-01-02, AAPL), 
        # (2024-01-01, GOOGL), (2024-01-02, GOOGL)
        assert len(partitioned) == 4
        
        # Check specific partitions
        aapl_jan1_key = (date(2024, 1, 1), "AAPL")
        aapl_jan2_key = (date(2024, 1, 2), "AAPL")
        googl_jan1_key = (date(2024, 1, 1), "GOOGL")
        googl_jan2_key = (date(2024, 1, 2), "GOOGL")
        
        assert aapl_jan1_key in partitioned
        assert aapl_jan2_key in partitioned
        assert googl_jan1_key in partitioned
        assert googl_jan2_key in partitioned
        
        # Each partition should have exactly one record
        for partition in partitioned.values():
            assert len(partition) == 1
    
    def test_get_file_path(self, temp_storage):
        """Test file path generation."""
        test_date = date(2024, 1, 15)
        symbol = "AAPL"
        
        file_path = temp_storage._get_file_path(test_date, symbol)
        
        expected_path = temp_storage.raw_data_path / "2024" / "01" / "15" / "AAPL.parquet"
        assert file_path == expected_path
    
    def test_records_to_dataframe(self, temp_storage):
        """Test conversion of records to DataFrame."""
        records = [
            PriceData("AAPL", datetime(2024, 1, 1, 9, 30), 150.0, 155.0, 149.0, 154.0, 1000000, 154.0),
            PriceData("AAPL", datetime(2024, 1, 2, 9, 30), 154.0, 158.0, 153.0, 157.0, 1100000, 157.0)
        ]
        
        df = temp_storage._records_to_dataframe(records)
        
        assert len(df) == 2
        assert list(df.columns) == [
            'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close'
        ]
        assert df.iloc[0]['symbol'] == 'AAPL'
        assert df.iloc[0]['open'] == 150.0
        assert df.iloc[1]['close'] == 157.0
    
    def test_load_data(self, temp_storage, sample_batch):
        """Test loading data from storage."""
        # First store the data
        temp_storage.store_batch(sample_batch)
        
        # Then load it back
        loaded_data = temp_storage.load_data("AAPL", date(2024, 1, 1), date(2024, 1, 2))
        
        assert len(loaded_data) == 2  # Two AAPL records
        
        # Check that loaded data matches original
        aapl_records = [record for record in sample_batch.data if record.symbol == "AAPL"]
        assert len(loaded_data) == len(aapl_records)
        
        # Sort both lists by timestamp for comparison
        loaded_data.sort(key=lambda x: x.timestamp)
        aapl_records.sort(key=lambda x: x.timestamp)
        
        for loaded, original in zip(loaded_data, aapl_records):
            assert loaded.symbol == original.symbol
            assert loaded.open == original.open
            assert loaded.close == original.close
    
    def test_load_data_single_date(self, temp_storage, sample_batch):
        """Test loading data for a single date."""
        temp_storage.store_batch(sample_batch)
        
        loaded_data = temp_storage.load_data("AAPL", date(2024, 1, 1))
        
        assert len(loaded_data) == 1
        assert loaded_data[0].symbol == "AAPL"
        assert loaded_data[0].timestamp.date() == date(2024, 1, 1)
    
    def test_load_data_nonexistent(self, temp_storage):
        """Test loading data that doesn't exist."""
        loaded_data = temp_storage.load_data("NONEXISTENT", date(2024, 1, 1))
        
        assert len(loaded_data) == 0
    
    def test_list_available_data(self, temp_storage, sample_batch):
        """Test listing available data."""
        temp_storage.store_batch(sample_batch)
        
        available_data = temp_storage.list_available_data()
        
        assert "AAPL" in available_data
        assert "GOOGL" in available_data
        
        # Check dates
        assert "2024-01-01" in available_data["AAPL"]
        assert "2024-01-02" in available_data["AAPL"]
        assert "2024-01-01" in available_data["GOOGL"]
        assert "2024-01-02" in available_data["GOOGL"]
        
        # Dates should be sorted
        for symbol_dates in available_data.values():
            assert symbol_dates == sorted(symbol_dates)
    
    def test_list_available_data_empty(self, temp_storage):
        """Test listing available data when storage is empty."""
        available_data = temp_storage.list_available_data()
        
        assert available_data == {}
    
    def test_get_storage_stats(self, temp_storage, sample_batch):
        """Test getting storage statistics."""
        # Initially empty
        stats = temp_storage.get_storage_stats()
        assert stats['total_files'] == 0
        assert stats['total_size_mb'] == 0
        assert stats['symbols'] == []
        
        # After storing data
        temp_storage.store_batch(sample_batch)
        stats = temp_storage.get_storage_stats()
        
        assert stats['total_files'] > 0
        assert stats['total_size_mb'] > 0
        assert set(stats['symbols']) == {"AAPL", "GOOGL"}
        assert stats['date_range']['earliest'] == "2024-01-01"
        assert stats['date_range']['latest'] == "2024-01-02"
    
    def test_dataframe_to_records(self, temp_storage):
        """Test conversion of DataFrame to records."""
        import pandas as pd
        
        df_data = {
            'symbol': ['AAPL', 'AAPL'],
            'timestamp': [datetime(2024, 1, 1, 9, 30), datetime(2024, 1, 2, 9, 30)],
            'open': [150.0, 154.0],
            'high': [155.0, 158.0],
            'low': [149.0, 153.0],
            'close': [154.0, 157.0],
            'volume': [1000000, 1100000],
            'adjusted_close': [154.0, 157.0]
        }
        
        df = pd.DataFrame(df_data)
        records = temp_storage._dataframe_to_records(df)
        
        assert len(records) == 2
        assert all(isinstance(record, PriceData) for record in records)
        assert records[0].symbol == 'AAPL'
        assert records[0].open == 150.0
        assert records[1].close == 157.0
    
    def test_store_metadata(self, temp_storage, sample_batch):
        """Test metadata storage."""
        files_written = ["test_file1.parquet", "test_file2.parquet"]
        metadata_file = temp_storage._store_metadata(sample_batch, files_written)
        
        assert metadata_file.exists()
        assert metadata_file.suffix == '.json'
        
        # Check metadata content
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['source'] == sample_batch.source
        assert metadata['symbols'] == sample_batch.get_symbols()
        assert metadata['record_count'] == len(sample_batch.data)
        assert metadata['files_written'] == files_written