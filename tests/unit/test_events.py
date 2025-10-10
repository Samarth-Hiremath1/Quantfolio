"""Unit tests for event publishing system."""

import pytest
import asyncio
import tempfile
import shutil
import json
from datetime import datetime
from pathlib import Path
from data_pipeline.ingestion.events import (
    InMemoryEventPublisher, FileEventPublisher, CompositeEventPublisher,
    EventBus, get_event_bus, set_event_bus, close_global_event_bus
)
from data_pipeline.ingestion.models import DataAvailableEvent


class TestInMemoryEventPublisher:
    """Test InMemoryEventPublisher."""
    
    @pytest.fixture
    def publisher(self):
        """Create in-memory event publisher."""
        return InMemoryEventPublisher()
    
    @pytest.fixture
    def sample_event(self):
        """Create sample event."""
        return DataAvailableEvent(
            symbols=["AAPL", "GOOGL"],
            date=datetime(2024, 1, 1),
            source="test_source",
            file_paths=["test_file1.parquet", "test_file2.parquet"],
            record_count=100
        )
    
    def test_initialization(self, publisher):
        """Test publisher initialization."""
        assert len(publisher.subscribers) == 0
        assert len(publisher.event_history) == 0
    
    def test_subscribe_unsubscribe(self, publisher):
        """Test subscriber management."""
        def callback(event):
            pass
        
        # Subscribe
        publisher.subscribe(callback)
        assert len(publisher.subscribers) == 1
        assert callback in publisher.subscribers
        
        # Unsubscribe
        publisher.unsubscribe(callback)
        assert len(publisher.subscribers) == 0
        assert callback not in publisher.subscribers
    
    @pytest.mark.asyncio
    async def test_publish_event(self, publisher, sample_event):
        """Test event publishing."""
        events_received = []
        
        def callback(event):
            events_received.append(event)
        
        publisher.subscribe(callback)
        
        result = await publisher.publish(sample_event)
        
        assert result is True
        assert len(events_received) == 1
        assert events_received[0] == sample_event
        assert len(publisher.event_history) == 1
    
    @pytest.mark.asyncio
    async def test_publish_to_multiple_subscribers(self, publisher, sample_event):
        """Test publishing to multiple subscribers."""
        events_received_1 = []
        events_received_2 = []
        
        def callback1(event):
            events_received_1.append(event)
        
        def callback2(event):
            events_received_2.append(event)
        
        publisher.subscribe(callback1)
        publisher.subscribe(callback2)
        
        result = await publisher.publish(sample_event)
        
        assert result is True
        assert len(events_received_1) == 1
        assert len(events_received_2) == 1
        assert events_received_1[0] == sample_event
        assert events_received_2[0] == sample_event
    
    @pytest.mark.asyncio
    async def test_publish_with_async_callback(self, publisher, sample_event):
        """Test publishing with async callback."""
        events_received = []
        
        async def async_callback(event):
            events_received.append(event)
        
        publisher.subscribe(async_callback)
        
        result = await publisher.publish(sample_event)
        
        assert result is True
        assert len(events_received) == 1
        assert events_received[0] == sample_event
    
    @pytest.mark.asyncio
    async def test_publish_with_failing_subscriber(self, publisher, sample_event):
        """Test publishing when a subscriber fails."""
        events_received = []
        
        def good_callback(event):
            events_received.append(event)
        
        def bad_callback(event):
            raise ValueError("Subscriber error")
        
        publisher.subscribe(good_callback)
        publisher.subscribe(bad_callback)
        
        result = await publisher.publish(sample_event)
        
        # Should still succeed despite one subscriber failing
        assert result is True
        assert len(events_received) == 1
        assert events_received[0] == sample_event
    
    def test_get_event_history(self, publisher):
        """Test getting event history."""
        history = publisher.get_event_history()
        assert isinstance(history, list)
        assert len(history) == 0
    
    def test_clear_history(self, publisher):
        """Test clearing event history."""
        # Add some dummy history
        publisher.event_history.append({"test": "event"})
        assert len(publisher.event_history) == 1
        
        publisher.clear_history()
        assert len(publisher.event_history) == 0
    
    @pytest.mark.asyncio
    async def test_close(self, publisher):
        """Test closing publisher."""
        def callback(event):
            pass
        
        publisher.subscribe(callback)
        assert len(publisher.subscribers) == 1
        
        await publisher.close()
        assert len(publisher.subscribers) == 0


class TestFileEventPublisher:
    """Test FileEventPublisher."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def publisher(self, temp_dir):
        """Create file event publisher."""
        return FileEventPublisher(events_dir=temp_dir)
    
    @pytest.fixture
    def sample_event(self):
        """Create sample event."""
        return DataAvailableEvent(
            symbols=["AAPL", "GOOGL"],
            date=datetime(2024, 1, 1),
            source="test_source",
            file_paths=["test_file1.parquet", "test_file2.parquet"],
            record_count=100
        )
    
    def test_initialization(self, publisher, temp_dir):
        """Test publisher initialization."""
        assert publisher.events_dir == Path(temp_dir)
        assert publisher.events_dir.exists()
    
    @pytest.mark.asyncio
    async def test_publish_event(self, publisher, sample_event, temp_dir):
        """Test event publishing to file."""
        result = await publisher.publish(sample_event)
        
        assert result is True
        
        # Check that file was created
        event_files = list(Path(temp_dir).glob("data_available_*.json"))
        assert len(event_files) == 1
        
        # Check file content
        with open(event_files[0], 'r') as f:
            event_data = json.load(f)
        
        assert event_data['symbols'] == sample_event.symbols
        assert event_data['source'] == sample_event.source
        assert event_data['record_count'] == sample_event.record_count
        assert 'published_at' in event_data
    
    def test_get_recent_events(self, publisher, temp_dir):
        """Test getting recent events."""
        # Create some test event files
        for i in range(3):
            event_file = Path(temp_dir) / f"data_available_test_{i}.json"
            event_data = {
                'symbols': [f'SYMBOL{i}'],
                'date': '2024-01-01T00:00:00',
                'source': 'test',
                'file_paths': [],
                'record_count': i * 10,
                'published_at': '2024-01-01T00:00:00'
            }
            with open(event_file, 'w') as f:
                json.dump(event_data, f)
        
        recent_events = publisher.get_recent_events(limit=2)
        
        assert len(recent_events) == 2
        assert all('symbols' in event for event in recent_events)
    
    def test_get_recent_events_empty(self, publisher):
        """Test getting recent events when no events exist."""
        recent_events = publisher.get_recent_events()
        assert len(recent_events) == 0
    
    @pytest.mark.asyncio
    async def test_close(self, publisher):
        """Test closing publisher."""
        await publisher.close()
        # File publisher doesn't have specific cleanup, so just ensure no errors


class TestCompositeEventPublisher:
    """Test CompositeEventPublisher."""
    
    @pytest.fixture
    def publishers(self):
        """Create multiple publishers."""
        return [
            InMemoryEventPublisher(),
            InMemoryEventPublisher()
        ]
    
    @pytest.fixture
    def composite_publisher(self, publishers):
        """Create composite publisher."""
        return CompositeEventPublisher(publishers)
    
    @pytest.fixture
    def sample_event(self):
        """Create sample event."""
        return DataAvailableEvent(
            symbols=["AAPL"],
            date=datetime(2024, 1, 1),
            source="test",
            file_paths=["test.parquet"],
            record_count=50
        )
    
    @pytest.mark.asyncio
    async def test_publish_to_all_publishers(self, composite_publisher, publishers, sample_event):
        """Test publishing to all publishers."""
        # Add subscribers to track events
        events_received_1 = []
        events_received_2 = []
        
        publishers[0].subscribe(lambda e: events_received_1.append(e))
        publishers[1].subscribe(lambda e: events_received_2.append(e))
        
        result = await composite_publisher.publish(sample_event)
        
        assert result is True
        assert len(events_received_1) == 1
        assert len(events_received_2) == 1
        assert events_received_1[0] == sample_event
        assert events_received_2[0] == sample_event
    
    @pytest.mark.asyncio
    async def test_publish_with_one_failing_publisher(self, sample_event):
        """Test publishing when one publisher fails."""
        good_publisher = InMemoryEventPublisher()
        
        # Create a mock failing publisher
        class FailingPublisher:
            async def publish(self, event):
                raise ValueError("Publisher failed")
            
            async def close(self):
                pass
        
        failing_publisher = FailingPublisher()
        composite_publisher = CompositeEventPublisher([good_publisher, failing_publisher])
        
        events_received = []
        good_publisher.subscribe(lambda e: events_received.append(e))
        
        result = await composite_publisher.publish(sample_event)
        
        # Should succeed because at least one publisher succeeded
        assert result is True
        assert len(events_received) == 1
    
    @pytest.mark.asyncio
    async def test_close_all_publishers(self, composite_publisher, publishers):
        """Test closing all publishers."""
        await composite_publisher.close()
        
        # Check that all publishers were closed (subscribers cleared)
        for publisher in publishers:
            assert len(publisher.subscribers) == 0


class TestEventBus:
    """Test EventBus."""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus."""
        return EventBus()
    
    @pytest.mark.asyncio
    async def test_publish_data_available(self, event_bus):
        """Test publishing data available event."""
        events_received = []
        
        # Subscribe to events
        if hasattr(event_bus.publisher, 'subscribe'):
            event_bus.publisher.subscribe(lambda e: events_received.append(e))
        
        result = await event_bus.publish_data_available(
            symbols=["AAPL", "GOOGL"],
            date=datetime(2024, 1, 1),
            source="test_source",
            file_paths=["test.parquet"],
            record_count=100
        )
        
        assert result is True
        if events_received:  # Only check if we have a subscribable publisher
            assert len(events_received) == 1
            event = events_received[0]
            assert event.symbols == ["AAPL", "GOOGL"]
            assert event.source == "test_source"
            assert event.record_count == 100
    
    @pytest.mark.asyncio
    async def test_publish_when_closed(self, event_bus):
        """Test publishing when event bus is closed."""
        await event_bus.close()
        
        result = await event_bus.publish_data_available(
            symbols=["AAPL"],
            date=datetime(2024, 1, 1),
            source="test",
            file_paths=[],
            record_count=0
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_close(self, event_bus):
        """Test closing event bus."""
        await event_bus.close()
        assert event_bus._closed is True


class TestGlobalEventBus:
    """Test global event bus functions."""
    
    @pytest.mark.asyncio
    async def test_get_global_event_bus(self):
        """Test getting global event bus."""
        # Clean up any existing global bus
        await close_global_event_bus()
        
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        
        # Should return the same instance
        assert bus1 is bus2
        
        # Clean up
        await close_global_event_bus()
    
    @pytest.mark.asyncio
    async def test_set_global_event_bus(self):
        """Test setting global event bus."""
        # Clean up any existing global bus
        await close_global_event_bus()
        
        custom_bus = EventBus()
        set_event_bus(custom_bus)
        
        retrieved_bus = get_event_bus()
        assert retrieved_bus is custom_bus
        
        # Clean up
        await close_global_event_bus()
    
    @pytest.mark.asyncio
    async def test_close_global_event_bus(self):
        """Test closing global event bus."""
        # Get a global bus
        bus = get_event_bus()
        assert bus is not None
        
        # Close it
        await close_global_event_bus()
        
        # Getting a new bus should create a new instance
        new_bus = get_event_bus()
        assert new_bus is not bus
        
        # Clean up
        await close_global_event_bus()