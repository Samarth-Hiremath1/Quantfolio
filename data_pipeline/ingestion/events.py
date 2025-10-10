"""Event publishing system for data availability notifications."""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path

from .models import DataAvailableEvent

logger = logging.getLogger(__name__)


class EventPublisher(ABC):
    """Abstract base class for event publishers."""
    
    @abstractmethod
    async def publish(self, event: DataAvailableEvent) -> bool:
        """Publish an event."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the publisher and cleanup resources."""
        pass


class InMemoryEventPublisher(EventPublisher):
    """In-memory event publisher for local development."""
    
    def __init__(self):
        self.subscribers: List[Callable[[DataAvailableEvent], None]] = []
        self.event_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    def subscribe(self, callback: Callable[[DataAvailableEvent], None]):
        """Subscribe to events."""
        self.subscribers.append(callback)
        logger.info(f"Added subscriber. Total subscribers: {len(self.subscribers)}")
    
    def unsubscribe(self, callback: Callable[[DataAvailableEvent], None]):
        """Unsubscribe from events."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info(f"Removed subscriber. Total subscribers: {len(self.subscribers)}")
    
    async def publish(self, event: DataAvailableEvent) -> bool:
        """Publish event to all subscribers."""
        async with self._lock:
            try:
                logger.info(f"Publishing event for {len(event.symbols)} symbols from {event.source}")
                
                # Store event in history
                event_dict = asdict(event)
                event_dict['date'] = event.date.isoformat()
                event_dict['published_at'] = datetime.now().isoformat()
                self.event_history.append(event_dict)
                
                # Notify all subscribers
                for subscriber in self.subscribers:
                    try:
                        if asyncio.iscoroutinefunction(subscriber):
                            await subscriber(event)
                        else:
                            subscriber(event)
                    except Exception as e:
                        logger.error(f"Error notifying subscriber: {e}")
                
                logger.info(f"Event published to {len(self.subscribers)} subscribers")
                return True
                
            except Exception as e:
                logger.error(f"Failed to publish event: {e}")
                return False
    
    async def close(self) -> None:
        """Close publisher."""
        self.subscribers.clear()
        logger.info("In-memory event publisher closed")
    
    def get_event_history(self) -> List[Dict[str, Any]]:
        """Get event history."""
        return self.event_history.copy()
    
    def clear_history(self):
        """Clear event history."""
        self.event_history.clear()


class FileEventPublisher(EventPublisher):
    """File-based event publisher that writes events to files."""
    
    def __init__(self, events_dir: str = "data/events"):
        self.events_dir = Path(events_dir)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
    
    async def publish(self, event: DataAvailableEvent) -> bool:
        """Publish event to file."""
        async with self._lock:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"data_available_{timestamp}.json"
                filepath = self.events_dir / filename
                
                event_dict = asdict(event)
                event_dict['date'] = event.date.isoformat()
                event_dict['published_at'] = datetime.now().isoformat()
                
                with open(filepath, 'w') as f:
                    json.dump(event_dict, f, indent=2)
                
                logger.info(f"Event published to file: {filepath}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to publish event to file: {e}")
                return False
    
    async def close(self) -> None:
        """Close publisher."""
        logger.info("File event publisher closed")
    
    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent events from files."""
        events = []
        
        try:
            event_files = sorted(
                self.events_dir.glob("data_available_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            for event_file in event_files[:limit]:
                try:
                    with open(event_file, 'r') as f:
                        event_data = json.load(f)
                        events.append(event_data)
                except Exception as e:
                    logger.warning(f"Failed to read event file {event_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
        
        return events


class CompositeEventPublisher(EventPublisher):
    """Composite publisher that publishes to multiple publishers."""
    
    def __init__(self, publishers: List[EventPublisher]):
        self.publishers = publishers
    
    async def publish(self, event: DataAvailableEvent) -> bool:
        """Publish to all publishers."""
        results = []
        
        for publisher in self.publishers:
            try:
                result = await publisher.publish(event)
                results.append(result)
            except Exception as e:
                logger.error(f"Publisher {type(publisher).__name__} failed: {e}")
                results.append(False)
        
        # Return True if at least one publisher succeeded
        success = any(results)
        logger.info(f"Composite publish result: {sum(results)}/{len(results)} publishers succeeded")
        return success
    
    async def close(self) -> None:
        """Close all publishers."""
        for publisher in self.publishers:
            try:
                await publisher.close()
            except Exception as e:
                logger.error(f"Error closing publisher {type(publisher).__name__}: {e}")


class EventBus:
    """Central event bus for managing event publishing."""
    
    def __init__(self, publisher: Optional[EventPublisher] = None):
        self.publisher = publisher or InMemoryEventPublisher()
        self._closed = False
    
    async def publish_data_available(self, symbols: List[str], date: datetime, 
                                   source: str, file_paths: List[str], 
                                   record_count: int) -> bool:
        """Publish data available event."""
        if self._closed:
            logger.warning("Event bus is closed, cannot publish event")
            return False
        
        event = DataAvailableEvent(
            symbols=symbols,
            date=date,
            source=source,
            file_paths=file_paths,
            record_count=record_count
        )
        
        return await self.publisher.publish(event)
    
    async def close(self):
        """Close the event bus."""
        if not self._closed:
            await self.publisher.close()
            self._closed = True
            logger.info("Event bus closed")
    
    def __del__(self):
        """Cleanup on deletion."""
        if not self._closed:
            # Note: This is not ideal for async cleanup, but provides a fallback
            logger.warning("Event bus was not properly closed")


# Global event bus instance for convenience
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def set_event_bus(event_bus: EventBus):
    """Set the global event bus instance."""
    global _global_event_bus
    _global_event_bus = event_bus


async def close_global_event_bus():
    """Close the global event bus."""
    global _global_event_bus
    if _global_event_bus is not None:
        await _global_event_bus.close()
        _global_event_bus = None