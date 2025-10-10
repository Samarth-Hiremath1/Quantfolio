"""Unit tests for circuit breaker."""

import pytest
import asyncio
from unittest.mock import AsyncMock
from data_pipeline.ingestion.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitBreakerError
)


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.success_threshold == 3
        assert config.timeout == 30.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
            timeout=15.0
        )
        
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30.0
        assert config.success_threshold == 2
        assert config.timeout == 15.0


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""
    
    def test_initial_state(self):
        """Test initial circuit breaker state."""
        config = CircuitBreakerConfig()
        cb = CircuitBreaker(config)
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.last_failure_time is None
        assert cb.is_open is False
    
    @pytest.mark.asyncio
    async def test_successful_call(self):
        """Test successful function call."""
        config = CircuitBreakerConfig()
        cb = CircuitBreaker(config)
        
        async def success_func():
            return "success"
        
        result = await cb.call(success_func)
        
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_failed_call(self):
        """Test failed function call."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(config)
        
        async def fail_func():
            raise ValueError("Test error")
        
        # First failure
        with pytest.raises(ValueError):
            await cb.call(fail_func)
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 1
        
        # Second failure should open the circuit
        with pytest.raises(ValueError):
            await cb.call(fail_func)
        
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 2
        assert cb.is_open is True
    
    @pytest.mark.asyncio
    async def test_circuit_open_blocks_calls(self):
        """Test that open circuit blocks calls."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config)
        
        async def fail_func():
            raise ValueError("Test error")
        
        # Fail once to open circuit
        with pytest.raises(ValueError):
            await cb.call(fail_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Next call should be blocked
        async def success_func():
            return "success"
        
        with pytest.raises(CircuitBreakerError):
            await cb.call(success_func)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling."""
        config = CircuitBreakerConfig(timeout=0.1)
        cb = CircuitBreaker(config)
        
        async def slow_func():
            await asyncio.sleep(0.2)  # Longer than timeout
            return "success"
        
        with pytest.raises(asyncio.TimeoutError):
            await cb.call(slow_func)
        
        assert cb.failure_count == 1
    
    @pytest.mark.asyncio
    async def test_half_open_recovery(self):
        """Test half-open state recovery."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
            success_threshold=2
        )
        cb = CircuitBreaker(config)
        
        async def fail_func():
            raise ValueError("Test error")
        
        async def success_func():
            return "success"
        
        # Open the circuit
        with pytest.raises(ValueError):
            await cb.call(fail_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(0.15)
        
        # First success should move to half-open
        result = await cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.success_count == 1
        
        # Second success should close the circuit
        result = await cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
    
    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self):
        """Test that failure in half-open state reopens circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1
        )
        cb = CircuitBreaker(config)
        
        async def fail_func():
            raise ValueError("Test error")
        
        async def success_func():
            return "success"
        
        # Open the circuit
        with pytest.raises(ValueError):
            await cb.call(fail_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(0.15)
        
        # Success moves to half-open
        await cb.call(success_func)
        assert cb.state == CircuitState.HALF_OPEN
        
        # Failure in half-open should reopen
        with pytest.raises(ValueError):
            await cb.call(fail_func)
        
        assert cb.state == CircuitState.OPEN
    
    def test_get_stats(self):
        """Test get_stats method."""
        config = CircuitBreakerConfig()
        cb = CircuitBreaker(config)
        
        stats = cb.get_stats()
        
        assert stats['state'] == 'closed'
        assert stats['failure_count'] == 0
        assert stats['success_count'] == 0
        assert stats['last_failure_time'] is None
        
        # Simulate some state changes
        cb.failure_count = 2
        cb.success_count = 1
        cb.last_failure_time = 1234567890.0
        
        stats = cb.get_stats()
        
        assert stats['failure_count'] == 2
        assert stats['success_count'] == 1
        assert stats['last_failure_time'] == 1234567890.0