"""Unit tests for API clients."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import aiohttp
from data_pipeline.ingestion.api_clients import (
    YahooFinanceClient, AlphaVantageClient, APIClientFactory,
    APIClientError, RateLimitError, DataNotFoundError
)
from data_pipeline.ingestion.circuit_breaker import CircuitBreakerConfig
from data_pipeline.ingestion.models import DataBatch, PriceData


class TestAPIClientFactory:
    """Test APIClientFactory."""
    
    def test_create_yahoo_client(self):
        """Test creating Yahoo Finance client."""
        client = APIClientFactory.create_client("yahoo", "test_key")
        assert isinstance(client, YahooFinanceClient)
        assert client.api_key == "test_key"
    
    def test_create_alpha_vantage_client(self):
        """Test creating Alpha Vantage client."""
        client = APIClientFactory.create_client("alpha_vantage", "test_key")
        assert isinstance(client, AlphaVantageClient)
        assert client.api_key == "test_key"
    
    def test_create_client_with_circuit_breaker_config(self):
        """Test creating client with custom circuit breaker config."""
        config = CircuitBreakerConfig(failure_threshold=2)
        client = APIClientFactory.create_client("yahoo", "test_key", config)
        
        assert isinstance(client, YahooFinanceClient)
        assert client.circuit_breaker.config.failure_threshold == 2
    
    def test_create_client_unsupported_provider(self):
        """Test creating client with unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported API provider"):
            APIClientFactory.create_client("unsupported", "test_key")


class TestYahooFinanceClient:
    """Test YahooFinanceClient."""
    
    @pytest.fixture
    def client(self):
        """Create Yahoo Finance client."""
        return YahooFinanceClient("test_key")
    
    @pytest.fixture
    def mock_yahoo_response(self):
        """Mock Yahoo Finance API response."""
        return {
            'chart': {
                'result': [{
                    'timestamp': [1704067800, 1704154200],  # Jan 1 and Jan 2, 2024
                    'indicators': {
                        'quote': [{
                            'open': [150.0, 154.0],
                            'high': [155.0, 158.0],
                            'low': [149.0, 153.0],
                            'close': [154.0, 157.0],
                            'volume': [1000000, 1100000]
                        }],
                        'adjclose': [{
                            'adjclose': [154.0, 157.0]
                        }]
                    }
                }]
            }
        }
    
    @pytest.mark.asyncio
    async def test_context_manager(self, client):
        """Test client as async context manager."""
        async with client:
            assert client.session is not None
            assert isinstance(client.session, aiohttp.ClientSession)
        
        # Session should be closed after exiting context
        assert client.session.closed
    
    @pytest.mark.asyncio
    async def test_fetch_daily_data_success(self, client, mock_yahoo_response):
        """Test successful data fetching."""
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_yahoo_response
            
            async with client:
                batch = await client.fetch_daily_data(["AAPL"], days=2)
            
            assert isinstance(batch, DataBatch)
            assert batch.source == "yahoo_finance"
            assert len(batch.data) == 2
            
            # Check first record
            first_record = batch.data[0]
            assert first_record.symbol == "AAPL"
            assert first_record.open == 150.0
            assert first_record.high == 155.0
            assert first_record.low == 149.0
            assert first_record.close == 154.0
            assert first_record.volume == 1000000
            assert first_record.adjusted_close == 154.0
    
    @pytest.mark.asyncio
    async def test_fetch_daily_data_multiple_symbols(self, client, mock_yahoo_response):
        """Test fetching data for multiple symbols."""
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_yahoo_response
            
            async with client:
                batch = await client.fetch_daily_data(["AAPL", "GOOGL"], days=1)
            
            # Should make two requests (one per symbol)
            assert mock_request.call_count == 2
            assert len(batch.data) == 4  # 2 records per symbol
    
    @pytest.mark.asyncio
    async def test_fetch_daily_data_with_none_values(self, client):
        """Test handling of None values in response."""
        response_with_nones = {
            'chart': {
                'result': [{
                    'timestamp': [1704067800, 1704154200],
                    'indicators': {
                        'quote': [{
                            'open': [150.0, None],  # None value
                            'high': [155.0, 158.0],
                            'low': [149.0, 153.0],
                            'close': [154.0, 157.0],
                            'volume': [1000000, 1100000]
                        }],
                        'adjclose': [{
                            'adjclose': [154.0, 157.0]
                        }]
                    }
                }]
            }
        }
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = response_with_nones
            
            async with client:
                batch = await client.fetch_daily_data(["AAPL"], days=2)
            
            # Should skip the record with None values
            assert len(batch.data) == 1
            assert batch.data[0].open == 150.0
    
    def test_parse_yahoo_response(self, client, mock_yahoo_response):
        """Test parsing Yahoo Finance response."""
        records = client._parse_yahoo_response("AAPL", mock_yahoo_response)
        
        assert len(records) == 2
        assert all(record.symbol == "AAPL" for record in records)
        assert records[0].open == 150.0
        assert records[1].close == 157.0
    
    def test_parse_yahoo_response_invalid_data(self, client):
        """Test parsing invalid Yahoo Finance response."""
        invalid_response = {'invalid': 'data'}
        
        with pytest.raises(APIClientError, match="Failed to parse Yahoo Finance response"):
            client._parse_yahoo_response("AAPL", invalid_response)
    
    @pytest.mark.asyncio
    async def test_make_request_success(self, client):
        """Test successful HTTP request."""
        mock_response_data = {"test": "data"}
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_get.return_value.__aenter__.return_value = mock_response
            
            async with client:
                result = await client._make_request("http://test.com", {"param": "value"})
            
            assert result == mock_response_data
    
    @pytest.mark.asyncio
    async def test_make_request_rate_limit(self, client):
        """Test handling rate limit response."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_get.return_value.__aenter__.return_value = mock_response
            
            async with client:
                with pytest.raises(RateLimitError):
                    await client._make_request("http://test.com", {})
    
    @pytest.mark.asyncio
    async def test_make_request_not_found(self, client):
        """Test handling not found response."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_get.return_value.__aenter__.return_value = mock_response
            
            async with client:
                with pytest.raises(DataNotFoundError):
                    await client._make_request("http://test.com", {})
    
    @pytest.mark.asyncio
    async def test_make_request_server_error(self, client):
        """Test handling server error response."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_get.return_value.__aenter__.return_value = mock_response
            
            async with client:
                with pytest.raises(APIClientError, match="API request failed with status 500"):
                    await client._make_request("http://test.com", {})
    
    @pytest.mark.asyncio
    async def test_make_request_without_session(self, client):
        """Test making request without initialized session."""
        with pytest.raises(APIClientError, match="Client session not initialized"):
            await client._make_request("http://test.com", {})


class TestAlphaVantageClient:
    """Test AlphaVantageClient."""
    
    @pytest.fixture
    def client(self):
        """Create Alpha Vantage client."""
        return AlphaVantageClient("test_key")
    
    @pytest.fixture
    def mock_alpha_vantage_response(self):
        """Mock Alpha Vantage API response."""
        return {
            'Time Series (Daily)': {
                '2024-01-02': {
                    '1. open': '154.0',
                    '2. high': '158.0',
                    '3. low': '153.0',
                    '4. close': '157.0',
                    '5. adjusted close': '157.0',
                    '6. volume': '1100000'
                },
                '2024-01-01': {
                    '1. open': '150.0',
                    '2. high': '155.0',
                    '3. low': '149.0',
                    '4. close': '154.0',
                    '5. adjusted close': '154.0',
                    '6. volume': '1000000'
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_fetch_daily_data_success(self, client, mock_alpha_vantage_response):
        """Test successful data fetching."""
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_alpha_vantage_response
            
            async with client:
                batch = await client.fetch_daily_data(["AAPL"], days=2)
            
            assert isinstance(batch, DataBatch)
            assert batch.source == "alpha_vantage"
            assert len(batch.data) == 2
            
            # Data should be sorted by date (most recent first)
            assert batch.data[0].timestamp.date().isoformat() == '2024-01-02'
            assert batch.data[1].timestamp.date().isoformat() == '2024-01-01'
    
    @pytest.mark.asyncio
    async def test_fetch_daily_data_with_delay(self, client, mock_alpha_vantage_response):
        """Test that client adds delay between requests."""
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_alpha_vantage_response
            
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                async with client:
                    await client.fetch_daily_data(["AAPL", "GOOGL"], days=1)
                
                # Should sleep between requests (but not after the last one)
                assert mock_sleep.call_count == 1
                mock_sleep.assert_called_with(12)  # 12 seconds delay
    
    def test_parse_alpha_vantage_response(self, client, mock_alpha_vantage_response):
        """Test parsing Alpha Vantage response."""
        records = client._parse_alpha_vantage_response("AAPL", mock_alpha_vantage_response)
        
        assert len(records) == 2
        assert all(record.symbol == "AAPL" for record in records)
        
        # Find records by date
        jan1_record = next(r for r in records if r.timestamp.date().isoformat() == '2024-01-01')
        jan2_record = next(r for r in records if r.timestamp.date().isoformat() == '2024-01-02')
        
        assert jan1_record.open == 150.0
        assert jan1_record.close == 154.0
        assert jan2_record.open == 154.0
        assert jan2_record.close == 157.0
    
    def test_parse_alpha_vantage_response_error_message(self, client):
        """Test parsing response with error message."""
        error_response = {
            'Error Message': 'Invalid API call'
        }
        
        with pytest.raises(APIClientError, match="Alpha Vantage error: Invalid API call"):
            client._parse_alpha_vantage_response("AAPL", error_response)
    
    def test_parse_alpha_vantage_response_rate_limit(self, client):
        """Test parsing response with rate limit note."""
        rate_limit_response = {
            'Note': 'Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute'
        }
        
        with pytest.raises(RateLimitError):
            client._parse_alpha_vantage_response("AAPL", rate_limit_response)
    
    def test_parse_alpha_vantage_response_invalid_data(self, client):
        """Test parsing invalid Alpha Vantage response."""
        invalid_response = {'invalid': 'data'}
        
        with pytest.raises(APIClientError, match="Failed to parse Alpha Vantage response"):
            client._parse_alpha_vantage_response("AAPL", invalid_response)