"""Async HTTP clients for financial data APIs."""

import asyncio
import aiohttp
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode

from .models import PriceData, DataBatch
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """Base exception for API client errors."""
    pass


class RateLimitError(APIClientError):
    """Raised when API rate limit is exceeded."""
    pass


class DataNotFoundError(APIClientError):
    """Raised when requested data is not found."""
    pass


class BaseAPIClient(ABC):
    """Base class for financial data API clients."""
    
    def __init__(self, api_key: str, circuit_breaker_config: Optional[CircuitBreakerConfig] = None):
        self.api_key = api_key
        self.circuit_breaker = CircuitBreaker(
            circuit_breaker_config or CircuitBreakerConfig()
        )
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'QuantFolio/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def fetch_daily_data(self, symbols: List[str], days: int = 1) -> DataBatch:
        """Fetch daily price data for symbols."""
        pass
    
    async def _make_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request with circuit breaker protection."""
        if not self.session:
            raise APIClientError("Client session not initialized")
        
        async def _request():
            async with self.session.get(url, params=params) as response:
                if response.status == 429:
                    raise RateLimitError("API rate limit exceeded")
                elif response.status == 404:
                    raise DataNotFoundError("Data not found")
                elif response.status != 200:
                    raise APIClientError(f"API request failed with status {response.status}")
                
                return await response.json()
        
        return await self.circuit_breaker.call(_request)


class YahooFinanceClient(BaseAPIClient):
    """Yahoo Finance API client."""
    
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    
    async def fetch_daily_data(self, symbols: List[str], days: int = 1) -> DataBatch:
        """Fetch daily data from Yahoo Finance."""
        logger.info(f"Fetching {days} days of data for {len(symbols)} symbols from Yahoo Finance")
        
        all_data = []
        fetch_timestamp = datetime.now()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 5)  # Extra buffer for weekends
        
        for symbol in symbols:
            try:
                data = await self._fetch_symbol_data(symbol, start_date, end_date)
                all_data.extend(data)
                # Add small delay to avoid rate limiting
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                # Continue with other symbols
                continue
        
        return DataBatch(
            data=all_data,
            fetch_timestamp=fetch_timestamp,
            source="yahoo_finance"
        )
    
    async def _fetch_symbol_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[PriceData]:
        """Fetch data for a single symbol."""
        params = {
            'symbol': symbol,
            'period1': int(start_date.timestamp()),
            'period2': int(end_date.timestamp()),
            'interval': '1d',
            'includePrePost': 'false',
            'events': 'div,splits'
        }
        
        url = f"{self.BASE_URL}/{symbol}"
        response_data = await self._make_request(url, params)
        
        return self._parse_yahoo_response(symbol, response_data)
    
    def _parse_yahoo_response(self, symbol: str, data: Dict[str, Any]) -> List[PriceData]:
        """Parse Yahoo Finance API response."""
        try:
            chart = data['chart']['result'][0]
            timestamps = chart['timestamp']
            quotes = chart['indicators']['quote'][0]
            adj_close = chart['indicators']['adjclose'][0]['adjclose']
            
            price_data = []
            for i, timestamp in enumerate(timestamps):
                # Skip if any required data is None
                if any(quotes[field][i] is None for field in ['open', 'high', 'low', 'close', 'volume']):
                    continue
                
                price_data.append(PriceData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(timestamp),
                    open=float(quotes['open'][i]),
                    high=float(quotes['high'][i]),
                    low=float(quotes['low'][i]),
                    close=float(quotes['close'][i]),
                    volume=int(quotes['volume'][i]),
                    adjusted_close=float(adj_close[i]) if adj_close[i] is not None else float(quotes['close'][i])
                ))
            
            return price_data
            
        except (KeyError, IndexError, TypeError) as e:
            raise APIClientError(f"Failed to parse Yahoo Finance response for {symbol}: {e}")


class AlphaVantageClient(BaseAPIClient):
    """Alpha Vantage API client."""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    async def fetch_daily_data(self, symbols: List[str], days: int = 1) -> DataBatch:
        """Fetch daily data from Alpha Vantage."""
        logger.info(f"Fetching {days} days of data for {len(symbols)} symbols from Alpha Vantage")
        
        all_data = []
        fetch_timestamp = datetime.now()
        
        for symbol in symbols:
            try:
                data = await self._fetch_symbol_data(symbol)
                # Filter to requested number of days
                recent_data = sorted(data, key=lambda x: x.timestamp, reverse=True)[:days]
                all_data.extend(recent_data)
                # Alpha Vantage has strict rate limits
                await asyncio.sleep(12)  # 5 calls per minute limit
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        return DataBatch(
            data=all_data,
            fetch_timestamp=fetch_timestamp,
            source="alpha_vantage"
        )
    
    async def _fetch_symbol_data(self, symbol: str) -> List[PriceData]:
        """Fetch data for a single symbol."""
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'compact'  # Last 100 data points
        }
        
        response_data = await self._make_request(self.BASE_URL, params)
        return self._parse_alpha_vantage_response(symbol, response_data)
    
    def _parse_alpha_vantage_response(self, symbol: str, data: Dict[str, Any]) -> List[PriceData]:
        """Parse Alpha Vantage API response."""
        try:
            if 'Error Message' in data:
                raise APIClientError(f"Alpha Vantage error: {data['Error Message']}")
            
            if 'Note' in data:
                raise RateLimitError("Alpha Vantage rate limit exceeded")
            
            time_series = data['Time Series (Daily)']
            price_data = []
            
            for date_str, daily_data in time_series.items():
                price_data.append(PriceData(
                    symbol=symbol,
                    timestamp=datetime.strptime(date_str, '%Y-%m-%d'),
                    open=float(daily_data['1. open']),
                    high=float(daily_data['2. high']),
                    low=float(daily_data['3. low']),
                    close=float(daily_data['4. close']),
                    volume=int(daily_data['6. volume']),
                    adjusted_close=float(daily_data['5. adjusted close'])
                ))
            
            return price_data
            
        except (KeyError, ValueError) as e:
            raise APIClientError(f"Failed to parse Alpha Vantage response for {symbol}: {e}")


class APIClientFactory:
    """Factory for creating API clients."""
    
    @staticmethod
    def create_client(provider: str, api_key: str, 
                     circuit_breaker_config: Optional[CircuitBreakerConfig] = None) -> BaseAPIClient:
        """Create API client for specified provider."""
        if provider.lower() == 'yahoo':
            return YahooFinanceClient(api_key, circuit_breaker_config)
        elif provider.lower() == 'alpha_vantage':
            return AlphaVantageClient(api_key, circuit_breaker_config)
        else:
            raise ValueError(f"Unsupported API provider: {provider}")