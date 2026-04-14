import os
import requests
import pandas as pd
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class AlphaVantageFetcher:
    """
    Fetches financial data from the Alpha Vantage API.
    Used as an alternative or supplementary source to yfinance.
    """
    
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        """
        :param api_key: Alpha Vantage API Key. If not provided, will look for ALPHA_VANTAGE_API_KEY environment var.
        """
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            logger.warning("Alpha Vantage API Key is missing. Fetches may fail.")

    def fetch_daily(self, ticker: str, outputsize: str = "compact") -> Optional[pd.DataFrame]:
        """
        Fetches daily time series for a single ticker.
        
        :param ticker: The stock ticker symbol.
        :param outputsize: 'compact' (latest 100 days) or 'full' (up to 20 years).
        :return: DataFrame with OHLCV data.
        """
        logger.info(f"Fetching Alpha Vantage daily data for {ticker}")
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "datatype": "json"
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                logger.error(f"Unexpected response from Alpha Vantage: {data}")
                return None
                
            ts_data = data["Time Series (Daily)"]
            
            # Parse into DataFrame
            df = pd.DataFrame.from_dict(ts_data, orient='index')
            df.index = pd.to_datetime(df.index).date
            df.index.name = 'trade_date'
            
            # Map columns
            rename_map = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adj_close',
                '6. volume': 'volume',
            }
            
            df = df.rename(columns=rename_map)
            df = df[['open', 'high', 'low', 'close', 'adj_close', 'volume']] # Drop dividend/split metadata for now
            df = df.astype(float)
            df['ticker'] = ticker
            df.reset_index(inplace=True)
            
            logger.info(f"Successfully fetched {len(df)} rows for {ticker} from Alpha Vantage.")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage for {ticker}: {str(e)}")
            return None

if __name__ == "__main__":
    fetcher = AlphaVantageFetcher("demo")
    df = fetcher.fetch_daily("IBM")
    if df is not None:
        print(df.head())
