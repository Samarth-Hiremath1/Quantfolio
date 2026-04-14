import yfinance as yf
import pandas as pd
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class YFinanceFetcher:
    """
    Fetches historical OHLCV data from Yahoo Finance.
    """
    
    def __init__(self, tickers: List[str]):
        """
        :param tickers: List of ticker symbols to fetch data for.
        """
        self.tickers = tickers

    def fetch_historical_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetches daily OHLCV data for the specified date range.
        
        :param start_date: Start date in YYYY-MM-DD format.
        :param end_date: End date in YYYY-MM-DD format.
        :return: A pandas DataFrame containing stacked historical data, or None if failed.
        """
        logger.info(f"Fetching yfinance data for {len(self.tickers)} tickers from {start_date} to {end_date}")
        try:
            # We use group_by='ticker' to get a clean multi-index, but standard way is often fine
            data = yf.download(self.tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)
            
            if data.empty:
                logger.warning("Returned data is empty from yfinance.")
                return None
            
            # Format the data into a long format suitable for database ingestion
            stacked_data = []
            
            if len(self.tickers) == 1:
                # If only one ticker, yfinance returns flat columns
                ticker = self.tickers[0]
                df = data.copy()
                df['Ticker'] = ticker
                df.reset_index(inplace=True)
                stacked_data.append(df)
            else:
                for ticker in self.tickers:
                    if ticker in data.columns.levels[0]:
                        df = data[ticker].copy()
                        df['Ticker'] = ticker
                        df.reset_index(inplace=True)
                        stacked_data.append(df)
            
            if not stacked_data:
                return None
                
            final_df = pd.concat(stacked_data, ignore_index=True)
            
            # Standardize column names
            # Map standard YF columns: Date, Open, High, Low, Close, Adj Close, Volume
            rename_map = {
                'Date': 'trade_date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume',
                'Ticker': 'ticker'
            }
            final_df.rename(columns=rename_map, inplace=True)
            
            # Ensure trade_date is a datetime date
            final_df['trade_date'] = pd.to_datetime(final_df['trade_date']).dt.date
            
            logger.info(f"Successfully processed {len(final_df)} rows of data.")
            return final_df
            
        except Exception as e:
            logger.error(f"Error fetching data from yfinance: {str(e)}")
            return None

if __name__ == "__main__":
    fetcher = YFinanceFetcher(["AAPL", "MSFT"])
    df = fetcher.fetch_historical_data("2023-01-01", "2023-01-10")
    print(df.head())
