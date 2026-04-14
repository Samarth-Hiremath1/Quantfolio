import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Handles missing values and outlier detection for financial time series data.
    """

    def __init__(self, method: str = 'ffill', outlier_std: float = 4.0):
        """
        :param method: Imputation method for missing gaps ('ffill' commonly used to prevent lookahead bias).
        :param outlier_std: Number of standard deviations away from the rolling mean to consider an outlier.
        """
        self.method = method
        self.outlier_std = outlier_std

    def clean_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans OHLCV dataframe. Assumes columns ['open', 'high', 'low', 'close', 'adj_close', 'volume'].
        
        :param df: The raw OHLCV DataFrame.
        :return: The cleaned DataFrame.
        """
        logger.info("Starting data cleaning process...")
        clean_df = df.copy()

        # Sort by date to ensure proper forward filling
        if 'trade_date' in clean_df.columns:
            clean_df = clean_df.sort_values(by=['ticker', 'trade_date']).reset_index(drop=True)

        # 1. Handle Missing Data
        # Forward fill up to 5 days, then drop the rest
        logger.info("Handling missing data...")
        price_cols = ['open', 'high', 'low', 'close', 'adj_close']
        
        for col in price_cols:
            if self.method == 'ffill':
                clean_df[col] = clean_df.groupby('ticker')[col].ffill(limit=5)
        
        clean_df['volume'] = clean_df.groupby('ticker')['volume'].ffill(limit=5).fillna(0)
        
        initial_len = len(clean_df)
        clean_df.dropna(subset=['close', 'adj_close'], inplace=True)
        dropped = initial_len - len(clean_df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows due to unrecoverable missing data.")

        # 2. Handle simple data errors (e.g., negative prices)
        for col in price_cols:
            invalid_prices = clean_df[clean_df[col] <= 0]
            if not invalid_prices.empty:
                logger.warning(f"Found {len(invalid_prices)} rows with negative or zero {col}. Filtering.")
                clean_df = clean_df[clean_df[col] > 0]

        # 3. Detect and optionally clip extreme price outliers in returns space (to avoid bias)
        # Often better done on returns during feature engineering, but we can do a sanity check on volume.
        # E.g., clip extreme volume spikes.
        clean_df['volume'] = clean_df.groupby('ticker')['volume'].apply(
            lambda x: x.clip(upper=x.rolling(window=21).mean() + self.outlier_std * x.rolling(window=21).std())
        ).reset_index(level=0, drop=True)

        logger.info("Data cleaning complete.")
        return clean_df
