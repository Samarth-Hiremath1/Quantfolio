import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Computes time-series features like returns, rolling volatility, and technical indicators.
    Enforces NO LOOKAHEAD BIAS by ensuring features at time t only use data up to t.
    """

    def __init__(self, price_col: str = 'adj_close'):
        """
        :param price_col: Which price column to use as the basis for features.
        """
        self.price_col = price_col

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a cleaned OHLCV DataFrame, calculates standard features per ticker.
        """
        feat_df = df.copy()
        
        # Ensure correct sorting
        feat_df = feat_df.sort_values(by=['ticker', 'trade_date']).reset_index(drop=True)

        # Apply features per ticker
        grouped = feat_df.groupby('ticker', group_keys=False)
        
        # 1. Arithmetic & Log Returns (1-day)
        feat_df['return_1d'] = grouped[self.price_col].pct_change()
        feat_df['log_return_1d'] = grouped[self.price_col].apply(lambda x: np.log(x / x.shift(1)))
        
        # 2. Rolling Volatility (21 days ~ 1 month), annualized
        feat_df['rolling_vol_21d'] = grouped['log_return_1d'].transform(lambda x: x.rolling(window=21).std() * np.sqrt(252))
        
        # 3. MACD (Moving Average Convergence Divergence)
        # Standard: 12-day EMA, 26-day EMA, 9-day signal EMA
        def calc_macd(group):
            ema_12 = group.ewm(span=12, adjust=False).mean()
            ema_26 = group.ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9, adjust=False).mean()
            return pd.DataFrame({'macd': macd, 'macd_signal': signal})
            
        macd_res = grouped[self.price_col].apply(calc_macd)
        if hasattr(macd_res, 'index') and isinstance(macd_res.index, pd.MultiIndex):
            macd_res = macd_res.reset_index(level=0, drop=True)
            
        feat_df['macd'] = macd_res['macd']
        feat_df['macd_signal'] = macd_res['macd_signal']

        # 4. RSI (Relative Strength Index) 14-day
        def calc_rsi(group, window=14):
            delta = group.diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        feat_df['rsi_14'] = grouped[self.price_col].apply(calc_rsi)
        
        return feat_df
