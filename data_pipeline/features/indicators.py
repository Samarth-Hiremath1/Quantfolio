"""Technical indicators calculation module."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

from ..ingestion.models import PriceData
from .models import TechnicalFeatures, FeatureComputationConfig

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators from price data."""
    
    def __init__(self, config: Optional[FeatureComputationConfig] = None):
        """Initialize with configuration."""
        self.config = config or FeatureComputationConfig()
    
    def compute_all_features(self, price_data: List[PriceData]) -> List[TechnicalFeatures]:
        """Compute all technical features for a list of price data."""
        if not price_data:
            return []
        
        # Convert to DataFrame for easier computation
        df = self._to_dataframe(price_data)
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        features_list = []
        
        for idx, row in df.iterrows():
            # Get historical data up to current point
            historical_df = df.iloc[:idx+1].copy()
            
            if len(historical_df) < 2:
                # Need at least 2 data points for most calculations
                continue
            
            features = TechnicalFeatures(
                symbol=row['symbol'],
                timestamp=row['timestamp']
            )
            
            # Compute returns
            features.returns = self._compute_returns(historical_df)
            
            # Compute moving averages
            features.moving_averages = self._compute_moving_averages(historical_df)
            
            # Compute momentum indicators
            features.momentum = self._compute_momentum_indicators(historical_df)
            
            # Compute volatility measures
            features.volatility = self._compute_volatility_measures(historical_df)
            
            # Compute price-based features
            features.price_features = self._compute_price_features(historical_df)
            
            features_list.append(features)
        
        return features_list
    
    def _to_dataframe(self, price_data: List[PriceData]) -> pd.DataFrame:
        """Convert price data to DataFrame."""
        data = []
        for item in price_data:
            data.append({
                'symbol': item.symbol,
                'timestamp': item.timestamp,
                'open': item.open,
                'high': item.high,
                'low': item.low,
                'close': item.close,
                'volume': item.volume,
                'adjusted_close': item.adjusted_close
            })
        
        return pd.DataFrame(data)
    
    def _compute_returns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute various types of returns."""
        returns = {}
        
        if len(df) < 2:
            return returns
        
        current_close = df.iloc[-1]['close']
        prev_close = df.iloc[-2]['close']
        current_adj_close = df.iloc[-1]['adjusted_close']
        prev_adj_close = df.iloc[-2]['adjusted_close']
        
        # Simple return
        if prev_close != 0:
            returns['simple_return'] = (current_close - prev_close) / prev_close
        
        # Log return
        if prev_close > 0 and current_close > 0:
            returns['log_return'] = np.log(current_close / prev_close)
        
        # Adjusted return
        if prev_adj_close != 0:
            returns['adjusted_return'] = (current_adj_close - prev_adj_close) / prev_adj_close
        
        return returns
    
    def _compute_moving_averages(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute moving averages."""
        ma_dict = {}
        closes = df['close'].values
        
        # Simple Moving Averages
        for period in self.config.sma_periods:
            if len(closes) >= period:
                sma = np.mean(closes[-period:])
                ma_dict[f'sma_{period}'] = sma
        
        # Exponential Moving Averages
        for period in self.config.ema_periods:
            if len(closes) >= period:
                ema = self._calculate_ema(closes, period)
                ma_dict[f'ema_{period}'] = ema
        
        return ma_dict
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _compute_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute momentum indicators."""
        momentum = {}
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # RSI
        if len(closes) >= self.config.rsi_period + 1:
            rsi = self._calculate_rsi(closes, self.config.rsi_period)
            momentum['rsi'] = rsi
        
        # MACD
        if len(closes) >= max(self.config.macd_fast, self.config.macd_slow):
            macd_line, macd_signal, macd_histogram = self._calculate_macd(
                closes, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
            )
            momentum['macd_line'] = macd_line
            momentum['macd_signal'] = macd_signal
            momentum['macd_histogram'] = macd_histogram
        
        # Stochastic Oscillator
        if len(closes) >= self.config.stoch_k_period:
            stoch_k, stoch_d = self._calculate_stochastic(
                highs, lows, closes, self.config.stoch_k_period, self.config.stoch_d_period
            )
            momentum['stoch_k'] = stoch_k
            momentum['stoch_d'] = stoch_d
        
        return momentum
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """Calculate Relative Strength Index."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        if len(gains) < period:
            return 50.0  # Neutral RSI
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[float, float, float]:
        """Calculate MACD indicators."""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        # For signal line, we need historical MACD values
        # Simplified calculation using current MACD as signal
        macd_signal = macd_line  # In practice, this would be EMA of MACD line
        macd_histogram = macd_line - macd_signal
        
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_stochastic(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                            k_period: int, d_period: int) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator."""
        if len(closes) < k_period:
            return 50.0, 50.0
        
        recent_highs = highs[-k_period:]
        recent_lows = lows[-k_period:]
        current_close = closes[-1]
        
        highest_high = np.max(recent_highs)
        lowest_low = np.min(recent_lows)
        
        if highest_high == lowest_low:
            stoch_k = 50.0
        else:
            stoch_k = 100 * (current_close - lowest_low) / (highest_high - lowest_low)
        
        # Simplified %D calculation
        stoch_d = stoch_k  # In practice, this would be SMA of %K
        
        return stoch_k, stoch_d
    
    def _compute_volatility_measures(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute volatility measures."""
        volatility = {}
        closes = df['close'].values
        
        if len(closes) < self.config.volatility_window:
            return volatility
        
        # Realized volatility (standard deviation of returns)
        returns = np.diff(np.log(closes[-self.config.volatility_window:]))
        if len(returns) > 0:
            volatility['realized_volatility'] = np.std(returns) * np.sqrt(252)  # Annualized
        
        # High-Low volatility estimator
        highs = df['high'].values[-self.config.volatility_window:]
        lows = df['low'].values[-self.config.volatility_window:]
        
        if len(highs) == len(lows) and len(highs) > 0:
            hl_ratios = np.log(highs / lows)
            volatility['hl_volatility'] = np.mean(hl_ratios) * np.sqrt(252)
        
        return volatility
    
    def _compute_price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute price-based features."""
        features = {}
        
        if len(df) == 0:
            return features
        
        current = df.iloc[-1]
        
        # High-Low ratio
        if current['low'] != 0:
            features['high_low_ratio'] = current['high'] / current['low']
        
        # Close position within high-low range
        if current['high'] != current['low']:
            features['close_position'] = (current['close'] - current['low']) / (current['high'] - current['low'])
        
        # Volume-weighted features
        if len(df) >= 2:
            recent_volume = df['volume'].iloc[-5:].mean() if len(df) >= 5 else df['volume'].mean()
            if recent_volume != 0:
                features['volume_ratio'] = current['volume'] / recent_volume
        
        # Price momentum (current vs N-day average)
        if len(df) >= 5:
            avg_close = df['close'].iloc[-5:].mean()
            if avg_close != 0:
                features['price_momentum'] = current['close'] / avg_close - 1
        
        return features