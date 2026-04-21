import pandas as pd
import numpy as np

class MovingAverageCrossover:
    """
    Generates trading signals based on Simple Moving Average (SMA) crossovers.
    Goes long when fast MA crosses above slow MA. Goes flat otherwise.
    """

    def __init__(self, fast_window: int = 10, slow_window: int = 50):
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates position weights (-1 to 1) for the given asset prices.
        """
        fast_ma = prices.rolling(window=self.fast_window).mean()
        slow_ma = prices.rolling(window=self.slow_window).mean()

        # Signal = 1.0 (long) if fast > slow, else 0.0 (flat)
        # Using np.where to build an array, then re-wrapping in DataFrame
        signals = pd.DataFrame(
            np.where(fast_ma > slow_ma, 1.0, 0.0),
            index=prices.index,
            columns=prices.columns
        )
        
        # We can't generate accurate signals until the slow MA window is filled
        signals.iloc[:self.slow_window] = 0.0
        
        return signals
