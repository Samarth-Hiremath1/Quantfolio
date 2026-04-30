from abc import ABC, abstractmethod
from queue import Queue
import pandas as pd
import numpy as np

from backtesting.events import SignalEvent
from backtesting.data_handler import DataHandler

class Strategy(ABC):
    """
    Base strategy interface.
    """
    @abstractmethod
    def calculate_signals(self, event):
        """
        Given a MarketEvent, generates SignalEvents.
        """
        pass

class MLForecastStrategy(Strategy):
    """
    Uses our PyTorch ML models to generate binary/fractional signals 
    based on predicted returns out-performing a given threshold.
    """
    def __init__(self, data_handler: DataHandler, events: Queue, tickers: list, model, threshold: float = 0.005):
        self.data_handler = data_handler
        self.events = events
        self.tickers = tickers
        self.model = model
        self.threshold = threshold
        
        self.strategy_id = "ML_Transformer_Forecast"

    def calculate_signals(self, event):
        if event.type != 'MARKET':
            return
            
        # In a real event driven loop, we would extract a sliding window of historical data 
        # from the data handler, feed it to self.model.predict(), and act on the results.
        # For simplicity in this demo structure, we will stub out the ML prediction response.
        
        # Pseudo-prediction logic
        latest_timestamp = self.data_handler.latest_symbol_data[-1][0]
        
        for ticker in self.tickers:
            # Assume model ran and returned a predicted 5-day return of 'pred_return'
            pred_return = np.random.normal(0.001, 0.02) # Mock prediction
            
            if pred_return > self.threshold:
                # Go Long
                signal = SignalEvent(self.strategy_id, ticker, latest_timestamp, 'LONG', 1.0)
                self.events.put(signal)
            elif pred_return < -self.threshold:
                # Go Short
                signal = SignalEvent(self.strategy_id, ticker, latest_timestamp, 'SHORT', 1.0)
                self.events.put(signal)

