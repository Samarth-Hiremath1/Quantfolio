import pandas as pd
from queue import Queue
from backtesting.events import MarketEvent

class DataHandler:
    """
    Simulates the exchange natively pushing historical market data 
    into the event queue row-by-row, identically to a live feed.
    """
    def __init__(self, events: Queue, historical_dataframe: pd.DataFrame):
        """
        :param events: The Event Queue.
        :param historical_dataframe: A multi-asset DataFrame where index is Date, and columns are a MultiIndex 
                                     (Ticker, OHLCV Feature).
        """
        self.events = events
        self.symbol_data = historical_dataframe
        
        # Sort index ensuring chronological order
        self.symbol_data.sort_index(inplace=True)
        
        self.continue_backtest = True
        self._data_generator = self._get_new_bar()
        self.latest_symbol_data = []

    def _get_new_bar(self):
        """
        Returns an iterator generating the next row of data.
        """
        for index, row in self.symbol_data.iterrows():
            yield index, dict(row)

    def update_bars(self):
        """
        Pushes the latest bar to the internal state and emits a MarketEvent.
        """
        try:
            index, bar = next(self._data_generator)
            self.latest_symbol_data.append((index, bar))
            self.events.put(MarketEvent())
        except StopIteration:
            self.continue_backtest = False
            
    def get_latest_bar(self, symbol: str) -> dict:
        """
        Returns the most recent dictionary of OHLCV data for the given symbol.
        """
        if not self.latest_symbol_data:
            return {}
        latest_dict = self.latest_symbol_data[-1][1]
        
        # Extract features for the specific symbol
        symbol_dict = {key[1]: latest_dict[key] for key in latest_dict.keys() if key[0] == symbol}
        return symbol_dict
        
    def get_latest_bar_value(self, symbol: str, val_type: str = 'adj_close') -> float:
        """
        Quick helper to get the latest close price for an asset.
        """
        bar = self.get_latest_bar(symbol)
        return bar.get(val_type, 0.0)
