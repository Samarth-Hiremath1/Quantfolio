from abc import ABC, abstractmethod
from collections import defaultdict, deque
from queue import Queue
from typing import Optional

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
    Generates LONG/SHORT signals from a real, deterministic signal source.

    Two modes:

    1. **Model-driven** — if `model` is supplied and exposes
       ``predict_next(history: np.ndarray) -> float`` (a forecast of the next
       period's return), the forecast is compared against `threshold`.
    2. **Momentum fallback** — otherwise a genuine SMA(fast/slow) crossover
       computed from the actual price history observed so far.

    Signals fire only on a **change of state** (a crossover / a flip in the
    forecast's sign relative to the threshold), not on every bar. Emitting a
    signal every bar produces unrealistic turnover and swamps the event queue.

    NOTE: this class previously returned ``np.random.normal(...)`` as a "mock
    prediction", which made every backtest result meaningless. See
    KNOWN_ISSUES.md D-04.
    """

    def __init__(self, data_handler: DataHandler, events: Queue, tickers: list,
                 model=None, threshold: float = 0.0,
                 fast_window: int = 10, slow_window: int = 50):
        self.data_handler = data_handler
        self.events = events
        self.tickers = tickers
        self.model = model
        self.threshold = threshold
        self.fast_window = fast_window
        self.slow_window = slow_window

        # Rolling price history per ticker, bounded to what the signal needs.
        self._prices = defaultdict(lambda: deque(maxlen=max(slow_window, 60) + 1))
        # Last emitted state per ticker: 1 long, -1 short, 0 flat/none.
        self._state = defaultdict(int)

        self.strategy_id = "ML_Forecast" if model is not None else "SMA_Crossover"

    def _desired_state(self, ticker: str) -> Optional[int]:
        """Returns the desired position state for a ticker, or None if undecidable yet."""
        prices = self._prices[ticker]

        if self.model is not None and hasattr(self.model, "predict_next"):
            if len(prices) < 2:
                return None
            arr = np.asarray(prices, dtype=float)
            rets = np.diff(arr) / arr[:-1]
            try:
                pred = float(self.model.predict_next(rets))
            except Exception:
                return None
            if not np.isfinite(pred):
                return None
            if pred > self.threshold:
                return 1
            if pred < -self.threshold:
                return -1
            return 0

        # Momentum fallback: SMA crossover on observed prices.
        if len(prices) < self.slow_window:
            return None
        arr = np.asarray(prices, dtype=float)
        fast = arr[-self.fast_window:].mean()
        slow = arr[-self.slow_window:].mean()
        return 1 if fast > slow else -1

    def calculate_signals(self, event):
        if event.type != 'MARKET':
            return

        latest_timestamp = self.data_handler.latest_symbol_data[-1][0]

        for ticker in self.tickers:
            price = self.data_handler.get_latest_bar_value(ticker, 'adj_close')
            if not price:  # 0.0 / missing -> nothing observable this bar
                continue
            self._prices[ticker].append(price)

            desired = self._desired_state(ticker)
            if desired is None or desired == self._state[ticker]:
                continue  # no change -> no signal (realistic turnover)

            self._state[ticker] = desired
            if desired == 1:
                self.events.put(SignalEvent(self.strategy_id, ticker, latest_timestamp, 'LONG', 1.0))
            elif desired == -1:
                self.events.put(SignalEvent(self.strategy_id, ticker, latest_timestamp, 'SHORT', 1.0))
            # desired == 0 -> flat: no new order under the current fixed-lot sizer


class BuyAndHoldStrategy(Strategy):
    """
    Reference benchmark: goes long every ticker on the first observable bar and
    never trades again. Used to give backtest results an honest comparison point.
    """

    def __init__(self, data_handler: DataHandler, events: Queue, tickers: list, model=None):
        self.data_handler = data_handler
        self.events = events
        self.tickers = tickers
        self.model = model
        self._bought = set()
        self.strategy_id = "BuyAndHold"

    def calculate_signals(self, event):
        if event.type != 'MARKET':
            return
        latest_timestamp = self.data_handler.latest_symbol_data[-1][0]
        for ticker in self.tickers:
            if ticker in self._bought:
                continue
            if not self.data_handler.get_latest_bar_value(ticker, 'adj_close'):
                continue
            self._bought.add(ticker)
            self.events.put(SignalEvent(self.strategy_id, ticker, latest_timestamp, 'LONG', 1.0))
