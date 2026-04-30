import logging
from queue import Queue
from backtesting.events import OrderEvent
from backtesting.data_handler import DataHandler

logger = logging.getLogger(__name__)

class Portfolio:
    """
    Handles position sizing and Order management based on SignalEvents.
    """
    def __init__(self, data_handler: DataHandler, events: Queue, initial_capital: float = 100000.0):
        self.data_handler = data_handler
        self.events = events
        self.current_cash = initial_capital
        
        self.current_positions = {} # ticker: quantity
        self.current_holdings = {}  # ticker: value in $

    def update_timeindex(self, event):
        """
        Updates the value of holdings based on the latest MARKET event prices.
        """
        for ticker, qty in self.current_positions.items():
            latest_price = self.data_handler.get_latest_bar_value(ticker, 'adj_close')
            self.current_holdings[ticker] = qty * latest_price

    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders.
        """
        if event.type == 'SIGNAL':
            order_event = self.generate_order(event)
            self.events.put(order_event)

    def generate_order(self, signal):
        """
        Simple Order generation logic:
        Given a LONG/SHORT signal, buys or sells an arbitrary set quantity (e.g., 100 shares).
        In a real MPT Phase 2 integrated pipeline, this would call `optimizer.py` to calculate exact weight distributions.
        """
        order_type = 'MKT'
        ticker = signal.ticker
        direction = signal.signal_type
        
        # Extremely basic position sizing logic (fixed 100 share lot)
        mkt_quantity = 100 
        
        if direction == 'LONG':
            logger.info(f"Signal generated: Buying {ticker}")
            return OrderEvent(ticker, order_type, mkt_quantity, 'BUY')
        elif direction == 'SHORT':
            logger.info(f"Signal generated: Selling {ticker}")
            return OrderEvent(ticker, order_type, mkt_quantity, 'SELL')

    def update_fill(self, event):
        """
        Updates the portfolio's current cash and positions from a FillEvent.
        """
        if event.type == 'FILL':
            fill_dir = 1 if event.direction == 'BUY' else -1
            
            # Update Position
            current_qty = self.current_positions.get(event.ticker, 0)
            self.current_positions[event.ticker] = current_qty + (fill_dir * event.quantity)
            
            # Update Cash
            cost = fill_dir * event.fill_cost * event.quantity
            self.current_cash -= (cost + event.commission)
            
            logger.info(f"Filled {event.direction} {event.quantity} {event.ticker} @ ${event.fill_cost:.2f}. Commission: ${event.commission:.2f}")
