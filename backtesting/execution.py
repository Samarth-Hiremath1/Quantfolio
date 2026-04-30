from queue import Queue
from backtesting.events import FillEvent
from backtesting.data_handler import DataHandler

class ExecutionHandler:
    """
    Simulates an interactive broker handling an OrderEvent.
    Calculates realistic slippage and returns a FillEvent.
    """
    def __init__(self, events: Queue, data_handler: DataHandler, slippage_pct: float = 0.0005):
        self.events = events
        self.data_handler = data_handler
        self.slippage_pct = slippage_pct

    def execute_order(self, event):
        """
        Mocks filling an order instantly at the current close price + slippage.
        """
        if event.type == 'ORDER':
            # Get latest explicit timestamp & price
            latest_timestamp = self.data_handler.latest_symbol_data[-1][0]
            price = self.data_handler.get_latest_bar_value(event.ticker, 'adj_close')
            
            if price == 0.0:
                print(f"Could not fill {event.direction} {event.ticker}. No price data found.")
                return
                
            # Apply slippage logically (buy higher, sell lower)
            if event.direction == 'BUY':
                fill_price = price * (1 + self.slippage_pct)
            else:
                fill_price = price * (1 - self.slippage_pct)
                
            fill_event = FillEvent(
                timeindex=latest_timestamp,
                ticker=event.ticker,
                exchange='SMART',
                quantity=event.quantity,
                direction=event.direction,
                fill_cost=fill_price
            )
            
            self.events.put(fill_event)
