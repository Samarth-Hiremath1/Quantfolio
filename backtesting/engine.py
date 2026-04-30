import queue
import logging
from typing import List
import pandas as pd

from backtesting.data_handler import DataHandler
from backtesting.strategy import Strategy
from backtesting.portfolio import Portfolio
from backtesting.execution import ExecutionHandler

logger = logging.getLogger(__name__)

class BacktestingEngine:
    """
    The core Event-Driven Engine. Contains the fundamental "while True" queue loop.
    Properly routes Market, Signal, Order, and Fill events completely destroying any lookahead bias natively.
    """
    def __init__(self, data: pd.DataFrame, tickers: List[str], strategy_class, model=None):
        self.events = queue.Queue()
        self.data_handler = DataHandler(self.events, data)
        
        # Initialize strategy (in our Phase 4 case, the ML Strategy)
        if model is not None:
            self.strategy = strategy_class(self.data_handler, self.events, tickers, model)
        else:
            self.strategy = strategy_class(self.data_handler, self.events, tickers)
            
        self.portfolio = Portfolio(self.data_handler, self.events)
        self.execution_handler = ExecutionHandler(self.events, self.data_handler)
        
        self.signals = 0
        self.orders = 0
        self.fills = 0

    def run(self):
        """
        Executes the backtest natively.
        """
        logger.info("Initializing Event-Driven Backtest...")
        
        while True:
            # Update the market bars
            if self.data_handler.continue_backtest == True:
                self.data_handler.update_bars()
            else:
                break
                
            # Process ALL events generated from this Market tick before moving to the next time tick
            while True:
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    break
                else:
                    if event is not None:
                        if event.type == 'MARKET':
                            self.strategy.calculate_signals(event)
                            self.portfolio.update_timeindex(event)
                            
                        elif event.type == 'SIGNAL':
                            self.signals += 1
                            self.portfolio.update_signal(event)
                            
                        elif event.type == 'ORDER':
                            self.orders += 1
                            self.execution_handler.execute_order(event)
                            
                        elif event.type == 'FILL':
                            self.fills += 1
                            self.portfolio.update_fill(event)

        logger.info("Backtest Complete.")
        logger.info(f"Final Cash: ${self.portfolio.current_cash:.2f}")
        total_val = self.portfolio.current_cash + sum(self.portfolio.current_holdings.values())
        logger.info(f"Final Total Portfolio Value: ${total_val:.2f}")
        logger.info(f"Signals: {self.signals}, Orders: {self.orders}, Fills: {self.fills}")
        
        return total_val
