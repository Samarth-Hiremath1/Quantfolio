import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VectorizedBacktester:
    """
    Simulates portfolio equity curves given fractional position targets.
    Optimized mathematically via Pandas instead of an event-driven loop.
    """

    def __init__(self, prices: pd.DataFrame, signals: pd.DataFrame, initial_capital: float = 10000.0, transaction_cost_pct: float = 0.001):
        """
        :param prices: DataFrame of asset prices (e.g. daily adjusted closes)
        :param signals: DataFrame of fractional target holdings (ranging from -1.0 to 1.0) for each asset.
                        Must perfectly match the index/columns of the `prices` DataFrame.
        :param initial_capital: Starting cash balance.
        :param transaction_cost_pct: Cost per trade (e.g. slippage + commissions). 0.001 = 0.1% per trade.
        """
        self.prices = prices.copy()
        
        # Shift signals 1 day forward: we can only trade tomorrow based on today's signal!
        self.target_weights = signals.shift(1).fillna(0)
        
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct

    def run(self) -> pd.DataFrame:
        """
        Executes the backtest and returns a timeseries of portfolio performance metrics.
        """
        logger.info("Running vectorized backtest engine...")
        
        # Asset daily arithmetic returns
        returns = self.prices.pct_change().fillna(0)

        # Portfolio return calculation
        # Gross portfolio daily return = sum of (weight * asset return)
        gross_returns = (self.target_weights * returns).sum(axis=1)

        # Transaction costs calculation
        # Any change in target weight implies a trade.
        weight_changes = self.target_weights.diff().fillna(0).abs()
        tx_costs = (weight_changes * self.transaction_cost_pct).sum(axis=1)
        
        # Net Returns
        net_returns = gross_returns - tx_costs

        # Cumulative track records
        cumulative_gross = (1 + gross_returns).cumprod() * self.initial_capital
        cumulative_net = (1 + net_returns).cumprod() * self.initial_capital

        performance_df = pd.DataFrame({
            "daily_gross_return": gross_returns,
            "daily_net_return": net_returns,
            "equity_gross": cumulative_gross,
            "equity_net": cumulative_net,
            "transaction_costs": tx_costs
        })

        return performance_df
