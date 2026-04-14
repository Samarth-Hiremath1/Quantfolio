import numpy as np
import pandas as pd

class RiskMetrics:
    """
    Computes portfolio risk metrics including Sharpe, Sortino, Drawdown, and VaR.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.daily_rf = self.risk_free_rate / 252

    def compute_sharpe_ratio(self, returns: pd.Series, annualized: bool = True) -> float:
        """
        Computes the Sharpe ratio of a returns series.
        """
        mean_return = returns.mean() - self.daily_rf
        std_dev = returns.std()
        if std_dev == 0 or np.isnan(std_dev):
            return 0.0
            
        sharpe = mean_return / std_dev
        if annualized:
            return sharpe * np.sqrt(252)
        return sharpe

    def compute_sortino_ratio(self, returns: pd.Series, annualized: bool = True) -> float:
        """
        Computes the Sortino ratio (punishes only downside volatility).
        """
        mean_return = returns.mean() - self.daily_rf
        downside_returns = returns[returns < 0]
        
        downside_std = downside_returns.std()
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
            
        sortino = mean_return / downside_std
        if annualized:
            return sortino * np.sqrt(252)
        return sortino

    def compute_max_drawdown(self, returns: pd.Series) -> float:
        """
        Computes the maximum drawdown of a returns series.
        """
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def compute_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Computes Value at Risk (Historical).
        """
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence_level) * 100)

    def compute_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Computes Conditional Value at Risk (Expected Shortfall).
        """
        var = self.compute_var(returns, confidence_level)
        tail_losses = returns[returns <= var]
        if len(tail_losses) == 0:
            return var
        return tail_losses.mean()

    def generate_report(self, returns: pd.Series) -> dict:
        """
        Generates a dictionary of all risk metrics for convenience.
        """
        return {
            "annualized_sharpe": self.compute_sharpe_ratio(returns),
            "annualized_sortino": self.compute_sortino_ratio(returns),
            "max_drawdown": self.compute_max_drawdown(returns),
            "var_95": self.compute_var(returns, 0.95),
            "cvar_95": self.compute_cvar(returns, 0.95)
        }
