import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Implements Markowitz Mean-Variance Optimization.
    """

    def __init__(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        :param expected_returns: Series of annualized expected returns for each asset.
        :param cov_matrix: DataFrame containing the annualized covariance matrix.
        :param risk_free_rate: Annual risk-free rate.
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(expected_returns)
        self.asset_names = expected_returns.index

    def _portfolio_annualized_performance(self, weights: np.ndarray):
        """
        Calculates expected portfolio return and volatility given weights.
        """
        returns = np.sum(self.expected_returns * weights)
        std_dev = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (returns - self.risk_free_rate) / std_dev if std_dev > 0 else 0
        return returns, std_dev, sharpe

    def _neg_sharpe_ratio(self, weights: np.ndarray):
        """
        Objective function for SciPy optimizer. We want to maximize Sharpe, so we minimize negative Sharpe.
        """
        return -1 * self._portfolio_annualized_performance(weights)[2]

    def _portfolio_volatility(self, weights: np.ndarray):
        return self._portfolio_annualized_performance(weights)[1]

    def maximize_sharpe(self, bounds: tuple = (0.0, 1.0)) -> pd.Series:
        """
        Finds the portfolio weights that maximize the Sharpe ratio.
        """
        args = ()
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Weights must sum to 1
        bound_params = tuple(bounds for _ in range(self.num_assets))
        initial_guess = self.num_assets * [1. / self.num_assets]

        result = minimize(
            self._neg_sharpe_ratio,
            initial_guess,
            args=args,
            method='SLSQP',
            bounds=bound_params,
            constraints=constraints
        )

        if not result.success:
            logger.warning(f"Sharpe maximization did not converge: {result.message}")

        return pd.Series(np.round(result.x, 4), index=self.asset_names)

    def minimize_volatility(self, bounds: tuple = (0.0, 1.0)) -> pd.Series:
        """
        Finds the portfolio weights that minimize volatility (Global Minimum Variance portfolio).
        """
        args = ()
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound_params = tuple(bounds for _ in range(self.num_assets))
        initial_guess = self.num_assets * [1. / self.num_assets]

        result = minimize(
            self._portfolio_volatility,
            initial_guess,
            args=args,
            method='SLSQP',
            bounds=bound_params,
            constraints=constraints
        )
        
        return pd.Series(np.round(result.x, 4), index=self.asset_names)

    def efficient_frontier(self, points: int = 50, bounds: tuple = (0.0, 1.0)) -> tuple:
        """
        Generates the efficient frontier.
        :return: Tuple of arrays (target_returns, target_vols, weights).
        """
        # Find minimum volatility and maximum return
        min_vol_weights = self.minimize_volatility(bounds)
        min_ret, min_vol, _ = self._portfolio_annualized_performance(min_vol_weights.values)
        
        # Max return is just fully weighting the asset with highest return
        # But we can approximate the max achievable return within bounds
        max_ret = self.expected_returns.max()
        
        target_returns = np.linspace(min_ret, max_ret, points)
        target_vols = []
        frontier_weights = []

        bound_params = tuple(bounds for _ in range(self.num_assets))
        
        for tr in target_returns:
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self._portfolio_annualized_performance(x)[0] - tr}
            )
            
            result = minimize(
                self._portfolio_volatility,
                self.num_assets * [1. / self.num_assets],
                method='SLSQP',
                bounds=bound_params,
                constraints=constraints
            )
            
            if result.success:
                target_vols.append(result.fun)
                frontier_weights.append(result.x)
            else:
                target_vols.append(np.nan)
                frontier_weights.append(np.nan)

        return np.array(target_returns), np.array(target_vols), frontier_weights
