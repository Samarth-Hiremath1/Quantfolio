import numpy as np
import pandas as pd
import statsmodels.api as sm
import logging

logger = logging.getLogger(__name__)

class FactorModel:
    """
    Factor-based risk decomposition. Uses OLS regression to decompose 
    portfolio returns into factor exposures (market beta, momentum, sector).
    """

    def __init__(self, portfolio_returns: pd.Series, factor_returns: pd.DataFrame):
        """
        :param portfolio_returns: Daily returns of the portfolio (aligned with factors).
        :param factor_returns: DataFrame of daily factor returns (e.g., SPY, momentum factor, sector ETFs).
        """
        # Ensure alignment
        aligned_data = pd.concat([portfolio_returns.rename("portfolio"), factor_returns], axis=1).dropna()
        self.portfolio_returns = aligned_data["portfolio"]
        self.factor_returns = aligned_data.drop(columns=["portfolio"])
        self.model = None
        self.results = None

    def fit(self) -> dict:
        """
        Fits the OLS regression model.
        Returns a dictionary of factor loadings, R-squared, and residual variance.
        """
        logger.info("Fitting multi-factor OLS model...")
        X = sm.add_constant(self.factor_returns)
        y = self.portfolio_returns

        self.model = sm.OLS(y, X)
        self.results = self.model.fit()

        summary = {
            "alpha": self.results.params.get("const", 0.0),
            "factor_loadings": self.results.params.drop("const", errors="ignore").to_dict(),
            "r_squared": self.results.rsquared,
            "adj_r_squared": self.results.rsquared_adj,
            "p_values": self.results.pvalues.to_dict(),
            "residual_variance": self.results.resid.var(),
            "idiosyncratic_risk": self.results.resid.std() * np.sqrt(252) # Annualized
        }
        
        logger.info(f"Model fit complete. R-squared: {summary['r_squared']:.4f}")
        return summary

    def decompose_risk(self) -> dict:
        """
        Decomposes total portfolio variance into systematic (factor) and idiosyncratic risk.
        Requires fit() to be run first.
        """
        if self.results is None:
            self.fit()

        factor_cov = self.factor_returns.cov()
        loadings = self.results.params.drop("const", errors="ignore")
        
        # Systematic variance = B * Cov(Factors) * B.T
        systematic_var = np.dot(loadings.T, np.dot(factor_cov, loadings))
        
        # Idiosyncratic variance
        idiosyncratic_var = self.results.resid.var()
        
        total_var = systematic_var + idiosyncratic_var
        
        decomposition = {
            "total_variance": total_var,
            "systematic_variance": systematic_var,
            "idiosyncratic_variance": idiosyncratic_var,
            "systematic_contribution_pct": (systematic_var / total_var) * 100 if total_var > 0 else 0,
            "idiosyncratic_contribution_pct": (idiosyncratic_var / total_var) * 100 if total_var > 0 else 0
        }
        
        return decomposition
