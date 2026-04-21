import pandas as pd
from sklearn.linear_model import Ridge
import logging

logger = logging.getLogger(__name__)

class BaselineForecaster:
    """
    A baseline Machine Learning model stub.
    Uses basic Ridge Regression to predict next day's returns based on today's technical features.
    """

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        
    def fit(self, features_df: pd.DataFrame, target_returns: pd.Series):
        """
        Trains the forecaster.
        :param features_df: DataFrame of indicators (e.g. RSI, MACD, Volatility)
        :param target_returns: Series of forward returns (e.g., Return(t+1))
        """
        # Ensure exact alignment and drop NaNs
        aligned = pd.concat([features_df, target_returns.rename('target')], axis=1).dropna()
        X = aligned.drop(columns=['target'])
        y = aligned['target']
        
        logger.info(f"Training Baseline Forecaster on {len(X)} samples.")
        self.model.fit(X, y)
        
    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Generates return forecasts.
        """
        X = features_df.dropna()
        if len(X) == 0:
            return pd.Series(dtype=float)
            
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=X.index)
