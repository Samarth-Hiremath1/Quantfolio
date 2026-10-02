import re

from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict
from datetime import date, datetime

# Defence-in-depth alongside the parameterized queries in api/queries.py:
# reject anything that isn't a plausible ticker symbol. See KNOWN_ISSUES.md D-05.
_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]{1,10}$")


def _validate_tickers(values: List[str]) -> List[str]:
    if not values:
        raise ValueError("at least one ticker is required")
    for v in values:
        if not _TICKER_RE.match(v):
            raise ValueError(f"invalid ticker symbol: {v!r}")
    return [v.upper() for v in values]

# Asset Metadata
class AssetMetadataBase(BaseModel):
    ticker: str
    name: Optional[str] = None
    sector: Optional[str] = None
    asset_class: Optional[str] = None

class AssetMetadataResponse(AssetMetadataBase):
    is_active: bool
    created_at: datetime
    
    model_config = {"from_attributes": True}

# Data Responses
class OHLCVRecord(BaseModel):
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int

    model_config = {"from_attributes": True}

class FeatureRecord(BaseModel):
    trade_date: date
    log_return_1d: Optional[float] = None
    rolling_vol_21d: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None

    model_config = {"from_attributes": True}

# Portfolio Optimization Requests
class OptimizationRequest(BaseModel):
    tickers: List[str]
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    objective: str = "sharpe" # 'sharpe', 'volatility', 'return'

    _v = field_validator("tickers")(_validate_tickers)

class OptimizationResponse(BaseModel):
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    risk_contributions: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]

class RiskResponse(BaseModel):
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float

class FactorDecompositionRequest(BaseModel):
    tickers: List[str]
    weights: Dict[str, float]
    factor_tickers: List[str] = ["SPY", "QQQ"] # Defaults to basic market and tech factor

    _v = field_validator("tickers", "factor_tickers")(_validate_tickers)

class FactorDecompositionResponse(BaseModel):
    alpha: float
    factor_loadings: Dict[str, float]
    r_squared: float
    idiosyncratic_risk: float

# Forecasting Requests
class ForecastRequest(BaseModel):
    tickers: List[str]
    model_type: str = "Ridge_AR"  # 'Ridge_AR' (default) or 'LSTM'

    _v = field_validator("tickers")(_validate_tickers)

class ForecastResponse(BaseModel):
    forecasts: Dict[str, List[float]]
    # Which model actually produced each forecast, and on how much history.
    # Prevents a short-history fallback being mistaken for a full LSTM fit.
    models_used: Dict[str, str]
    n_observations: Dict[str, int]

# Backtesting Requests
class BacktestRequest(BaseModel):
    tickers: List[str]
    strategy: str = "ML_Forecast"
    initial_capital: float = 100000.0

    _v = field_validator("tickers")(_validate_tickers)

class BacktestResponse(BaseModel):
    final_value: float
    total_return_pct: float
    total_trades: int
    sharpe_ratio: float
    max_drawdown: float
    bars_processed: int
