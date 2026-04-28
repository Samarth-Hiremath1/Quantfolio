from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import date, datetime

# Asset Metadata
class AssetMetadataBase(BaseModel):
    ticker: str
    name: Optional[str] = None
    sector: Optional[str] = None
    asset_class: Optional[str] = None

class AssetMetadataResponse(AssetMetadataBase):
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# Data Responses
class OHLCVRecord(BaseModel):
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int

    class Config:
        from_attributes = True

class FeatureRecord(BaseModel):
    trade_date: date
    log_return_1d: Optional[float] = None
    rolling_vol_21d: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None

    class Config:
        from_attributes = True

# Portfolio Optimization Requests
class OptimizationRequest(BaseModel):
    tickers: List[str]
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    objective: str = "sharpe" # 'sharpe', 'volatility', 'return'

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

class FactorDecompositionResponse(BaseModel):
    alpha: float
    factor_loadings: Dict[str, float]
    r_squared: float
    idiosyncratic_risk: float
