from fastapi import APIRouter, HTTPException
import pandas as pd
from typing import List

from api.database import engine
from api import schemas
from portfolio.optimizer import PortfolioOptimizer
from portfolio.risk import PortfolioRisk
from portfolio.factor_model import FactorModel

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

def _fetch_aligned_returns(tickers: List[str]) -> pd.DataFrame:
    """Helper to fetch 1d log returns from Postgres via pandas."""
    if not tickers:
        raise HTTPException(status_code=400, detail="Must provide at least one ticker.")
        
    tickers_str = "','".join(tickers)
    query = f"""
    SELECT trade_date, ticker, log_return_1d 
    FROM daily_features 
    WHERE ticker IN ('{tickers_str}')
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for the provided tickers.")
        
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    # Pivot so each column is a ticker's returns
    pivot_df = df.pivot(index='trade_date', columns='ticker', values='log_return_1d').dropna()
    return pivot_df

@router.post("/optimize", response_model=schemas.OptimizationResponse)
def optimize_portfolio(request: schemas.OptimizationRequest):
    rets_df = _fetch_aligned_returns(request.tickers)
    
    # Calculate expected returns (annualized arithmetic mean approximation) 
    expected_returns = rets_df.mean() * 252
    cov_matrix = rets_df.cov() * 252
    
    optimizer = PortfolioOptimizer(expected_returns, cov_matrix)
    
    if request.objective == "volatility":
        weights = optimizer.minimize_volatility()
    else:
        weights = optimizer.maximize_sharpe()
        
    ret, vol, sharpe = optimizer._portfolio_annualized_performance(weights.values)
    risk_contributions = optimizer.get_risk_contributions(weights)
    corr_matrix = rets_df.corr().to_dict()
    
    return {
        "weights": weights.to_dict(),
        "expected_return": ret,
        "expected_volatility": vol,
        "sharpe_ratio": sharpe,
        "risk_contributions": risk_contributions.to_dict(),
        "correlation_matrix": corr_matrix
    }

@router.post("/risk", response_model=schemas.RiskResponse)
def calculate_risk(request: schemas.FactorDecompositionRequest):
    # Using FactorDecompositionRequest since it has tickers and weights
    rets_df = _fetch_aligned_returns(request.tickers)
    
    weights_series = pd.Series(request.weights)
    # Ensure alignment
    aligned_weights = weights_series.reindex(rets_df.columns).fillna(0.0)
    
    # Calculate portfolio return timeseries
    port_returns = rets_df.dot(aligned_weights)
    
    risk_engine = PortfolioRisk(port_returns)
    metrics = risk_engine.compute_all_metrics()
    
    return {
        "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
        "sortino_ratio": metrics.get("sortino_ratio", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "var_95": metrics.get("value_at_risk_95", 0.0),
        "cvar_95": metrics.get("conditional_var_95", 0.0)
    }

@router.post("/factor", response_model=schemas.FactorDecompositionResponse)
def compute_factors(request: schemas.FactorDecompositionRequest):
    all_tickers = list(set(request.tickers + request.factor_tickers))
    rets_df = _fetch_aligned_returns(all_tickers)
    
    # Portfolio returns
    port_cols = [t for t in request.tickers if t in rets_df.columns]
    factor_cols = [t for t in request.factor_tickers if t in rets_df.columns]
    
    if not factor_cols:
        raise HTTPException(status_code=400, detail="Requested factor tickers have no data.")
        
    weights_series = pd.Series(request.weights).reindex(port_cols).fillna(0.0)
    port_returns = rets_df[port_cols].dot(weights_series)
    factor_returns = rets_df[factor_cols]
    
    model = FactorModel(port_returns, factor_returns)
    summary = model.fit()
    
    return {
        "alpha": summary["alpha"],
        "factor_loadings": summary["factor_loadings"],
        "r_squared": summary["r_squared"],
        "idiosyncratic_risk": summary["idiosyncratic_risk"]
    }
