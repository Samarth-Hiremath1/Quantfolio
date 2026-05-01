from fastapi import APIRouter, HTTPException
import pandas as pd
from typing import List

from api.database import engine
from api import schemas
from backtesting.engine import BacktestingEngine
from backtesting.strategy import MLForecastStrategy

router = APIRouter(prefix="/backtest", tags=["backtest"])

def _fetch_full_historical_data(tickers: List[str]) -> pd.DataFrame:
    """Fetches OHLCV and Features for the backtester DataHandler."""
    tickers_str = "','".join(tickers)
    query = f"SELECT trade_date, ticker, adj_close, volume, log_return_1d, rsi_14, macd FROM daily_features JOIN daily_ohlcv USING (trade_date, ticker) WHERE ticker IN ('{tickers_str}')"
    df = pd.read_sql(query, engine)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")
    
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index(['trade_date', 'ticker'], inplace=True)
    return df.unstack(level=1)

@router.post("/", response_model=schemas.BacktestResponse)
def run_event_driven_backtest(request: schemas.BacktestRequest):
    """
    Initializes the Event-Driven Backtesting loop, feeding historical data
    and simulating realistic Interactive Brokers fills.
    """
    # 1. Load Data
    historical_data = _fetch_full_historical_data(request.tickers)
    
    # 2. Setup Event Engine
    # (Mocking model object to None for Strategy instantiation)
    backtester = BacktestingEngine(
        data=historical_data, 
        tickers=request.tickers, 
        strategy_class=MLForecastStrategy,
        model=None 
    )
    
    # 3. Inject cash
    backtester.portfolio.current_cash = request.initial_capital
    
    # 4. Run Loop
    final_val = backtester.run()
    
    # 5. Return Stats
    total_ret = ((final_val - request.initial_capital) / request.initial_capital) * 100.0
    
    return {
        "final_value": final_val,
        "total_return_pct": total_ret,
        "total_trades": backtester.fills
    }
