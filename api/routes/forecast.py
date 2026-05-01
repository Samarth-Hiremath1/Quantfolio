from fastapi import APIRouter, HTTPException
import pandas as pd
from typing import List

from api.database import engine
from api import schemas

# Mock importing the PyTorch models for the API layer
# from models.forecasting.lstm_model import UnivariateLSTM
# from models.forecasting.transformer_model import MultiAssetTransformer

router = APIRouter(prefix="/forecast", tags=["forecast"])

def _fetch_aligned_returns(tickers: List[str]) -> pd.DataFrame:
    tickers_str = "','".join(tickers)
    query = f"SELECT trade_date, ticker, log_return_1d FROM daily_features WHERE ticker IN ('{tickers_str}')"
    df = pd.read_sql(query, engine)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df.pivot(index='trade_date', columns='ticker', values='log_return_1d').dropna()

@router.post("/", response_model=schemas.ForecastResponse)
def generate_forecast(request: schemas.ForecastRequest):
    """
    Invokes the PyTorch Machine Learning models to predict the next 5 days of returns.
    """
    rets_df = _fetch_aligned_returns(request.tickers)
    
    # In production, we would load the serialized .pt weights and call model.predict()
    # model = UnivariateLSTM() if request.model_type == 'LSTM' else MultiAssetTransformer()
    # predictions = model(rets_df.values)
    
    # Mocking response for API demonstration
    forecasts = {}
    for ticker in request.tickers:
        # Mock 5-day trajectory
        forecasts[ticker] = [0.001, 0.002, -0.001, 0.005, 0.003]
        
    return {"forecasts": forecasts}
