"""
Forecasting endpoint.

Previously this returned a hardcoded list of five floats for every ticker and
never touched a model (KNOWN_ISSUES.md D-06). It now fits a real model on the
requested tickers' return history and returns genuine recursive forecasts.
"""
import numpy as np
from fastapi import APIRouter, HTTPException

from api import schemas
from api.queries import fetch_aligned_returns

router = APIRouter(prefix="/forecast", tags=["forecast"])

HORIZON = 5
AR_LAGS = 5
MIN_OBS = AR_LAGS + 10          # minimum history to fit the AR model at all
MIN_LSTM_OBS = 120              # below this, an LSTM fit is not defensible


def _ridge_ar_forecast(series: np.ndarray, horizon: int = HORIZON, lags: int = AR_LAGS):
    """Ridge AR(lags) fit, forecast `horizon` steps recursively."""
    from sklearn.linear_model import Ridge

    X = np.column_stack([series[i:len(series) - lags + i] for i in range(lags)])
    y = series[lags:]
    model = Ridge(alpha=1.0).fit(X, y)

    window = list(series[-lags:])
    preds = []
    for _ in range(horizon):
        nxt = float(model.predict(np.array(window[-lags:]).reshape(1, -1))[0])
        preds.append(nxt)
        window.append(nxt)
    return preds


def _lstm_forecast(series: np.ndarray, horizon: int = HORIZON, seq_len: int = 20, epochs: int = 40):
    """Train the repo's UnivariateLSTM on this series and forecast `horizon` steps."""
    import torch
    from models.forecasting.lstm_model import UnivariateLSTM, create_sequences

    torch.manual_seed(42)
    X, Y = create_sequences(series.astype(np.float32), seq_len, horizon)
    if len(X) == 0:
        raise ValueError("insufficient history for LSTM sequences")

    model = UnivariateLSTM(input_size=1, hidden_size=32, num_layers=1, output_size=horizon)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    Xt, Yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), Yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        last = torch.tensor(series[-seq_len:].astype(np.float32)).reshape(1, seq_len, 1)
        return [float(v) for v in model(last).numpy().ravel()]


@router.post("/", response_model=schemas.ForecastResponse)
def generate_forecast(request: schemas.ForecastRequest):
    """
    Fits a model per ticker on stored return history and returns a real 5-day
    forecast. `model_type` accepts "Ridge_AR" (default, fast) or "LSTM".

    The response reports which model actually ran and how many observations it
    was fitted on, so a short-history fallback can't be mistaken for a full fit.
    """
    rets_df = fetch_aligned_returns(request.tickers)

    requested = (request.model_type or "Ridge_AR").upper()
    if requested not in {"RIDGE_AR", "LSTM"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_type '{request.model_type}'. Options: Ridge_AR, LSTM.",
        )

    forecasts, models_used, n_obs = {}, {}, {}
    for ticker in rets_df.columns:
        series = rets_df[ticker].to_numpy(dtype=float)
        n_obs[ticker] = int(len(series))

        if len(series) < MIN_OBS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Insufficient history for {ticker}: {len(series)} observations, "
                    f"need at least {MIN_OBS}."
                ),
            )

        if requested == "LSTM" and len(series) >= MIN_LSTM_OBS:
            try:
                forecasts[ticker] = _lstm_forecast(series)
                models_used[ticker] = "UnivariateLSTM"
                continue
            except Exception as exc:  # fall back rather than 500
                models_used[ticker] = f"Ridge_AR (LSTM failed: {type(exc).__name__})"
        elif requested == "LSTM":
            models_used[ticker] = (
                f"Ridge_AR (fallback: {len(series)} obs < {MIN_LSTM_OBS} required for LSTM)"
            )
        else:
            models_used[ticker] = "Ridge_AR"

        forecasts[ticker] = _ridge_ar_forecast(series)

    return {"forecasts": forecasts, "models_used": models_used, "n_observations": n_obs}
