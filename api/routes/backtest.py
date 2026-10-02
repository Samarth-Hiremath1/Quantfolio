from fastapi import APIRouter, HTTPException

from api import schemas
from api.queries import fetch_backtest_frame
from backtesting.engine import BacktestingEngine
from backtesting.strategy import MLForecastStrategy, BuyAndHoldStrategy
from portfolio.risk import RiskMetrics

router = APIRouter(prefix="/backtest", tags=["backtest"])

_STRATEGIES = {
    "ML_Forecast": MLForecastStrategy,   # SMA-crossover signal when no model injected
    "SMA_Crossover": MLForecastStrategy,
    "BuyAndHold": BuyAndHoldStrategy,
}


@router.post("/", response_model=schemas.BacktestResponse)
def run_event_driven_backtest(request: schemas.BacktestRequest):
    """
    Runs the event-driven backtest loop over stored history, simulating
    slippage and Interactive-Brokers-style commissions.

    Signals come from a real SMA crossover (or an injected model), not from
    random draws -- see KNOWN_ISSUES.md D-04.
    """
    strategy_class = _STRATEGIES.get(request.strategy)
    if strategy_class is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{request.strategy}'. Options: {sorted(_STRATEGIES)}",
        )

    historical_data = fetch_backtest_frame(request.tickers)

    backtester = BacktestingEngine(
        data=historical_data,
        tickers=[t.upper() for t in request.tickers],
        strategy_class=strategy_class,
        model=None,
    )
    backtester.portfolio.current_cash = request.initial_capital

    final_val = backtester.run()
    total_ret = ((final_val - request.initial_capital) / request.initial_capital) * 100.0

    # Risk-adjusted stats from the recorded equity curve (KNOWN_ISSUES.md D-07).
    curve = backtester.portfolio.get_equity_curve()
    if len(curve) > 1:
        rm = RiskMetrics()
        rets = curve["returns"]
        sharpe = float(rm.compute_sharpe_ratio(rets))
        max_dd = float(rm.compute_max_drawdown(rets))
    else:
        sharpe, max_dd = 0.0, 0.0

    return {
        "final_value": final_val,
        "total_return_pct": total_ret,
        "total_trades": backtester.fills,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "bars_processed": len(curve),
    }
