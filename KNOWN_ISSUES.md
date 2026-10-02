# Defect log

Bugs found while benchmarking the project. Each fixed entry has a regression test in `tests/`.

## Fixed

### D-01: `/api/v1/backtest` raised `TypeError` on every request

`BacktestingEngine` called the strategy with three arguments when `model` was `None`. `MLForecastStrategy.__init__` required four. The route always passed `model=None`, so it always failed.

Fix: `model` is now optional on the strategy, and the engine passes it in every case.

### D-02: `/api/v1/backtest` returned zero trades without an error

The route built its frame with `unstack(level=1)`, which orders the columns as `(feature, ticker)`. `DataHandler.get_latest_bar` selects on `key[0] == symbol` and expects `(ticker, feature)`. Every price lookup missed and fell back to `0.0`. The execution handler treats a zero price as "no data" and drops the order. The result was a clean response reporting a 0% return.

Fix: the query helper calls `swaplevel(axis=1)`. `DataHandler` now checks the column orientation in its constructor and raises if it is wrong.

This one was worse than D-01. A crash is obvious. A believable zero is not.

### D-03: `/api/v1/portfolio/risk` imported a class that does not exist

The route imported `PortfolioRisk` and called `compute_all_metrics()`. The module defines `RiskMetrics` and `generate_report()`. The dictionary keys differed too, and the route read them with `.get(key, 0.0)`, so fixing only the import would have returned all zeros.

Fix: correct class and method, and explicit key lookups that fail loudly.

### D-04: the backtest strategy traded on random numbers

`calculate_signals` used `np.random.normal(0.001, 0.02)` as its prediction, redrawn per ticker per bar. Every result from the backtester was noise. The strategy also signalled on most bars, about 5.6 signals per bar with seven tickers.

Fix: an SMA(10/50) crossover computed from observed prices, or a forecast from an injected model exposing `predict_next`. Signals fire only on a change of state.

### D-05: SQL injection in three routes

`portfolio.py`, `backtest.py` and `forecast.py` joined ticker strings from the request body into SQL with an f-string. `data.py` uses the ORM and was not affected.

Fix: `api/queries.py` uses SQLAlchemy bound parameters with an expanding `IN` clause. The request schemas also reject tickers that do not match `^[A-Za-z0-9.\-]{1,10}$`.

### D-06: `/api/v1/forecast` returned hardcoded values

Every ticker received `[0.001, 0.002, -0.001, 0.005, 0.003]`. The model call was commented out.

Fix: the route fits a Ridge AR(5) model on the stored return history and forecasts five steps recursively. `model_type="LSTM"` trains `UnivariateLSTM` when at least 120 observations exist and falls back to Ridge otherwise. The response names the model that ran and the number of observations it saw. The LSTM branch has no test yet.

### D-07: the portfolio kept no equity curve

`Portfolio` tracked current cash and holdings only, so a backtest produced one final value and no way to compute Sharpe or drawdown.

Fix: `update_timeindex` appends `(timestamp, total_value)` on each bar. `get_equity_curve()` returns it as a DataFrame with a returns column.

### D-08: no tests

`requirements.txt` listed pytest and the repository had no tests. There are now 14, covering the fixes above.

## Open

### D-09: the optimizer returns weights that did not converge

At 100 assets SLSQP reaches its iteration limit. `maximize_sharpe` logs a warning and returns the weights regardless. Options: raise `maxiter`, supply an analytic gradient, or raise on failure.

### D-10: sample covariance without shrinkage

The optimizer uses the plain sample covariance. With many assets and a short history that matrix is poorly conditioned. Ledoit-Wolf shrinkage would be the usual remedy.

### D-11: the Alpha Vantage fetcher is unused

`data/ingestion/alpha_vantage_fetcher.py` defines a fetcher, but the Airflow DAG only calls `YFinanceFetcher`.

### D-12: forecasting models are not connected to the backtester

`UnivariateLSTM` and `MultiAssetTransformer` are defined, and the LSTM is evaluated in the walk-forward benchmark. Neither drives the strategy. The strategy accepts any object with `predict_next`, so the hook exists.
