"""
Regression tests for the defects catalogued in KNOWN_ISSUES.md.
Each test names the defect it guards.
"""
import queue

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestingEngine
from backtesting.data_handler import DataHandler
from backtesting.strategy import MLForecastStrategy, BuyAndHoldStrategy


def _make_data(n_tickers=3, n_days=200, seed=0):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n_tickers)]
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    frames = {}
    for tk in tickers:
        price = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n_days)))
        frames[(tk, "adj_close")] = price
        frames[(tk, "close")] = price
        frames[(tk, "volume")] = rng.integers(1e6, 5e6, n_days).astype(float)
    df = pd.DataFrame(frames, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df, tickers


# ---- D-01: engine no longer raises with model=None -------------------------
def test_d01_engine_runs_with_model_none():
    data, tickers = _make_data()
    eng = BacktestingEngine(data=data, tickers=tickers,
                            strategy_class=MLForecastStrategy, model=None)
    final = eng.run()
    assert final > 0


# ---- D-02: (feature, ticker) orientation is rejected loudly -----------------
def test_d02_wrong_orientation_raises():
    data, _ = _make_data()
    swapped = data.swaplevel(axis=1).sort_index(axis=1)  # -> (feature, ticker)
    with pytest.raises(ValueError, match="feature, ticker"):
        DataHandler(queue.Queue(), swapped)


def test_d02_correct_orientation_ok():
    data, _ = _make_data()
    DataHandler(queue.Queue(), data)  # (ticker, feature) -> no raise


# ---- D-04: signals are deterministic, not random ----------------------------
def test_d04_signals_are_deterministic():
    finals = []
    for _ in range(3):
        data, tickers = _make_data(seed=7)
        eng = BacktestingEngine(data=data, tickers=tickers,
                                strategy_class=MLForecastStrategy, model=None)
        finals.append(eng.run())
    assert len(set(round(f, 6) for f in finals)) == 1, "backtest must be reproducible"


def test_d04_crossover_generates_trades_on_trending_data():
    # A steadily rising series must trigger at least one SMA crossover -> fills.
    dates = pd.bdate_range("2020-01-01", periods=200)
    price = np.linspace(100, 200, 200)
    df = pd.DataFrame({("T0", "adj_close"): price, ("T0", "close"): price,
                       ("T0", "volume"): 1e6})
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    eng = BacktestingEngine(data=df, tickers=["T0"],
                            strategy_class=MLForecastStrategy, model=None)
    eng.run()
    assert eng.fills > 0


def test_d04_model_predict_next_is_used():
    class UpModel:
        def predict_next(self, history):
            return 0.05  # always bullish
    data, tickers = _make_data(n_tickers=1)
    eng = BacktestingEngine(data=data, tickers=tickers,
                            strategy_class=MLForecastStrategy, model=UpModel())
    eng.run()
    assert eng.strategy.strategy_id == "ML_Forecast"
    assert eng.fills > 0


# ---- D-07: equity curve is recorded -----------------------------------------
def test_d07_equity_curve_recorded():
    data, tickers = _make_data(n_days=120)
    eng = BacktestingEngine(data=data, tickers=tickers,
                            strategy_class=BuyAndHoldStrategy, model=None)
    eng.run()
    curve = eng.portfolio.get_equity_curve()
    assert len(curve) == 120
    assert "returns" in curve.columns
    assert curve["total_value"].iloc[0] > 0
