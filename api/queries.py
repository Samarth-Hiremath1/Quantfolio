"""
Parameterized data-access helpers shared by the API routes.

All ticker lists are passed as SQLAlchemy bound parameters (expanding IN clause)
rather than interpolated into the SQL string. See KNOWN_ISSUES.md D-05.
"""
from typing import List

import pandas as pd
from fastapi import HTTPException
from sqlalchemy import bindparam, text

from api.database import engine

_RETURNS_SQL = """
    SELECT trade_date, ticker, log_return_1d
    FROM daily_features
    WHERE ticker IN :tickers
"""

_BACKTEST_SQL = """
    SELECT f.trade_date, f.ticker, o.adj_close, o.volume,
           f.log_return_1d, f.rsi_14, f.macd
    FROM daily_features f
    JOIN daily_ohlcv o
      ON o.trade_date = f.trade_date AND o.ticker = f.ticker
    WHERE f.ticker IN :tickers
"""


def _run(sql: str, tickers: List[str]) -> pd.DataFrame:
    if not tickers:
        raise HTTPException(status_code=400, detail="Must provide at least one ticker.")
    stmt = text(sql).bindparams(bindparam("tickers", expanding=True))
    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn, params={"tickers": [t.upper() for t in tickers]})
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for the provided tickers.")
    return df


def fetch_aligned_returns(tickers: List[str]) -> pd.DataFrame:
    """Wide frame of 1-day log returns: index=trade_date, columns=ticker."""
    df = _run(_RETURNS_SQL, tickers)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    pivot = df.pivot(index="trade_date", columns="ticker", values="log_return_1d").dropna()
    if pivot.empty:
        raise HTTPException(
            status_code=422,
            detail="No overlapping dates with complete return data across the requested tickers.",
        )
    return pivot.astype(float)


def fetch_backtest_frame(tickers: List[str]) -> pd.DataFrame:
    """
    OHLCV+features shaped for the backtester's DataHandler, oriented as
    (ticker, feature) -- NOT (feature, ticker). See KNOWN_ISSUES.md D-02.
    """
    df = _run(_BACKTEST_SQL, tickers)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index(["trade_date", "ticker"]).astype(float)
    wide = df.unstack(level=1)              # -> (feature, ticker)
    wide = wide.swaplevel(axis=1)           # -> (ticker, feature)
    return wide.sort_index(axis=1).ffill().dropna(how="all")
