"""
Test fixtures: seed an in-memory-ish SQLite DB from the repo's real CSVs, then
point the API at it so the routes can be exercised without Docker/Postgres.
"""
import glob
import os

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "tests", "_test.db")
DB_URL = f"sqlite:///{DB_PATH}"

# Must be set BEFORE api.* modules import their engine.
os.environ["POSTGRES_URL"] = DB_URL


def _seed(engine):
    raw_files = [os.path.join(ROOT, "tests", "fixtures", "raw_ohlcv_sample.csv")]
    feat_files = [os.path.join(ROOT, "tests", "fixtures", "feat_ohlcv_sample.csv")]
    raw = pd.read_csv(raw_files[-1])
    feat = pd.read_csv(feat_files[-1])

    ohlcv = raw[["ticker", "trade_date", "open", "high", "low", "close", "adj_close", "volume"]].copy()
    ohlcv.insert(0, "id", range(1, len(ohlcv) + 1))
    ohlcv.to_sql("daily_ohlcv", engine, if_exists="replace", index=False)

    fcols = ["ticker", "trade_date", "log_return_1d", "rolling_vol_21d", "rsi_14", "macd", "macd_signal"]
    features = feat[fcols].copy()
    features.insert(0, "id", range(1, len(features) + 1))
    features.to_sql("daily_features", engine, if_exists="replace", index=False)

    meta = pd.DataFrame({
        "ticker": sorted(raw["ticker"].unique()),
    })
    meta["name"] = meta["ticker"]
    meta["sector"] = "n/a"
    meta["asset_class"] = "equity"
    meta["is_active"] = True
    meta["created_at"] = pd.Timestamp.utcnow().isoformat()
    meta.to_sql("asset_metadata", engine, if_exists="replace", index=False)


@pytest.fixture(scope="session")
def client():
    from sqlalchemy import create_engine
    from fastapi.testclient import TestClient

    engine = create_engine(DB_URL)
    _seed(engine)

    from api.main import app
    with TestClient(app) as c:
        yield c

    engine.dispose()
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


@pytest.fixture(scope="session")
def tickers():
    feat_files = [os.path.join(ROOT, "tests", "fixtures", "feat_ohlcv_sample.csv")]
    return sorted(pd.read_csv(feat_files[-1])["ticker"].unique().tolist())
