-- Initialize the schema for QuantFolio

CREATE TABLE IF NOT EXISTS asset_metadata (
    ticker VARCHAR(20) PRIMARY KEY,
    name VARCHAR(255),
    sector VARCHAR(100),
    asset_class VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS daily_ohlcv (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) REFERENCES asset_metadata(ticker),
    trade_date DATE,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    adj_close NUMERIC,
    volume BIGINT,
    UNIQUE(ticker, trade_date)
);

CREATE TABLE IF NOT EXISTS daily_features (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) REFERENCES asset_metadata(ticker),
    trade_date DATE,
    log_return_1d NUMERIC,
    rolling_vol_21d NUMERIC,
    rsi_14 NUMERIC,
    macd NUMERIC,
    macd_signal NUMERIC,
    UNIQUE(ticker, trade_date)
);

-- Note: We will insert some basic asset_metadata later.
