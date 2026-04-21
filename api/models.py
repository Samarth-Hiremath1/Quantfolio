from sqlalchemy import Column, Integer, String, Boolean, Numeric, Date, BigInteger, ForeignKey, DateTime
from sqlalchemy.sql import func
from api.database import Base

class AssetMetadata(Base):
    __tablename__ = "asset_metadata"

    ticker = Column(String(20), primary_key=True, index=True)
    name = Column(String(255))
    sector = Column(String(100))
    asset_class = Column(String(50))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())

class DailyOHLCV(Base):
    __tablename__ = "daily_ohlcv"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), ForeignKey("asset_metadata.ticker"))
    trade_date = Column(Date, index=True)
    open = Column(Numeric)
    high = Column(Numeric)
    low = Column(Numeric)
    close = Column(Numeric)
    adj_close = Column(Numeric)
    volume = Column(BigInteger)

class DailyFeatures(Base):
    __tablename__ = "daily_features"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), ForeignKey("asset_metadata.ticker"))
    trade_date = Column(Date, index=True)
    log_return_1d = Column(Numeric)
    rolling_vol_21d = Column(Numeric)
    rsi_14 = Column(Numeric)
    macd = Column(Numeric)
    macd_signal = Column(Numeric)
