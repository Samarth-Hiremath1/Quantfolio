from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from api.database import get_db
from api import models, schemas

router = APIRouter(prefix="/data", tags=["data"])

@router.get("/symbols", response_model=List[schemas.AssetMetadataResponse])
def get_symbols(db: Session = Depends(get_db)):
    """Fetch all active assets stored in the platform."""
    return db.query(models.AssetMetadata).filter(models.AssetMetadata.is_active == True).all()

@router.get("/ohlcv/{ticker}", response_model=List[schemas.OHLCVRecord])
def get_ohlcv(ticker: str, limit: int = 252, db: Session = Depends(get_db)):
    """Fetch recent OHLCV pricing records for a single ticker."""
    records = db.query(models.DailyOHLCV)\
                .filter(models.DailyOHLCV.ticker == ticker.upper())\
                .order_by(models.DailyOHLCV.trade_date.desc())\
                .limit(limit)\
                .all()
    if not records:
        raise HTTPException(status_code=404, detail=f"No pricing data found for ticker {ticker}")
    return records

@router.get("/features/{ticker}", response_model=List[schemas.FeatureRecord])
def get_features(ticker: str, limit: int = 252, db: Session = Depends(get_db)):
    """Fetch recent technical indicator features for a single ticker."""
    records = db.query(models.DailyFeatures)\
                .filter(models.DailyFeatures.ticker == ticker.upper())\
                .order_by(models.DailyFeatures.trade_date.desc())\
                .limit(limit)\
                .all()
    if not records:
        raise HTTPException(status_code=404, detail=f"No feature data found for ticker {ticker}")
    return records
