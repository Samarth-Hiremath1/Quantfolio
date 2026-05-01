from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import data, portfolio, forecast, backtest
from api.database import engine, Base

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="QuantFolio API", description="Quantitative Portfolio Research API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router, prefix="/api/v1")
app.include_router(portfolio.router, prefix="/api/v1")
app.include_router(forecast.router, prefix="/api/v1")
app.include_router(backtest.router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
