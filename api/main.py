from fastapi import FastAPI

app = FastAPI(title="QuantFolio API", description="Quantitative Portfolio Research & Forecasting API")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
