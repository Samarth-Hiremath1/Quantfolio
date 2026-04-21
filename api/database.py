import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Expects docker-compose to set POSTGRES_URL format: postgresql://quant:quant_pass@postgres:5432/quantfolio
SQLALCHEMY_DATABASE_URL = os.getenv(
    "POSTGRES_URL", 
    "postgresql://quant:quant_pass@localhost:5432/quantfolio" # fallback for pure local dev outside docker
)

# Create SQLAlchemy engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create a scoped session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all ORM models
Base = declarative_base()

# Dependency for FastAPI to get DB sessions per request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
