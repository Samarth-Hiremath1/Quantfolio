#!/usr/bin/env python3
"""
Setup script for QuantFolio ML Pipeline
Creates necessary directories and initializes the project structure
"""

import os
import sys
from pathlib import Path
import subprocess


def create_directories():
    """Create necessary directories for the project"""
    directories = [
        "data/raw/daily",
        "data/processed",
        "data/features/technical",
        "data/features/time_series/windows_30d",
        "data/features/time_series/windows_60d",
        "data/models/pytorch",
        "data/models/tensorflow",
        "data/models/metadata",
        "data/portfolios/allocations",
        "data/portfolios/backtests",
        "logs",
        "mlruns",
        "artifacts",
        "models/artifacts",
        "orchestration/dags",
        "monitoring/dashboards",
        "infrastructure/terraform",
        "tests/unit",
        "tests/integration",
        "tests/performance",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
.venv/
.env

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Data and models
data/
!data/.gitkeep
mlruns/
artifacts/
models/artifacts/
*.pkl
*.joblib
*.h5
*.pth
*.ckpt

# Logs
logs/
*.log

# Configuration
.env.local
.env.development
.env.production
config/secrets.yaml

# Docker
.dockerignore

# Terraform
infrastructure/terraform/.terraform/
infrastructure/terraform/*.tfstate
infrastructure/terraform/*.tfstate.backup
infrastructure/terraform/.terraform.lock.hcl

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# MLflow
mlruns/
mlartifacts/

# Monitoring
monitoring/data/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())
    
    print("Created .gitignore file")


def create_env_template():
    """Create environment template file"""
    env_template = """
# Environment Configuration Template
# Copy this file to .env and fill in your values

# Environment
ENVIRONMENT=local
DEBUG=true
SERVICE_NAME=quantfolio
LOG_LEVEL=INFO

# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=quantfolio_local
DATABASE_USER=postgres
DATABASE_PASSWORD=password

# External APIs
ALPHA_VANTAGE_API_KEY=your_api_key_here

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# ML Configuration
PYTORCH_DEVICE=cpu
TENSORFLOW_DEVICE=/CPU:0

# Security (change in production)
JWT_SECRET_KEY=your-secret-key-change-in-production
"""
    
    with open(".env.template", "w") as f:
        f.write(env_template.strip())
    
    print("Created .env.template file")


def create_makefile():
    """Create Makefile for common operations"""
    makefile_content = """
.PHONY: help install dev-install test lint format clean docker-build docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  dev-install  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean build artifacts"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-up    - Start services with Docker Compose"
	@echo "  docker-down  - Stop services"

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=. --cov-report=html

lint:
	flake8 .
	mypy .

format:
	black .
	isort .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

setup-dev:
	python setup.py
	cp .env.template .env
	@echo "Development environment setup complete!"
	@echo "Please edit .env file with your configuration"
"""
    
    with open("Makefile", "w") as f:
        f.write(makefile_content.strip())
    
    print("Created Makefile")


def create_dev_requirements():
    """Create development requirements file"""
    dev_requirements = """
# Development dependencies
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.1
pre-commit==3.6.0
"""
    
    with open("requirements-dev.txt", "w") as f:
        f.write(dev_requirements.strip())
    
    print("Created requirements-dev.txt")


def main():
    """Main setup function"""
    print("Setting up QuantFolio ML Pipeline project structure...")
    
    create_directories()
    create_gitignore()
    create_env_template()
    create_makefile()
    create_dev_requirements()
    
    # Create .gitkeep files for empty directories
    empty_dirs = [
        "data/raw/daily",
        "data/processed",
        "logs",
        "mlruns",
        "artifacts",
        "models/artifacts",
        "orchestration/dags",
        "monitoring/dashboards",
    ]
    
    for directory in empty_dirs:
        gitkeep_file = Path(directory) / ".gitkeep"
        gitkeep_file.touch()
    
    print("\n✅ Project structure setup complete!")
    print("\nNext steps:")
    print("1. Copy .env.template to .env and configure your settings")
    print("2. Install dependencies: make dev-install")
    print("3. Start services: make docker-up")
    print("4. Run tests: make test")


if __name__ == "__main__":
    main()