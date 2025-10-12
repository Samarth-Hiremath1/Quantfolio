"""
Integration test for PyTorch training service
"""

import pytest
import torch
import numpy as np
import asyncio
import tempfile
import os
import sys

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

from pytorch_service import (
    PyTorchTrainingService, ModelConfig, create_model_config
)


@pytest.mark.asyncio
async def test_lstm_training_integration():
    """Test LSTM training end-to-end"""
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 8
    
    features = np.random.randn(n_samples, n_features)
    targets = np.sum(features[:, :3], axis=1) * 0.1 + np.random.randn(n_samples) * 0.05
    returns = np.random.randn(n_samples) * 0.02 + 0.001
    
    # Create training service
    with tempfile.TemporaryDirectory() as temp_dir:
        mlflow_uri = f"file://{temp_dir}/mlruns"
        training_service = PyTorchTrainingService(mlflow_tracking_uri=mlflow_uri)
        
        # Create config
        config = create_model_config(
            input_size=n_features,
            hidden_size=32,
            num_layers=1,
            num_epochs=5,  # Small for testing
            sequence_length=10,
            batch_size=16
        )
        
        # Train LSTM model
        result = await training_service.train_lstm_model(config, features, targets, returns)
        
        # Verify results
        assert result is not None
        assert 'sharpe_ratio' in result.metrics
        assert 'accuracy' in result.metrics
        assert 'mse' in result.metrics
        assert os.path.exists(result.model_path)


@pytest.mark.asyncio
async def test_gru_training_integration():
    """Test GRU training end-to-end"""
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 6
    
    features = np.random.randn(n_samples, n_features)
    targets = np.sum(features[:, :2], axis=1) * 0.1 + np.random.randn(n_samples) * 0.05
    returns = np.random.randn(n_samples) * 0.02 + 0.001
    
    # Create training service
    with tempfile.TemporaryDirectory() as temp_dir:
        mlflow_uri = f"file://{temp_dir}/mlruns"
        training_service = PyTorchTrainingService(mlflow_tracking_uri=mlflow_uri)
        
        # Create config
        config = create_model_config(
            input_size=n_features,
            hidden_size=24,
            num_layers=1,
            num_epochs=5,  # Small for testing
            sequence_length=15,
            batch_size=8
        )
        
        # Train GRU model
        result = await training_service.train_gru_model(config, features, targets, returns)
        
        # Verify results
        assert result is not None
        assert 'sharpe_ratio' in result.metrics
        assert 'accuracy' in result.metrics
        assert 'mse' in result.metrics
        assert os.path.exists(result.model_path)


def test_model_evaluation():
    """Test model evaluation functionality"""
    
    from pytorch_service import LSTMWithAttention
    
    # Create simple test data
    np.random.seed(42)
    features = np.random.randn(100, 5)
    targets = np.random.randn(100)
    returns = np.random.randn(100) * 0.02
    
    config = create_model_config(
        input_size=5,
        hidden_size=16,
        num_layers=1,
        sequence_length=10
    )
    
    model = LSTMWithAttention(config)
    training_service = PyTorchTrainingService()
    
    # Test evaluation
    async def run_evaluation():
        metrics = await training_service.evaluate_model(
            model, features, targets, returns, config
        )
        return metrics
    
    metrics = asyncio.run(run_evaluation())
    
    assert metrics.mse >= 0
    assert metrics.mae >= 0
    assert isinstance(metrics.sharpe_ratio, float)
    assert 0 <= metrics.accuracy <= 1


if __name__ == "__main__":
    pytest.main([__file__])