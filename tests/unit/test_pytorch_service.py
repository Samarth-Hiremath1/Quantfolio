"""
Unit tests for PyTorch Training Service

Tests model architectures, training loops, evaluation metrics, and MLflow integration.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from pathlib import Path
import mlflow

# Import the modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

from pytorch_service import (
    PyTorchTrainingService, LSTMWithAttention, GRUModel, AttentionMechanism,
    ModelConfig, TrainingResult, EvaluationMetrics, create_model_config, compare_models
)


class TestModelConfig:
    """Test ModelConfig dataclass"""
    
    def test_model_config_creation(self):
        config = ModelConfig(
            input_size=10,
            hidden_size=64,
            num_layers=2,
            dropout_rate=0.2,
            learning_rate=0.001,
            batch_size=32,
            num_epochs=100,
            sequence_length=30
        )
        
        assert config.input_size == 10
        assert config.hidden_size == 64
        assert config.num_layers == 2
        assert config.dropout_rate == 0.2
        assert config.output_size == 1  # default value
        assert config.device in ["cuda", "cpu"]
    
    def test_create_model_config_with_defaults(self):
        config = create_model_config(input_size=15)
        
        assert config.input_size == 15
        assert config.hidden_size == 128  # default
        assert config.num_layers == 2  # default
        assert config.sequence_length == 30  # default
    
    def test_create_model_config_with_overrides(self):
        config = create_model_config(
            input_size=20,
            hidden_size=256,
            learning_rate=0.01
        )
        
        assert config.input_size == 20
        assert config.hidden_size == 256
        assert config.learning_rate == 0.01
        assert config.num_layers == 2  # still default


class TestAttentionMechanism:
    """Test attention mechanism component"""
    
    def test_attention_forward_pass(self):
        hidden_size = 64
        batch_size = 8
        seq_len = 30
        
        attention = AttentionMechanism(hidden_size)
        lstm_outputs = torch.randn(batch_size, seq_len, hidden_size)
        
        attended_output, attention_weights = attention(lstm_outputs)
        
        # Check output shapes
        assert attended_output.shape == (batch_size, hidden_size)
        assert attention_weights.shape == (batch_size, seq_len, 1)
        
        # Check attention weights sum to 1
        attention_sums = torch.sum(attention_weights, dim=1)
        assert torch.allclose(attention_sums, torch.ones(batch_size, 1), atol=1e-6)
    
    def test_attention_weights_properties(self):
        attention = AttentionMechanism(32)
        lstm_outputs = torch.randn(4, 10, 32)
        
        _, attention_weights = attention(lstm_outputs)
        
        # All weights should be positive (softmax output)
        assert torch.all(attention_weights >= 0)
        
        # Weights should sum to 1 for each sequence
        sums = torch.sum(attention_weights, dim=1)
        assert torch.allclose(sums, torch.ones(4, 1))


class TestLSTMWithAttention:
    """Test LSTM model with attention mechanism"""
    
    def test_lstm_model_creation(self):
        config = ModelConfig(
            input_size=10,
            hidden_size=64,
            num_layers=2,
            dropout_rate=0.2,
            learning_rate=0.001,
            batch_size=32,
            num_epochs=100,
            sequence_length=30
        )
        
        model = LSTMWithAttention(config)
        
        assert isinstance(model.lstm, nn.LSTM)
        assert isinstance(model.attention, AttentionMechanism)
        assert model.lstm.input_size == 10
        assert model.lstm.hidden_size == 64
        assert model.lstm.num_layers == 2
    
    def test_lstm_forward_pass(self):
        config = ModelConfig(
            input_size=15,
            hidden_size=32,
            num_layers=1,
            dropout_rate=0.1,
            learning_rate=0.001,
            batch_size=16,
            num_epochs=50,
            sequence_length=20
        )
        
        model = LSTMWithAttention(config)
        batch_size = 8
        seq_len = 20
        input_size = 15
        
        x = torch.randn(batch_size, seq_len, input_size)
        output, attention_weights = model(x)
        
        # Check output shapes
        assert output.shape == (batch_size, 1)
        assert attention_weights.shape == (batch_size, seq_len, 1)
        
        # Check output is finite
        assert torch.all(torch.isfinite(output))
    
    def test_lstm_gradient_flow(self):
        config = ModelConfig(
            input_size=5,
            hidden_size=16,
            num_layers=1,
            dropout_rate=0.0,
            learning_rate=0.001,
            batch_size=4,
            num_epochs=10,
            sequence_length=10
        )
        
        model = LSTMWithAttention(config)
        x = torch.randn(2, 10, 5, requires_grad=True)
        target = torch.randn(2, 1)
        
        output, _ = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert torch.any(param.grad != 0)


class TestGRUModel:
    """Test GRU model with dropout and batch normalization"""
    
    def test_gru_model_creation(self):
        config = ModelConfig(
            input_size=12,
            hidden_size=48,
            num_layers=3,
            dropout_rate=0.3,
            learning_rate=0.001,
            batch_size=32,
            num_epochs=100,
            sequence_length=25
        )
        
        model = GRUModel(config)
        
        assert isinstance(model.gru, nn.GRU)
        assert isinstance(model.batch_norm, nn.BatchNorm1d)
        assert model.gru.input_size == 12
        assert model.gru.hidden_size == 48
        assert model.gru.num_layers == 3
    
    def test_gru_forward_pass(self):
        config = ModelConfig(
            input_size=8,
            hidden_size=24,
            num_layers=2,
            dropout_rate=0.2,
            learning_rate=0.001,
            batch_size=16,
            num_epochs=50,
            sequence_length=15
        )
        
        model = GRUModel(config)
        batch_size = 6
        seq_len = 15
        input_size = 8
        
        x = torch.randn(batch_size, seq_len, input_size)
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 1)
        
        # Check output is finite
        assert torch.all(torch.isfinite(output))
    
    def test_gru_batch_normalization(self):
        config = ModelConfig(
            input_size=6,
            hidden_size=32,
            num_layers=1,
            dropout_rate=0.1,
            learning_rate=0.001,
            batch_size=8,
            num_epochs=20,
            sequence_length=12
        )
        
        model = GRUModel(config)
        
        # Test with different batch sizes
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 12, 6)
            output = model(x)
            assert output.shape == (batch_size, 1)
            assert torch.all(torch.isfinite(output))


class TestPyTorchTrainingService:
    """Test PyTorch training service"""
    
    @pytest.fixture
    def training_service(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow_uri = f"file://{temp_dir}/mlruns"
            service = PyTorchTrainingService(mlflow_tracking_uri=mlflow_uri)
            yield service
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        # Generate synthetic financial data
        features = np.random.randn(n_samples, n_features)
        # Add some trend and correlation
        for i in range(1, n_features):
            features[:, i] = 0.7 * features[:, i-1] + 0.3 * features[:, i]
        
        # Generate targets (returns) with some correlation to features
        targets = np.sum(features[:, :3], axis=1) * 0.1 + np.random.randn(n_samples) * 0.05
        
        # Generate return series for metrics calculation
        returns = np.random.randn(n_samples) * 0.02 + 0.001
        
        return features, targets, returns
    
    def test_prepare_data(self, training_service, sample_data):
        features, targets, _ = sample_data
        sequence_length = 30
        
        train_loader, val_loader = training_service._prepare_data(
            features, targets, sequence_length, train_split=0.8
        )
        
        # Check data loaders are created
        assert train_loader is not None
        assert val_loader is not None
        
        # Check batch shapes
        for batch_X, batch_y in train_loader:
            assert batch_X.shape[1] == sequence_length
            assert batch_X.shape[2] == features.shape[1]
            assert batch_y.shape[1] == 1
            break  # Just check first batch
    
    def test_calculate_financial_metrics(self, training_service):
        # Create test data
        predictions = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        actual = np.array([0.015, -0.018, 0.025, -0.012, 0.018])
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        
        metrics = training_service._calculate_financial_metrics(predictions, actual, returns)
        
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.mse >= 0
        assert metrics.mae >= 0
        assert metrics.rmse >= 0
        assert isinstance(metrics.sharpe_ratio, float)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.directional_accuracy <= 1
        assert metrics.volatility_prediction_error >= 0
        assert metrics.max_drawdown <= 0
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('mlflow.pytorch.log_model')
    @patch('mlflow.log_artifact')
    def test_train_lstm_model_integration(self, mock_log_artifact, mock_log_model, 
                                        mock_log_metrics, mock_log_params, mock_start_run,
                                        training_service, sample_data):
        # Mock MLflow run context
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_run.info.experiment_id = "test_experiment_id"
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        features, targets, returns = sample_data
        
        config = ModelConfig(
            input_size=features.shape[1],
            hidden_size=32,
            num_layers=1,
            dropout_rate=0.1,
            learning_rate=0.01,
            batch_size=16,
            num_epochs=2,  # Small for testing
            sequence_length=10
        )
        
        # This would normally be async, but for testing we'll call it directly
        # In a real async test, we'd use pytest-asyncio
        import asyncio
        
        async def run_test():
            result = await training_service.train_lstm_model(config, features, targets, returns)
            return result
        
        # For this test, we'll mock the heavy computation parts
        with patch.object(training_service, '_train_model') as mock_train:
            mock_model = LSTMWithAttention(config)
            mock_history = {'train_loss': [0.1, 0.05], 'val_loss': [0.12, 0.06]}
            mock_train.return_value = (mock_model, mock_history)
            
            result = asyncio.run(run_test())
            
            assert isinstance(result, TrainingResult)
            assert result.run_id == "test_run_id"
            assert result.experiment_id == "test_experiment_id"
            assert 'mse' in result.metrics
            assert 'sharpe_ratio' in result.metrics
    
    def test_save_and_load_checkpoint(self, training_service):
        config = ModelConfig(
            input_size=5,
            hidden_size=16,
            num_layers=1,
            dropout_rate=0.1,
            learning_rate=0.001,
            batch_size=8,
            num_epochs=10,
            sequence_length=10
        )
        
        model = LSTMWithAttention(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            checkpoint_path = tmp_file.name
        
        try:
            # Save checkpoint
            training_service.save_model_checkpoint(
                model, optimizer, epoch=5, loss=0.1, config=config, 
                checkpoint_path=checkpoint_path
            )
            
            # Load checkpoint
            loaded_model, loaded_optimizer, epoch, loss = training_service.load_model_checkpoint(
                checkpoint_path, LSTMWithAttention
            )
            
            assert epoch == 5
            assert loss == 0.1
            assert isinstance(loaded_model, LSTMWithAttention)
            assert isinstance(loaded_optimizer, torch.optim.Adam)
            
        finally:
            os.unlink(checkpoint_path)
    
    def test_model_evaluation(self, training_service, sample_data):
        features, targets, returns = sample_data
        
        config = ModelConfig(
            input_size=features.shape[1],
            hidden_size=16,
            num_layers=1,
            dropout_rate=0.0,
            learning_rate=0.001,
            batch_size=8,
            num_epochs=1,
            sequence_length=10
        )
        
        model = LSTMWithAttention(config)
        
        # Use a subset of data for testing
        test_features = features[:100]
        test_targets = targets[:100]
        test_returns = returns[:100]
        
        import asyncio
        
        async def run_evaluation():
            metrics = await training_service.evaluate_model(
                model, test_features, test_targets, test_returns, config
            )
            return metrics
        
        metrics = asyncio.run(run_evaluation())
        
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.mse >= 0
        assert metrics.mae >= 0


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_compare_models(self):
        # Create mock training results
        config1 = ModelConfig(input_size=10, hidden_size=32, num_layers=1, 
                             dropout_rate=0.1, learning_rate=0.001, batch_size=16,
                             num_epochs=10, sequence_length=20)
        
        config2 = ModelConfig(input_size=10, hidden_size=64, num_layers=2,
                             dropout_rate=0.2, learning_rate=0.001, batch_size=32,
                             num_epochs=20, sequence_length=30)
        
        result1 = TrainingResult(
            model_path="model1.pth",
            metrics={'sharpe_ratio': 1.2, 'accuracy': 0.65, 'mse': 0.01},
            training_history={'train_loss': [0.1, 0.05]},
            model_config=config1,
            experiment_id="exp1",
            run_id="run1"
        )
        
        result2 = TrainingResult(
            model_path="model2.pth",
            metrics={'sharpe_ratio': 1.5, 'accuracy': 0.70, 'mse': 0.008},
            training_history={'train_loss': [0.12, 0.04]},
            model_config=config2,
            experiment_id="exp2",
            run_id="run2"
        )
        
        comparison = compare_models([result1, result2])
        
        assert len(comparison['models']) == 2
        assert comparison['best_model']['run_id'] == "run2"  # Higher Sharpe ratio
        assert comparison['best_metric'] == 'sharpe_ratio'
        assert comparison['best_model']['metrics']['sharpe_ratio'] == 1.5


class TestModelArchitectures:
    """Test model architecture specific functionality"""
    
    def test_lstm_attention_mechanism_integration(self):
        config = ModelConfig(
            input_size=8,
            hidden_size=32,
            num_layers=2,
            dropout_rate=0.1,
            learning_rate=0.001,
            batch_size=16,
            num_epochs=10,
            sequence_length=20
        )
        
        model = LSTMWithAttention(config)
        x = torch.randn(4, 20, 8)
        
        output, attention_weights = model(x)
        
        # Test that attention weights are properly shaped and normalized
        assert attention_weights.shape == (4, 20, 1)
        attention_sums = torch.sum(attention_weights, dim=1)
        assert torch.allclose(attention_sums, torch.ones(4, 1), atol=1e-6)
        
        # Test that different inputs produce different attention patterns
        x2 = torch.randn(4, 20, 8)
        _, attention_weights2 = model(x2)
        
        # Attention weights should be different for different inputs
        assert not torch.allclose(attention_weights, attention_weights2, atol=1e-3)
    
    def test_gru_batch_norm_integration(self):
        config = ModelConfig(
            input_size=6,
            hidden_size=24,
            num_layers=1,
            dropout_rate=0.2,
            learning_rate=0.001,
            batch_size=8,
            num_epochs=5,
            sequence_length=15
        )
        
        model = GRUModel(config)
        
        # Test with training mode (batch norm should work)
        model.train()
        x_train = torch.randn(8, 15, 6)
        output_train = model(x_train)
        assert output_train.shape == (8, 1)
        
        # Test with eval mode (batch norm should use running stats)
        model.eval()
        x_eval = torch.randn(1, 15, 6)  # Different batch size
        output_eval = model(x_eval)
        assert output_eval.shape == (1, 1)
    
    def test_model_parameter_counts(self):
        config = ModelConfig(
            input_size=10,
            hidden_size=64,
            num_layers=2,
            dropout_rate=0.2,
            learning_rate=0.001,
            batch_size=32,
            num_epochs=50,
            sequence_length=30
        )
        
        lstm_model = LSTMWithAttention(config)
        gru_model = GRUModel(config)
        
        lstm_params = sum(p.numel() for p in lstm_model.parameters())
        gru_params = sum(p.numel() for p in gru_model.parameters())
        
        # Both models should have reasonable parameter counts
        assert lstm_params > 1000  # Should have substantial parameters
        assert gru_params > 1000
        
        # LSTM typically has more parameters than GRU due to additional gates
        # But this depends on architecture, so we just check they're both reasonable
        assert lstm_params > 0
        assert gru_params > 0


if __name__ == "__main__":
    pytest.main([__file__])