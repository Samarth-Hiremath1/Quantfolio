"""
Example usage of PyTorch Training Service

This script demonstrates how to use the PyTorch training service with MLflow integration
for training LSTM and GRU models on financial data.
"""

import sys
import os
import numpy as np
import pandas as pd
import asyncio
from pathlib import Path

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_pipeline', 'features'))

from pytorch_service import (
    PyTorchTrainingService, ModelConfig, create_model_config, compare_models
)
from hyperparameter_optimizer import HyperparameterOptimizer, ModelEnsemble

# Import feature engineering if available
try:
    from service import FeatureEngineeringService
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    print("Feature engineering service not available, using synthetic data")


def generate_synthetic_financial_data(n_samples: int = 2000, n_assets: int = 5) -> tuple:
    """Generate synthetic financial data for demonstration"""
    
    np.random.seed(42)
    
    # Generate price data
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Simulate correlated asset returns
    correlation_matrix = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Generate returns using multivariate normal distribution
    mean_returns = np.random.uniform(-0.001, 0.002, n_assets)
    returns = np.random.multivariate_normal(mean_returns, correlation_matrix * 0.0004, n_samples)
    
    # Convert to prices
    prices = np.zeros((n_samples, n_assets))
    prices[0] = 100  # Starting price
    
    for i in range(1, n_samples):
        prices[i] = prices[i-1] * (1 + returns[i])
    
    # Create DataFrame
    asset_names = [f'ASSET_{i}' for i in range(n_assets)]
    price_df = pd.DataFrame(prices, index=dates, columns=asset_names)
    returns_df = pd.DataFrame(returns, index=dates, columns=asset_names)
    
    return price_df, returns_df


def create_features_from_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    """Create technical features from price data"""
    
    features_df = pd.DataFrame(index=price_df.index)
    
    for asset in price_df.columns:
        prices = price_df[asset]
        
        # Returns
        features_df[f'{asset}_return_1d'] = prices.pct_change()
        features_df[f'{asset}_return_5d'] = prices.pct_change(5)
        features_df[f'{asset}_return_20d'] = prices.pct_change(20)
        
        # Moving averages
        features_df[f'{asset}_sma_5'] = prices.rolling(5).mean() / prices
        features_df[f'{asset}_sma_20'] = prices.rolling(20).mean() / prices
        features_df[f'{asset}_sma_50'] = prices.rolling(50).mean() / prices
        
        # Volatility
        features_df[f'{asset}_vol_5'] = prices.pct_change().rolling(5).std()
        features_df[f'{asset}_vol_20'] = prices.pct_change().rolling(20).std()
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features_df[f'{asset}_rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        features_df[f'{asset}_macd'] = (ema_12 - ema_26) / prices
        
    # Drop NaN values
    features_df = features_df.dropna()
    
    return features_df


async def train_single_models_example():
    """Example of training individual LSTM and GRU models"""
    
    print("=== Single Model Training Example ===")
    
    # Generate or load data
    print("Generating synthetic financial data...")
    price_df, returns_df = generate_synthetic_financial_data(n_samples=1500, n_assets=3)
    
    # Create features
    print("Creating technical features...")
    features_df = create_features_from_prices(price_df)
    
    # Prepare data for ML
    features = features_df.values
    # Predict next day return for first asset - align indices first
    aligned_returns = returns_df.loc[features_df.index]
    target_returns = aligned_returns.iloc[:, 0].shift(-1).dropna()
    
    # Align features and targets
    min_length = min(len(features), len(target_returns))
    features = features[:min_length]
    targets = target_returns.values[:min_length]
    returns = aligned_returns.iloc[:min_length, 0].values
    
    print(f"Data shape: Features {features.shape}, Targets {targets.shape}")
    
    # Initialize training service
    training_service = PyTorchTrainingService(mlflow_tracking_uri="file:./mlruns")
    
    # Create model configurations
    lstm_config = create_model_config(
        input_size=features.shape[1],
        hidden_size=128,
        num_layers=2,
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=50,
        sequence_length=30
    )
    
    gru_config = create_model_config(
        input_size=features.shape[1],
        hidden_size=128,
        num_layers=2,
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=50,
        sequence_length=30
    )
    
    # Train LSTM model
    print("Training LSTM model...")
    lstm_result = await training_service.train_lstm_model(lstm_config, features, targets, returns)
    print(f"LSTM Results - Sharpe Ratio: {lstm_result.metrics['sharpe_ratio']:.4f}, "
          f"Accuracy: {lstm_result.metrics['accuracy']:.4f}")
    
    # Train GRU model
    print("Training GRU model...")
    gru_result = await training_service.train_gru_model(gru_config, features, targets, returns)
    print(f"GRU Results - Sharpe Ratio: {gru_result.metrics['sharpe_ratio']:.4f}, "
          f"Accuracy: {gru_result.metrics['accuracy']:.4f}")
    
    # Compare models
    comparison = compare_models([lstm_result, gru_result])
    print(f"Best model: {comparison['best_model']['run_id']} "
          f"(Sharpe Ratio: {comparison['best_model']['metrics']['sharpe_ratio']:.4f})")
    
    return lstm_result, gru_result


async def hyperparameter_optimization_example():
    """Example of hyperparameter optimization"""
    
    print("\n=== Hyperparameter Optimization Example ===")
    
    # Generate data
    print("Generating data for optimization...")
    price_df, returns_df = generate_synthetic_financial_data(n_samples=1000, n_assets=2)
    features_df = create_features_from_prices(price_df)
    
    features = features_df.values
    aligned_returns = returns_df.loc[features_df.index]
    target_returns = aligned_returns.iloc[:, 0].shift(-1).dropna()
    
    min_length = min(len(features), len(target_returns))
    features = features[:min_length]
    targets = target_returns.values[:min_length]
    returns = aligned_returns.iloc[:min_length, 0].values
    
    # Initialize services
    training_service = PyTorchTrainingService(mlflow_tracking_uri="file:./mlruns")
    optimizer = HyperparameterOptimizer(training_service)
    
    # Optimize LSTM hyperparameters (reduced trials for demo)
    print("Optimizing LSTM hyperparameters...")
    lstm_optimization = optimizer.optimize_lstm_hyperparameters(
        features, targets, returns, n_trials=10, timeout=300  # 5 minutes max
    )
    
    print(f"Best LSTM params: {lstm_optimization['best_params']}")
    print(f"Best LSTM score: {lstm_optimization['best_value']:.4f}")
    
    # Optimize GRU hyperparameters
    print("Optimizing GRU hyperparameters...")
    gru_optimization = optimizer.optimize_gru_hyperparameters(
        features, targets, returns, n_trials=10, timeout=300
    )
    
    print(f"Best GRU params: {gru_optimization['best_params']}")
    print(f"Best GRU score: {gru_optimization['best_value']:.4f}")
    
    # Compare optimized models
    print("Comparing optimized models...")
    comparison = optimizer.compare_optimized_models(
        features, targets, returns,
        lstm_optimization['best_params'],
        gru_optimization['best_params']
    )
    
    print(f"Winner: {comparison['winner']}")
    print(f"Performance difference: {comparison['comparison_metrics']}")
    
    return lstm_optimization, gru_optimization, comparison


def model_evaluation_example():
    """Example of model evaluation and checkpointing"""
    
    print("\n=== Model Evaluation and Checkpointing Example ===")
    
    # Generate test data
    price_df, returns_df = generate_synthetic_financial_data(n_samples=500, n_assets=2)
    features_df = create_features_from_prices(price_df)
    
    features = features_df.values
    target_returns = returns_df.iloc[features_df.index].iloc[:, 0].shift(-1).dropna()
    
    min_length = min(len(features), len(target_returns))
    test_features = features[:min_length]
    test_targets = target_returns.values[:min_length]
    test_returns = returns_df.iloc[:min_length, 0].values
    
    # Initialize service
    training_service = PyTorchTrainingService()
    
    # Create and train a simple model
    config = create_model_config(
        input_size=test_features.shape[1],
        hidden_size=64,
        num_layers=1,
        num_epochs=10,
        sequence_length=20
    )
    
    from pytorch_service import LSTMWithAttention
    import torch
    
    model = LSTMWithAttention(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Save checkpoint
    checkpoint_path = "models/artifacts/example_checkpoint.pth"
    training_service.save_model_checkpoint(
        model, optimizer, epoch=5, loss=0.1, config=config, 
        checkpoint_path=checkpoint_path
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Load checkpoint
    loaded_model, loaded_optimizer, epoch, loss = training_service.load_model_checkpoint(
        checkpoint_path, LSTMWithAttention
    )
    print(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}")
    
    # Evaluate model
    async def evaluate():
        metrics = await training_service.evaluate_model(
            loaded_model, test_features, test_targets, test_returns, config
        )
        return metrics
    
    metrics = asyncio.run(evaluate())
    print(f"Evaluation metrics:")
    print(f"  MSE: {metrics.mse:.6f}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
    print(f"  Accuracy: {metrics.accuracy:.4f}")
    print(f"  Directional Accuracy: {metrics.directional_accuracy:.4f}")
    
    # Clean up
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


async def ensemble_example():
    """Example of model ensemble"""
    
    print("\n=== Model Ensemble Example ===")
    
    # This is a simplified example - in practice, you'd use the optimization results
    from pytorch_service import LSTMWithAttention, GRUModel
    import torch
    
    # Create dummy models for demonstration
    config1 = create_model_config(input_size=10, hidden_size=32, sequence_length=20)
    config2 = create_model_config(input_size=10, hidden_size=64, sequence_length=20)
    
    model1 = LSTMWithAttention(config1)
    model2 = GRUModel(config2)
    
    # Create ensemble (in practice, these would be trained models with real weights)
    ensemble = ModelEnsemble([
        (model1, config1, 0.6),  # 60% weight
        (model2, config2, 0.4)   # 40% weight
    ])
    
    # Generate test data
    test_features = np.random.randn(100, 10)
    test_targets = np.random.randn(80)  # 100 - 20 (sequence_length)
    test_returns = np.random.randn(80) * 0.02
    
    # Make ensemble predictions
    predictions = ensemble.predict(test_features, sequence_length=20)
    print(f"Ensemble predictions shape: {predictions.shape}")
    
    # Evaluate ensemble
    training_service = PyTorchTrainingService()
    ensemble_metrics = ensemble.evaluate_ensemble(
        test_features, test_targets, test_returns, 
        sequence_length=20, training_service=training_service
    )
    
    print(f"Ensemble Sharpe Ratio: {ensemble_metrics.sharpe_ratio:.4f}")
    print(f"Ensemble Accuracy: {ensemble_metrics.accuracy:.4f}")


def main():
    """Main function to run all examples"""
    
    print("PyTorch Training Service Examples")
    print("=" * 50)
    
    # Ensure directories exist
    os.makedirs("models/artifacts", exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    
    try:
        # Run single model training example
        lstm_result, gru_result = asyncio.run(train_single_models_example())
        
        # Run hyperparameter optimization example (commented out for speed)
        # lstm_opt, gru_opt, comparison = asyncio.run(hyperparameter_optimization_example())
        
        # Run evaluation example
        model_evaluation_example()
        
        # Run ensemble example
        asyncio.run(ensemble_example())
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()