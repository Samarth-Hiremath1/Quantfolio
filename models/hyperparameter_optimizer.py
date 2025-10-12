"""
Hyperparameter Optimization for PyTorch Models

This module provides utilities for optimizing hyperparameters using Optuna
and comparing different model configurations.
"""

import optuna
import mlflow
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_service import (
    PyTorchTrainingService, LSTMWithAttention, GRUModel, 
    ModelConfig, TrainingResult, EvaluationMetrics
)

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna with MLflow integration"""
    
    def __init__(self, training_service: PyTorchTrainingService, 
                 mlflow_experiment_name: str = "hyperparameter_optimization"):
        self.training_service = training_service
        self.mlflow_experiment_name = mlflow_experiment_name
        mlflow.set_experiment(mlflow_experiment_name)
        
    def optimize_lstm_hyperparameters(self, features: np.ndarray, targets: np.ndarray, 
                                    returns: np.ndarray, n_trials: int = 100,
                                    timeout: Optional[int] = None) -> Dict[str, Any]:
        """Optimize LSTM hyperparameters using Optuna"""
        
        def objective(trial):
            # Suggest hyperparameters
            config = ModelConfig(
                input_size=features.shape[1],
                hidden_size=trial.suggest_int('hidden_size', 32, 512, step=32),
                num_layers=trial.suggest_int('num_layers', 1, 4),
                dropout_rate=trial.suggest_float('dropout_rate', 0.0, 0.5),
                learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                batch_size=trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                num_epochs=trial.suggest_int('num_epochs', 20, 100),
                sequence_length=trial.suggest_int('sequence_length', 10, 60)
            )
            
            try:
                with mlflow.start_run(nested=True):
                    # Log trial parameters
                    mlflow.log_params({
                        'trial_number': trial.number,
                        'model_type': 'LSTM_Optimization',
                        **config.__dict__
                    })
                    
                    # Quick training for optimization
                    train_loader, val_loader = self.training_service._prepare_data(
                        features, targets, config.sequence_length, train_split=0.8
                    )
                    
                    model = LSTMWithAttention(config).to(self.training_service.device)
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
                    
                    best_val_loss = float('inf')
                    patience_counter = 0
                    patience = 10
                    
                    for epoch in range(config.num_epochs):
                        # Training
                        model.train()
                        train_loss = 0.0
                        for batch_X, batch_y in train_loader:
                            batch_X = batch_X.to(self.training_service.device)
                            batch_y = batch_y.to(self.training_service.device)
                            
                            optimizer.zero_grad()
                            outputs, _ = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            train_loss += loss.item()
                        
                        # Validation
                        model.eval()
                        val_loss = 0.0
                        with torch.no_grad():
                            for batch_X, batch_y in val_loader:
                                batch_X = batch_X.to(self.training_service.device)
                                batch_y = batch_y.to(self.training_service.device)
                                outputs, _ = model(batch_X)
                                loss = criterion(outputs, batch_y)
                                val_loss += loss.item()
                        
                        avg_val_loss = val_loss / len(val_loader)
                        scheduler.step(avg_val_loss)
                        
                        # Early stopping
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            
                        if patience_counter >= patience:
                            break
                        
                        # Report intermediate value for pruning
                        trial.report(avg_val_loss, epoch)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                    
                    # Calculate financial metrics for final evaluation
                    model.eval()
                    predictions, actuals = [], []
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X = batch_X.to(self.training_service.device)
                            batch_y = batch_y.to(self.training_service.device)
                            outputs, _ = model(batch_X)
                            predictions.extend(outputs.cpu().numpy())
                            actuals.extend(batch_y.cpu().numpy())
                    
                    predictions = np.array(predictions)
                    actuals = np.array(actuals)
                    
                    metrics = self.training_service._calculate_financial_metrics(
                        predictions, actuals, returns[-len(predictions):]
                    )
                    
                    # Log final metrics
                    mlflow.log_metrics({
                        'final_val_loss': best_val_loss,
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'accuracy': metrics.accuracy,
                        'directional_accuracy': metrics.directional_accuracy
                    })
                    
                    # Optimize for Sharpe ratio (higher is better, so return negative)
                    return -metrics.sharpe_ratio if metrics.sharpe_ratio > -10 else 10
                    
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                return float('inf')
        
        # Create study with pruning
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Log best results
        with mlflow.start_run():
            mlflow.log_params({
                'optimization_type': 'LSTM_best_params',
                **study.best_params
            })
            mlflow.log_metrics({
                'best_objective_value': study.best_value,
                'n_trials': len(study.trials)
            })
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study,
            'n_trials': len(study.trials)
        }
    
    def optimize_gru_hyperparameters(self, features: np.ndarray, targets: np.ndarray, 
                                   returns: np.ndarray, n_trials: int = 100,
                                   timeout: Optional[int] = None) -> Dict[str, Any]:
        """Optimize GRU hyperparameters using Optuna"""
        
        def objective(trial):
            config = ModelConfig(
                input_size=features.shape[1],
                hidden_size=trial.suggest_int('hidden_size', 32, 512, step=32),
                num_layers=trial.suggest_int('num_layers', 1, 4),
                dropout_rate=trial.suggest_float('dropout_rate', 0.0, 0.5),
                learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                batch_size=trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                num_epochs=trial.suggest_int('num_epochs', 20, 100),
                sequence_length=trial.suggest_int('sequence_length', 10, 60)
            )
            
            try:
                with mlflow.start_run(nested=True):
                    mlflow.log_params({
                        'trial_number': trial.number,
                        'model_type': 'GRU_Optimization',
                        **config.__dict__
                    })
                    
                    train_loader, val_loader = self.training_service._prepare_data(
                        features, targets, config.sequence_length, train_split=0.8
                    )
                    
                    model = GRUModel(config).to(self.training_service.device)
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
                    
                    best_val_loss = float('inf')
                    patience_counter = 0
                    patience = 10
                    
                    for epoch in range(config.num_epochs):
                        # Training
                        model.train()
                        for batch_X, batch_y in train_loader:
                            batch_X = batch_X.to(self.training_service.device)
                            batch_y = batch_y.to(self.training_service.device)
                            
                            optimizer.zero_grad()
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                        
                        # Validation
                        model.eval()
                        val_loss = 0.0
                        with torch.no_grad():
                            for batch_X, batch_y in val_loader:
                                batch_X = batch_X.to(self.training_service.device)
                                batch_y = batch_y.to(self.training_service.device)
                                outputs = model(batch_X)
                                loss = criterion(outputs, batch_y)
                                val_loss += loss.item()
                        
                        avg_val_loss = val_loss / len(val_loader)
                        scheduler.step(avg_val_loss)
                        
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            
                        if patience_counter >= patience:
                            break
                        
                        trial.report(avg_val_loss, epoch)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                    
                    # Calculate financial metrics
                    model.eval()
                    predictions, actuals = [], []
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X = batch_X.to(self.training_service.device)
                            batch_y = batch_y.to(self.training_service.device)
                            outputs = model(batch_X)
                            predictions.extend(outputs.cpu().numpy())
                            actuals.extend(batch_y.cpu().numpy())
                    
                    predictions = np.array(predictions)
                    actuals = np.array(actuals)
                    
                    metrics = self.training_service._calculate_financial_metrics(
                        predictions, actuals, returns[-len(predictions):]
                    )
                    
                    mlflow.log_metrics({
                        'final_val_loss': best_val_loss,
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'accuracy': metrics.accuracy,
                        'directional_accuracy': metrics.directional_accuracy
                    })
                    
                    return -metrics.sharpe_ratio if metrics.sharpe_ratio > -10 else 10
                    
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                return float('inf')
        
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        with mlflow.start_run():
            mlflow.log_params({
                'optimization_type': 'GRU_best_params',
                **study.best_params
            })
            mlflow.log_metrics({
                'best_objective_value': study.best_value,
                'n_trials': len(study.trials)
            })
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study,
            'n_trials': len(study.trials)
        }
    
    def compare_optimized_models(self, features: np.ndarray, targets: np.ndarray, 
                               returns: np.ndarray, lstm_params: Dict, gru_params: Dict) -> Dict[str, Any]:
        """Compare optimized LSTM and GRU models"""
        
        results = {}
        
        # Train LSTM with optimized parameters
        lstm_config = ModelConfig(
            input_size=features.shape[1],
            **lstm_params
        )
        
        import asyncio
        
        async def train_models():
            lstm_result = await self.training_service.train_lstm_model(
                lstm_config, features, targets, returns
            )
            
            gru_config = ModelConfig(
                input_size=features.shape[1],
                **gru_params
            )
            
            gru_result = await self.training_service.train_gru_model(
                gru_config, features, targets, returns
            )
            
            return lstm_result, gru_result
        
        lstm_result, gru_result = asyncio.run(train_models())
        
        # Compare results
        comparison = {
            'lstm_result': {
                'metrics': lstm_result.metrics,
                'config': lstm_result.model_config.__dict__,
                'run_id': lstm_result.run_id
            },
            'gru_result': {
                'metrics': gru_result.metrics,
                'config': gru_result.model_config.__dict__,
                'run_id': gru_result.run_id
            },
            'winner': None,
            'comparison_metrics': {}
        }
        
        # Determine winner based on Sharpe ratio
        lstm_sharpe = lstm_result.metrics['sharpe_ratio']
        gru_sharpe = gru_result.metrics['sharpe_ratio']
        
        if lstm_sharpe > gru_sharpe:
            comparison['winner'] = 'LSTM'
        elif gru_sharpe > lstm_sharpe:
            comparison['winner'] = 'GRU'
        else:
            comparison['winner'] = 'Tie'
        
        # Calculate comparison metrics
        comparison['comparison_metrics'] = {
            'sharpe_ratio_diff': lstm_sharpe - gru_sharpe,
            'accuracy_diff': lstm_result.metrics['accuracy'] - gru_result.metrics['accuracy'],
            'mse_diff': lstm_result.metrics['mse'] - gru_result.metrics['mse'],
            'directional_accuracy_diff': (
                lstm_result.metrics['directional_accuracy'] - 
                gru_result.metrics['directional_accuracy']
            )
        }
        
        # Log comparison to MLflow
        with mlflow.start_run():
            mlflow.log_params({
                'comparison_type': 'LSTM_vs_GRU_optimized',
                'winner': comparison['winner']
            })
            mlflow.log_metrics(comparison['comparison_metrics'])
        
        return comparison


class ModelEnsemble:
    """Ensemble of optimized PyTorch models"""
    
    def __init__(self, models: List[Tuple[nn.Module, ModelConfig, float]]):
        """
        Initialize ensemble with list of (model, config, weight) tuples
        """
        self.models = models
        self.total_weight = sum(weight for _, _, weight in models)
        
    def predict(self, features: np.ndarray, sequence_length: int, device: str = "cpu") -> np.ndarray:
        """Make ensemble predictions"""
        
        # Prepare data
        X = []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
        X = torch.FloatTensor(np.array(X)).to(device)
        
        ensemble_predictions = []
        
        for model, config, weight in self.models:
            model.eval()
            model = model.to(device)
            
            predictions = []
            with torch.no_grad():
                for i in range(0, len(X), config.batch_size):
                    batch = X[i:i + config.batch_size]
                    if isinstance(model, LSTMWithAttention):
                        outputs, _ = model(batch)
                    else:
                        outputs = model(batch)
                    predictions.extend(outputs.cpu().numpy())
            
            weighted_predictions = np.array(predictions) * (weight / self.total_weight)
            ensemble_predictions.append(weighted_predictions)
        
        # Combine predictions
        final_predictions = np.sum(ensemble_predictions, axis=0)
        return final_predictions
    
    def evaluate_ensemble(self, features: np.ndarray, targets: np.ndarray, 
                         returns: np.ndarray, sequence_length: int, 
                         training_service: PyTorchTrainingService) -> EvaluationMetrics:
        """Evaluate ensemble performance"""
        
        predictions = self.predict(features, sequence_length, training_service.device)
        actual = targets[sequence_length:]
        
        return training_service._calculate_financial_metrics(
            predictions, actual, returns[sequence_length:]
        )


def create_ensemble_from_optimization_results(
    optimization_results: List[Dict[str, Any]], 
    features: np.ndarray, targets: np.ndarray, returns: np.ndarray,
    training_service: PyTorchTrainingService,
    top_k: int = 3
) -> ModelEnsemble:
    """Create ensemble from top optimization results"""
    
    # Sort results by performance (assuming negative Sharpe ratio was optimized)
    sorted_results = sorted(optimization_results, key=lambda x: x['best_value'])
    top_results = sorted_results[:top_k]
    
    models = []
    
    for i, result in enumerate(top_results):
        # Create config from best params
        config = ModelConfig(
            input_size=features.shape[1],
            **result['best_params']
        )
        
        # Train model with best params
        import asyncio
        
        async def train_model():
            if 'lstm' in str(result).lower():
                return await training_service.train_lstm_model(config, features, targets, returns)
            else:
                return await training_service.train_gru_model(config, features, targets, returns)
        
        training_result = asyncio.run(train_model())
        
        # Load trained model
        if 'lstm' in str(result).lower():
            model = LSTMWithAttention(config)
        else:
            model = GRUModel(config)
        
        model.load_state_dict(torch.load(training_result.model_path, map_location='cpu'))
        
        # Weight based on performance (higher Sharpe ratio = higher weight)
        weight = max(0.1, training_result.metrics['sharpe_ratio'])
        
        models.append((model, config, weight))
    
    return ModelEnsemble(models)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 2000
    n_features = 15
    
    features = np.random.randn(n_samples, n_features)
    targets = np.sum(features[:, :5], axis=1) * 0.1 + np.random.randn(n_samples) * 0.05
    returns = np.random.randn(n_samples) * 0.02 + 0.001
    
    # Initialize services
    training_service = PyTorchTrainingService()
    optimizer = HyperparameterOptimizer(training_service)
    
    # Optimize hyperparameters
    print("Optimizing LSTM hyperparameters...")
    lstm_results = optimizer.optimize_lstm_hyperparameters(
        features, targets, returns, n_trials=20
    )
    
    print("Optimizing GRU hyperparameters...")
    gru_results = optimizer.optimize_gru_hyperparameters(
        features, targets, returns, n_trials=20
    )
    
    # Compare optimized models
    print("Comparing optimized models...")
    comparison = optimizer.compare_optimized_models(
        features, targets, returns, 
        lstm_results['best_params'], 
        gru_results['best_params']
    )
    
    print(f"Winner: {comparison['winner']}")
    print(f"Sharpe ratio difference: {comparison['comparison_metrics']['sharpe_ratio_diff']:.4f}")