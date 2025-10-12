"""
PyTorch Training Service with MLflow Integration

This module implements PyTorch-based neural network models for financial return prediction
with comprehensive MLflow experiment tracking and model management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import mlflow
import mlflow.pytorch
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import pickle
from datetime import datetime
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for PyTorch models"""
    input_size: int
    hidden_size: int
    num_layers: int
    dropout_rate: float
    learning_rate: float
    batch_size: int
    num_epochs: int
    sequence_length: int
    output_size: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingResult:
    """Results from model training"""
    model_path: str
    metrics: Dict[str, float]
    training_history: Dict[str, List[float]]
    model_config: ModelConfig
    experiment_id: str
    run_id: str


@dataclass
class EvaluationMetrics:
    """Financial evaluation metrics"""
    mse: float
    mae: float
    rmse: float
    sharpe_ratio: float
    accuracy: float
    volatility_prediction_error: float
    directional_accuracy: float
    max_drawdown: float


class AttentionMechanism(nn.Module):
    """Attention mechanism for LSTM models"""
    
    def __init__(self, hidden_size: int):
        super(AttentionMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_outputs):
        # lstm_outputs shape: (batch_size, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_outputs), dim=1)
        # attention_weights shape: (batch_size, seq_len, 1)
        
        # Apply attention weights
        attended_output = torch.sum(attention_weights * lstm_outputs, dim=1)
        # attended_output shape: (batch_size, hidden_size)
        
        return attended_output, attention_weights


class LSTMWithAttention(nn.Module):
    """LSTM model with attention mechanism for return prediction"""
    
    def __init__(self, config: ModelConfig):
        super(LSTMWithAttention, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout_rate if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = AttentionMechanism(config.hidden_size)
        
        # Output layers
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.fc2 = nn.Linear(config.hidden_size // 2, config.output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        attended_out, attention_weights = self.attention(lstm_out)
        
        # Final prediction layers
        out = self.dropout(attended_out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, attention_weights


class GRUModel(nn.Module):
    """GRU model with dropout and batch normalization"""
    
    def __init__(self, config: ModelConfig):
        super(GRUModel, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout_rate if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        
        # Output layers with dropout
        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.fc2 = nn.Linear(config.hidden_size // 2, config.hidden_size // 4)
        self.dropout3 = nn.Dropout(config.dropout_rate)
        self.fc3 = nn.Linear(config.hidden_size // 4, config.output_size)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # GRU forward pass
        gru_out, _ = self.gru(x, h0)
        
        # Take the last output
        last_output = gru_out[:, -1, :]
        
        # Apply batch normalization (only if batch size > 1 and in training mode)
        if self.training and batch_size > 1:
            normalized = self.batch_norm(last_output)
        else:
            normalized = last_output
        
        # Forward through fully connected layers with dropout
        out = self.dropout1(normalized)
        out = self.leaky_relu(self.fc1(out))
        out = self.dropout2(out)
        out = self.relu(self.fc2(out))
        out = self.dropout3(out)
        out = self.fc3(out)
        
        return out


class PyTorchTrainingService:
    """PyTorch training service with MLflow integration"""
    
    def __init__(self, mlflow_tracking_uri: str = "file:./mlruns"):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def _prepare_data(self, features: np.ndarray, targets: np.ndarray, 
                     sequence_length: int, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders for training and validation"""
        
        # Create sequences
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(targets[i + sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train/validation split
        split_idx = int(len(X) * train_split)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        return train_loader, val_loader
    
    def _calculate_financial_metrics(self, predictions: np.ndarray, 
                                   actual: np.ndarray, returns: np.ndarray) -> EvaluationMetrics:
        """Calculate financial evaluation metrics"""
        
        # Basic regression metrics
        mse = mean_squared_error(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        pred_direction = np.sign(predictions.flatten())
        actual_direction = np.sign(actual.flatten())
        directional_accuracy = np.mean(pred_direction == actual_direction)
        
        # Accuracy (within 1% threshold)
        accuracy = np.mean(np.abs(predictions.flatten() - actual.flatten()) < 0.01)
        
        # Sharpe ratio calculation
        portfolio_returns = predictions.flatten() * returns[-len(predictions):]
        if np.std(portfolio_returns) > 0:
            sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Volatility prediction error
        pred_vol = np.std(predictions.flatten())
        actual_vol = np.std(actual.flatten())
        volatility_prediction_error = abs(pred_vol - actual_vol)
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return EvaluationMetrics(
            mse=mse,
            mae=mae,
            rmse=rmse,
            sharpe_ratio=sharpe_ratio,
            accuracy=accuracy,
            volatility_prediction_error=volatility_prediction_error,
            directional_accuracy=directional_accuracy,
            max_drawdown=max_drawdown
        )
    
    def _train_model(self, model: nn.Module, train_loader: DataLoader, 
                    val_loader: DataLoader, config: ModelConfig, 
                    experiment_name: str) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """Train a PyTorch model with MLflow tracking"""
        
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                if isinstance(model, LSTMWithAttention):
                    outputs, _ = model(batch_X)
                else:
                    outputs = model(batch_X)
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    if isinstance(model, LSTMWithAttention):
                        outputs, _ = model(batch_X)
                    else:
                        outputs = model(batch_X)
                    
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['learning_rate'].append(current_lr)
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': current_lr
            }, step=epoch)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model checkpoint
                checkpoint_path = f"models/artifacts/best_model_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': best_val_loss,
                    'config': config
                }, checkpoint_path)
                mlflow.log_artifact(checkpoint_path)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{config.num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        return model, history   
 
    async def train_lstm_model(self, config: ModelConfig, features: np.ndarray, 
                              targets: np.ndarray, returns: np.ndarray) -> TrainingResult:
        """Train LSTM model with attention mechanism"""
        
        experiment_name = "lstm_attention_training"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"lstm_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            # Log parameters
            mlflow.log_params({
                'model_type': 'LSTM_Attention',
                'input_size': config.input_size,
                'hidden_size': config.hidden_size,
                'num_layers': config.num_layers,
                'dropout_rate': config.dropout_rate,
                'learning_rate': config.learning_rate,
                'batch_size': config.batch_size,
                'num_epochs': config.num_epochs,
                'sequence_length': config.sequence_length,
                'device': config.device
            })
            
            # Prepare data
            train_loader, val_loader = self._prepare_data(features, targets, config.sequence_length)
            
            # Create and train model
            model = LSTMWithAttention(config)
            trained_model, history = self._train_model(model, train_loader, val_loader, config, "lstm")
            
            # Evaluate model
            trained_model.eval()
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs, _ = trained_model(batch_X)
                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(batch_y.cpu().numpy())
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Calculate financial metrics
            metrics = self._calculate_financial_metrics(predictions, actuals, returns)
            
            # Log evaluation metrics
            mlflow.log_metrics({
                'final_mse': metrics.mse,
                'final_mae': metrics.mae,
                'final_rmse': metrics.rmse,
                'sharpe_ratio': metrics.sharpe_ratio,
                'accuracy': metrics.accuracy,
                'volatility_prediction_error': metrics.volatility_prediction_error,
                'directional_accuracy': metrics.directional_accuracy,
                'max_drawdown': metrics.max_drawdown
            })
            
            # Save model
            model_path = f"models/artifacts/lstm_model_{run.info.run_id}.pth"
            torch.save(trained_model.state_dict(), model_path)
            
            # Log model to MLflow
            mlflow.pytorch.log_model(trained_model, "model")
            mlflow.log_artifact(model_path)
            
            # Save training history
            history_path = f"models/artifacts/lstm_history_{run.info.run_id}.json"
            with open(history_path, 'w') as f:
                json.dump(history, f)
            mlflow.log_artifact(history_path)
            
            logger.info(f"LSTM training completed. Sharpe Ratio: {metrics.sharpe_ratio:.4f}, Accuracy: {metrics.accuracy:.4f}")
            
            return TrainingResult(
                model_path=model_path,
                metrics=metrics.__dict__,
                training_history=history,
                model_config=config,
                experiment_id=run.info.experiment_id,
                run_id=run.info.run_id
            )
    
    async def train_gru_model(self, config: ModelConfig, features: np.ndarray, 
                             targets: np.ndarray, returns: np.ndarray) -> TrainingResult:
        """Train GRU model with dropout and batch normalization"""
        
        experiment_name = "gru_training"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"gru_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            # Log parameters
            mlflow.log_params({
                'model_type': 'GRU',
                'input_size': config.input_size,
                'hidden_size': config.hidden_size,
                'num_layers': config.num_layers,
                'dropout_rate': config.dropout_rate,
                'learning_rate': config.learning_rate,
                'batch_size': config.batch_size,
                'num_epochs': config.num_epochs,
                'sequence_length': config.sequence_length,
                'device': config.device
            })
            
            # Prepare data
            train_loader, val_loader = self._prepare_data(features, targets, config.sequence_length)
            
            # Create and train model
            model = GRUModel(config)
            trained_model, history = self._train_model(model, train_loader, val_loader, config, "gru")
            
            # Evaluate model
            trained_model.eval()
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = trained_model(batch_X)
                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(batch_y.cpu().numpy())
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Calculate financial metrics
            metrics = self._calculate_financial_metrics(predictions, actuals, returns)
            
            # Log evaluation metrics
            mlflow.log_metrics({
                'final_mse': metrics.mse,
                'final_mae': metrics.mae,
                'final_rmse': metrics.rmse,
                'sharpe_ratio': metrics.sharpe_ratio,
                'accuracy': metrics.accuracy,
                'volatility_prediction_error': metrics.volatility_prediction_error,
                'directional_accuracy': metrics.directional_accuracy,
                'max_drawdown': metrics.max_drawdown
            })
            
            # Save model
            model_path = f"models/artifacts/gru_model_{run.info.run_id}.pth"
            torch.save(trained_model.state_dict(), model_path)
            
            # Log model to MLflow
            mlflow.pytorch.log_model(trained_model, "model")
            mlflow.log_artifact(model_path)
            
            # Save training history
            history_path = f"models/artifacts/gru_history_{run.info.run_id}.json"
            with open(history_path, 'w') as f:
                json.dump(history, f)
            mlflow.log_artifact(history_path)
            
            logger.info(f"GRU training completed. Sharpe Ratio: {metrics.sharpe_ratio:.4f}, Accuracy: {metrics.accuracy:.4f}")
            
            return TrainingResult(
                model_path=model_path,
                metrics=metrics.__dict__,
                training_history=history,
                model_config=config,
                experiment_id=run.info.experiment_id,
                run_id=run.info.run_id
            )
    
    async def evaluate_model(self, model: nn.Module, test_features: np.ndarray, 
                           test_targets: np.ndarray, returns: np.ndarray, 
                           config: ModelConfig) -> EvaluationMetrics:
        """Evaluate a trained model on test data"""
        
        model.eval()
        model = model.to(self.device)
        
        # Prepare test data
        X_test, y_test = [], []
        for i in range(len(test_features) - config.sequence_length):
            X_test.append(test_features[i:i + config.sequence_length])
            y_test.append(test_targets[i + config.sequence_length])
        
        X_test = torch.FloatTensor(np.array(X_test)).to(self.device)
        y_test = np.array(y_test)
        
        # Generate predictions
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X_test), config.batch_size):
                batch = X_test[i:i + config.batch_size]
                if isinstance(model, LSTMWithAttention):
                    outputs, _ = model(batch)
                else:
                    outputs = model(batch)
                predictions.extend(outputs.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Calculate and return metrics
        return self._calculate_financial_metrics(predictions, y_test, returns)
    
    def save_model_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                             epoch: int, loss: float, config: ModelConfig, 
                             checkpoint_path: str) -> None:
        """Save model checkpoint for recovery"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': config.__dict__
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_model_checkpoint(self, checkpoint_path: str, model_class) -> Tuple[nn.Module, optim.Optimizer, int, float]:
        """Load model checkpoint for recovery"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Reconstruct config
        config_dict = checkpoint['config']
        config = ModelConfig(**config_dict)
        
        # Create model and optimizer
        model = model_class(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}, epoch: {epoch}, loss: {loss}")
        
        return model, optimizer, epoch, loss
    
    def optimize_hyperparameters(self, features: np.ndarray, targets: np.ndarray, 
                               returns: np.ndarray, model_type: str = "lstm", 
                               n_trials: int = 50) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            # Suggest hyperparameters
            config = ModelConfig(
                input_size=features.shape[1],
                hidden_size=trial.suggest_int('hidden_size', 32, 256),
                num_layers=trial.suggest_int('num_layers', 1, 4),
                dropout_rate=trial.suggest_float('dropout_rate', 0.1, 0.5),
                learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
                num_epochs=50,  # Reduced for optimization
                sequence_length=trial.suggest_int('sequence_length', 10, 60)
            )
            
            try:
                # Prepare data
                train_loader, val_loader = self._prepare_data(features, targets, config.sequence_length)
                
                # Create model
                if model_type == "lstm":
                    model = LSTMWithAttention(config)
                else:
                    model = GRUModel(config)
                
                # Train model (simplified for optimization)
                model = model.to(self.device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
                
                # Quick training
                for epoch in range(config.num_epochs):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        
                        optimizer.zero_grad()
                        if isinstance(model, LSTMWithAttention):
                            outputs, _ = model(batch_X)
                        else:
                            outputs = model(batch_X)
                        
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                
                # Evaluate
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        if isinstance(model, LSTMWithAttention):
                            outputs, _ = model(batch_X)
                        else:
                            outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                return val_loss / len(val_loader)
                
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best validation loss: {study.best_value}")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }


# Utility functions for model management
def create_model_config(input_size: int, **kwargs) -> ModelConfig:
    """Create a model configuration with sensible defaults"""
    
    defaults = {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 100,
        'sequence_length': 30,
        'output_size': 1
    }
    
    defaults.update(kwargs)
    defaults['input_size'] = input_size
    
    return ModelConfig(**defaults)


def compare_models(results: List[TrainingResult]) -> Dict[str, Any]:
    """Compare multiple model training results"""
    
    comparison = {
        'models': [],
        'best_model': None,
        'best_metric': 'sharpe_ratio'
    }
    
    best_score = float('-inf')
    
    for result in results:
        model_info = {
            'run_id': result.run_id,
            'model_path': result.model_path,
            'metrics': result.metrics,
            'config': result.model_config.__dict__
        }
        comparison['models'].append(model_info)
        
        # Find best model based on Sharpe ratio
        if result.metrics['sharpe_ratio'] > best_score:
            best_score = result.metrics['sharpe_ratio']
            comparison['best_model'] = model_info
    
    return comparison