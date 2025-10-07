"""
MLOps Model Training with Experiment Tracking

This module provides model training with MLflow experiment tracking,
model versioning, and automated logging.

Author: Gabriel Demetrios Lafis
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from typing import Dict, Any
import argparse
from loguru import logger


class MLOpsTrainer:
    """
    Model trainer with MLflow integration for experiment tracking.
    """
    
    def __init__(
        self,
        experiment_name: str = "default-experiment",
        tracking_uri: str = "http://localhost:5000"
    ):
        """
        Initialize MLOps trainer.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
        """
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"Initialized MLOps trainer for experiment: {experiment_name}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_params: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Train model with MLflow tracking.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            model_params: Model hyperparameters
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_params is None:
            model_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(model_params)
            
            # Train model
            logger.info("Training model...")
            model = RandomForestClassifier(**model_params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'test_accuracy': accuracy_score(y_test, y_pred_test),
                'precision': precision_score(y_test, y_pred_test, average='weighted'),
                'recall': recall_score(y_test, y_pred_test, average='weighted'),
                'f1_score': f1_score(y_test, y_pred_test, average='weighted')
            }
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log feature importance
            if hasattr(model, 'feature_importances_'):
                importance_dict = {
                    f'feature_{i}': imp 
                    for i, imp in enumerate(model.feature_importances_)
                }
                mlflow.log_params(importance_dict)
            
            logger.success(f"Model trained successfully. Test accuracy: {metrics['test_accuracy']:.4f}")
            
            return metrics
    
    def register_model(self, run_id: str, model_name: str):
        """
        Register model in MLflow Model Registry.
        
        Args:
            run_id: MLflow run ID
            model_name: Name for registered model
        """
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, model_name)
        logger.info(f"Model registered as: {model_name}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train model with MLflow tracking')
    parser.add_argument('--experiment-name', type=str, default='default-experiment')
    parser.add_argument('--data-path', type=str, default='data/processed/dataset.csv')
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    args = parser.parse_args()
    
    # Load data (example with synthetic data)
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    trainer = MLOpsTrainer(experiment_name=args.experiment_name)
    
    model_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'random_state': 42
    }
    
    metrics = trainer.train(X_train, y_train, X_test, y_test, model_params)
    
    print("\nTraining Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
