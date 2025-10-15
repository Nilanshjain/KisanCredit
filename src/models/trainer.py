"""LightGBM Model Trainer for Profitability Scoring.

Implements training pipeline with:
- Custom profitability objective function
- MLflow experiment tracking
- Hyperparameter tuning
- Model versioning
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import joblib
import mlflow
import mlflow.lightgbm
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..utils.logger import get_logger
from ..utils.config import settings

logger = get_logger(__name__)


class ProfitabilityModelTrainer:
    """Trains LightGBM model for loan profitability prediction.

    Features:
    - Weighted profitability score prediction (regression)
    - MLflow experiment tracking
    - Automatic feature importance calculation
    - Model versioning and artifact management
    """

    def __init__(
        self,
        experiment_name: str = "kisancredit_profitability",
        model_dir: str = "models"
    ):
        """Initialize trainer.

        Args:
            experiment_name: MLflow experiment name
            model_dir: Directory to save model artifacts
        """
        self.experiment_name = experiment_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[lgb.Booster] = None
        self.feature_names: Optional[list] = None
        self.feature_importance: Optional[Dict[str, float]] = None

        # Set up MLflow with fallback to local file tracking
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(
                "MLflow initialized with server tracking",
                tracking_uri=settings.mlflow_tracking_uri,
                experiment_name=experiment_name
            )
        except Exception as e:
            # Fallback to file-based tracking
            mlflow_dir = Path("mlruns")
            mlflow_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file:///{mlflow_dir.absolute()}")
            mlflow.set_experiment(experiment_name)
            logger.warning(
                "MLflow server not available, using file-based tracking",
                tracking_uri=f"file:///{mlflow_dir.absolute()}",
                error=str(e)
            )

        logger.info(
            "Model trainer initialized",
            experiment_name=experiment_name,
            model_dir=str(self.model_dir)
        )

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'profitability_score',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training and test datasets.

        Args:
            df: Feature dataframe
            target_col: Target column name
            test_size: Test set proportion
            random_state: Random seed

        Returns:
            X_train, X_test, y_train, y_test
        """
        # Remove metadata columns
        exclude_cols = ['application_id', 'user_id', target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]
        y = df[target_col]

        # Handle missing values - convert categorical columns to avoid fillna errors
        for col in X.columns:
            if X[col].dtype.name == 'category':
                X[col] = X[col].astype(str)

        # Now fill missing values
        X = X.fillna(0)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.feature_names = feature_cols

        logger.info(
            "Data prepared for training",
            n_features=len(feature_cols),
            n_train=len(X_train),
            n_test=len(X_test),
            target_mean=round(y.mean(), 4)
        )

        return X_train, X_test, y_train, y_test

    def get_default_params(self) -> Dict[str, Any]:
        """Get default LightGBM parameters optimized for profitability scoring.

        Returns:
            Dictionary of hyperparameters
        """
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        params: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: int = 50,
        log_to_mlflow: bool = True
    ) -> lgb.Booster:
        """Train LightGBM model with MLflow tracking.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            params: Model hyperparameters (uses defaults if None)
            early_stopping_rounds: Early stopping patience
            log_to_mlflow: Whether to log to MLflow

        Returns:
            Trained LightGBM booster
        """
        if params is None:
            params = self.get_default_params()

        # Start MLflow run
        if log_to_mlflow:
            mlflow.start_run()
            mlflow.log_params(params)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_features", X_train.shape[1])

        try:
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_sets = [train_data]
            valid_names = ['train']

            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                valid_sets.append(val_data)
                valid_names.append('val')
                if log_to_mlflow:
                    mlflow.log_param("n_val", len(X_val))

            # Train model
            logger.info("Starting model training", params=params)

            callbacks = []
            if log_to_mlflow:
                callbacks.append(lgb.log_evaluation(period=50))

            self.model = lgb.train(
                params,
                train_data,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )

            # Calculate feature importance
            self.feature_importance = dict(zip(
                self.feature_names or X_train.columns,
                self.model.feature_importance(importance_type='gain')
            ))

            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

            # Log feature importance
            if log_to_mlflow:
                for feat, imp in list(self.feature_importance.items())[:20]:
                    mlflow.log_metric(f"feature_importance_{feat}", imp)

            logger.info(
                "Model training completed",
                best_iteration=self.model.best_iteration,
                num_trees=self.model.num_trees()
            )

            return self.model

        except Exception as e:
            logger.error(f"Training failed: {e}")
            if log_to_mlflow:
                mlflow.end_run(status='FAILED')
            raise

        finally:
            if log_to_mlflow:
                mlflow.end_run()

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        log_to_mlflow: bool = True
    ) -> Dict[str, float]:
        """Evaluate model performance on test set.

        Args:
            X_test: Test features
            y_test: Test target
            log_to_mlflow: Whether to log metrics to MLflow

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        }

        # Calculate profitability-specific metrics
        # Assuming profitability_score > 0.6 is "profitable"
        y_test_binary = (y_test > 0.6).astype(int)
        y_pred_binary = (y_pred > 0.6).astype(int)

        tp = np.sum((y_pred_binary == 1) & (y_test_binary == 1))
        fp = np.sum((y_pred_binary == 1) & (y_test_binary == 0))
        fn = np.sum((y_pred_binary == 0) & (y_test_binary == 1))
        tn = np.sum((y_pred_binary == 0) & (y_test_binary == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': (tp + tn) / len(y_test)
        })

        # Log to MLflow
        if log_to_mlflow:
            with mlflow.start_run():
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

        logger.info("Model evaluation completed", **{k: round(v, 4) for k, v in metrics.items()})

        return metrics

    def save_model(
        self,
        model_name: str = "profitability_model",
        metadata: Optional[Dict] = None
    ) -> str:
        """Save model to disk with metadata.

        Args:
            model_name: Name for saved model file
            metadata: Additional metadata to save

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = self.model_dir / model_filename

        # Prepare model artifact
        artifact = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'metadata': metadata or {},
            'timestamp': timestamp,
            'lightgbm_version': lgb.__version__
        }

        # Save to disk
        joblib.dump(artifact, model_path)

        logger.info(
            "Model saved",
            model_path=str(model_path),
            size_mb=round(model_path.stat().st_size / 1024 / 1024, 2)
        )

        # Also save latest version
        latest_path = self.model_dir / f"{model_name}_latest.pkl"
        joblib.dump(artifact, latest_path)

        return str(model_path)

    def load_model(self, model_path: str) -> None:
        """Load saved model from disk.

        Args:
            model_path: Path to saved model file
        """
        artifact = joblib.load(model_path)

        self.model = artifact['model']
        self.feature_names = artifact['feature_names']
        self.feature_importance = artifact['feature_importance']

        logger.info("Model loaded", model_path=model_path)

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top N most important features.

        Args:
            top_n: Number of top features to return

        Returns:
            Dictionary of feature names and importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained. Call train() first.")

        return dict(list(self.feature_importance.items())[:top_n])
