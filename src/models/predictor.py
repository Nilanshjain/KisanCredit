"""Profitability Score Prediction Module.

Handles model loading and inference for production use.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path
import joblib
import time

from ..utils.logger import get_logger
from ..utils.config import settings

logger = get_logger(__name__)


class ProfitabilityPredictor:
    """Production predictor for loan profitability scoring.

    Features:
    - Fast inference (<10ms per prediction)
    - Batch prediction support
    - Feature validation
    - Confidence scores
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize predictor.

        Args:
            model_path: Path to saved model artifact. If None, uses default path.
        """
        self.model: Optional[lgb.Booster] = None
        self.feature_names: Optional[List[str]] = None
        self.feature_importance: Optional[Dict[str, float]] = None
        self.metadata: Optional[Dict] = None

        if model_path:
            self.load_model(model_path)
        else:
            # Try to load latest model
            default_path = Path("models") / "profitability_model_latest.pkl"
            if default_path.exists():
                self.load_model(str(default_path))
            else:
                logger.warning("No model loaded. Call load_model() before prediction.")

    def load_model(self, model_path: str) -> None:
        """Load trained model from disk.

        Args:
            model_path: Path to saved model artifact
        """
        start_time = time.time()

        artifact = joblib.load(model_path)

        self.model = artifact['model']
        self.feature_names = artifact['feature_names']
        self.feature_importance = artifact.get('feature_importance')
        self.metadata = artifact.get('metadata', {})

        load_time = time.time() - start_time

        logger.info(
            "Model loaded successfully",
            model_path=model_path,
            n_features=len(self.feature_names),
            load_time_ms=round(load_time * 1000, 2)
        )

    def _validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Validate and align features with model expectations.

        Args:
            features: Input features

        Returns:
            Validated and aligned features
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(features.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}. Filling with zeros.")
            for feat in missing_features:
                features[feat] = 0

        # Select and order features correctly
        features = features[self.feature_names]

        # Handle missing values
        features = features.fillna(0)

        return features

    def predict(
        self,
        features: Union[Dict, pd.DataFrame],
        return_confidence: bool = False
    ) -> Union[float, Dict[str, float]]:
        """Predict profitability score for a single application.

        Args:
            features: Feature dictionary or DataFrame (single row)
            return_confidence: Whether to return confidence metrics

        Returns:
            Profitability score (0-1), or dict with score and confidence
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        # Convert dict to DataFrame if needed
        if isinstance(features, dict):
            features = pd.DataFrame([features])

        # Validate features
        features = self._validate_features(features)

        # Make prediction
        score = self.model.predict(features)[0]

        # Clip to valid range [0, 1]
        score = np.clip(score, 0.0, 1.0)

        prediction_time = time.time() - start_time

        logger.debug(
            "Prediction completed",
            score=round(score, 4),
            prediction_time_ms=round(prediction_time * 1000, 2)
        )

        if return_confidence:
            # Calculate confidence based on feature quality
            confidence = self._calculate_confidence(features)
            return {
                'score': float(score),
                'confidence': float(confidence),
                'decision': 'approve' if score > 0.6 else 'reject',
                'prediction_time_ms': round(prediction_time * 1000, 2)
            }

        return float(score)

    def predict_batch(
        self,
        features: pd.DataFrame,
        return_confidence: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Predict profitability scores for multiple applications.

        Args:
            features: Feature DataFrame (multiple rows)
            return_confidence: Whether to return confidence metrics

        Returns:
            Array of scores, or DataFrame with scores and confidence
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        # Validate features
        features = self._validate_features(features)

        # Make predictions
        scores = self.model.predict(features)

        # Clip to valid range [0, 1]
        scores = np.clip(scores, 0.0, 1.0)

        prediction_time = time.time() - start_time
        throughput = len(scores) / prediction_time if prediction_time > 0 else 0

        logger.info(
            "Batch prediction completed",
            n_predictions=len(scores),
            avg_score=round(scores.mean(), 4),
            prediction_time_sec=round(prediction_time, 2),
            throughput_per_sec=round(throughput, 2)
        )

        if return_confidence:
            # Calculate confidence for each prediction
            confidences = features.apply(
                lambda row: self._calculate_confidence(pd.DataFrame([row])),
                axis=1
            )

            results = pd.DataFrame({
                'score': scores,
                'confidence': confidences,
                'decision': ['approve' if s > 0.6 else 'reject' for s in scores]
            })

            return results

        return scores

    def _calculate_confidence(self, features: pd.DataFrame) -> float:
        """Calculate prediction confidence based on feature quality.

        Args:
            features: Single row DataFrame

        Returns:
            Confidence score (0-1)
        """
        # Simple confidence metric based on feature completeness and values
        # This is a heuristic; more sophisticated approaches could use uncertainty estimation

        # Count non-zero features (indicator of data richness)
        non_zero_ratio = (features.iloc[0] != 0).sum() / len(features.columns)

        # Check for extreme values (outliers reduce confidence)
        normalized = (features - features.mean()) / (features.std() + 1e-10)
        has_extreme = (np.abs(normalized) > 3).any().any()

        confidence = non_zero_ratio * 0.7
        if has_extreme:
            confidence *= 0.8

        return np.clip(confidence, 0.3, 0.95)  # Bounded confidence

    def explain_prediction(
        self,
        features: Union[Dict, pd.DataFrame],
        top_n: int = 10
    ) -> Dict[str, any]:
        """Explain prediction using feature contributions.

        Args:
            features: Feature dictionary or DataFrame (single row)
            top_n: Number of top contributing features to return

        Returns:
            Dictionary with prediction and top feature contributions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Convert dict to DataFrame if needed
        if isinstance(features, dict):
            features = pd.DataFrame([features])

        # Validate features
        features = self._validate_features(features)

        # Make prediction
        score = self.predict(features, return_confidence=False)

        # Get feature values and importance
        feature_values = features.iloc[0].to_dict()

        # Calculate contribution based on feature importance and values
        contributions = {}
        for feat, value in feature_values.items():
            importance = self.feature_importance.get(feat, 0) if self.feature_importance else 0
            # Normalize contribution
            contributions[feat] = float(value * importance)

        # Sort by absolute contribution
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]

        return {
            'score': float(score),
            'decision': 'approve' if score > 0.6 else 'reject',
            'top_contributors': [
                {
                    'feature': feat,
                    'value': float(feature_values[feat]),
                    'contribution': float(contrib),
                    'importance': float(self.feature_importance.get(feat, 0)) if self.feature_importance else 0
                }
                for feat, contrib in sorted_contributions
            ]
        }

    def get_model_info(self) -> Dict[str, any]:
        """Get model metadata and information.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        return {
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'num_trees': self.model.num_trees(),
            'metadata': self.metadata,
            'lightgbm_version': lgb.__version__
        }

    def health_check(self) -> Dict[str, any]:
        """Perform health check on predictor.

        Returns:
            Dictionary with health status
        """
        is_healthy = self.model is not None

        # Test prediction if model is loaded
        prediction_latency = None
        if is_healthy:
            try:
                # Create dummy features
                dummy_features = pd.DataFrame([{feat: 0 for feat in self.feature_names}])

                start_time = time.time()
                _ = self.predict(dummy_features)
                prediction_latency = (time.time() - start_time) * 1000  # ms

            except Exception as e:
                is_healthy = False
                logger.error(f"Health check failed: {e}")

        return {
            'is_healthy': is_healthy,
            'model_loaded': self.model is not None,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'prediction_latency_ms': round(prediction_latency, 2) if prediction_latency else None
        }
