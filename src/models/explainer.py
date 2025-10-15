"""SHAP-based Model Explainability.

Provides interpretable explanations for profitability predictions using SHAP values.
"""

import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
from pathlib import Path
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelExplainer:
    """SHAP-based explainability for profitability scoring model.

    Features:
    - Individual prediction explanations
    - Global feature importance
    - Feature interaction analysis
    - Visualization support
    """

    def __init__(self, model, feature_names: List[str]):
        """Initialize explainer.

        Args:
            model: Trained LightGBM model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer: Optional[shap.TreeExplainer] = None
        self.base_value: Optional[float] = None

        self._initialize_explainer()

    def _initialize_explainer(self) -> None:
        """Initialize SHAP TreeExplainer."""
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.base_value = float(self.explainer.expected_value)

            logger.info(
                "SHAP explainer initialized",
                base_value=round(self.base_value, 4),
                n_features=len(self.feature_names)
            )
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            raise

    def explain_prediction(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        top_n: int = 10
    ) -> Dict[str, any]:
        """Explain a single prediction using SHAP values.

        Args:
            features: Feature values (single row)
            top_n: Number of top features to return

        Returns:
            Dictionary with explanation details
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        # Ensure features is 2D
        if isinstance(features, pd.Series):
            features = features.to_frame().T
        elif isinstance(features, np.ndarray) and features.ndim == 1:
            features = features.reshape(1, -1)

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(features)

        # Handle both single and batch predictions
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        if shap_values.ndim > 1:
            shap_values = shap_values[0]

        # Get feature values
        if isinstance(features, pd.DataFrame):
            feature_values = features.iloc[0].values
        else:
            feature_values = features[0]

        # Create feature contributions
        contributions = []
        for i, (feat_name, feat_value, shap_value) in enumerate(
            zip(self.feature_names, feature_values, shap_values)
        ):
            contributions.append({
                'feature': feat_name,
                'value': float(feat_value),
                'shap_value': float(shap_value),
                'abs_shap_value': float(abs(shap_value))
            })

        # Sort by absolute SHAP value
        contributions.sort(key=lambda x: x['abs_shap_value'], reverse=True)

        # Calculate prediction
        prediction = self.base_value + np.sum(shap_values)

        explanation = {
            'prediction': float(prediction),
            'base_value': float(self.base_value),
            'top_contributions': contributions[:top_n],
            'all_contributions': contributions,
            'decision': 'approve' if prediction > 0.6 else 'reject'
        }

        logger.debug(
            "Prediction explained",
            prediction=round(prediction, 4),
            top_feature=contributions[0]['feature']
        )

        return explanation

    def explain_batch(
        self,
        features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Calculate SHAP values for batch of predictions.

        Args:
            features: Feature values (multiple rows)

        Returns:
            SHAP values array (n_samples x n_features)
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        shap_values = self.explainer.shap_values(features)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        logger.info(
            "Batch explanations generated",
            n_samples=len(features),
            shap_shape=shap_values.shape
        )

        return shap_values

    def get_global_importance(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        top_n: int = 20
    ) -> Dict[str, float]:
        """Calculate global feature importance using SHAP values.

        Args:
            features: Feature values for representative sample
            top_n: Number of top features to return

        Returns:
            Dictionary of feature names and mean absolute SHAP values
        """
        shap_values = self.explain_batch(features)

        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Create importance dictionary
        importance = dict(zip(self.feature_names, mean_abs_shap))

        # Sort by importance
        importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )

        logger.info(
            "Global feature importance calculated",
            top_feature=list(importance.keys())[0],
            top_importance=round(list(importance.values())[0], 4)
        )

        return importance

    def plot_waterfall(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        output_path: Optional[str] = None,
        max_display: int = 15
    ) -> None:
        """Create waterfall plot for single prediction.

        Args:
            features: Feature values (single row)
            output_path: Path to save plot (optional)
            max_display: Maximum number of features to display
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        # Ensure features is 2D
        if isinstance(features, pd.Series):
            features = features.to_frame().T
        elif isinstance(features, np.ndarray) and features.ndim == 1:
            features = features.reshape(1, -1)

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(features)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[0] if shap_values.ndim > 1 else shap_values,
            base_values=self.base_value,
            data=features.iloc[0].values if isinstance(features, pd.DataFrame) else features[0],
            feature_names=self.feature_names
        )

        # Create waterfall plot
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(explanation, max_display=max_display, show=False)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Waterfall plot saved to {output_path}")

        plt.close()

    def plot_force(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        output_path: Optional[str] = None
    ) -> None:
        """Create force plot for single prediction.

        Args:
            features: Feature values (single row)
            output_path: Path to save plot (optional)
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        # Ensure features is 2D
        if isinstance(features, pd.Series):
            features = features.to_frame().T
        elif isinstance(features, np.ndarray) and features.ndim == 1:
            features = features.reshape(1, -1)

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(features)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Create force plot
        shap.force_plot(
            self.base_value,
            shap_values[0] if shap_values.ndim > 1 else shap_values,
            features.iloc[0] if isinstance(features, pd.DataFrame) else features[0],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Force plot saved to {output_path}")

        plt.close()

    def plot_summary(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        output_path: Optional[str] = None,
        max_display: int = 20
    ) -> None:
        """Create summary plot showing global feature importance.

        Args:
            features: Feature values for representative sample
            output_path: Path to save plot (optional)
            max_display: Maximum number of features to display
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        shap_values = self.explain_batch(features)

        # Create summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            features,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Summary plot saved to {output_path}")

        plt.close()

    def plot_bar(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        output_path: Optional[str] = None,
        max_display: int = 20
    ) -> None:
        """Create bar plot showing mean absolute SHAP values.

        Args:
            features: Feature values for representative sample
            output_path: Path to save plot (optional)
            max_display: Maximum number of features to display
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        shap_values = self.explain_batch(features)

        # Create bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            features,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False
        )

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Bar plot saved to {output_path}")

        plt.close()

    def plot_dependence(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        feature_name: str,
        interaction_feature: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> None:
        """Create dependence plot showing how a feature affects predictions.

        Args:
            features: Feature values for representative sample
            feature_name: Name of feature to analyze
            interaction_feature: Name of interaction feature (optional)
            output_path: Path to save plot (optional)
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in model features")

        shap_values = self.explain_batch(features)

        feature_idx = self.feature_names.index(feature_name)
        interaction_idx = None

        if interaction_feature:
            if interaction_feature not in self.feature_names:
                raise ValueError(f"Feature '{interaction_feature}' not found")
            interaction_idx = self.feature_names.index(interaction_feature)

        # Create dependence plot
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_values,
            features,
            feature_names=self.feature_names,
            interaction_index=interaction_idx,
            show=False
        )

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dependence plot saved to {output_path}")

        plt.close()

    def save_explanation(
        self,
        explanation: Dict,
        output_path: str
    ) -> None:
        """Save explanation to JSON file.

        Args:
            explanation: Explanation dictionary from explain_prediction()
            output_path: Path to save explanation
        """
        with open(output_path, 'w') as f:
            json.dump(explanation, f, indent=2)

        logger.info(f"Explanation saved to {output_path}")
