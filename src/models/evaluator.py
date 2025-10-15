"""Model Evaluation and Performance Analysis.

Comprehensive evaluation metrics for profitability scoring model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for profitability scoring.

    Features:
    - Regression metrics (RMSE, MAE, R2, MAPE)
    - Classification metrics (precision, recall, F1, AUC)
    - Business metrics (profitability, cost-benefit analysis)
    - Visualization and reporting
    """

    def __init__(self, threshold: float = 0.6):
        """Initialize evaluator.

        Args:
            threshold: Score threshold for profitable/unprofitable classification
        """
        self.threshold = threshold
        self.results: Dict = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """Comprehensive model evaluation.

        Args:
            y_true: True profitability scores
            y_pred: Predicted profitability scores
            feature_names: List of feature names (for feature analysis)

        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("Starting model evaluation")

        results = {}

        # Regression metrics
        results['regression'] = self._evaluate_regression(y_true, y_pred)

        # Classification metrics (using threshold)
        results['classification'] = self._evaluate_classification(y_true, y_pred)

        # Distribution analysis
        results['distribution'] = self._analyze_distribution(y_true, y_pred)

        # Business metrics
        results['business'] = self._calculate_business_metrics(y_true, y_pred)

        self.results = results

        logger.info(
            "Model evaluation completed",
            rmse=round(results['regression']['rmse'], 4),
            precision=round(results['classification']['precision'], 4),
            recall=round(results['classification']['recall'], 4)
        )

        return results

    def _evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of regression metrics
        """
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100,
            'max_error': np.max(np.abs(y_true - y_pred)),
            'mean_prediction': float(np.mean(y_pred)),
            'std_prediction': float(np.std(y_pred))
        }

        return metrics

    def _evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate classification metrics using threshold.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of classification metrics
        """
        # Convert to binary classification
        y_true_binary = (y_true > self.threshold).astype(int)
        y_pred_binary = (y_pred > self.threshold).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Specificity and NPV
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # ROC AUC
        try:
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred)
            roc_auc = auc(fpr, tpr)
        except:
            roc_auc = 0.0

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            'npv': npv,
            'roc_auc': roc_auc,
            'confusion_matrix': {
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn)
            }
        }

        return metrics

    def _analyze_distribution(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Analyze prediction distribution.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with distribution statistics
        """
        # Residuals
        residuals = y_true - y_pred

        # Percentiles
        percentiles = [10, 25, 50, 75, 90]
        pred_percentiles = {
            f'p{p}': float(np.percentile(y_pred, p))
            for p in percentiles
        }

        metrics = {
            'residual_mean': float(np.mean(residuals)),
            'residual_std': float(np.std(residuals)),
            'residual_skewness': float(pd.Series(residuals).skew()),
            'residual_kurtosis': float(pd.Series(residuals).kurtosis()),
            'prediction_percentiles': pred_percentiles,
            'true_mean': float(np.mean(y_true)),
            'true_std': float(np.std(y_true)),
            'pred_mean': float(np.mean(y_pred)),
            'pred_std': float(np.std(y_pred))
        }

        return metrics

    def _calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate business-relevant metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of business metrics
        """
        # Binary classification for business metrics
        y_true_binary = (y_true > self.threshold).astype(int)
        y_pred_binary = (y_pred > self.threshold).astype(int)

        tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
        tn = np.sum((y_pred_binary == 0) & (y_true_binary == 0))

        total = len(y_true)

        # Business metrics
        # Assuming cost of false positive >> cost of false negative
        # Cost assumptions (in rupees):
        # - False positive (bad loan approved): ₹50,000 average loss
        # - False negative (good loan rejected): ₹1,000 opportunity cost
        # - True positive (good loan approved): ₹5,000 profit
        # - True negative (bad loan rejected): ₹100 processing savings

        fp_cost = 50000
        fn_cost = 1000
        tp_profit = 5000
        tn_saving = 100

        total_cost = fp * fp_cost + fn * fn_cost
        total_profit = tp * tp_profit + tn * tn_saving
        net_profit = total_profit - total_cost

        metrics = {
            'approval_rate': float((tp + fp) / total) if total > 0 else 0.0,
            'profitable_loans_approved': int(tp),
            'unprofitable_loans_approved': int(fp),
            'profitable_loans_rejected': int(fn),
            'unprofitable_loans_rejected': int(tn),
            'estimated_total_profit_inr': float(total_profit),
            'estimated_total_cost_inr': float(total_cost),
            'estimated_net_profit_inr': float(net_profit),
            'profit_per_application_inr': float(net_profit / total) if total > 0 else 0.0,
            'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        }

        return metrics

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report.

        Args:
            output_path: Path to save report (optional)

        Returns:
            Report as formatted string
        """
        if not self.results:
            raise ValueError("No evaluation results. Call evaluate() first.")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PROFITABILITY MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Regression metrics
        report_lines.append("REGRESSION METRICS")
        report_lines.append("-" * 80)
        reg = self.results['regression']
        report_lines.append(f"  RMSE:                    {reg['rmse']:.4f}")
        report_lines.append(f"  MAE:                     {reg['mae']:.4f}")
        report_lines.append(f"  R²:                      {reg['r2']:.4f}")
        report_lines.append(f"  MAPE:                    {reg['mape']:.2f}%")
        report_lines.append(f"  Max Error:               {reg['max_error']:.4f}")
        report_lines.append("")

        # Classification metrics
        report_lines.append("CLASSIFICATION METRICS (Threshold = 0.6)")
        report_lines.append("-" * 80)
        clf = self.results['classification']
        report_lines.append(f"  Precision:               {clf['precision']:.4f}")
        report_lines.append(f"  Recall:                  {clf['recall']:.4f}")
        report_lines.append(f"  F1 Score:                {clf['f1']:.4f}")
        report_lines.append(f"  Accuracy:                {clf['accuracy']:.4f}")
        report_lines.append(f"  ROC AUC:                 {clf['roc_auc']:.4f}")
        report_lines.append("")
        cm = clf['confusion_matrix']
        report_lines.append("  Confusion Matrix:")
        report_lines.append(f"    TP: {cm['tp']:6d}  |  FP: {cm['fp']:6d}")
        report_lines.append(f"    FN: {cm['fn']:6d}  |  TN: {cm['tn']:6d}")
        report_lines.append("")

        # Business metrics
        report_lines.append("BUSINESS METRICS")
        report_lines.append("-" * 80)
        biz = self.results['business']
        report_lines.append(f"  Approval Rate:           {biz['approval_rate']*100:.2f}%")
        report_lines.append(f"  Profitable Loans (Approved):    {biz['profitable_loans_approved']}")
        report_lines.append(f"  Unprofitable Loans (Approved):  {biz['unprofitable_loans_approved']}")
        report_lines.append(f"  Profitable Loans (Rejected):    {biz['profitable_loans_rejected']}")
        report_lines.append(f"  Estimated Net Profit:    ₹{biz['estimated_net_profit_inr']:,.2f}")
        report_lines.append(f"  Profit per Application:  ₹{biz['profit_per_application_inr']:,.2f}")
        report_lines.append("")

        # Distribution
        report_lines.append("DISTRIBUTION ANALYSIS")
        report_lines.append("-" * 80)
        dist = self.results['distribution']
        report_lines.append(f"  Prediction Mean:         {dist['pred_mean']:.4f}")
        report_lines.append(f"  Prediction Std:          {dist['pred_std']:.4f}")
        report_lines.append(f"  Residual Mean:           {dist['residual_mean']:.4f}")
        report_lines.append(f"  Residual Std:            {dist['residual_std']:.4f}")
        report_lines.append("")

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        # Save to file if path provided
        if output_path:
            Path(output_path).write_text(report)
            logger.info(f"Report saved to {output_path}")

        return report

    def save_results(self, output_path: str) -> None:
        """Save evaluation results to JSON file.

        Args:
            output_path: Path to save results
        """
        if not self.results:
            raise ValueError("No evaluation results. Call evaluate() first.")

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def plot_results(self, y_true: np.ndarray, y_pred: np.ndarray, output_dir: str = "plots") -> None:
        """Generate evaluation plots.

        Args:
            y_true: True values
            y_pred: Predicted values
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")

        # 1. Predicted vs Actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
        plt.xlabel('True Profitability Score')
        plt.ylabel('Predicted Profitability Score')
        plt.title('Predicted vs Actual Profitability Scores')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'predicted_vs_actual.png', dpi=300)
        plt.close()

        # 2. Residual plot
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Profitability Score')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.tight_layout()
        plt.savefig(output_path / 'residuals.png', dpi=300)
        plt.close()

        # 3. Distribution comparison
        plt.figure(figsize=(10, 6))
        plt.hist(y_true, bins=30, alpha=0.5, label='True', density=True)
        plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted', density=True)
        plt.xlabel('Profitability Score')
        plt.ylabel('Density')
        plt.title('Distribution of True vs Predicted Scores')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'distribution.png', dpi=300)
        plt.close()

        # 4. ROC Curve
        y_true_binary = (y_true > self.threshold).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'roc_curve.png', dpi=300)
        plt.close()

        logger.info(f"Plots saved to {output_dir}/")

    def compare_models(
        self,
        model_results: Dict[str, Dict],
        metrics: List[str] = ['rmse', 'precision', 'recall', 'f1']
    ) -> pd.DataFrame:
        """Compare multiple model evaluation results.

        Args:
            model_results: Dictionary mapping model names to evaluation results
            metrics: List of metrics to compare

        Returns:
            DataFrame with comparison results
        """
        comparison = {}

        for model_name, results in model_results.items():
            model_metrics = {}
            for metric in metrics:
                # Look in regression metrics
                if metric in results.get('regression', {}):
                    model_metrics[metric] = results['regression'][metric]
                # Look in classification metrics
                elif metric in results.get('classification', {}):
                    model_metrics[metric] = results['classification'][metric]
                # Look in business metrics
                elif metric in results.get('business', {}):
                    model_metrics[metric] = results['business'][metric]

            comparison[model_name] = model_metrics

        df = pd.DataFrame(comparison).T
        return df
