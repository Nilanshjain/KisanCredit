"""Comprehensive tests for ML models module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.models import (
    ProfitabilityModelTrainer,
    ProfitabilityPredictor,
    ModelEvaluator,
    ModelExplainer
)


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)

    n_samples = 1000
    n_features = 45

    # Generate random features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Generate target with some correlation to features
    y = pd.Series(
        0.3 + 0.4 * X['feature_0'] + 0.3 * X['feature_1'] + np.random.randn(n_samples) * 0.1
    )
    # Clip to [0, 1] range
    y = y.clip(0, 1)

    # Add profitability_score column
    X['profitability_score'] = y

    return X


@pytest.fixture
def trained_model(sample_data, tmp_path):
    """Train a model for testing."""
    trainer = ProfitabilityModelTrainer(model_dir=str(tmp_path))

    X_train, X_test, y_train, y_test = trainer.prepare_data(
        sample_data,
        test_size=0.2,
        random_state=42
    )

    # Use minimal params for fast testing
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 50,
        'verbose': -1,
        'random_state': 42
    }

    model = trainer.train(
        X_train, y_train,
        X_test, y_test,
        params=params,
        log_to_mlflow=False
    )

    return trainer, model, X_test, y_test


class TestProfitabilityModelTrainer:
    """Tests for ProfitabilityModelTrainer class."""

    def test_initialization(self, tmp_path):
        """Test trainer initialization."""
        trainer = ProfitabilityModelTrainer(model_dir=str(tmp_path))

        assert trainer.model is None
        assert trainer.feature_names is None
        assert trainer.model_dir.exists()

    def test_prepare_data(self, sample_data):
        """Test data preparation."""
        trainer = ProfitabilityModelTrainer()

        X_train, X_test, y_train, y_test = trainer.prepare_data(
            sample_data,
            test_size=0.2,
            random_state=42
        )

        assert len(X_train) == 800
        assert len(X_test) == 200
        assert len(y_train) == 800
        assert len(y_test) == 200
        assert len(trainer.feature_names) == 45
        assert 'profitability_score' not in X_train.columns

    def test_get_default_params(self):
        """Test default parameters."""
        trainer = ProfitabilityModelTrainer()
        params = trainer.get_default_params()

        assert params['objective'] == 'regression'
        assert 'learning_rate' in params
        assert 'num_leaves' in params
        assert params['random_state'] == 42

    def test_train(self, sample_data, tmp_path):
        """Test model training."""
        trainer = ProfitabilityModelTrainer(model_dir=str(tmp_path))

        X_train, X_test, y_train, y_test = trainer.prepare_data(sample_data)

        params = {
            'objective': 'regression',
            'n_estimators': 10,
            'verbose': -1,
            'random_state': 42
        }

        model = trainer.train(
            X_train, y_train,
            params=params,
            log_to_mlflow=False
        )

        assert model is not None
        assert trainer.model is not None
        assert trainer.feature_importance is not None
        assert len(trainer.feature_importance) > 0

    def test_evaluate(self, trained_model):
        """Test model evaluation."""
        trainer, model, X_test, y_test = trained_model

        metrics = trainer.evaluate(X_test, y_test, log_to_mlflow=False)

        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics

        # Check metrics are reasonable
        assert 0 <= metrics['rmse'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1

    def test_save_and_load_model(self, trained_model, tmp_path):
        """Test model saving and loading."""
        trainer, model, X_test, y_test = trained_model

        # Save model
        model_path = trainer.save_model(model_name="test_model")

        assert Path(model_path).exists()

        # Load model
        new_trainer = ProfitabilityModelTrainer(model_dir=str(tmp_path))
        new_trainer.load_model(model_path)

        assert new_trainer.model is not None
        assert new_trainer.feature_names is not None
        assert new_trainer.feature_importance is not None

        # Make sure predictions match
        pred_original = model.predict(X_test)
        pred_loaded = new_trainer.model.predict(X_test)

        np.testing.assert_array_almost_equal(pred_original, pred_loaded)

    def test_get_feature_importance(self, trained_model):
        """Test feature importance retrieval."""
        trainer, model, X_test, y_test = trained_model

        importance = trainer.get_feature_importance(top_n=10)

        assert len(importance) == 10
        assert all(isinstance(v, (int, float)) for v in importance.values())


class TestProfitabilityPredictor:
    """Tests for ProfitabilityPredictor class."""

    def test_initialization_no_model(self):
        """Test predictor initialization without model."""
        predictor = ProfitabilityPredictor()

        # Should not raise error, just log warning
        assert predictor.model is None

    def test_load_model(self, trained_model, tmp_path):
        """Test model loading."""
        trainer, model, X_test, y_test = trained_model

        # Save model first
        model_path = trainer.save_model(model_name="test_model")

        # Load with predictor
        predictor = ProfitabilityPredictor(model_path=model_path)

        assert predictor.model is not None
        assert predictor.feature_names is not None

    def test_predict_single(self, trained_model, tmp_path):
        """Test single prediction."""
        trainer, model, X_test, y_test = trained_model

        # Save and load model
        model_path = trainer.save_model(model_name="test_model")
        predictor = ProfitabilityPredictor(model_path=model_path)

        # Make prediction on first test sample
        features = X_test.iloc[0:1]
        score = predictor.predict(features)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_predict_with_confidence(self, trained_model, tmp_path):
        """Test prediction with confidence."""
        trainer, model, X_test, y_test = trained_model

        model_path = trainer.save_model(model_name="test_model")
        predictor = ProfitabilityPredictor(model_path=model_path)

        features = X_test.iloc[0:1]
        result = predictor.predict(features, return_confidence=True)

        assert isinstance(result, dict)
        assert 'score' in result
        assert 'confidence' in result
        assert 'decision' in result
        assert result['decision'] in ['approve', 'reject']

    def test_predict_batch(self, trained_model, tmp_path):
        """Test batch prediction."""
        trainer, model, X_test, y_test = trained_model

        model_path = trainer.save_model(model_name="test_model")
        predictor = ProfitabilityPredictor(model_path=model_path)

        scores = predictor.predict_batch(X_test)

        assert len(scores) == len(X_test)
        assert all(0 <= s <= 1 for s in scores)

    def test_predict_batch_with_confidence(self, trained_model, tmp_path):
        """Test batch prediction with confidence."""
        trainer, model, X_test, y_test = trained_model

        model_path = trainer.save_model(model_name="test_model")
        predictor = ProfitabilityPredictor(model_path=model_path)

        results = predictor.predict_batch(X_test, return_confidence=True)

        assert isinstance(results, pd.DataFrame)
        assert 'score' in results.columns
        assert 'confidence' in results.columns
        assert 'decision' in results.columns
        assert len(results) == len(X_test)

    def test_explain_prediction(self, trained_model, tmp_path):
        """Test prediction explanation."""
        trainer, model, X_test, y_test = trained_model

        model_path = trainer.save_model(model_name="test_model")
        predictor = ProfitabilityPredictor(model_path=model_path)

        features = X_test.iloc[0:1]
        explanation = predictor.explain_prediction(features, top_n=5)

        assert isinstance(explanation, dict)
        assert 'score' in explanation
        assert 'decision' in explanation
        assert 'top_contributors' in explanation
        assert len(explanation['top_contributors']) == 5

    def test_get_model_info(self, trained_model, tmp_path):
        """Test model info retrieval."""
        trainer, model, X_test, y_test = trained_model

        model_path = trainer.save_model(model_name="test_model")
        predictor = ProfitabilityPredictor(model_path=model_path)

        info = predictor.get_model_info()

        assert 'n_features' in info
        assert 'feature_names' in info
        assert 'num_trees' in info
        assert info['n_features'] == 45

    def test_health_check(self, trained_model, tmp_path):
        """Test health check."""
        trainer, model, X_test, y_test = trained_model

        model_path = trainer.save_model(model_name="test_model")
        predictor = ProfitabilityPredictor(model_path=model_path)

        health = predictor.health_check()

        assert health['is_healthy'] is True
        assert health['model_loaded'] is True
        assert health['n_features'] == 45
        assert health['prediction_latency_ms'] is not None


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator(threshold=0.6)

        assert evaluator.threshold == 0.6
        assert evaluator.results == {}

    def test_evaluate(self, trained_model):
        """Test comprehensive evaluation."""
        trainer, model, X_test, y_test = trained_model

        y_pred = model.predict(X_test)

        evaluator = ModelEvaluator()
        results = evaluator.evaluate(y_test.values, y_pred)

        assert 'regression' in results
        assert 'classification' in results
        assert 'distribution' in results
        assert 'business' in results

        # Check regression metrics
        assert 'rmse' in results['regression']
        assert 'mae' in results['regression']
        assert 'r2' in results['regression']

        # Check classification metrics
        assert 'precision' in results['classification']
        assert 'recall' in results['classification']
        assert 'f1' in results['classification']

        # Check business metrics
        assert 'approval_rate' in results['business']
        assert 'estimated_net_profit_inr' in results['business']

    def test_generate_report(self, trained_model, tmp_path):
        """Test report generation."""
        trainer, model, X_test, y_test = trained_model

        y_pred = model.predict(X_test)

        evaluator = ModelEvaluator()
        evaluator.evaluate(y_test.values, y_pred)

        report = evaluator.generate_report()

        assert isinstance(report, str)
        assert 'PROFITABILITY MODEL EVALUATION REPORT' in report
        assert 'RMSE' in report
        assert 'Precision' in report

        # Test saving report
        report_path = tmp_path / "report.txt"
        evaluator.generate_report(output_path=str(report_path))
        assert report_path.exists()

    def test_save_results(self, trained_model, tmp_path):
        """Test results saving."""
        trainer, model, X_test, y_test = trained_model

        y_pred = model.predict(X_test)

        evaluator = ModelEvaluator()
        evaluator.evaluate(y_test.values, y_pred)

        results_path = tmp_path / "results.json"
        evaluator.save_results(str(results_path))

        assert results_path.exists()


class TestModelExplainer:
    """Tests for ModelExplainer class."""

    def test_initialization(self, trained_model):
        """Test explainer initialization."""
        trainer, model, X_test, y_test = trained_model

        explainer = ModelExplainer(model, trainer.feature_names)

        assert explainer.model is not None
        assert explainer.feature_names is not None
        assert explainer.explainer is not None
        assert explainer.base_value is not None

    def test_explain_prediction(self, trained_model):
        """Test single prediction explanation."""
        trainer, model, X_test, y_test = trained_model

        explainer = ModelExplainer(model, trainer.feature_names)

        features = X_test.iloc[0:1]
        explanation = explainer.explain_prediction(features, top_n=5)

        assert isinstance(explanation, dict)
        assert 'prediction' in explanation
        assert 'base_value' in explanation
        assert 'top_contributions' in explanation
        assert len(explanation['top_contributions']) == 5

        # Check contribution structure
        for contrib in explanation['top_contributions']:
            assert 'feature' in contrib
            assert 'value' in contrib
            assert 'shap_value' in contrib

    def test_explain_batch(self, trained_model):
        """Test batch explanation."""
        trainer, model, X_test, y_test = trained_model

        explainer = ModelExplainer(model, trainer.feature_names)

        shap_values = explainer.explain_batch(X_test[:10])

        assert shap_values.shape == (10, len(trainer.feature_names))

    def test_get_global_importance(self, trained_model):
        """Test global importance calculation."""
        trainer, model, X_test, y_test = trained_model

        explainer = ModelExplainer(model, trainer.feature_names)

        importance = explainer.get_global_importance(X_test[:100], top_n=10)

        assert isinstance(importance, dict)
        assert len(importance) == 10
        assert all(isinstance(v, (int, float, np.number)) for v in importance.values())


def test_end_to_end_workflow(sample_data, tmp_path):
    """Test complete end-to-end workflow."""
    # 1. Train model
    trainer = ProfitabilityModelTrainer(model_dir=str(tmp_path))

    X_train, X_test, y_train, y_test = trainer.prepare_data(sample_data)

    params = {
        'objective': 'regression',
        'n_estimators': 20,
        'verbose': -1,
        'random_state': 42
    }

    model = trainer.train(X_train, y_train, X_test, y_test, params=params, log_to_mlflow=False)

    # 2. Evaluate model
    metrics = trainer.evaluate(X_test, y_test, log_to_mlflow=False)
    assert 'rmse' in metrics

    # 3. Save model
    model_path = trainer.save_model(model_name="workflow_test")
    assert Path(model_path).exists()

    # 4. Load model with predictor
    predictor = ProfitabilityPredictor(model_path=model_path)

    # 5. Make predictions
    scores = predictor.predict_batch(X_test)
    assert len(scores) == len(X_test)

    # 6. Explain predictions
    explanation = predictor.explain_prediction(X_test.iloc[0:1])
    assert 'score' in explanation

    # 7. Run evaluator
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(y_test.values, scores)
    assert 'regression' in results

    # 8. Generate report
    report = evaluator.generate_report()
    assert 'RMSE' in report

    # 9. SHAP explainer
    explainer = ModelExplainer(model, trainer.feature_names)
    shap_explanation = explainer.explain_prediction(X_test.iloc[0:1])
    assert 'prediction' in shap_explanation
