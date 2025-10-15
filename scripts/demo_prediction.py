"""Demo script showcasing the KisanCredit prediction system.

Demonstrates:
- Loading the trained model
- Making predictions on sample data
- SHAP explanations for interpretability
- Batch prediction capabilities
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.predictor import ProfitabilityPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_sample_applicant(applicant_type: str) -> dict:
    """Create sample applicant features based on profile type.

    Args:
        applicant_type: 'good', 'average', or 'risky'
    """

    if applicant_type == "good":
        # Financially stable applicant
        return {
            "income_monthly_avg": 35000,
            "income_consistency_score": 0.85,
            "income_growth_trend": 0.12,
            "income_source_diversity": 3,
            "income_credit_ratio": 0.95,
            "income_seasonal_variance": 0.15,
            "income_regularity_score": 0.90,
            "income_upi_percentage": 0.75,
            "income_largest_transaction": 45000,

            "expense_monthly_avg": 20000,
            "expense_to_income_ratio": 0.57,
            "expense_essential_ratio": 0.70,
            "expense_luxury_ratio": 0.10,
            "expense_savings_potential": 0.40,
            "expense_debt_burden": 0.15,
            "expense_volatility": 0.20,
            "expense_category_diversity": 8,
            "expense_bill_timeliness": 0.95,

            "social_network_strength": 0.85,
            "social_total_contacts": 350,
            "social_family_size": 30,
            "social_business_contacts": 75,
            "social_government_contacts": 5,
            "social_communication_frequency": 60,
            "social_contact_diversity": 0.80,
            "social_network_depth": 4,

            "discipline_overall_score": 0.88,
            "discipline_emi_regularity": 0.95,
            "discipline_bill_payment_score": 0.92,
            "discipline_failed_transactions": 0,
            "discipline_overdraft_frequency": 0,
            "discipline_savings_consistency": 0.85,

            "behavioral_risk_score": 0.10,
            "behavioral_gambling_indicator": 0,
            "behavioral_location_changes": 2,
            "behavioral_night_transaction_ratio": 0.08,
            "behavioral_financial_literacy": 0.85,
            "behavioral_app_usage_score": 8.5,

            "location_stability_score": 0.90,
            "location_mobility_score": 0.20,
            "location_travel_frequency": 3,
            "location_distance_from_center": 15,
            "location_urban_score": 0.40,
            "location_unique_places": 8,
            "location_consistency_score": 0.88
        }

    elif applicant_type == "average":
        # Moderate risk applicant
        return {
            "income_monthly_avg": 18000,
            "income_consistency_score": 0.65,
            "income_growth_trend": 0.03,
            "income_source_diversity": 2,
            "income_credit_ratio": 0.75,
            "income_seasonal_variance": 0.35,
            "income_regularity_score": 0.70,
            "income_upi_percentage": 0.50,
            "income_largest_transaction": 22000,

            "expense_monthly_avg": 14000,
            "expense_to_income_ratio": 0.78,
            "expense_essential_ratio": 0.65,
            "expense_luxury_ratio": 0.15,
            "expense_savings_potential": 0.20,
            "expense_debt_burden": 0.25,
            "expense_volatility": 0.35,
            "expense_category_diversity": 6,
            "expense_bill_timeliness": 0.75,

            "social_network_strength": 0.60,
            "social_total_contacts": 150,
            "social_family_size": 18,
            "social_business_contacts": 30,
            "social_government_contacts": 2,
            "social_communication_frequency": 25,
            "social_contact_diversity": 0.55,
            "social_network_depth": 2,

            "discipline_overall_score": 0.65,
            "discipline_emi_regularity": 0.70,
            "discipline_bill_payment_score": 0.68,
            "discipline_failed_transactions": 2,
            "discipline_overdraft_frequency": 1,
            "discipline_savings_consistency": 0.55,

            "behavioral_risk_score": 0.25,
            "behavioral_gambling_indicator": 0,
            "behavioral_location_changes": 8,
            "behavioral_night_transaction_ratio": 0.18,
            "behavioral_financial_literacy": 0.60,
            "behavioral_app_usage_score": 5.5,

            "location_stability_score": 0.65,
            "location_mobility_score": 0.45,
            "location_travel_frequency": 10,
            "location_distance_from_center": 50,
            "location_urban_score": 0.55,
            "location_unique_places": 15,
            "location_consistency_score": 0.60
        }

    else:  # risky
        # High-risk applicant
        return {
            "income_monthly_avg": 9000,
            "income_consistency_score": 0.40,
            "income_growth_trend": -0.05,
            "income_source_diversity": 1,
            "income_credit_ratio": 0.55,
            "income_seasonal_variance": 0.60,
            "income_regularity_score": 0.45,
            "income_upi_percentage": 0.30,
            "income_largest_transaction": 12000,

            "expense_monthly_avg": 8500,
            "expense_to_income_ratio": 0.94,
            "expense_essential_ratio": 0.50,
            "expense_luxury_ratio": 0.25,
            "expense_savings_potential": 0.05,
            "expense_debt_burden": 0.45,
            "expense_volatility": 0.55,
            "expense_category_diversity": 4,
            "expense_bill_timeliness": 0.50,

            "social_network_strength": 0.35,
            "social_total_contacts": 70,
            "social_family_size": 8,
            "social_business_contacts": 12,
            "social_government_contacts": 0,
            "social_communication_frequency": 12,
            "social_contact_diversity": 0.30,
            "social_network_depth": 1,

            "discipline_overall_score": 0.42,
            "discipline_emi_regularity": 0.55,
            "discipline_bill_payment_score": 0.48,
            "discipline_failed_transactions": 5,
            "discipline_overdraft_frequency": 3,
            "discipline_savings_consistency": 0.25,

            "behavioral_risk_score": 0.45,
            "behavioral_gambling_indicator": 1,
            "behavioral_location_changes": 18,
            "behavioral_night_transaction_ratio": 0.35,
            "behavioral_financial_literacy": 0.35,
            "behavioral_app_usage_score": 3.2,

            "location_stability_score": 0.40,
            "location_mobility_score": 0.68,
            "location_travel_frequency": 18,
            "location_distance_from_center": 120,
            "location_urban_score": 0.75,
            "location_unique_places": 25,
            "location_consistency_score": 0.35
        }


def demo_single_predictions(predictor: ProfitabilityPredictor):
    """Demonstrate single predictions for different applicant types."""

    print("\n" + "="*70)
    print("DEMO: SINGLE PREDICTIONS")
    print("="*70)

    applicant_types = ["good", "average", "risky"]

    for app_type in applicant_types:
        print(f"\n--- {app_type.upper()} APPLICANT ---")

        features = create_sample_applicant(app_type)
        features_df = pd.DataFrame([features])

        # Make prediction with confidence
        result = predictor.predict(features_df, return_confidence=True)

        print(f"Profitability Score: {result['score']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Decision: {result['decision'].upper()}")

        # Get explanation
        try:
            explanation = predictor.explain_prediction(features_df, top_n=3)
            if 'top_contributors' in explanation:
                print(f"\nTop Contributing Factors:")
                contributors = explanation['top_contributors']
                if isinstance(contributors, dict):
                    for feature, value in list(contributors.items())[:3]:
                        print(f"  - {feature}: {value:+.3f}")
                elif isinstance(contributors, list):
                    for item in contributors[:3]:
                        if isinstance(item, dict) and 'feature' in item:
                            print(f"  - {item['feature']}: {item.get('contribution', 0):+.3f}")
                        else:
                            print(f"  - {item}")
        except Exception as e:
            print(f"\n  (Explanation unavailable: {str(e)})")


def demo_batch_prediction(predictor: ProfitabilityPredictor):
    """Demonstrate batch prediction capabilities."""

    print("\n" + "="*70)
    print("DEMO: BATCH PREDICTIONS")
    print("="*70)

    # Create batch of mixed applicants
    batch_features = []
    for _ in range(3):
        batch_features.append(create_sample_applicant("good"))
    for _ in range(4):
        batch_features.append(create_sample_applicant("average"))
    for _ in range(3):
        batch_features.append(create_sample_applicant("risky"))

    batch_df = pd.DataFrame(batch_features)

    print(f"\nProcessing batch of {len(batch_df)} applications...")

    # Batch prediction with confidence
    results = predictor.predict_batch(batch_df, return_confidence=True)

    print(f"\nBatch Results:")
    print(f"  Total applications: {len(results)}")
    print(f"  Approved: {(results['decision'] == 'approve').sum()}")
    print(f"  Rejected: {(results['decision'] == 'reject').sum()}")
    print(f"  Avg Score: {results['score'].mean():.3f}")
    print(f"  Avg Confidence: {results['confidence'].mean():.3f}")

    print(f"\nScore Distribution:")
    print(results['score'].describe()[['min', '25%', '50%', '75%', 'max']].to_string())


def demo_model_info(predictor: ProfitabilityPredictor):
    """Display model information."""

    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)

    info = predictor.get_model_info()

    print(f"\nModel Details:")
    print(f"  Number of features: {info['n_features']}")
    print(f"  Number of trees: {info.get('num_trees', 'N/A')}")
    print(f"  Model type: LightGBM Gradient Boosting")

    print(f"\nTop 10 Most Important Features:")
    if hasattr(predictor, 'feature_importance') and predictor.feature_importance:
        importance = dict(list(predictor.feature_importance.items())[:10])
        for i, (feature, imp) in enumerate(importance.items(), 1):
            print(f"  {i:2d}. {feature:40s} {imp:10.1f}")
    else:
        print("  Feature importance not available")


def main():
    """Run KisanCredit prediction demo."""

    print("\n" + "="*70)
    print("KISANCREDIT - AI LOAN UNDERWRITING DEMO")
    print("="*70)
    print("\nAlternative data-driven profitability scoring for rural India")
    print("Predicts loan profitability using SMS, contacts, location, and behavior")

    # Find latest model
    model_dir = Path("models")
    model_files = list(model_dir.glob("profitability_model_*_latest.pkl"))

    if not model_files:
        model_files = list(model_dir.glob("profitability_model_*.pkl"))

    if not model_files:
        print("\n[ERROR] No trained model found. Run train_model.py first.")
        return

    model_path = str(sorted(model_files, key=lambda x: x.stat().st_mtime)[-1])
    print(f"\nLoading model: {Path(model_path).name}")

    # Initialize predictor
    predictor = ProfitabilityPredictor(model_path=model_path)

    # Run demos
    demo_model_info(predictor)
    demo_single_predictions(predictor)
    demo_batch_prediction(predictor)

    # Health check
    health = predictor.health_check()

    print("\n" + "="*70)
    print("SYSTEM HEALTH CHECK")
    print("="*70)
    print(f"\nModel Status: {'HEALTHY' if health['is_healthy'] else 'UNHEALTHY'}")
    print(f"Prediction Latency: {health['prediction_latency_ms']:.2f}ms")
    print(f"Features Loaded: {health['n_features']}")

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  - Start Docker stack: docker-compose up -d")
    print("  - Access API docs: http://localhost:8000/docs")
    print("  - Run benchmark: python scripts/benchmark_latency.py")
    print("  - Generate more data: python scripts/generate_data.py")


if __name__ == "__main__":
    main()
