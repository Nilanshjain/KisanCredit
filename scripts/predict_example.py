"""Example script for making predictions with trained model.

Usage:
    python scripts/predict_example.py --model_path models/profitability_model_latest.pkl
"""

import argparse
from pathlib import Path
import pandas as pd
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ProfitabilityPredictor
from src.features import FeatureEngineeringPipeline
from src.pipeline.data_generator import DataGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def predict_application(model_path: str, application_data: dict = None) -> None:
    """Make prediction for a loan application.

    Args:
        model_path: Path to trained model
        application_data: Application data dictionary (if None, uses synthetic example)
    """
    logger.info("Loading trained model...")
    predictor = ProfitabilityPredictor(model_path=model_path)

    # Check model health
    health = predictor.health_check()
    logger.info(f"Model health: {health}")

    if not health['is_healthy']:
        logger.error("Model is not healthy. Exiting.")
        return

    # Generate example application if not provided
    if application_data is None:
        logger.info("Generating example application...")
        generator = DataGenerator()
        applications = generator.generate_batch(1)
        application_data = applications[0]

    # Extract features
    logger.info("Extracting features from application...")
    pipeline = FeatureEngineeringPipeline()
    features = pipeline.extract_features(application_data)

    # Convert to DataFrame (remove metadata)
    features_df = pd.DataFrame([features])
    metadata_cols = ['application_id', 'user_id']
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    features_df = features_df[feature_cols]

    logger.info(f"Extracted {len(feature_cols)} features")

    # Make prediction with confidence
    logger.info("Making prediction...")
    result = predictor.predict(features_df, return_confidence=True)

    # Display results
    print("\n" + "=" * 80)
    print("PROFITABILITY PREDICTION RESULTS")
    print("=" * 80)
    print(f"Application ID:     {application_data.get('application_id', 'N/A')}")
    print(f"User ID:            {application_data.get('user_id', 'N/A')}")
    print(f"\nPrediction Score:   {result['score']:.4f}")
    print(f"Confidence:         {result['confidence']:.4f}")
    print(f"Decision:           {result['decision'].upper()}")
    print(f"Processing Time:    {result['prediction_time_ms']:.2f} ms")
    print("=" * 80)

    # Get detailed explanation
    logger.info("Generating prediction explanation...")
    explanation = predictor.explain_prediction(features_df, top_n=10)

    print("\nTOP 10 CONTRIBUTING FEATURES:")
    print("-" * 80)
    for i, contrib in enumerate(explanation['top_contributors'], 1):
        print(f"{i:2d}. {contrib['feature']:35s} | Value: {contrib['value']:8.4f} | Contrib: {contrib['contribution']:8.4f}")
    print("-" * 80)

    # Business interpretation
    print("\nBUSINESS INTERPRETATION:")
    print("-" * 80)
    if result['decision'] == 'approve':
        print("✓ RECOMMEND APPROVAL")
        print(f"  - High profitability score ({result['score']:.2%})")
        print(f"  - Confidence level: {result['confidence']:.2%}")
        print("  - Expected positive return on investment")
    else:
        print("✗ RECOMMEND REJECTION")
        print(f"  - Low profitability score ({result['score']:.2%})")
        print(f"  - Confidence level: {result['confidence']:.2%}")
        print("  - High risk of unprofitable outcome")
    print("=" * 80)


def batch_predict(model_path: str, n_applications: int = 10) -> None:
    """Make batch predictions for multiple applications.

    Args:
        model_path: Path to trained model
        n_applications: Number of applications to predict
    """
    logger.info(f"Loading trained model...")
    predictor = ProfitabilityPredictor(model_path=model_path)

    # Generate batch of applications
    logger.info(f"Generating {n_applications} test applications...")
    generator = DataGenerator()
    applications = generator.generate_batch(n_applications)

    # Extract features
    logger.info("Extracting features from applications...")
    pipeline = FeatureEngineeringPipeline()
    features_df = pipeline.extract_batch(applications)

    # Remove metadata
    metadata_cols = ['application_id', 'user_id']
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    X = features_df[feature_cols]

    # Make batch predictions
    logger.info("Making batch predictions...")
    results = predictor.predict_batch(X, return_confidence=True)

    # Add metadata back
    results['application_id'] = features_df['application_id'].values
    results['user_id'] = features_df['user_id'].values

    # Display summary
    print("\n" + "=" * 80)
    print("BATCH PREDICTION RESULTS")
    print("=" * 80)
    print(f"Total Applications:     {len(results)}")
    print(f"Approved:               {(results['decision'] == 'approve').sum()} ({(results['decision'] == 'approve').sum() / len(results) * 100:.1f}%)")
    print(f"Rejected:               {(results['decision'] == 'reject').sum()} ({(results['decision'] == 'reject').sum() / len(results) * 100:.1f}%)")
    print(f"\nAverage Score:          {results['score'].mean():.4f}")
    print(f"Average Confidence:     {results['confidence'].mean():.4f}")
    print(f"Score Range:            {results['score'].min():.4f} - {results['score'].max():.4f}")
    print("=" * 80)

    # Show top 5 and bottom 5
    print("\nTOP 5 APPLICATIONS (Highest Profitability):")
    print("-" * 80)
    top5 = results.nlargest(5, 'score')
    for idx, row in top5.iterrows():
        print(f"  {row['application_id']:20s} | Score: {row['score']:.4f} | Decision: {row['decision']}")

    print("\nBOTTOM 5 APPLICATIONS (Lowest Profitability):")
    print("-" * 80)
    bottom5 = results.nsmallest(5, 'score')
    for idx, row in bottom5.iterrows():
        print(f"  {row['application_id']:20s} | Score: {row['score']:.4f} | Decision: {row['decision']}")

    print("=" * 80)

    # Save results
    output_path = Path("outputs") / "batch_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"Results saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Make predictions with trained model")

    parser.add_argument(
        '--model_path',
        type=str,
        default='models/profitability_model_latest.pkl',
        help='Path to trained model'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'batch'],
        default='single',
        help='Prediction mode: single or batch'
    )

    parser.add_argument(
        '--n_applications',
        type=int,
        default=10,
        help='Number of applications for batch prediction'
    )

    args = parser.parse_args()

    if not Path(args.model_path).exists():
        logger.error(f"Model not found at: {args.model_path}")
        logger.info("Please train a model first using: python scripts/train_model.py")
        sys.exit(1)

    if args.mode == 'single':
        predict_application(args.model_path)
    else:
        batch_predict(args.model_path, args.n_applications)


if __name__ == '__main__':
    main()
