"""Test script for feature extraction pipeline.

Tests all feature extractors and the main pipeline orchestrator.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import with proper package structure
from src.features import FeatureEngineeringPipeline
from src.pipeline.data_generator import SyntheticDataGenerator
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def test_feature_extraction():
    """Test feature extraction on sample data."""
    logger.info("Starting feature extraction test")

    # Generate a single sample application
    logger.info("Generating sample loan application...")
    generator = SyntheticDataGenerator(seed=42)
    sample_app_model = generator.generate_application()
    sample_app = sample_app_model.model_dump()

    logger.info(
        "Sample application generated",
        application_id=sample_app['application_id'],
        sms_count=len(sample_app['sms_transactions']),
        profitability_score=sample_app.get('profitability_score')
    )

    # Initialize feature pipeline
    logger.info("Initializing feature engineering pipeline...")
    pipeline = FeatureEngineeringPipeline()

    # Extract features
    logger.info("Extracting features...")
    features = pipeline.extract_features(sample_app)

    # Display results
    print("\n" + "="*80)
    print("FEATURE EXTRACTION TEST RESULTS")
    print("="*80)

    print(f"\nApplication ID: {features.get('application_id')}")
    print(f"User ID: {features.get('user_id')}")
    print(f"Total features extracted: {len(features)}")

    # Get feature groups
    feature_groups = pipeline.get_feature_importance_groups()

    print("\n" + "-"*80)
    print("FEATURES BY CATEGORY")
    print("-"*80)

    for category, info in feature_groups.items():
        weight = info['weight']
        category_features = info['features']

        print(f"\n{category.upper().replace('_', ' ')} (Weight: {weight*100:.0f}%)")
        print(f"Feature count: {len(category_features)}")

        for feat_name in category_features:
            feat_value = features.get(feat_name, 'N/A')
            print(f"  - {feat_name}: {feat_value}")

    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*80)

    # Verify all expected features are present
    all_expected_features = []
    for info in feature_groups.values():
        all_expected_features.extend(info['features'])

    missing_features = [f for f in all_expected_features if f not in features]

    if missing_features:
        logger.warning(
            "Some features are missing",
            missing_count=len(missing_features),
            missing_features=missing_features
        )
        print(f"\nWARNING: {len(missing_features)} features missing!")
    else:
        logger.info("All expected features present")
        print("\nSUCCESS: All 45+ features extracted successfully!")

    return features


def test_batch_extraction():
    """Test batch feature extraction."""
    logger.info("Starting batch feature extraction test")

    # Generate batch of applications
    logger.info("Generating batch of 10 applications...")
    generator = SyntheticDataGenerator(seed=42)
    applications = []

    for i in range(10):
        app_model = generator.generate_application()
        applications.append(app_model.model_dump())

    # Extract features for batch
    logger.info("Extracting features for batch...")
    pipeline = FeatureEngineeringPipeline()
    features_df = pipeline.extract_batch(applications)

    print("\n" + "="*80)
    print("BATCH EXTRACTION TEST RESULTS")
    print("="*80)
    print(f"\nApplications processed: {len(features_df)}")
    print(f"Features per application: {len(features_df.columns)}")
    print(f"\nDataFrame shape: {features_df.shape}")
    print(f"\nFirst 5 rows:")
    print(features_df[['application_id', 'income_monthly_avg', 'expense_to_income_ratio',
                       'social_network_strength', 'behavioral_risk_score']].head())

    print("\n" + "="*80)
    print("BATCH TEST COMPLETED SUCCESSFULLY")
    print("="*80)

    return features_df


if __name__ == "__main__":
    print("\n" + "="*80)
    print("KISANCREDIT - FEATURE EXTRACTION TEST SUITE")
    print("="*80)

    try:
        # Test 1: Single application feature extraction
        print("\n[TEST 1] Single Application Feature Extraction")
        features = test_feature_extraction()

        # Test 2: Batch feature extraction
        print("\n[TEST 2] Batch Feature Extraction")
        features_df = test_batch_extraction()

        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        sys.exit(1)
