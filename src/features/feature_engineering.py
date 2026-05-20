"""Main Feature Engineering Pipeline.

Orchestrates all feature extractors to create complete feature vectors
for the profitability scoring model.
"""

import pandas as pd
from typing import Dict, List
import time

from .income_features import IncomeFeatureExtractor
from .expense_features import ExpenseFeatureExtractor
from .social_features import SocialFeatureExtractor
from .behavioral_features import BehavioralFeatureExtractor
from .location_features import LocationFeatureExtractor
from .discipline_features import FinancialDisciplineExtractor

from ..utils.logger import get_logger
from ..utils.config import settings

logger = get_logger(__name__)


class FeatureExtractionError(Exception):
    """Raised when a feature extractor fails on validated input.

    Individual extractors are expected to handle empty/missing inputs gracefully
    (returning zero features). This error signals an actual extraction bug
    (TypeError, KeyError on a documented field, etc.) that the API should
    surface as a 422 rather than silently masking with zeros.
    """

    def __init__(self, extractor: str, original: Exception):
        self.extractor = extractor
        self.original = original
        super().__init__(f"{extractor} extraction failed: {original.__class__.__name__}: {original}")


class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline for loan profitability scoring.

    Extracts 45+ features across 6 categories:
    - Income Stability (40% weight): 9 features
    - Expense Management (25% weight): 9 features
    - Social Network (15% weight): 8 features
    - Financial Discipline (10% weight): 6 features
    - Behavioral Patterns (10% weight): 6 features
    - Location Stability: 7 features
    """

    def __init__(self):
        """Initialize all feature extractors."""
        self.income_extractor = IncomeFeatureExtractor()
        self.expense_extractor = ExpenseFeatureExtractor()
        self.social_extractor = SocialFeatureExtractor()
        self.behavioral_extractor = BehavioralFeatureExtractor()
        self.location_extractor = LocationFeatureExtractor()
        self.discipline_extractor = FinancialDisciplineExtractor()

        logger.info("Feature engineering pipeline initialized")

    def extract_features(self, application_data: Dict) -> Dict:
        """Extract all features from a single loan application.

        Args:
            application_data: Dictionary with loan application data
                {
                    'application_id': str,
                    'sms_transactions': List[dict],
                    'contact_metadata': dict,
                    'location_pattern': dict,
                    'behavioral_data': dict,
                }

        Returns:
            Dictionary with all extracted features (45+ features)
        """
        start_time = time.time()

        features = {}

        extractors = [
            ("income", self.income_extractor.extract_features, application_data.get('sms_transactions', [])),
            ("expense", self.expense_extractor.extract_features, application_data.get('sms_transactions', [])),
            ("social", self.social_extractor.extract_features, application_data.get('contact_metadata', {})),
            ("behavioral", self.behavioral_extractor.extract_features, application_data.get('behavioral_data', {})),
            ("location", self.location_extractor.extract_features, application_data.get('location_pattern', {})),
            ("discipline", self.discipline_extractor.extract_features, application_data.get('sms_transactions', [])),
        ]

        for name, extract, payload in extractors:
            try:
                features.update(extract(payload))
            except Exception as e:
                # Per root-cause discipline: don't paper over with zeros.
                # Individual extractors must handle empty/missing inputs internally;
                # anything that bubbles out is a real bug — surface it.
                logger.error(
                    f"{name} feature extraction failed",
                    application_id=application_data.get('application_id'),
                    error_type=e.__class__.__name__,
                    error=str(e),
                    exc_info=True,
                )
                raise FeatureExtractionError(name, e) from e

        # Add metadata
        features['application_id'] = application_data.get('application_id', 'unknown')
        features['user_id'] = application_data.get('user_id', 'unknown')

        extraction_time = time.time() - start_time
        logger.info(
            "Features extracted",
            application_id=application_data.get('application_id'),
            feature_count=len(features),
            extraction_time_ms=round(extraction_time * 1000, 2)
        )

        return features

    def extract_batch(self, applications: List[Dict]) -> pd.DataFrame:
        """Extract features for a batch of applications.

        Args:
            applications: List of loan application dictionaries

        Returns:
            DataFrame with all features for all applications
        """
        start_time = time.time()

        feature_list = []
        for app in applications:
            features = self.extract_features(app)
            feature_list.append(features)

        df = pd.DataFrame(feature_list)

        processing_time = time.time() - start_time
        records_per_sec = len(applications) / processing_time if processing_time > 0 else 0

        logger.info(
            "Batch feature extraction completed",
            total_applications=len(applications),
            feature_count=len(df.columns),
            processing_time_sec=round(processing_time, 2),
            records_per_sec=round(records_per_sec, 2)
        )

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        # Generate a sample to get feature names
        sample = {
            'application_id': 'sample',
            'user_id': 'sample',
            'sms_transactions': [],
            'contact_metadata': {},
            'location_pattern': {},
            'behavioral_data': {},
        }

        features = self.extract_features(sample)
        return list(features.keys())

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get features grouped by category with their weights.

        Returns:
            Dictionary mapping category to feature names and weights
        """
        return {
            'income_stability': {
                'weight': settings.income_weight,
                'features': [
                    'income_monthly_avg',
                    'income_consistency_score',
                    'income_source_diversity',
                    'income_transaction_frequency',
                    'income_growth_trend',
                    'income_regularity',
                    'income_max_transaction',
                    'income_min_transaction',
                    'income_quartile_ratio',
                ]
            },
            'expense_management': {
                'weight': settings.expense_weight,
                'features': [
                    'expense_monthly_avg',
                    'expense_to_income_ratio',
                    'essential_expense_ratio',
                    'expense_consistency',
                    'bill_payment_regularity',
                    'savings_potential',
                    'expense_spike_count',
                    'discretionary_spending_ratio',
                    'expense_volatility',
                ]
            },
            'social_network': {
                'weight': settings.social_weight,
                'features': [
                    'social_total_contacts',
                    'social_network_strength',
                    'social_family_size',
                    'social_business_size',
                    'social_government_connections',
                    'social_communication_frequency',
                    'social_network_diversity',
                    'social_business_ratio',
                ]
            },
            'financial_discipline': {
                'weight': settings.discipline_weight,
                'features': [
                    'discipline_emi_regularity',
                    'discipline_bill_timeliness',
                    'discipline_failed_transaction_ratio',
                    'discipline_savings_behavior',
                    'discipline_credit_usage_score',
                    'discipline_overall_score',
                ]
            },
            'behavioral_patterns': {
                'weight': settings.behavioral_weight,
                'features': [
                    'behavioral_gambling_flag',
                    'behavioral_location_changes',
                    'behavioral_location_stability',
                    'behavioral_night_transaction_ratio',
                    'behavioral_financial_literacy',
                    'behavioral_risk_score',
                ]
            },
            'location_stability': {
                'weight': 0.0,  # Supportive features, not directly weighted
                'features': [
                    'location_unique_count',
                    'location_stability_score',
                    'location_avg_distance_from_home',
                    'location_travel_radius',
                    'location_area_type_urban',
                    'location_mobility_score',
                    'location_diversity_score',
                ]
            },
        }
