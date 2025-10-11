"""Data validation and quality checks."""

import time
from typing import List, Tuple
import pandas as pd
from pydantic import ValidationError

from src.pipeline.schemas import LoanApplication, DataQualityMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validate loan application data using Pydantic schemas."""

    def __init__(self):
        """Initialize validator."""
        self.validation_errors: List[str] = []

    def validate_record(self, record: dict) -> Tuple[bool, str]:
        """Validate a single record.

        Args:
            record: Dictionary containing loan application data

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            LoanApplication(**record)
            return True, ""
        except ValidationError as e:
            error_msg = f"Validation error: {str(e)}"
            return False, error_msg

    def validate_dataframe(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Validate entire dataframe with quality metrics.

        Args:
            df: DataFrame containing loan applications

        Returns:
            DataQualityMetrics with validation results
        """
        start_time = time.time()
        logger.info(f"Starting validation of {len(df)} records")

        valid_records = 0
        invalid_records = 0
        validation_errors = []

        for idx, row in df.iterrows():
            is_valid, error_msg = self.validate_record(row.to_dict())
            if is_valid:
                valid_records += 1
            else:
                invalid_records += 1
                validation_errors.append(f"Row {idx}: {error_msg}")

                # Limit error messages to first 100
                if len(validation_errors) >= 100:
                    validation_errors.append("... (additional errors truncated)")
                    break

        # Calculate null counts
        null_counts = df.isnull().sum().to_dict()

        processing_time = time.time() - start_time

        metrics = DataQualityMetrics(
            total_records=len(df),
            valid_records=valid_records,
            invalid_records=invalid_records,
            null_counts=null_counts,
            validation_errors=validation_errors,
            processing_time_seconds=round(processing_time, 2)
        )

        logger.info(
            f"Validation complete: {valid_records}/{len(df)} valid "
            f"({metrics.quality_score:.2f}% quality score)"
        )

        return metrics

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe by removing invalid records.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame with only valid records
        """
        logger.info(f"Cleaning dataframe with {len(df)} records")

        valid_rows = []
        for idx, row in df.iterrows():
            is_valid, _ = self.validate_record(row.to_dict())
            if is_valid:
                valid_rows.append(row)

        cleaned_df = pd.DataFrame(valid_rows)
        logger.info(
            f"Cleaning complete: {len(cleaned_df)}/{len(df)} records retained "
            f"({len(df) - len(cleaned_df)} removed)"
        )

        return cleaned_df
