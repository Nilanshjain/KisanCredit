"""ETL data processor with vectorized operations.

Target: Process 1,000 records in <2 seconds using vectorized pandas operations.
"""

import time
from typing import Optional
import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger(__name__)


class DataProcessor:
    """High-performance ETL processor using vectorized operations."""

    def __init__(self):
        """Initialize processor."""
        self.feature_weights = {
            "income": settings.income_weight,
            "expense": settings.expense_weight,
            "social": settings.social_weight,
            "discipline": settings.discipline_weight,
            "behavioral": settings.behavioral_weight
        }

    def process_batch(
        self,
        df: pd.DataFrame,
        batch_size: int = 1000
    ) -> pd.DataFrame:
        """Process dataframe in batches for optimal performance.

        Args:
            df: Input DataFrame
            batch_size: Number of records per batch (default 1000)

        Returns:
            Processed DataFrame
        """
        logger.info(f"Processing {len(df)} records in batches of {batch_size}")
        start_time = time.time()

        processed_batches = []
        num_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)

        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(df))
            batch = df.iloc[batch_start:batch_end]

            processed_batch = self._process_vectorized(batch)
            processed_batches.append(processed_batch)

            # Log progress every 10 batches
            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                elapsed = time.time() - start_time
                records_processed = batch_end
                rate = records_processed / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Batch {i+1}/{num_batches}: {records_processed}/{len(df)} "
                    f"records processed ({rate:.0f} records/sec)"
                )

        result_df = pd.concat(processed_batches, ignore_index=True)

        total_time = time.time() - start_time
        rate = len(df) / total_time if total_time > 0 else 0

        logger.info(
            f"✓ Processing complete: {len(df)} records in {total_time:.2f}s "
            f"({rate:.0f} records/sec)"
        )

        # Check if we met the target (1K records / 2 seconds)
        target_rate = 500  # records per second
        if rate >= target_rate:
            logger.info(f"✓ Performance target met: {rate:.0f} >= {target_rate} records/sec")
        else:
            logger.warning(f"⚠ Performance below target: {rate:.0f} < {target_rate} records/sec")

        return result_df

    def _process_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process batch using vectorized pandas operations.

        Args:
            df: Batch DataFrame

        Returns:
            Processed DataFrame with derived features
        """
        # Create a copy to avoid modifying original
        processed = df.copy()

        # Extract nested data using vectorized operations
        # (Feature engineering will be done in Phase 3)

        # For now, just ensure data types are optimized
        processed = self._optimize_dtypes(processed)

        return processed

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting dtypes.

        Args:
            df: Input DataFrame

        Returns:
            Memory-optimized DataFrame
        """
        # Downcast numeric columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')

        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')

        # Convert object columns to category where appropriate
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')

        return df

    def save_processed_data(
        self,
        df: pd.DataFrame,
        output_path: str,
        format: str = 'parquet'
    ) -> None:
        """Save processed data to disk.

        Args:
            df: DataFrame to save
            output_path: Output file path
            format: File format ('parquet', 'csv')
        """
        logger.info(f"Saving processed data to {output_path}")

        if format == 'parquet':
            df.to_parquet(output_path, index=False, compression='snappy')
        elif format == 'csv':
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        file_size_mb = pd.io.common.get_file_size(output_path) / (1024 ** 2)
        logger.info(f"✓ Saved {len(df)} records ({file_size_mb:.2f} MB) to {output_path}")
