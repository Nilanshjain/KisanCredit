"""Script to generate synthetic loan application dataset.

Usage:
    python scripts/generate_data.py --num-applications 100000
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.data_generator import SyntheticDataGenerator
from src.pipeline.data_validator import DataValidator
from src.pipeline.data_processor import DataProcessor
from src.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Generate synthetic dataset with validation and processing."""
    parser = argparse.ArgumentParser(description="Generate synthetic loan data")
    parser.add_argument(
        "--num-applications",
        type=int,
        default=100000,
        help="Number of applications to generate (default: 100000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Output directory (default: data/synthetic)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("KisanCredit Synthetic Data Generation")
    logger.info("="*60)
    logger.info(f"Number of applications: {args.num_applications}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("="*60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate synthetic data
    logger.info("\n[Step 1/4] Generating synthetic data...")
    generator = SyntheticDataGenerator(seed=args.seed)
    df = generator.generate_dataset(num_applications=args.num_applications)

    raw_output = output_dir / "loan_applications_raw.parquet"
    generator.save_dataset(df, str(raw_output))

    # Step 2: Validate data
    logger.info("\n[Step 2/4] Validating data quality...")
    validator = DataValidator()
    metrics = validator.validate_dataframe(df)

    logger.info(f"Data Quality Metrics:")
    logger.info(f"  Total records: {metrics.total_records}")
    logger.info(f"  Valid records: {metrics.valid_records}")
    logger.info(f"  Invalid records: {metrics.invalid_records}")
    logger.info(f"  Quality score: {metrics.quality_score:.2f}%")
    logger.info(f"  Processing time: {metrics.processing_time_seconds}s")

    # Clean data if needed
    if metrics.invalid_records > 0:
        logger.warning(f"Found {metrics.invalid_records} invalid records, cleaning...")
        df = validator.clean_dataframe(df)

    # Step 3: Process data with vectorized operations
    logger.info("\n[Step 3/4] Processing data (vectorized operations)...")
    processor = DataProcessor()
    processed_df = processor.process_batch(df, batch_size=1000)

    processed_output = output_dir / "loan_applications_processed.parquet"
    processor.save_processed_data(processed_df, str(processed_output))

    # Step 4: Generate summary statistics
    logger.info("\n[Step 4/4] Generating summary statistics...")
    logger.info(f"\nDataset Summary:")
    logger.info(f"  Total applications: {len(processed_df)}")
    logger.info(f"  Approved: {processed_df['approved'].sum()}")
    logger.info(f"  Approval rate: {processed_df['approved'].mean()*100:.2f}%")
    logger.info(f"  Avg profitability score: {processed_df['profitability_score'].mean():.2f}")
    logger.info(f"  Avg loan amount: ₹{processed_df['loan_amount_requested'].mean():.2f}")
    logger.info(f"  Avg monthly income: ₹{processed_df['monthly_income'].mean():.2f}")

    logger.info("\n" + "="*60)
    logger.info("✓ Data generation complete!")
    logger.info("="*60)
    logger.info(f"Raw data: {raw_output}")
    logger.info(f"Processed data: {processed_output}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
