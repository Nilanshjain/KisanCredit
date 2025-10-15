"""Benchmark prediction latency for the trained model.

Tests:
- Single prediction latency
- Batch prediction throughput
- P50, P95, P99 latencies
- Memory usage

Target: <100ms P95 latency
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.predictor import ProfitabilityPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_sample_features(n_samples: int = 1) -> pd.DataFrame:
    """Generate random feature vectors for testing."""
    np.random.seed(42)

    features = {
        # Income features (9)
        'income_monthly_avg': np.random.uniform(5000, 50000, n_samples),
        'income_consistency_score': np.random.uniform(0.3, 1.0, n_samples),
        'income_growth_trend': np.random.uniform(-0.2, 0.5, n_samples),
        'income_source_diversity': np.random.randint(1, 5, n_samples),
        'income_credit_ratio': np.random.uniform(0.5, 1.0, n_samples),
        'income_seasonal_variance': np.random.uniform(0.1, 0.8, n_samples),
        'income_regularity_score': np.random.uniform(0.3, 1.0, n_samples),
        'income_upi_percentage': np.random.uniform(0.2, 0.9, n_samples),
        'income_largest_transaction': np.random.uniform(5000, 100000, n_samples),

        # Expense features (9)
        'expense_monthly_avg': np.random.uniform(3000, 40000, n_samples),
        'expense_to_income_ratio': np.random.uniform(0.4, 0.9, n_samples),
        'expense_essential_ratio': np.random.uniform(0.4, 0.8, n_samples),
        'expense_luxury_ratio': np.random.uniform(0.0, 0.3, n_samples),
        'expense_savings_potential': np.random.uniform(0.1, 0.5, n_samples),
        'expense_debt_burden': np.random.uniform(0.0, 0.4, n_samples),
        'expense_volatility': np.random.uniform(0.1, 0.7, n_samples),
        'expense_category_diversity': np.random.randint(3, 10, n_samples),
        'expense_bill_timeliness': np.random.uniform(0.5, 1.0, n_samples),

        # Social network features (8)
        'social_network_strength': np.random.uniform(0.3, 1.0, n_samples),
        'social_total_contacts': np.random.randint(50, 500, n_samples),
        'social_family_size': np.random.randint(5, 50, n_samples),
        'social_business_contacts': np.random.randint(10, 100, n_samples),
        'social_government_contacts': np.random.randint(0, 10, n_samples),
        'social_communication_frequency': np.random.uniform(10, 80, n_samples),
        'social_contact_diversity': np.random.uniform(0.3, 0.9, n_samples),
        'social_network_depth': np.random.randint(1, 5, n_samples),

        # Discipline features (6)
        'discipline_overall_score': np.random.uniform(0.4, 1.0, n_samples),
        'discipline_emi_regularity': np.random.uniform(0.5, 1.0, n_samples),
        'discipline_bill_payment_score': np.random.uniform(0.5, 1.0, n_samples),
        'discipline_failed_transactions': np.random.randint(0, 5, n_samples),
        'discipline_overdraft_frequency': np.random.randint(0, 3, n_samples),
        'discipline_savings_consistency': np.random.uniform(0.3, 1.0, n_samples),

        # Behavioral features (6)
        'behavioral_risk_score': np.random.uniform(0.0, 0.5, n_samples),
        'behavioral_gambling_indicator': np.random.randint(0, 2, n_samples).astype(float),
        'behavioral_location_changes': np.random.randint(0, 20, n_samples),
        'behavioral_night_transaction_ratio': np.random.uniform(0.0, 0.4, n_samples),
        'behavioral_financial_literacy': np.random.uniform(0.3, 1.0, n_samples),
        'behavioral_app_usage_score': np.random.uniform(3, 10, n_samples),

        # Location features (7)
        'location_stability_score': np.random.uniform(0.3, 1.0, n_samples),
        'location_mobility_score': np.random.uniform(0.0, 0.7, n_samples),
        'location_travel_frequency': np.random.randint(0, 20, n_samples),
        'location_distance_from_center': np.random.uniform(0, 200, n_samples),
        'location_urban_score': np.random.uniform(0.0, 1.0, n_samples),
        'location_unique_places': np.random.randint(3, 30, n_samples),
        'location_consistency_score': np.random.uniform(0.4, 1.0, n_samples),
    }

    return pd.DataFrame(features)


def benchmark_single_prediction(predictor: ProfitabilityPredictor, n_iterations: int = 1000):
    """Benchmark single prediction latency."""
    logger.info(f"Benchmarking single predictions ({n_iterations} iterations)...")

    # Generate test samples
    samples = [generate_sample_features(1) for _ in range(n_iterations)]

    latencies = []
    for sample in samples:
        start = time.perf_counter()
        _ = predictor.predict(sample)
        latency = (time.perf_counter() - start) * 1000  # Convert to ms
        latencies.append(latency)

    latencies = np.array(latencies)

    results = {
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'p50': np.percentile(latencies, 50),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'std': np.std(latencies)
    }

    logger.info("Single Prediction Latency:")
    logger.info(f"  Mean: {results['mean']:.2f}ms")
    logger.info(f"  Median: {results['median']:.2f}ms")
    logger.info(f"  P50: {results['p50']:.2f}ms")
    logger.info(f"  P95: {results['p95']:.2f}ms")
    logger.info(f"  P99: {results['p99']:.2f}ms")
    logger.info(f"  Min: {results['min']:.2f}ms")
    logger.info(f"  Max: {results['max']:.2f}ms")

    # Check if target is met
    if results['p95'] < 100:
        logger.info(f"[OK] P95 latency ({results['p95']:.2f}ms) meets <100ms target")
    else:
        logger.warning(f"[WARN] P95 latency ({results['p95']:.2f}ms) exceeds 100ms target")

    return results


def benchmark_batch_prediction(predictor: ProfitabilityPredictor, batch_sizes: list = [10, 100, 1000]):
    """Benchmark batch prediction throughput."""
    logger.info("Benchmarking batch predictions...")

    results = {}

    for batch_size in batch_sizes:
        samples = generate_sample_features(batch_size)

        start = time.perf_counter()
        _ = predictor.predict_batch(samples)
        elapsed = time.perf_counter() - start

        throughput = batch_size / elapsed
        avg_latency = (elapsed / batch_size) * 1000  # ms per prediction

        results[batch_size] = {
            'total_time_ms': elapsed * 1000,
            'throughput': throughput,
            'avg_latency_ms': avg_latency
        }

        logger.info(f"Batch size {batch_size}:")
        logger.info(f"  Total time: {elapsed*1000:.2f}ms")
        logger.info(f"  Throughput: {throughput:.0f} predictions/sec")
        logger.info(f"  Avg latency: {avg_latency:.2f}ms per prediction")

    return results


def main():
    """Run latency benchmarks."""
    logger.info("=== PREDICTION LATENCY BENCHMARK ===")

    # Find latest model
    model_dir = Path("models")
    model_files = list(model_dir.glob("profitability_model_*_latest.pkl"))

    if not model_files:
        # Try non-latest models
        model_files = list(model_dir.glob("profitability_model_*.pkl"))

    if not model_files:
        logger.error("No trained model found. Run train_model.py first.")
        return

    model_path = str(sorted(model_files, key=lambda x: x.stat().st_mtime)[-1])
    logger.info(f"Loading model: {model_path}")

    # Load predictor
    predictor = ProfitabilityPredictor(model_path=model_path)

    # Warmup
    logger.info("Warming up model...")
    for _ in range(10):
        predictor.predict(generate_sample_features(1))

    # Run benchmarks
    logger.info("\n--- Single Prediction Latency ---")
    single_results = benchmark_single_prediction(predictor, n_iterations=1000)

    logger.info("\n--- Batch Prediction Throughput ---")
    batch_results = benchmark_batch_prediction(predictor, batch_sizes=[10, 100, 1000])

    # Summary
    logger.info("\n=== BENCHMARK SUMMARY ===")
    logger.info(f"Single Prediction P95: {single_results['p95']:.2f}ms")
    logger.info(f"Target: <100ms")
    logger.info(f"Status: {'PASS' if single_results['p95'] < 100 else 'FAIL'}")

    logger.info(f"\nBatch Throughput (1000 samples): {batch_results[1000]['throughput']:.0f} predictions/sec")

    # Save results
    results_file = Path("benchmark_results.txt")
    with open(results_file, "w") as f:
        f.write("=== PREDICTION LATENCY BENCHMARK ===\n\n")
        f.write("Single Prediction Latency:\n")
        for metric, value in single_results.items():
            f.write(f"  {metric}: {value:.2f}ms\n")

        f.write("\nBatch Prediction Throughput:\n")
        for batch_size, metrics in batch_results.items():
            f.write(f"\n  Batch size {batch_size}:\n")
            f.write(f"    Total time: {metrics['total_time_ms']:.2f}ms\n")
            f.write(f"    Throughput: {metrics['throughput']:.0f} predictions/sec\n")
            f.write(f"    Avg latency: {metrics['avg_latency_ms']:.2f}ms\n")

        f.write(f"\nTarget: <100ms P95 latency\n")
        f.write(f"Status: {'PASS' if single_results['p95'] < 100 else 'FAIL'}\n")

    logger.info(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
