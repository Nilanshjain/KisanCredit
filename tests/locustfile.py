"""Load testing script for KisanCredit API using Locust.

Run with: locust -f tests/locustfile.py --host=http://localhost:8000
Access UI: http://localhost:8089
"""

import json
import random
from locust import HttpUser, task, between


class KisanCreditUser(HttpUser):
    """Simulates a user interacting with KisanCredit API."""

    # Wait 1-3 seconds between requests
    wait_time = between(1, 3)

    def on_start(self):
        """Called when a simulated user starts."""
        self.application_ids = []

    @task(5)
    def health_check(self):
        """Check API health (most frequent operation)."""
        with self.client.get(
            "/api/v1/health",
            catch_response=True,
            name="/api/v1/health"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(3)
    def get_metrics(self):
        """Get API metrics."""
        with self.client.get(
            "/api/v1/metrics",
            catch_response=True,
            name="/api/v1/metrics"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics failed: {response.status_code}")

    @task(10)
    def submit_application(self):
        """Submit a loan application (primary workflow)."""
        payload = self._generate_application_payload()

        with self.client.post(
            "/api/v1/applications",
            json=payload,
            catch_response=True,
            name="/api/v1/applications"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "application_id" in data:
                        self.application_ids.append(data["application_id"])
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 429:
                # Rate limit hit - expected behavior
                response.success()
            else:
                response.failure(f"Application submission failed: {response.status_code}")

    @task(7)
    def predict_direct(self):
        """Make a direct prediction from features."""
        payload = {
            "application_id": f"APP_{random.randint(10000, 99999)}",
            "features": self._generate_features()
        }

        with self.client.post(
            "/api/v1/predictions",
            json=payload,
            catch_response=True,
            name="/api/v1/predictions"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                response.success()
            else:
                response.failure(f"Prediction failed: {response.status_code}")

    @task(2)
    def batch_predict(self):
        """Submit batch predictions (less frequent)."""
        # Create batch of 5-20 predictions
        batch_size = random.randint(5, 20)
        batch_payload = [
            {
                "application_id": f"APP_{random.randint(10000, 99999)}",
                "features": self._generate_features()
            }
            for _ in range(batch_size)
        ]

        with self.client.post(
            "/api/v1/predictions/batch",
            json=batch_payload,
            catch_response=True,
            name="/api/v1/predictions/batch"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                response.success()
            else:
                response.failure(f"Batch prediction failed: {response.status_code}")

    @task(1)
    def explain_prediction(self):
        """Get SHAP explanation (least frequent, CPU intensive)."""
        if not self.application_ids:
            return

        app_id = random.choice(self.application_ids)

        with self.client.get(
            f"/api/v1/predictions/{app_id}/explain",
            catch_response=True,
            name="/api/v1/predictions/{id}/explain"
        ) as response:
            if response.status_code in [200, 404]:
                # 404 is okay - application might not exist
                response.success()
            elif response.status_code == 429:
                response.success()
            else:
                response.failure(f"Explanation failed: {response.status_code}")

    def _generate_application_payload(self):
        """Generate realistic application payload."""
        return {
            "user_id": f"USER_{random.randint(10000, 99999)}",
            "loan_amount": random.randint(10000, 200000),
            "loan_purpose": random.choice([
                "Agricultural equipment",
                "Crop cultivation",
                "Livestock purchase",
                "Farm improvement",
                "Working capital"
            ]),
            "sms_transactions": self._generate_sms_transactions(),
            "contact_metadata": self._generate_contact_metadata(),
            "location_pattern": self._generate_location_pattern(),
            "behavioral_data": self._generate_behavioral_data()
        }

    def _generate_sms_transactions(self):
        """Generate SMS transaction data."""
        transactions = []
        num_transactions = random.randint(50, 200)

        for _ in range(num_transactions):
            transactions.append({
                "timestamp": "2024-01-01T12:00:00Z",
                "sender": f"SENDER{random.randint(1000, 9999)}",
                "message": f"Transaction {random.randint(100, 10000)}",
                "amount": random.uniform(100, 5000),
                "type": random.choice(["credit", "debit"])
            })

        return transactions

    def _generate_contact_metadata(self):
        """Generate contact metadata."""
        return {
            "total_contacts": random.randint(50, 500),
            "unique_contacts": random.randint(30, 300),
            "contact_frequency": random.uniform(1.0, 10.0),
            "avg_call_duration": random.uniform(30, 300)
        }

    def _generate_location_pattern(self):
        """Generate location pattern data."""
        return {
            "home_latitude": random.uniform(8.0, 35.0),
            "home_longitude": random.uniform(68.0, 97.0),
            "work_latitude": random.uniform(8.0, 35.0),
            "work_longitude": random.uniform(68.0, 97.0),
            "unique_locations": random.randint(5, 50),
            "avg_distance_km": random.uniform(1.0, 50.0)
        }

    def _generate_behavioral_data(self):
        """Generate behavioral data."""
        return {
            "app_usage_hours": random.uniform(1.0, 12.0),
            "financial_literacy_score": random.uniform(0.0, 1.0),
            "risk_taking_behavior": random.uniform(0.0, 1.0),
            "gambling_indicators": random.choice([True, False])
        }

    def _generate_features(self):
        """Generate 45 feature values."""
        return {
            f"feature_{i}": random.uniform(0.0, 100.0)
            for i in range(1, 46)
        }


class StressTestUser(KisanCreditUser):
    """Aggressive user for stress testing."""

    # Almost no wait time
    wait_time = between(0.1, 0.5)


class HeavyBatchUser(HttpUser):
    """User that primarily does batch operations."""

    wait_time = between(2, 5)

    @task
    def large_batch_predict(self):
        """Submit large batch predictions."""
        batch_size = random.randint(50, 100)  # Max allowed
        batch_payload = [
            {
                "application_id": f"APP_{random.randint(10000, 99999)}",
                "features": {f"feature_{i}": random.uniform(0.0, 100.0) for i in range(1, 46)}
            }
            for _ in range(batch_size)
        ]

        with self.client.post(
            "/api/v1/predictions/batch",
            json=batch_payload,
            catch_response=True,
            name="/api/v1/predictions/batch [large]"
        ) as response:
            if response.status_code in [200, 429]:
                response.success()
            else:
                response.failure(f"Large batch failed: {response.status_code}")
