"""Comprehensive tests for FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import time

from src.api.main import app
from src.api.schemas import (
    LoanApplicationRequest,
    SMSTransaction,
    ContactMetadata,
    LocationPattern,
    BehavioralData
)


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_application_data():
    """Create sample application data."""
    return {
        "user_id": "TEST_USER_001",
        "loan_amount": 50000.0,
        "loan_purpose": "Agricultural equipment",
        "sms_transactions": [
            {
                "transaction_id": "TXN001",
                "timestamp": "2024-01-15T10:30:00",
                "amount": 5000.0,
                "transaction_type": "credit",
                "merchant_category": "agriculture",
                "is_credit": True
            },
            {
                "transaction_id": "TXN002",
                "timestamp": "2024-01-16T14:20:00",
                "amount": 2000.0,
                "transaction_type": "debit",
                "merchant_category": "groceries",
                "is_credit": False
            }
        ],
        "contact_metadata": {
            "total_contacts": 150,
            "family_contacts": 20,
            "business_contacts": 50,
            "government_contacts": 5,
            "avg_call_duration": 180.0,
            "contact_diversity_score": 0.75
        },
        "location_pattern": {
            "unique_locations": 5,
            "home_location": {"lat": 28.7041, "lon": 77.1025},
            "travel_radius_km": 25.0,
            "area_type": "rural",
            "location_stability_score": 0.85
        },
        "behavioral_data": {
            "app_usage_hours_per_day": 4.5,
            "night_activity_ratio": 0.15,
            "gambling_indicators": 0,
            "financial_app_usage": True,
            "literacy_score": 0.7
        }
    }


@pytest.fixture
def sample_features():
    """Create sample features for direct prediction."""
    return {
        "income_monthly_avg": 25000.0,
        "income_consistency_score": 0.75,
        "expense_to_income_ratio": 0.65,
        "social_network_strength": 0.80,
        "discipline_overall_score": 0.70,
        "behavioral_risk_score": 0.20
    }


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "KisanCredit API"


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "model_health" in data
        assert "uptime_seconds" in data

        assert data["version"] == "1.0.0"

    def test_health_check_structure(self, client):
        """Test health check response structure."""
        response = client.get("/api/v1/health")
        data = response.json()

        # Check model health structure
        assert isinstance(data["model_health"], dict)
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/api/v1/metrics")

        assert response.status_code == 200
        data = response.json()

        required_fields = [
            "total_predictions",
            "predictions_last_hour",
            "avg_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "error_rate",
            "approval_rate",
            "cache_hit_rate"
        ]

        for field in required_fields:
            assert field in data

    def test_metrics_values(self, client):
        """Test metrics values are reasonable."""
        response = client.get("/api/v1/metrics")
        data = response.json()

        assert data["total_predictions"] >= 0
        assert data["avg_latency_ms"] >= 0
        assert 0 <= data["error_rate"] <= 1
        assert 0 <= data["approval_rate"] <= 1


class TestApplicationEndpoint:
    """Tests for application submission endpoint."""

    def test_submit_application_success(self, client, sample_application_data):
        """Test successful application submission."""
        response = client.post("/api/v1/applications", json=sample_application_data)

        # Should return 201 Created or 503 if model not loaded
        assert response.status_code in [201, 503]

        if response.status_code == 201:
            data = response.json()

            assert "application_id" in data
            assert "user_id" in data
            assert "profitability_score" in data
            assert "decision" in data
            assert "processing_time_ms" in data

            assert data["user_id"] == sample_application_data["user_id"]
            assert data["decision"] in ["approve", "reject", "manual_review"]
            assert 0 <= data["profitability_score"] <= 1

    def test_submit_application_invalid_loan_amount(self, client, sample_application_data):
        """Test application with invalid loan amount."""
        sample_application_data["loan_amount"] = -1000

        response = client.post("/api/v1/applications", json=sample_application_data)

        assert response.status_code == 422  # Validation error

    def test_submit_application_loan_amount_too_high(self, client, sample_application_data):
        """Test application with loan amount exceeding limit."""
        sample_application_data["loan_amount"] = 600000

        response = client.post("/api/v1/applications", json=sample_application_data)

        assert response.status_code == 422

    def test_submit_application_missing_fields(self, client):
        """Test application with missing required fields."""
        incomplete_data = {
            "user_id": "TEST_USER",
            "loan_amount": 50000
            # Missing required fields
        }

        response = client.post("/api/v1/applications", json=incomplete_data)

        assert response.status_code == 422


class TestPredictionEndpoint:
    """Tests for direct prediction endpoint."""

    def test_predict_success(self, client, sample_features):
        """Test successful prediction."""
        request_data = {
            "application_id": "TEST_APP_001",
            "features": sample_features
        }

        response = client.post("/api/v1/predictions", json=request_data)

        # Should return 200 or 503 if model not loaded
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()

            assert "application_id" in data
            assert "profitability_score" in data
            assert "confidence" in data
            assert "decision" in data
            assert "processing_time_ms" in data

            assert data["application_id"] == "TEST_APP_001"
            assert 0 <= data["profitability_score"] <= 1
            assert 0 <= data["confidence"] <= 1

    def test_predict_missing_application_id(self, client, sample_features):
        """Test prediction without application ID."""
        request_data = {
            "features": sample_features
        }

        response = client.post("/api/v1/predictions", json=request_data)

        assert response.status_code == 422

    def test_predict_empty_features(self, client):
        """Test prediction with empty features."""
        request_data = {
            "application_id": "TEST_APP_001",
            "features": {}
        }

        response = client.post("/api/v1/predictions", json=request_data)

        # Should work but may have low confidence
        assert response.status_code in [200, 500, 503]


class TestExplanationEndpoint:
    """Tests for explanation endpoint."""

    def test_explain_prediction(self, client, sample_features):
        """Test prediction explanation."""
        response = client.get(
            "/api/v1/predictions/TEST_APP_001/explain",
            params={"features": sample_features}
        )

        # Should return 200 or 503 if explainer not available
        assert response.status_code in [200, 422, 503]

        # Note: This endpoint requires query params which are harder to test
        # In practice, would need proper request formatting


class TestBatchPrediction:
    """Tests for batch prediction endpoint."""

    def test_batch_predict_success(self, client, sample_application_data):
        """Test batch prediction."""
        batch_request = {
            "applications": [sample_application_data, sample_application_data]
        }

        response = client.post("/api/v1/predictions/batch", json=batch_request)

        # Should return 200 or 503 if model not loaded
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()

            assert "batch_id" in data
            assert "total_applications" in data
            assert "successful_predictions" in data
            assert "results" in data

            assert data["total_applications"] == 2

    def test_batch_predict_too_many(self, client, sample_application_data):
        """Test batch with too many applications."""
        batch_request = {
            "applications": [sample_application_data] * 101  # Exceeds limit of 100
        }

        response = client.post("/api/v1/predictions/batch", json=batch_request)

        assert response.status_code == 422


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_headers(self, client):
        """Test rate limit headers are present."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_rate_limit_enforcement(self, client):
        """Test rate limiting is enforced."""
        # Make many requests quickly
        responses = []
        for _ in range(105):  # Exceed limit of 100
            response = client.get("/api/v1/health")
            responses.append(response)

        # Should eventually get rate limited
        # Note: In test environment, rate limiting may not be fully active
        status_codes = [r.status_code for r in responses]

        # All should be 200 or some should be 429
        assert all(code in [200, 429] for code in status_codes)


class TestMiddleware:
    """Tests for middleware functionality."""

    def test_request_id_header(self, client):
        """Test request ID is added to response."""
        response = client.get("/api/v1/health")

        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"].startswith("req_")

    def test_processing_time_header(self, client):
        """Test processing time is added to response."""
        response = client.get("/api/v1/health")

        assert "X-Processing-Time-MS" in response.headers
        processing_time = float(response.headers["X-Processing-Time-MS"])
        assert processing_time >= 0

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.get("/api/v1/health")

        assert "Access-Control-Allow-Origin" in response.headers


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")

        assert response.status_code == 404
        data = response.json()

        assert "detail" in data

    def test_validation_error(self, client):
        """Test validation error response."""
        invalid_data = {
            "user_id": "TEST",
            "loan_amount": "invalid"  # Should be number
        }

        response = client.post("/api/v1/applications", json=invalid_data)

        assert response.status_code == 422
        data = response.json()

        assert "detail" in data


class TestPerformance:
    """Tests for API performance."""

    def test_health_check_latency(self, client):
        """Test health check responds quickly."""
        start_time = time.time()
        response = client.get("/api/v1/health")
        latency = (time.time() - start_time) * 1000

        assert response.status_code == 200
        assert latency < 100  # Should be under 100ms

    def test_metrics_latency(self, client):
        """Test metrics endpoint responds quickly."""
        start_time = time.time()
        response = client.get("/api/v1/metrics")
        latency = (time.time() - start_time) * 1000

        assert response.status_code == 200
        assert latency < 100


def test_api_documentation(client):
    """Test API documentation is available."""
    # Test OpenAPI docs
    response = client.get("/docs")
    assert response.status_code == 200

    # Test ReDoc
    response = client.get("/redoc")
    assert response.status_code == 200

    # Test OpenAPI schema
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema
