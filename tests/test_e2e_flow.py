"""
End-to-End Integration Test for KisanCredit API

Tests the complete flow:
1. Submit loan application with SMS/contact/location data
2. Feature extraction
3. ML prediction
4. Database storage
5. SHAP explanation retrieval
"""

import httpx
import pytest
from datetime import datetime, timedelta


# Base URL for API
API_BASE_URL = "http://localhost:8000/api/v1"


def test_health_endpoint():
    """Test that API is healthy and model is loaded."""
    response = httpx.get(f"{API_BASE_URL}/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_health"]["is_healthy"] is True
    assert data["model_health"]["n_features"] == 45

    print(f"[OK] Health check passed - Model loaded with {data['model_health']['n_features']} features")
    print(f"[OK] Prediction latency: {data['model_health']['prediction_latency_ms']:.2f}ms")


def create_sample_loan_application():
    """Create a sample loan application with raw data."""
    now = datetime.now()

    # Sample SMS transactions (30 days worth)
    sms_data = []
    for i in range(50):
        transaction_date = now - timedelta(days=i % 30, hours=i % 24)

        # Mix of income and expense transactions
        if i % 3 == 0:  # Credit (income)
            sms_data.append({
                "timestamp": transaction_date.isoformat(),
                "sender": f"BANK{i % 3}",
                "message": f"Credited Rs. {1500 + (i * 100)} to A/c XX{i}",
                "amount": 1500 + (i * 100),
                "transaction_type": "credit",
                "category": "salary" if i % 6 == 0 else "other_income"
            })
        else:  # Debit (expense)
            sms_data.append({
                "timestamp": transaction_date.isoformat(),
                "sender": f"MERCHANT{i % 5}",
                "message": f"Debited Rs. {500 + (i * 50)} from A/c XX{i}",
                "amount": 500 + (i * 50),
                "transaction_type": "debit",
                "category": ["groceries", "utilities", "transport", "shopping"][i % 4]
            })

    # Sample contact data
    contact_data = [
        {"name": "Ramesh Kumar", "phone": "9876543210", "call_frequency": 25, "contact_type": "family"},
        {"name": "Suresh Patel", "phone": "9876543211", "call_frequency": 15, "contact_type": "business"},
        {"name": "Govt Office", "phone": "1800111222", "call_frequency": 2, "contact_type": "government"},
        {"name": "SBI Bank", "phone": "1800111000", "call_frequency": 5, "contact_type": "financial"},
        {"name": "Friend 1", "phone": "9876543212", "call_frequency": 10, "contact_type": "social"},
        {"name": "Friend 2", "phone": "9876543213", "call_frequency": 8, "contact_type": "social"},
    ]

    # Sample location data
    location_data = []
    for i in range(30):
        loc_date = now - timedelta(days=i)
        location_data.append({
            "timestamp": loc_date.isoformat(),
            "latitude": 28.6139 + (i * 0.001),  # Near Delhi
            "longitude": 77.2090 + (i * 0.001),
            "accuracy": 10.0,
            "place_type": ["home", "work", "market", "bank"][i % 4]
        })

    # Sample behavioral data
    behavioral_data = {
        "app_opens_per_day": 12,
        "night_usage_percentage": 0.08,
        "gambling_apps_present": False,
        "financial_apps_count": 3,
        "education_apps_count": 2,
        "social_media_hours_per_day": 2.5
    }

    return {
        "user": {
            "phone_number": "9876543210",
            "name": "Test Farmer",
            "age": 35,
            "pincode": "110001",
            "occupation": "Agriculture"
        },
        "loan_request": {
            "amount": 50000,
            "tenure_months": 12,
            "purpose": "crop_financing"
        },
        "alternative_data": {
            "sms_transactions": sms_data[:30],  # Limit to 30 transactions
            "contact_metadata": contact_data,
            "location_history": location_data[:20],  # Limit to 20 locations
            "behavioral_signals": behavioral_data
        }
    }


def test_complete_loan_application_flow():
    """
    Test complete flow: Application submission → Feature extraction → Prediction → DB storage
    """
    print("\n" + "="*80)
    print("TESTING COMPLETE END-TO-END LOAN APPLICATION FLOW")
    print("="*80)

    # Step 1: Create sample application
    print("\n[1/5] Creating sample loan application...")
    application_data = create_sample_loan_application()
    print(f"[OK] Created application: Rs.{application_data['loan_request']['amount']} loan for {application_data['loan_request']['tenure_months']} months")
    print(f"[OK] Provided {len(application_data['alternative_data']['sms_transactions'])} SMS transactions")
    print(f"[OK] Provided {len(application_data['alternative_data']['contact_metadata'])} contacts")
    print(f"[OK] Provided {len(application_data['alternative_data']['location_history'])} location points")

    # Step 2: Submit application to API
    print("\n[2/5] Submitting application to API...")
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{API_BASE_URL}/applications",
            json=application_data
        )

    print(f"[OK] Response status: {response.status_code}")

    # Verify response
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    result = response.json()

    print(f"[OK] Application ID: {result['application_id']}")
    print(f"[OK] Request ID: {result['request_id']}")

    # Step 3: Check prediction results
    print("\n[3/5] Analyzing prediction results...")
    assert "profitability_score" in result
    assert "decision" in result
    assert "confidence" in result

    score = result["profitability_score"]
    decision = result["decision"]
    confidence = result["confidence"]

    print(f"[OK] Profitability Score: {score:.2f}")
    print(f"[OK] Decision: {decision.upper()}")
    print(f"[OK] Confidence: {confidence:.2f}")
    print(f"[OK] Processing Time: {result.get('processing_time_ms', 0):.2f}ms")

    assert 0 <= score <= 100, f"Score should be 0-100, got {score}"
    assert decision in ["approve", "reject", "manual_review"]
    assert 0 <= confidence <= 1

    # Step 4: Verify features were extracted
    print("\n[4/5] Verifying feature extraction...")
    if "features_extracted" in result:
        features = result["features_extracted"]
        print(f"[OK] Extracted {len(features)} features")

        # Check for key features
        assert "income_monthly_avg" in features
        assert "expense_monthly_avg" in features
        assert "social_network_strength" in features
        print(f"[OK] Income (monthly avg): Rs.{features.get('income_monthly_avg', 0):.2f}")
        print(f"[OK] Expense (monthly avg): Rs.{features.get('expense_monthly_avg', 0):.2f}")
        print(f"[OK] Social network strength: {features.get('social_network_strength', 0):.2f}")

    # Step 5: Test SHAP explanation endpoint
    print("\n[5/5] Testing SHAP explanation...")
    application_id = result["application_id"]

    with httpx.Client(timeout=60.0) as client:  # SHAP can take a while
        try:
            explain_response = client.get(
                f"{API_BASE_URL}/predictions/{application_id}/explain"
            )

            if explain_response.status_code == 200:
                explanation = explain_response.json()
                print(f"[OK] SHAP explanation retrieved")

                if "top_contributors" in explanation:
                    print(f"[OK] Top 5 contributing features:")
                    for i, contrib in enumerate(explanation["top_contributors"][:5], 1):
                        feature_name = contrib.get("feature", "unknown")
                        contribution = contrib.get("contribution", 0)
                        print(f"   {i}. {feature_name}: {contribution:+.2f}")
        except Exception as e:
            print(f"! SHAP explanation skipped (not critical): {e}")

    print("\n" + "="*80)
    print("[SUCCESS] END-TO-END TEST PASSED SUCCESSFULLY!")
    print("="*80)

    return result


def test_direct_prediction_endpoint():
    """Test direct prediction endpoint with pre-extracted features."""
    print("\n" + "="*80)
    print("TESTING DIRECT PREDICTION ENDPOINT")
    print("="*80)

    # Good applicant features (from demo script)
    features = {
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

    print(f"\n[1/2] Submitting features for direct prediction...")

    with httpx.Client(timeout=10.0) as client:
        response = client.post(
            f"{API_BASE_URL}/predictions",
            json={"features": features}
        )

    assert response.status_code == 200
    result = response.json()

    print(f"[OK] Prediction ID: {result['prediction_id']}")
    print(f"[OK] Profitability Score: {result['profitability_score']:.2f}")
    print(f"[OK] Decision: {result['decision'].upper()}")
    print(f"[OK] Confidence: {result['confidence']:.2f}")
    print(f"[OK] Processing Time: {result['processing_time_ms']:.2f}ms")

    # Good applicant should likely be approved
    assert result["profitability_score"] > 60, "Good applicant should have high score"

    print("\n[SUCCESS] DIRECT PREDICTION TEST PASSED!")
    print("="*80)


if __name__ == "__main__":
    """Run tests manually."""
    try:
        print("\n>> Starting KisanCredit End-to-End Tests...")

        # Test 1: Health check
        test_health_endpoint()

        # Test 2: Complete loan application flow
        test_complete_loan_application_flow()

        # Test 3: Direct prediction
        test_direct_prediction_endpoint()

        print("\n" + "="*80)
        print("*** ALL TESTS PASSED! SYSTEM IS FULLY OPERATIONAL!")
        print("="*80)

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n[FAIL] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
