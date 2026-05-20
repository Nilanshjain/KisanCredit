# KisanCredit - API Reference

> Complete API documentation with examples and error codes

**Base URL (Local):** `http://localhost:8000/api/v1`
**Base URL (Production):** `https://kisancredit-api.onrender.com/api/v1`

---

## Table of Contents
1. [Authentication](#authentication)
2. [Health Check](#health-check)
3. [Predictions](#predictions)
4. [Error Codes](#error-codes)
5. [Rate Limiting](#rate-limiting)

---

## Authentication

**Current Status:** No authentication required (public demo)

**Future:** JWT tokens for production
```http
Authorization: Bearer <token>
```

---

## Health Check

### GET `/health`

Check if the API is running and get system information.

#### Request
```bash
curl http://localhost:8000/api/v1/health
```

#### Response (200 OK)
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_health": "operational",
  "model_path": "models/profitability_model_latest.pkl",
  "feature_count": 45,
  "uptime_seconds": 3456.78,
  "timestamp": "2025-10-22T10:30:45.123456"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Overall API status (`healthy` or `unhealthy`) |
| `model_loaded` | boolean | Whether ML model is loaded in memory |
| `model_health` | string | Model status (`operational`, `degraded`, `failed`) |
| `model_path` | string | Path to loaded model file |
| `feature_count` | integer | Number of features model expects (should be 45) |
| `uptime_seconds` | float | Seconds since API started |
| `timestamp` | string | Current server time (ISO 8601) |

#### Use Cases
- **Monitoring:** Check if API is responsive
- **Debugging:** Verify model is loaded correctly
- **Load Balancer:** Health check endpoint for auto-scaling

---

## Predictions

### POST `/predictions`

Predict loan profitability based on applicant features.

#### Request

**Headers:**
```http
Content-Type: application/json
```

**Body:**
```json
{
  "features": {
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
}
```

#### Complete cURL Example
```bash
curl -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
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
  }'
```

#### Response (200 OK)
```json
{
  "profitability_score": 78.45,
  "confidence": 0.92,
  "processing_time_ms": 2.34,
  "timestamp": "2025-10-22T10:35:12.456789"
}
```

#### Response Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `profitability_score` | float | 0-100 | Predicted loan profitability (higher = better) |
| `confidence` | float | 0-1 | Model confidence in prediction |
| `processing_time_ms` | float | >0 | Time taken for prediction (milliseconds) |
| `timestamp` | string | ISO 8601 | When prediction was made |

#### Decision Thresholds

| Score Range | Decision | Action |
|-------------|----------|--------|
| 60-100 | **APPROVED** | Auto-approve loan |
| 40-59 | **MANUAL_REVIEW** | Send to human reviewer |
| 0-39 | **REJECTED** | Auto-reject loan |

#### Example: Approved Application
```json
{
  "profitability_score": 78.45,  // > 60 → APPROVED
  "confidence": 0.92,             // High confidence
  "processing_time_ms": 2.34,
  "timestamp": "2025-10-22T10:35:12.456789"
}
```

#### Example: Manual Review
```json
{
  "profitability_score": 52.10,  // 40-59 → MANUAL_REVIEW
  "confidence": 0.75,             // Moderate confidence
  "processing_time_ms": 3.12,
  "timestamp": "2025-10-22T10:36:45.123456"
}
```

#### Example: Rejected
```json
{
  "profitability_score": 28.67,  // < 40 → REJECTED
  "confidence": 0.88,             // High confidence in rejection
  "processing_time_ms": 2.89,
  "timestamp": "2025-10-22T10:37:22.789012"
}
```

---

## Feature Requirements

All 45 features are **required**. Missing fields will result in 422 Validation Error.

### Feature Categories

**Income Features (9):**
- `income_monthly_avg` (float, > 0): Average monthly income in ₹
- `income_consistency_score` (float, 0-1): How consistent income is
- `income_growth_trend` (float, -1 to 1): Income growth rate
- `income_source_diversity` (int, 1-10): Number of income sources
- `income_credit_ratio` (float, 0-1): Creditworthiness indicator
- `income_seasonal_variance` (float, 0-1): Income seasonality
- `income_regularity_score` (float, 0-1): Payment regularity
- `income_upi_percentage` (float, 0-1): % of income via UPI
- `income_largest_transaction` (float, > 0): Largest single income

**Expense Features (9):**
- `expense_monthly_avg` (float, > 0): Average monthly expenses in ₹
- `expense_to_income_ratio` (float, 0-1): Expenses / Income
- `expense_essential_ratio` (float, 0-1): % essential expenses
- `expense_luxury_ratio` (float, 0-1): % luxury expenses
- `expense_savings_potential` (float, 0-1): Savings capacity
- `expense_debt_burden` (float, 0-1): Debt payment ratio
- `expense_volatility` (float, 0-1): Expense consistency
- `expense_category_diversity` (int, 1-20): Expense categories
- `expense_bill_timeliness` (float, 0-1): On-time bill payment

**Social Features (8):**
- `social_network_strength` (float, 0-1): Contact network quality
- `social_total_contacts` (int, >= 0): Total phone contacts
- `social_family_size` (int, >= 0): Family contacts
- `social_business_contacts` (int, >= 0): Business contacts
- `social_government_contacts` (int, >= 0): Govt/bank contacts
- `social_communication_frequency` (int, >= 0): Calls/msgs per month
- `social_contact_diversity` (float, 0-1): Network diversity
- `social_network_depth` (int, 1-10): Connection levels

**Discipline Features (6):**
- `discipline_overall_score` (float, 0-1): Financial discipline
- `discipline_emi_regularity` (float, 0-1): Loan payment history
- `discipline_bill_payment_score` (float, 0-1): Bill payment score
- `discipline_failed_transactions` (int, >= 0): Failed transaction count
- `discipline_overdraft_frequency` (int, >= 0): Overdraft count
- `discipline_savings_consistency` (float, 0-1): Savings regularity

**Behavioral Features (6):**
- `behavioral_risk_score` (float, 0-1): Risk indicator (higher = riskier)
- `behavioral_gambling_indicator` (int, 0 or 1): Gambling detected (binary)
- `behavioral_location_changes` (int, >= 0): Residence changes
- `behavioral_night_transaction_ratio` (float, 0-1): % night transactions
- `behavioral_financial_literacy` (float, 0-1): Financial knowledge
- `behavioral_app_usage_score` (float, 0-10): Digital engagement

**Location Features (7):**
- `location_stability_score` (float, 0-1): Location consistency
- `location_mobility_score` (float, 0-1): Travel frequency
- `location_travel_frequency` (int, >= 0): Trips per month
- `location_distance_from_center` (float, >= 0): Distance from urban center (km)
- `location_urban_score` (float, 0-1): Urban vs rural
- `location_unique_places` (int, >= 0): Unique locations visited
- `location_consistency_score` (float, 0-1): Location pattern regularity

---

## Error Codes

### 400 Bad Request
**Cause:** Malformed JSON or missing `features` key

**Example:**
```json
{
  "detail": "Request body must contain 'features' field"
}
```

**How to Fix:** Ensure JSON is valid and contains `features` object.

---

### 422 Validation Error
**Cause:** Invalid feature values (missing fields, wrong types, out of range)

**Example Response:**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "features", "income_monthly_avg"],
      "msg": "Field required",
      "input": {...}
    },
    {
      "type": "float_type",
      "loc": ["body", "features", "income_consistency_score"],
      "msg": "Input should be a valid number",
      "input": "invalid"
    }
  ]
}
```

**Common Causes:**
1. Missing feature (forgot to include a field)
2. Wrong type (sent string instead of number)
3. Out of range (score > 1.0 or < 0)

**How to Fix:**
- Check all 45 features are present
- Verify types (numbers not strings)
- Ensure scores are between 0-1
- Ensure counts are >= 0

---

### 429 Rate Limit Exceeded
**Cause:** Too many requests from same IP

**Response:**
```json
{
  "detail": "Rate limit exceeded: 100 requests per 15 minutes"
}
```

**Current Limit:** 100 requests per 15 minutes per IP

**How to Fix:**
- Wait 15 minutes
- Implement client-side rate limiting
- For production, contact for higher limits

---

### 500 Internal Server Error
**Cause:** Server or model error

**Response:**
```json
{
  "detail": "Internal server error processing prediction"
}
```

**Common Causes:**
- Model failed to load
- Database connection lost
- Out of memory

**How to Fix:**
- Check `/health` endpoint to verify model status
- Report to support if persistent

---

## Rate Limiting

### Current Limits
- **Per IP:** 100 requests per 15 minutes
- **Sliding Window:** Requests tracked over rolling 15-minute period

### Headers
Rate limit information is included in response headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1698765432
```

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Maximum requests allowed in window |
| `X-RateLimit-Remaining` | Requests remaining in current window |
| `X-RateLimit-Reset` | Unix timestamp when limit resets |

### Best Practices
1. **Check headers:** Monitor remaining requests
2. **Implement backoff:** Exponential backoff on 429 errors
3. **Cache results:** Don't repeat identical requests
4. **Batch processing:** For multiple predictions, space them out

---

## CORS (Cross-Origin Requests)

### Allowed Origins
- Development: `http://localhost:3000`
- Production: `https://kisancredit.vercel.app`

### Allowed Methods
- GET, POST, OPTIONS

### Allowed Headers
- Content-Type, Authorization

### Example Preflight Request
```http
OPTIONS /api/v1/predictions HTTP/1.1
Origin: http://localhost:3000
Access-Control-Request-Method: POST
Access-Control-Request-Headers: Content-Type
```

**Response:**
```http
HTTP/1.1 200 OK
Access-Control-Allow-Origin: http://localhost:3000
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

---

## Code Examples

### JavaScript/TypeScript (Frontend)
```typescript
async function predictLoan(features: Features) {
  try {
    const response = await fetch('http://localhost:8000/api/v1/predictions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ features }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Prediction failed:', error);
    throw error;
  }
}
```

### Python
```python
import requests
import json

def predict_loan(features: dict) -> dict:
    url = "http://localhost:8000/api/v1/predictions"
    headers = {"Content-Type": "application/json"}
    payload = {"features": features}

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    return response.json()

# Example usage
features = {
    "income_monthly_avg": 35000,
    "income_consistency_score": 0.85,
    # ... (all 45 features)
}

result = predict_loan(features)
print(f"Score: {result['profitability_score']}")
```

### cURL (Testing)
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Prediction (use @ to load from file)
curl -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  -d @features.json
```

---

## Testing Tips

### 1. Start with Health Check
Always verify API is running:
```bash
curl http://localhost:8000/api/v1/health
```

### 2. Use Test Features
Create a `test_features.json` file with valid features:
```json
{
  "features": {
    "income_monthly_avg": 30000,
    "income_consistency_score": 0.8,
    ...
  }
}
```

### 3. Check Response Time
Measure API latency:
```bash
time curl -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  -d @test_features.json
```

Target: < 100ms total

### 4. Test Error Cases
- Missing features: Remove a field
- Invalid types: Send string instead of number
- Out of range: Use score > 1.0

---

## API Versioning

**Current Version:** `v1`

Future versions will be served at different paths:
- `/api/v1/...` (current)
- `/api/v2/...` (future)

Backwards compatibility maintained for 6 months after new version release.

---

## Support

**Issues:** https://github.com/yourusername/KisanCredit/issues
**Email:** support@kisancredit.com (production)

**Response Times:**
- P0 (API down): 1 hour
- P1 (degraded): 4 hours
- P2 (feature request): 2 days
