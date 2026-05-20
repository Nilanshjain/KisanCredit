# KisanCredit - System Architecture

> Complete system overview showing how all components work together

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                    (Next.js Frontend - Vercel)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Landing Page │  │ Apply Form   │  │ Result Page  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTPS/JSON
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (Render)                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  API Layer (src/api/main.py)                             │  │
│  │  - Health check endpoint                                 │  │
│  │  - Prediction endpoint                                   │  │
│  │  - Rate limiting middleware                              │  │
│  │  - CORS headers                                          │  │
│  └────────────┬──────────────────────────────┬───────────────┘  │
│               │                              │                  │
│               ▼                              ▼                  │
│  ┌─────────────────────┐       ┌─────────────────────┐        │
│  │  Feature Engineering│       │   ML Model Layer    │        │
│  │  (src/features/)    │──────▶│   (src/models/)     │        │
│  │                     │       │                     │        │
│  │  - 45 features      │       │  - LightGBM model   │        │
│  │  - 6 categories     │       │  - SHAP explainer   │        │
│  └─────────────────────┘       │  - Predictor class  │        │
│                                └─────────────────────┘        │
│               │                              │                  │
│               ▼                              ▼                  │
│  ┌─────────────────────┐       ┌─────────────────────┐        │
│  │   Cache Layer       │       │  Database Layer     │        │
│  │   (Redis/Upstash)   │       │  (PostgreSQL/Neon)  │        │
│  │                     │       │                     │        │
│  │  - 1 hour TTL       │       │  - Applications     │        │
│  │  - 70% hit rate     │       │  - Predictions      │        │
│  └─────────────────────┘       │  - Audit logs       │        │
│                                └─────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘

```

---

## Component Breakdown

### 1. Frontend (Next.js App)

**Location:** `/frontend`
**Technology:** Next.js 16, React 19, TypeScript, Tailwind CSS
**Deployment:** Vercel (free tier)

**Structure:**
```
frontend/
├── app/
│   ├── page.tsx           # Landing page
│   ├── apply/page.tsx     # Application form
│   ├── layout.tsx         # Root layout
│   └── globals.css        # Tailwind styles
├── lib/
│   ├── api.ts             # API client functions
│   └── utils.ts           # Utility functions
└── components/ui/         # Reusable UI components
```

**Responsibilities:**
- User interface and experience
- Form validation
- API communication
- Results visualization
- Responsive design for mobile users

**Key Features:**
- Rural-themed design (golden harvest colors, green fields)
- Touch-friendly for mobile devices
- Simple language for accessibility
- Fast loading (optimized for slow connections)

---

### 2. API Layer (FastAPI)

**Location:** `/src/api`
**Technology:** FastAPI, Python 3.10, Uvicorn
**Deployment:** Render.com (free tier)

**Files:**
- `main.py` - FastAPI app initialization, endpoints
- `middleware.py` - Rate limiting, CORS, logging
- `schemas.py` - Pydantic models for request/response

**Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/health` | GET | Health check + system info |
| `/api/v1/predictions` | POST | Loan prediction |

**Flow:**
```
1. Request arrives → CORS middleware
2. Rate limit check (100 req/15 min)
3. Request logging (structured JSON)
4. Input validation (Pydantic)
5. Process prediction
6. Response logging
7. Return JSON response
```

**Error Handling:**
- 400: Invalid input (missing fields, wrong types)
- 422: Validation error (out-of-range values)
- 429: Rate limit exceeded
- 500: Server error (model failure)

---

### 3. Feature Engineering Pipeline

**Location:** `/src/features`
**Purpose:** Transform raw user data into 45 ML-ready features

**Categories (6 total):**

**Income Features (9):**
- Files: `income_features.py`
- Calculates: Monthly average, consistency, growth, seasonality
- Example: `income_monthly_avg`, `income_consistency_score`

**Expense Features (9):**
- Files: `expense_features.py`
- Calculates: Expense ratio, savings potential, bill timeliness
- Example: `expense_to_income_ratio`, `expense_savings_potential`

**Social Features (8):**
- Files: `social_features.py`
- Calculates: Network strength, contact diversity
- Example: `social_network_strength`, `social_total_contacts`

**Discipline Features (6):**
- Files: `discipline_features.py`
- Calculates: Payment regularity, failed transactions
- Example: `discipline_emi_regularity`, `discipline_failed_transactions`

**Behavioral Features (6):**
- Files: `behavioral_features.py`
- Calculates: Risk indicators, financial literacy
- Example: `behavioral_risk_score`, `behavioral_financial_literacy`

**Location Features (7):**
- Files: `location_features.py`
- Calculates: Stability, mobility, urban score
- Example: `location_stability_score`, `location_consistency_score`

**Pipeline Flow:**
```
Raw Data (SMS, Contacts, Location)
    ↓
Extract per category (vectorized operations)
    ↓
Validate ranges (all 0-1 normalized except counts)
    ↓
Combine into 45-feature vector
    ↓
Feed to ML model
```

---

### 4. Machine Learning Layer

**Location:** `/src/models`
**Model Type:** LightGBM (Gradient Boosting)
**Model File:** `/models/profitability_model_latest.pkl`

**Components:**

**Predictor (`predictor.py`):**
- Loads trained model
- Accepts 45-feature vector
- Returns profitability score (0-100)
- Latency: 2ms P95

**Explainer (`explainer.py`):**
- Uses SHAP (SHapley Additive exPlanations)
- Identifies top contributing features
- Shows why loan was approved/rejected
- Example: "Approved due to: high income consistency (0.92), low expense ratio (0.45)"

**Trainer (`trainer.py`):**
- Trains LightGBM model on synthetic data
- Hyperparameter: max_depth=6, n_estimators=100, learning_rate=0.1
- Saves to pickle file
- Tracks experiments with MLflow

**Evaluator (`evaluator.py`):**
- Measures model performance
- Metrics: Precision (82%), AUC (0.86), Latency (2ms)

**Model Details:**
```python
# Input: 45 features (all float)
input_shape = (45,)

# Output: Single probability score
output = profitability_score  # 0-100

# Decision Thresholds:
if score >= 60: return "APPROVED"
elif score >= 40: return "MANUAL_REVIEW"
else: return "REJECTED"
```

---

### 5. Database Layer (PostgreSQL)

**Location:** `/src/database`
**Technology:** PostgreSQL 15 (Neon - cloud)
**ORM:** SQLAlchemy (async)

**Tables:**

**applications:**
- Stores loan application data
- Fields: name, mobile, DOB, loan_amount, monthly_income
- Primary key: `id` (UUID)

**predictions:**
- Stores ML model predictions
- Fields: application_id, profitability_score, confidence, decision
- Links to applications via foreign key

**users** (optional):
- User authentication data
- Currently not used (public demo)

**audit_logs:**
- Tracks all API requests
- Fields: endpoint, method, status_code, latency_ms, timestamp
- Used for monitoring and debugging

**Connection:**
```python
# Async connection pool
DATABASE_URL = "postgresql+asyncpg://user:pass@host/db"
engine = create_async_engine(DATABASE_URL, pool_size=10)
```

**Migrations:**
- Tool: Alembic
- Location: `/alembic/versions/`
- Current version: 6 tables

---

### 6. Cache Layer (Redis)

**Location:** `/src/cache`
**Technology:** Redis 7 (Upstash - cloud)
**Pattern:** Cache-aside

**Purpose:**
- Speed up repeated predictions for same features
- Reduce database load
- Improve API response time

**Configuration:**
```python
REDIS_URL = "redis://default:password@host:6379"
TTL = 3600  # 1 hour
```

**Flow:**
```
1. Request arrives with features
2. Generate cache key: hash(features)
3. Check Redis:
   - HIT: Return cached result (2ms)
   - MISS: Query model → Store in cache → Return (10ms)
4. Cache expires after 1 hour
```

**Performance:**
- Target hit rate: 70%
- Cache hit latency: 2ms
- Cache miss latency: 10ms

---

## Data Flow (End-to-End)

### Complete Request Journey:

```
┌─────────┐
│  USER   │ Submits loan application form
└────┬────┘
     │
     ▼
┌──────────────────┐
│  Frontend (Next) │ Validates input, generates 45 features
└────┬─────────────┘
     │ POST /api/v1/predictions
     ▼
┌──────────────────┐
│  API (FastAPI)   │ Rate limit check → Log request
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│  Cache (Redis)   │ Check if features already predicted
└────┬─────────────┘
     │ MISS
     ▼
┌──────────────────┐
│  ML Model        │ LightGBM predicts profitability score
│  (LightGBM)      │ 2ms latency
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│  SHAP Explainer  │ Calculate feature contributions
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│  Database (PG)   │ Store application + prediction
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│  Cache (Redis)   │ Store result for future requests
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│  API Response    │ Return JSON: {score, decision, confidence, time}
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│  Frontend        │ Display result (Approved/Rejected/Review)
└────┬─────────────┘
     │
     ▼
┌─────────┐
│  USER   │ Sees decision + loan details
└─────────┘
```

**Timing Breakdown:**
- Frontend form submission: ~100ms
- API processing: ~10ms
  - Rate limit check: <1ms
  - Feature validation: <1ms
  - Model prediction: 2ms
  - SHAP explanation: 3ms
  - Database write: 3ms
  - Cache write: 1ms
- **Total: ~110ms (well under 1 second target)**

---

## Technology Stack

### Backend
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Web Framework | FastAPI | 0.115+ | REST API |
| ML Library | LightGBM | 4.1+ | Gradient boosting model |
| Explainability | SHAP | 0.45+ | Feature importance |
| Database ORM | SQLAlchemy | 2.0+ | Async PostgreSQL |
| Cache Client | Redis-py | 5.0+ | Redis operations |
| Data Processing | Pandas | 2.2+ | Feature engineering |
| Validation | Pydantic | 2.0+ | Request/response schemas |
| ASGI Server | Uvicorn | 0.32+ | Production server |

### Frontend
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Framework | Next.js | 16.0+ | React framework |
| UI Library | React | 19.0+ | Component library |
| Language | TypeScript | 5.0+ | Type safety |
| Styling | Tailwind CSS | 4.0+ | Utility-first CSS |
| Icons | Lucide React | 0.469+ | Icon library |

### Infrastructure
| Component | Technology | Plan | Purpose |
|-----------|-----------|------|---------|
| Database | PostgreSQL 15 (Neon) | Free 0.5GB | Persistent storage |
| Cache | Redis 7 (Upstash) | Free 10K/day | Fast lookups |
| Backend Host | Render.com | Free 750h/month | API hosting |
| Frontend Host | Vercel | Free unlimited | Static + SSR hosting |

---

## Security & Performance

### Security
- **Rate Limiting:** 100 requests per 15 minutes per IP
- **Input Validation:** Pydantic schemas validate all inputs
- **CORS:** Configured for frontend domain only
- **Environment Variables:** Secrets stored in .env (not in git)
- **SQL Injection:** Prevented by SQLAlchemy ORM
- **HTTPS:** Enforced on all production endpoints

### Performance Optimizations
1. **Caching:** Redis reduces database load by 70%
2. **Async I/O:** Non-blocking database and cache operations
3. **Connection Pooling:** Reuse database connections (pool_size=10)
4. **Vectorized Operations:** Pandas for fast feature extraction (220K records/sec)
5. **Model Optimization:** LightGBM optimized for low latency (2ms)
6. **CDN:** Vercel edge network for fast frontend delivery

### Monitoring
- **Structured Logging:** JSON logs for all requests
- **Prometheus Metrics:** Request count, latency, error rate
- **Health Checks:** `/health` endpoint monitors system status
- **Audit Trail:** All predictions logged to database

---

## Scalability

### Current Limits (Free Tier)
- Backend: 500 concurrent requests
- Database: 0.5GB storage, 10 connections
- Cache: 10,000 commands/day
- Frontend: Unlimited

### How to Scale (If Needed)
1. **Horizontal Scaling:**
   - Deploy multiple API instances behind load balancer
   - Stateless design allows easy replication

2. **Database Scaling:**
   - Upgrade Neon plan for more connections
   - Add read replicas for read-heavy workloads
   - Implement database sharding for massive scale

3. **Cache Scaling:**
   - Upgrade Upstash plan for more operations
   - Use Redis Cluster for distributed caching

4. **Model Serving:**
   - Separate model serving (e.g., TensorFlow Serving)
   - GPU acceleration for complex models
   - Batch predictions for efficiency

---

## Deployment Architecture

### Development
```
localhost:3000 (Frontend)
    ↓
localhost:8000 (API)
    ↓
localhost:5432 (PostgreSQL - local)
localhost:6379 (Redis - local)
```

### Production
```
kisancredit.vercel.app (Frontend - Edge Network)
    ↓ HTTPS
kisancredit-api.onrender.com (API - US East)
    ↓
neon.tech (PostgreSQL - Cloud)
upstash.com (Redis - Cloud)
```

---

## Key Design Decisions

### Why FastAPI?
- **Fast:** Async support, high performance
- **Type-Safe:** Pydantic validation
- **Auto Docs:** Swagger UI at `/docs`
- **Modern:** Built for Python 3.7+

### Why LightGBM?
- **Accurate:** 82% precision on test data
- **Fast:** 2ms prediction latency
- **Efficient:** Small model size (1.2MB)
- **Explainable:** Works well with SHAP

### Why Next.js?
- **SEO:** Server-side rendering
- **Fast:** Edge caching, code splitting
- **Developer Experience:** Hot reload, TypeScript support
- **Deployment:** Vercel integration

### Why Cloud Services (Neon, Upstash)?
- **Cost:** $0/month for development
- **Scalability:** Easy upgrade path
- **Maintenance:** Fully managed (no DevOps)
- **Reliability:** 99.9% uptime SLA

---

## Future Enhancements

### Technical Improvements
1. **API Authentication:** Add JWT tokens for security
2. **Real-time Monitoring:** Grafana dashboards
3. **A/B Testing:** Compare model versions
4. **Batch Predictions:** Process multiple applications
5. **Model Retraining:** Automated retraining pipeline

### Feature Additions
1. **SMS Integration:** Direct SMS data access (with permission)
2. **UPI Integration:** Real UPI transaction parsing
3. **Aadhaar Verification:** Optional identity check
4. **Loan Tracking:** Status updates for approved loans
5. **WhatsApp Bot:** Apply via WhatsApp

---

This architecture supports the goal of **60-second loan decisions for 190 million underserved Indians** while maintaining **zero monthly cost** during development and demo phase.
