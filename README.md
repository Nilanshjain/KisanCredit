# KisanCredit - AI Loan Underwriting Platform

Production-grade AI-powered loan underwriting system for rural India, processing credit applications in under 60 seconds using alternative data sources.

## Project Structure

```
KisanCredit/
├── src/
│   ├── pipeline/      # Data ETL pipeline
│   ├── features/      # Feature engineering (45+ features)
│   ├── models/        # ML models (LightGBM profitability scorer)
│   ├── api/           # FastAPI application
│   ├── database/      # PostgreSQL models
│   ├── cache/         # Redis caching
│   └── utils/         # Utilities (logging, config)
├── tests/             # Test suite (unit, integration, load)
├── notebooks/         # Jupyter notebooks for EDA
├── data/              # Data storage
├── models/            # Saved model artifacts
├── config/            # Configuration files
├── requirements.txt   # Production dependencies
└── requirements-dev.txt  # Development dependencies
```

## Key Features

1. **Profitability Score Model**: Predicts profitable repayment likelihood (not just risk)
2. **Alternative Data**: SMS transactions, contact graphs, location patterns, behavioral signals
3. **Weighted Scoring**: Income (40%) + Expenses (25%) + Social (15%) + Discipline (10%) + Behavioral (10%)
4. **Production-Ready**: FastAPI, async architecture, PostgreSQL, Redis, MLflow, monitoring
5. **High Performance**: <40ms P95 latency, 10K+ predictions/day, 500+ concurrent users

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run the application (coming in Phase 5)
uvicorn src.api.main:app --reload
```

## Development Phases

- **Phase 1**: Foundation & Infrastructure ⏳ (In Progress)
- **Phase 2**: Data Pipeline Engineering
- **Phase 3**: Feature Engineering (45+ features)
- **Phase 4**: ML Model Development
- **Phase 5**: Backend API Development
- **Phase 6**: Database & Caching
- **Phase 7**: Testing & Quality (90%+ coverage)
- **Phase 8**: Monitoring & Observability
- **Phase 9**: Containerization & Deployment
- **Phase 10**: Documentation & Portfolio

See `DEVELOPMENT_ROADMAP.md` for detailed plan.

## Tech Stack

**Backend**: Python 3.11, FastAPI, Uvicorn, Pydantic
**Database**: PostgreSQL, Redis, SQLAlchemy
**ML/Data**: LightGBM, pandas, NumPy, scikit-learn, SHAP, MLflow
**Testing**: pytest, pytest-cov, Locust
**Monitoring**: Prometheus, Grafana, Structlog

## Target Metrics

- API Latency: <40ms P95
- Throughput: 10K+ predictions/day
- Scalability: 500+ concurrent requests
- Reliability: 99.9% uptime
- Test Coverage: 90%+
- Processing Speed: 1K records/2sec

## Business Impact

- 99% faster (14 days → 60 seconds)
- 98% cost reduction (₹450 → ₹8 per application)
- Scalable to ₹10 Crore annual lending capacity
- Targeting 190M credit-invisible Indians

## License

MIT

## Author

Nilansh
