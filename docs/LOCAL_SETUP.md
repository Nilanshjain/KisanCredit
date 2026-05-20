# KisanCredit - Local Development Setup

> Complete guide to run the project on your machine

---

## Quick Start (3 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 2. Start backend
python -m uvicorn src.api.main:app --reload

# 3. Start frontend (in new terminal)
cd frontend && npm run dev
```

**Done!**
- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

---

## Prerequisites

### Required Software

| Software | Version | Purpose | Installation |
|----------|---------|---------|--------------|
| **Python** | 3.10+ | Backend & ML | [python.org](https://python.org) |
| **Node.js** | 18+ | Frontend | [nodejs.org](https://nodejs.org) |
| **npm** | 9+ | Package manager | Comes with Node.js |
| **Git** | Latest | Version control | [git-scm.com](https://git-scm.com) |

**Check versions:**
```bash
python --version   # Should be 3.10 or higher
node --version     # Should be v18 or higher
npm --version      # Should be 9 or higher
git --version      # Any recent version
```

### Optional (Already Configured)
- ✅ **PostgreSQL** - Using Neon (cloud) - credentials in `.env`
- ✅ **Redis** - Using Upstash (cloud) - credentials in `.env`

You don't need to install these locally! The project uses free cloud services.

---

## Step-by-Step Setup

### 1. Clone Repository

```bash
cd C:/Users/YourName/Desktop  # Or your preferred location
git clone https://github.com/yourusername/KisanCredit.git
cd KisanCredit
```

### 2. Backend Setup

#### 2.1 Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

#### 2.2 Install Python Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- FastAPI (web framework)
- LightGBM (ML library)
- SHAP (explainability)
- SQLAlchemy (database ORM)
- Pandas, NumPy (data processing)
- Pydantic (validation)
- Uvicorn (ASGI server)
- Redis client
- PostgreSQL driver

**Installation time:** ~2-3 minutes

#### 2.3 Verify Environment Variables

Check `.env` file exists in project root:

```bash
cat .env  # Mac/Linux
type .env  # Windows
```

Should contain:
```env
DATABASE_URL=postgresql+asyncpg://user:pass@host/db?sslmode=require
DATABASE_URL_SYNC=postgresql://user:pass@host/db?sslmode=require
REDIS_URL=redis://default:password@host:6379
MODEL_PATH=models/profitability_model_latest.pkl
```

✅ **Already configured!** No changes needed.

#### 2.4 Verify Model File Exists

```bash
ls models/profitability_model_latest.pkl  # Mac/Linux
dir models\profitability_model_latest.pkl  # Windows
```

Should show file size (~1.2 MB).

#### 2.5 Start Backend Server

```bash
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Will watch for changes in these directories: ['C:\\...\\KisanCredit']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using WatchFiles
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
2025-10-22 10:00:00 [info] Rate limiter initialized
2025-10-22 10:00:00 [info] Starting KisanCredit API...
2025-10-22 10:00:00 [info] Model loaded successfully
2025-10-22 10:00:01 [info] SHAP explainer initialized
2025-10-22 10:00:01 [info] [OK] KisanCredit API started successfully
INFO:     Application startup complete.
```

✅ **Backend is running!**

#### 2.6 Test Backend

Open new terminal:

```bash
# Test health endpoint
curl http://localhost:8000/api/v1/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_health": "operational",
  "model_path": "models/profitability_model_latest.pkl",
  "feature_count": 45,
  "uptime_seconds": 10.5,
  "timestamp": "2025-10-22T10:00:10.123456"
}
```

✅ **API working!**

---

### 3. Frontend Setup

#### 3.1 Install Node.js Dependencies

Open **new terminal** (keep backend running):

```bash
cd frontend
npm install
```

**What gets installed:**
- Next.js (React framework)
- React & React DOM
- TypeScript
- Tailwind CSS
- Lucide React (icons)
- Other utilities

**Installation time:** ~1-2 minutes

#### 3.2 Verify Environment Variables

Check `.env.local` exists:

```bash
cat .env.local  # Mac/Linux
type .env.local  # Windows
```

Should contain:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

✅ **Already configured!**

#### 3.3 Start Frontend Server

```bash
npm run dev
```

**Expected output:**
```
  ▲ Next.js 16.0.0
  - Local:        http://localhost:3000
  - Network:      http://192.168.1.100:3000

 ✓ Ready in 1.5s
```

✅ **Frontend is running!**

---

### 4. Verify Complete Setup

#### 4.1 Open Browser

Navigate to:
- **Frontend:** http://localhost:3000
- **API Docs:** http://localhost:8000/docs

#### 4.2 Test Application Flow

1. Go to http://localhost:3000
2. Click "Apply for Loan Now"
3. Fill form with test data:
   - Name: Ramesh Kumar
   - Mobile: 9876543210
   - DOB: 1985-05-15
   - Gender: Male
   - Occupation: Farmer
   - Pincode: 110001
   - Loan Amount: ₹50,000
   - Purpose: Agriculture
   - Monthly Income: ₹35,000
   - Monthly Expenses: ₹20,000
4. Click Submit
5. Should see result in ~2 seconds

✅ **Everything working!**

---

## Project Structure

```
KisanCredit/
├── frontend/               # Next.js frontend
│   ├── app/
│   │   ├── page.tsx       # Landing page
│   │   ├── apply/
│   │   │   └── page.tsx   # Application form
│   │   ├── layout.tsx     # Root layout
│   │   └── globals.css    # Styles
│   ├── lib/
│   │   ├── api.ts         # API client
│   │   └── utils.ts       # Utilities
│   ├── components/        # UI components
│   ├── package.json       # Dependencies
│   ├── tsconfig.json      # TypeScript config
│   ├── tailwind.config.ts # Tailwind config
│   ├── next.config.js     # Next.js config
│   └── .env.local         # Environment variables
│
├── src/                   # Python backend
│   ├── api/
│   │   ├── main.py        # FastAPI app
│   │   ├── schemas.py     # Pydantic models
│   │   └── middleware.py  # CORS, rate limiting
│   ├── models/
│   │   ├── predictor.py   # ML prediction
│   │   ├── explainer.py   # SHAP explanations
│   │   ├── trainer.py     # Model training
│   │   └── evaluator.py   # Model evaluation
│   ├── features/
│   │   ├── income_features.py
│   │   ├── expense_features.py
│   │   ├── social_features.py
│   │   ├── discipline_features.py
│   │   ├── behavioral_features.py
│   │   ├── location_features.py
│   │   └── feature_engineering.py
│   ├── database/
│   │   ├── models.py      # SQLAlchemy models
│   │   ├── connection.py  # DB connection
│   │   └── repositories.py # Data access
│   ├── cache/
│   │   └── redis_cache.py # Redis operations
│   └── utils/
│       ├── config.py      # Configuration
│       ├── logger.py      # Structured logging
│       └── metrics.py     # Prometheus metrics
│
├── models/
│   └── profitability_model_latest.pkl  # Trained LightGBM model
│
├── data/                  # Training data
│   └── applications.parquet
│
├── docs/                  # Documentation
│   ├── ARCHITECTURE.md    # System architecture
│   ├── API_REFERENCE.md   # API documentation
│   ├── FEATURES.md        # Feature engineering
│   ├── WORKFLOW.md        # Data flow
│   └── LOCAL_SETUP.md     # This file
│
├── tests/                 # Test suite
│   ├── test_api.py
│   └── test_models.py
│
├── scripts/               # Utility scripts
│   ├── train_model.py
│   └── demo_prediction.py
│
├── alembic/               # Database migrations
│
├── .env                   # Backend environment variables
├── requirements.txt       # Python dependencies
├── README.md              # Project overview
└── DEPLOYMENT.md          # Deployment guide
```

---

## Common Issues & Solutions

### Issue 1: Port Already in Use

**Error:**
```
Error: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Kill process
kill -9 <PID>  # Mac/Linux
taskkill /PID <PID> /F  # Windows

# Or use different port
uvicorn src.api.main:app --reload --port 8001
```

---

### Issue 2: Module Not Found

**Error:**
```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution:**
```bash
# Make sure virtual environment is activated
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

### Issue 3: Database Connection Error

**Error:**
```
asyncpg.exceptions.InvalidAuthorizationSpecificationError
```

**Solution:**
Check `.env` file has correct DATABASE_URL. The provided credentials should work (free Neon tier).

If persistent:
```bash
# Test connection
python -c "import asyncpg; asyncpg.connect('postgresql://...')"
```

---

### Issue 4: Redis Connection Error

**Error:**
```
redis.exceptions.ConnectionError
```

**Solution:**
Check `.env` file has correct REDIS_URL. The provided credentials should work (free Upstash tier).

If persistent:
```bash
# Test connection
python -c "import redis; r = redis.from_url('redis://...'); r.ping()"
```

---

### Issue 5: Frontend Can't Connect to API

**Error in browser console:**
```
Failed to fetch: http://localhost:8000/api/v1/predictions
```

**Solution:**
1. Check backend is running (http://localhost:8000/api/v1/health)
2. Check CORS settings in `src/api/main.py`:
   ```python
   origins = ["http://localhost:3000", ...]
   ```
3. Check frontend API URL in `frontend/.env.local`:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
   ```

---

### Issue 6: Model File Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/profitability_model_latest.pkl'
```

**Solution:**
```bash
# Check if file exists
ls models/

# If missing, train new model
python scripts/train_model.py

# Or download from GitHub releases
```

---

## Development Workflow

### Making Changes

**Backend changes:**
1. Edit Python files in `src/`
2. Server auto-reloads (thanks to `--reload` flag)
3. Test at http://localhost:8000/docs

**Frontend changes:**
1. Edit TypeScript files in `frontend/app/` or `frontend/lib/`
2. Browser auto-refreshes (Hot Module Replacement)
3. Test at http://localhost:3000

### Running Tests

```bash
# Backend tests
pytest

# With coverage
pytest --cov=src tests/

# Specific test file
pytest tests/test_api.py

# Frontend tests (if configured)
cd frontend && npm test
```

### Code Formatting

```bash
# Python (Black)
pip install black
black src/

# Python (isort - imports)
pip install isort
isort src/

# TypeScript/JavaScript (Prettier)
cd frontend
npx prettier --write .
```

---

## Environment Variables Reference

### Backend (`.env`)

```env
# Database (Neon - Cloud PostgreSQL)
DATABASE_URL=postgresql+asyncpg://user:pass@host/db?sslmode=require
DATABASE_URL_SYNC=postgresql://user:pass@host/db?sslmode=require

# Cache (Upstash - Cloud Redis)
REDIS_URL=redis://default:password@host:6379

# Model
MODEL_PATH=models/profitability_model_latest.pkl

# API Settings
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=900  # 15 minutes in seconds

# Logging
LOG_LEVEL=INFO
```

### Frontend (`frontend/.env.local`)

```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1

# For production, change to:
# NEXT_PUBLIC_API_URL=https://kisancredit-api.onrender.com/api/v1
```

---

## Performance Tuning

### Backend Optimization

**1. Connection Pooling:**
```python
# src/database/connection.py
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,        # Default: 5
    max_overflow=10,     # Default: 10
    pool_pre_ping=True,  # Verify connections
)
```

**2. Redis Connection Pool:**
```python
# src/cache/redis_cache.py
redis_client = redis.ConnectionPool(
    host=REDIS_HOST,
    port=6379,
    max_connections=50,  # Default: 50
    decode_responses=True
)
```

**3. Uvicorn Workers:**
```bash
# Multiple workers for production
uvicorn src.api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### Frontend Optimization

**1. Build for Production:**
```bash
cd frontend
npm run build
npm start  # Production server
```

**2. Analyze Bundle Size:**
```bash
cd frontend
npx @next/bundle-analyzer
```

---

## Monitoring & Debugging

### Check Logs

**Backend logs:**
```bash
# Structured JSON logs written to stdout
python -m uvicorn src.api.main:app --reload 2>&1 | jq .

# Or redirect to file
python -m uvicorn src.api.main:app --reload > api.log 2>&1
```

**Frontend logs:**
```bash
# Browser console (F12)
# Or terminal where `npm run dev` is running
```

### Health Checks

```bash
# API health
curl http://localhost:8000/api/v1/health | jq .

# Model status
curl http://localhost:8000/api/v1/health | jq .model_health

# Database connection
python -c "from src.database.connection import test_connection; test_connection()"
```

### Performance Profiling

```bash
# API latency
curl -w "@curl-format.txt" http://localhost:8000/api/v1/health

# Model prediction time
python scripts/benchmark_latency.py
```

---

## Next Steps

Once local setup is working:

1. **Read Documentation:**
   - `docs/ARCHITECTURE.md` - Understand system design
   - `docs/FEATURES.md` - Learn about 45 features
   - `docs/WORKFLOW.md` - Follow data flow

2. **Explore API:**
   - Open http://localhost:8000/docs
   - Try interactive Swagger UI
   - Test prediction endpoint

3. **Customize:**
   - Modify feature engineering (`src/features/`)
   - Adjust model thresholds (`src/models/predictor.py`)
   - Enhance frontend UI (`frontend/app/`)

4. **Deploy:**
   - Follow `DEPLOYMENT.md` for production deployment
   - Deploy backend to Render.com
   - Deploy frontend to Vercel

---

## Troubleshooting Checklist

Before asking for help, verify:

- [ ] Python 3.10+ installed (`python --version`)
- [ ] Node.js 18+ installed (`node --version`)
- [ ] Virtual environment activated (`(venv)` in prompt)
- [ ] Dependencies installed (`pip list | grep fastapi`, `npm list next`)
- [ ] `.env` file exists in project root
- [ ] `.env.local` exists in `frontend/` directory
- [ ] Model file exists (`models/profitability_model_latest.pkl`)
- [ ] Backend running on port 8000
- [ ] Frontend running on port 3000
- [ ] Can access http://localhost:8000/api/v1/health
- [ ] Can access http://localhost:3000

---

## Getting Help

- **GitHub Issues:** https://github.com/yourusername/KisanCredit/issues
- **Documentation:** `docs/` folder
- **API Docs:** http://localhost:8000/docs (when running)
- **Community:** Discussions tab on GitHub

---

**Happy coding! 🚀**

You now have a fully functional local development environment for KisanCredit.
