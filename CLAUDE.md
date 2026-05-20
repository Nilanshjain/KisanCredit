# KisanCredit — Claude Reference

AI-powered alternative credit scoring for rural India. Uses UPI/SMS/contact/location data to score 190M+ underserved users.

**Status:** Backend 100% • ML 100% • DB 100% • Frontend ~80% • Deploy-ready

## Stack
- **Backend:** FastAPI 0.104, SQLAlchemy 2.0 (async, asyncpg), Alembic, Redis, Pydantic 2.5, python-jose (JWT), structlog, Prometheus
- **ML:** LightGBM 4.1 + SHAP 0.43, pandas/numpy. Model at `models/profitability_model_latest.pkl` (1.2MB, 47 features, ~2ms P95)
- **Frontend:** Next.js 16 (App Router), React 19, TS 5.9, Tailwind 4, Framer Motion, Zustand, TanStack Query
- **Infra:** Render (API) + Vercel (FE) + Neon Postgres + Upstash Redis — all free tiers

## Layout
```
src/
  api/         main.py (8 endpoints) · schemas.py · auth.py · users.py · middleware.py
  auth/        jwt_handler.py · otp_manager.py · dependencies.py · password_utils.py
  models/      predictor.py · trainer.py · evaluator.py · explainer.py
  features/    feature_engineering.py + {income,expense,social,discipline,behavioral,location}_features.py
  database/    models.py (6 tables) · repositories.py · connection.py
  cache/       redis_cache.py
  pipeline/    data_{generator,processor,validator}.py
  utils/       config.py · logger.py · metrics.py
frontend/
  app/         page.tsx · login/ · apply/ · dashboard/ + applications/[id]/ · contact/ · privacy/ · terms/
  components/  ui/{Button,Card,Input,Alert,Badge,Skeleton} · ErrorBoundary.tsx · Navbar.tsx
  lib/         api.ts · authStore.ts
alembic/versions/  196a9b156de2 (initial 6-table schema) · 60834f3aed34 (password_hash)
tests/             test_e2e_flow.py · test_api.py · test_models.py · locustfile.py
scripts/           train_model.py
```

## Database (6 tables)
- **users** — id, user_id, phone_number, email, full_name, aadhaar/pan, kyc fields, password_hash, timestamps
- **applications** — FK users; loan_amount/purpose/tenure; status (submitted|processing|approved|rejected); JSON cols for sms_transactions, contact_metadata, location_pattern, behavioral_data, extracted_features
- **predictions** — FK applications; profitability_score, confidence, decision, model_version, SHAP feature_contributions/top_features, prediction_latency_ms
- **audit_logs** — event_type/category, actor, event_data JSON, ip/user_agent
- **model_metrics** — period rollups: approval/rejection rates, latency p95/p99, loan totals
- **cache_metrics** — hit_rate, latency, memory usage

40+ indexes. Migrations in `alembic/versions/`.

## API (prefix `/api/v1`)
| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/applications` | yes | Submit loan app |
| GET  | `/applications/{id}` | yes | App details |
| POST | `/predictions` | no | Direct prediction |
| POST | `/predictions/batch` | no | Batch (max 100) |
| GET  | `/predictions/{id}/explain` | no | SHAP explanation |
| POST | `/auth/send-otp` | no | Send OTP |
| POST | `/auth/verify-otp` | no | Verify + issue tokens |
| POST | `/auth/refresh` | no | Refresh access token |
| GET  | `/auth/validate` | yes | Validate token |
| POST | `/auth/logout` | yes | Logout |
| GET  | `/health` | no | Health + model status |
| GET  | `/metrics` | no | Prometheus |

Docs: `/docs`, `/redoc`.

## Auth
- JWT: access 60min, refresh 7d, both in localStorage (`access_token`, `refresh_token`)
- OTP: 6-digit, 5min expiry, currently in-memory dict; `send_sms()` in `src/auth/otp_manager.py` is a placeholder (logs to console) — Twilio/MSG91 ready
- Phone validation: 10 digits starting with 6-9
- Flow: phone → OTP → verify → tokens → dashboard

## ML Model
- **Algorithm:** LightGBM, trained on 10K synthetic apps, R²=1.0 on synthetic (pipeline demo)
- **Features (47):** Income (9, weight 40%) · Expense (9, 25%) · Social (8, 15%) · Discipline (6, 10%) · Behavioral (6, 10%) · Location (7) · Metadata (2). Full list in `FEATURES.md`.
- **Decision:** score > 0.6 approve · < 0.4 reject · else manual_review
- **Explainability:** SHAP TreeExplainer; top factors returned in response
- **Feature engineering throughput:** ~220K records/sec

## Deployment
- **render.yaml** — Python web service, free plan, `uvicorn src.api.main:app`, healthcheck `/api/v1/health`
- **frontend/vercel.json** — Next.js with `/api/*` rewrite to backend
- **Env vars (backend):** `DATABASE_URL` (asyncpg), `DATABASE_URL_SYNC`, `REDIS_URL`, `SECRET_KEY` (32+ chars), `MODEL_PATH`, `ACCESS_TOKEN_EXPIRE_MINUTES=60`, `REFRESH_TOKEN_EXPIRE_DAYS=7`, `ENVIRONMENT`
- **Env vars (frontend):** `NEXT_PUBLIC_API_URL`
- **Free-tier caveats:** Render spins down after 15min idle (~50s cold start); Neon sleeps after 5min; Upstash 10K cmds/day

## Outstanding Work
**Critical path to MVP (~8–15h):**
1. Frontend API integration — replace hardcoded `localhost:8000` with `NEXT_PUBLIC_API_URL` across `login/apply/dashboard` pages and `lib/api.ts`; add network error handling/retries
2. `.env.example` files + env validation on startup
3. SMS gateway (Twilio/MSG91) wired into `otp_manager.send_sms()` — or email OTP via SendGrid as MVP fallback
4. Toast notifications + 401 auto-logout + 500 graceful handling
5. Mobile responsiveness pass (iOS Safari, Android Chrome)
6. Security: CORS whitelist (not `*`), per-IP rate limiting, CSRF on forms
7. Deploy + run migrations + load test (Locust 500+ concurrent)

**Post-MVP:** admin dashboard, email notifications, real SMS/UPI/banking integrations, GitHub Actions CI/CD, Grafana/Sentry.

## Common Commands
```bash
# Backend
python -m uvicorn src.api.main:app --reload
pytest                                  # or: pytest --cov=src tests/
alembic upgrade head
alembic revision --autogenerate -m "msg"
python scripts/train_model.py
locust -f tests/locustfile.py

# Frontend
cd frontend && npm install && npm run dev
npm run build && npm run lint && npm run type-check
```

## Working Conventions
- Check this file first; update it for significant structural changes
- Don't introduce new libraries without discussion — stick to the stack above
- New endpoint → add to `src/api/main.py` + schema in `schemas.py` + test
- New model feature → add extractor under `src/features/`, update `FEATURES.md`
- New page → `frontend/app/<name>/page.tsx`, add to nav
- Always validate input, use SQLAlchemy (no raw SQL), use parameterized queries
- Auth-protected frontend routes check `access_token` in localStorage, redirect to `/login`

## Targets to Monitor
- Prediction P95 < 50ms · API P95 < 200ms · error rate < 1% · cache hit rate > 70%
- FE page load < 2s · TTI < 3s · JS bundle < 500KB

Full docs: `README.md`, `docs/{PROJECT_OVERVIEW,ARCHITECTURE,FEATURES,DEPLOYMENT,API_REFERENCE,WORKFLOW,LOCAL_SETUP}.md`.
