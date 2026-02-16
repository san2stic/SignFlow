# SignFlow

SignFlow is a full-stack platform for real-time sign language translation, continuous model training, and an interactive dictionary/knowledge graph.

## What is in this repository

- Backend API (FastAPI + WebSocket + ML pipeline + training orchestration)
- Frontend app (React + TypeScript + Vite + Tailwind + Zustand)
- Video labeling and few-shot training workflow (API complete, advanced UI present in code)
- Model versioning, activation, export, canary/shadow routing support
- Dockerized local and production-like stacks

## Current implementation status

Implemented and usable today:
- Auth: register/login/profile (`/api/v1/auth/*`)
- Real-time translation stream (`WS /api/v1/translate/stream`)
- Sign CRUD + sign videos upload/list
- Dictionary graph/search/export/import
- Video labeling endpoints (`unlabeled`, `label`, `suggestions`, `bulk-label`)
- Training sessions + live WS metrics + deploy endpoint
- Model listing/activation/export
- Dashboard stats endpoints

Frontend note:
- Active routes currently use:
  - `Dashboard.tsx` (dashboard shell + live metrics sections)
  - `TranslatePage.tsx` (live camera + WS)
  - `DictionaryPage.tsx`, `TrainPage.tsx`, `SettingsPage.tsx`, `Profile.tsx`
  - Compatibility alias route: `/train` -> same screen as `/training`

## Architecture

### Backend

- Python 3.11+
- FastAPI (REST + WebSocket)
- SQLAlchemy + Alembic
- PostgreSQL default in Docker (SQLite also supported)
- Redis + Celery for async training
- Elasticsearch for sign/dictionary search (optional, enabled in Docker stacks)
- PyTorch + MediaPipe + NumPy + scikit-learn
- Optional TorchServe + ONNX Runtime + MLflow registry

### Frontend

- React 18 + TypeScript + Vite
- Tailwind CSS + Zustand
- MediaPipe JS + WebSocket
- D3/Recharts for graph and dashboard visualizations
- PWA manifest + service worker cache

## Quick start (Docker, recommended)

```bash
cp .env.example .env
docker compose up --build
```

Endpoints:
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Elasticsearch: `http://localhost:9200`
- MLflow: `http://localhost:5001`
- TorchServe ping: `http://localhost:8080/ping`

## Native development

### Backend

```bash
cd backend
python3 -m pip install -e .[dev]
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite in this repo is configured on port `3000`.

## Compose variants

- `docker-compose.yml`: full dev stack
- `docker-compose.arm64.yml`: Apple Silicon override
- `docker-compose.cpu.yml`: CPU-only TorchServe override
- `docker-compose.gpu.yml`: NVIDIA GPU override
- `docker-compose.prod.yml`: production-like stack with Caddy

## Core API map

Base prefix: `/api/v1`

- Auth: `/auth/register`, `/auth/login`, `/auth/me`
- Translate: `WS /translate/stream`, active-learning queue endpoints
- Signs: CRUD + `/signs/{id}/videos` + `/signs/{id}/backlinks`
- Media: `/media/{video_id}`, `/media/{video_id}/stream`
- Videos labeling: `/videos/unlabeled`, `/videos/{id}/label`, `/videos/{id}/suggestions`, `/videos/bulk-label`
- Training: `/training/sessions*` + `WS /training/sessions/{id}/live`
- Models: `/models`, `/models/active`, `/models/{id}/activate`, `/models/{id}/export`
- Dictionary: `/dictionary/graph`, `/dictionary/search`, `/dictionary/export`, `/dictionary/import`
- Search admin: `/search/reindex`
- Stats: `/stats/overview`, `/stats/accuracy-history`, `/stats/signs-per-category`

Health and metrics:
- `GET /healthz`
- `GET /metrics`

## Environment

Primary template: `.env.example`

Important variables:
- Runtime/data: `ENV`, `DATABASE_URL`, `REDIS_URL`, `MODEL_DIR`, `VIDEO_DIR`, `EXPORT_DIR`
- Search: `SEARCH_BACKEND`, `ELASTICSEARCH_URL`, `ELASTICSEARCH_INDEX`, `ELASTICSEARCH_TIMEOUT_MS`
- Search resilience/bootstrap: `ELASTICSEARCH_REINDEX_ON_STARTUP`, `ELASTICSEARCH_FAIL_OPEN`, `ELASTICSEARCH_VERIFY_CERTS`
- API controls: `CORS_ORIGINS`, `TRUSTED_HOSTS`, `ENABLE_DOCS`
- Limits: `MAX_REQUEST_MB`, `RATE_LIMIT_PER_MINUTE`, `WRITE_RATE_LIMIT_PER_MINUTE`
- Translation tuning: `TRANSLATE_*`
- Optional serving: `USE_TORCHSERVE`, `TORCHSERVE_URL`, `TORCHSERVE_TIMEOUT_MS`
- Canary/shadow routing: `CANARY_*`, `SHADOW_*`
- Active learning/drift: `ACTIVE_LEARNING_*`, `DRIFT_*`
- Auth: `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`

## Dataset/bootstrap scripts

Seed base signs:

```bash
python scripts/seed_data.py
```

WLASL subset manifest:

```bash
python scripts/download_dataset.py --dataset wlasl --max-signs 100 --clips-per-sign 20 --dry-run
```

With clip download:

```bash
python scripts/download_dataset.py --dataset wlasl --max-signs 100 --clips-per-sign 20 --download-videos --skip-existing
```

## Tests

Backend:

```bash
cd backend
python3 -m pytest -q
```

Frontend:

```bash
cd frontend
npm run test -- --run
npm run build
```

## Production-like run (Caddy)

```bash
cp .env.example .env
# set SIGNFLOW_DOMAIN, POSTGRES_PASSWORD, and REDIS_PASSWORD
docker compose -f docker-compose.prod.yml up --build
```

Caddy provides:
- single-origin frontend + `/api/*` proxy
- security headers (HSTS, nosniff, frame deny, permissions policy)

## Known gaps

- Frontend has two API base strategies (`src/lib/api.ts` and `src/api/client.ts`) and should be consolidated.

For deeper technical details, see `AGENTS.md`.
