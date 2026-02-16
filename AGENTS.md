# SignFlow - AGENTS.md

Last updated: 2026-02-16

This document is the implementation-level reference for humans and coding agents working in this repository.
It describes what exists in code today, how to run it, and what is partially implemented.

## 1. Project Overview

SignFlow is a full-stack platform for sign language translation and continuous learning.

Core product targets:
- Real-time translation from camera landmarks over WebSocket.
- Few-shot training for new signs with model versioning and deployment gates.
- Dictionary with graph relations, markdown notes, backlinks, import/export.
- Video labeling workflow (manual + similarity suggestions + bulk label).
- Mobile-aware frontend and PWA caching.

Primary language target in current docs/data: LSFB.

## 2. Current Status (Important)

The backend is feature-rich and exposes most v1 API capabilities.

The frontend currently has two UI tracks in code:
- Active routes mounted in `frontend/src/routes.tsx`:
  - `/dashboard` -> `frontend/src/pages/Dashboard.tsx` (static/demo-like UI)
  - `/translate` -> `frontend/src/pages/TranslatePage.tsx` (live camera + WS)
  - `/dictionary` -> `frontend/src/pages/Dictionary.tsx` (placeholder)
  - `/training` -> `frontend/src/pages/Training.tsx` (placeholder)
  - `/settings` -> `frontend/src/pages/Settings.tsx` (placeholder)
  - `/profile` -> `frontend/src/pages/Profile.tsx`
- Advanced pages implemented but not wired by default:
  - `frontend/src/pages/DashboardPage.tsx`
  - `frontend/src/pages/DictionaryPage.tsx`
  - `frontend/src/pages/TrainPage.tsx`
  - `frontend/src/pages/SettingsPage.tsx`

Known frontend routing mismatch in current code:
- `TranslatePage` and `DictionaryPage` navigate to `/train`, but router exposes `/training`.
- Result: this redirection path is currently broken unless router is updated.

## 3. Repository Map

Top-level:
- `backend/` FastAPI + ML + SQLAlchemy + Celery + TorchServe integration.
- `frontend/` React + TypeScript + Vite + Tailwind + Zustand.
- `docs/` user guides, test reports, plans.
- `scripts/` setup/bootstrap utility scripts.
- `docker-compose*.yml` deployment profiles.
- `Caddyfile` reverse proxy and security headers for production-like stack.

Backend main modules:
- `backend/app/api/` REST + WebSocket endpoints.
- `backend/app/models/` SQLAlchemy models.
- `backend/app/schemas/` Pydantic schemas.
- `backend/app/services/` domain services.
- `backend/app/ml/` training/inference/augmentation/monitoring pipeline.
- `backend/app/auth/` JWT auth utilities and dependencies.
- `backend/app/tasks/` Celery tasks.

Frontend main modules:
- `frontend/src/pages/` route-level pages.
- `frontend/src/components/` UI components.
- `frontend/src/api/` typed API clients.
- `frontend/src/hooks/` camera, MediaPipe, WebSocket hooks.
- `frontend/src/stores/` Zustand state stores.
- `frontend/public/sw.js` service worker cache strategies.

## 4. Tech Stack

Backend:
- Python 3.11+
- FastAPI
- SQLAlchemy + Alembic
- PostgreSQL (Docker default), SQLite supported
- Redis + Celery for async training
- Optional Elasticsearch for search indexing/query relevance
- PyTorch + NumPy + scikit-learn
- MediaPipe (backend processing path exists)
- Optional TorchServe and ONNX runtime
- Optional MLflow model registry integration

Frontend:
- React 18 + TypeScript
- Vite
- Tailwind CSS
- Zustand
- MediaPipe JS packages
- Framer Motion, Recharts, D3
- PWA manifest + service worker

Infra:
- Docker + docker compose
- Caddy in production-like stack

## 5. Backend Runtime Architecture

### 5.1 App entrypoint

- File: `backend/app/main.py`
- Prefix: `API_V1_PREFIX` default `/api/v1`
- Health endpoint: `GET /healthz`
- Metrics endpoint: `GET /metrics` (Prometheus)
- CORS and TrustedHost middleware enabled
- Request-size guard for POST/PUT/PATCH
- Startup behavior:
  - imports models
  - `Base.metadata.create_all(...)`
  - runtime-safe schema patches for SQLite compatibility
  - optional Elasticsearch index bootstrap
  - preloads inference pipeline

### 5.2 Data models

- `Sign` (`backend/app/models/sign.py`)
  - metadata, tags, variants, notes, usage_count, accuracy, related graph edges.
- `Video` (`backend/app/models/video.py`)
  - optional sign linkage, file paths, landmarks extraction metadata, quality fields.
- `TrainingSession` (`backend/app/models/training.py`)
  - mode/status/progress/config/metrics, produced model version info.
- `ModelVersion` (`backend/app/models/model_version.py`)
  - versioning, active flag, class labels, metadata, artifact path/size, lineage.
- `User` (`backend/app/models/user.py`)
  - auth profile model (email/username/password hash/state).

### 5.3 API surface (v1)

Base prefix: `/api/v1`

Auth (`/auth`):
- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`
- `PATCH /auth/me`

Translate (`/translate`):
- `WS /translate/stream`
- `GET /translate/active-learning/queue`
- `POST /translate/active-learning/queue/{sample_id}/resolve`

Signs (`/signs`):
- `GET /signs`
- `GET /signs/{sign_id}`
- `GET /signs/{sign_id}/backlinks`
- `POST /signs`
- `PUT /signs/{sign_id}`
- `DELETE /signs/{sign_id}`
- `POST /signs/{sign_id}/videos`
- `GET /signs/{sign_id}/videos`

Media (`/media`):
- `DELETE /media/{video_id}`
- `GET /media/{video_id}/stream`

Video labeling (`/videos`):
- `GET /videos/unlabeled`
- `PATCH /videos/{video_id}/label`
- `POST /videos/{video_id}/suggestions`
- `PATCH /videos/bulk-label`

Training (`/training`):
- `POST /training/sessions`
- `GET /training/sessions/{session_id}`
- `GET /training/sessions`
- `POST /training/sessions/{session_id}/stop`
- `POST /training/sessions/{session_id}/deploy`
- `WS /training/sessions/{session_id}/live`

Models (`/models`):
- `GET /models`
- `GET /models/active`
- `POST /models/{model_id}/activate`
- `GET /models/{model_id}/export?format=pt|onnx`

Dictionary (`/dictionary`):
- `GET /dictionary/graph`
- `GET /dictionary/search`
- `POST /dictionary/export`
- `POST /dictionary/import`

Search admin (`/search`):
- `POST /search/reindex`

Stats (`/stats`):
- `GET /stats/overview`
- `GET /stats/accuracy-history`
- `GET /stats/signs-per-category`

### 5.4 Inference pipeline

Core file: `backend/app/ml/pipeline.py`

Highlights:
- Sliding-window landmark inference state machine (`IDLE`, `RECORDING`, `INFERRING`).
- Transformer model (`SignTransformer`, `backend/app/ml/model.py`) with:
  - positional encoding
  - multiscale temporal stem
  - optional cosine head blending
  - CLS + masked pooling logic
- Post-processing:
  - confidence thresholding
  - prediction smoothing
  - sentence buffer
  - rejection reasons/diagnostics
- Optional TTA settings via env and runtime config.
- Optional TorchServe client path in translate endpoint.
- Optional drift detection and active-learning uncertainty queue.
- Canary/shadow model routing support (`backend/app/api/router_v2.py`, `backend/app/api/shadow_mode.py`).

### 5.5 Training pipeline

Core orchestration:
- API: `backend/app/api/training.py`
- Service: `backend/app/services/training_service.py`
- Celery task: `backend/app/tasks/training_tasks.py`

Capabilities:
- `few-shot` and `full-retrain` modes.
- quality gating from video landmark extraction and detection rate.
- stratified split and train-only augmentation (prevents validation leakage).
- advanced training config schema with many tunables (`backend/app/schemas/training.py`).
- deployment gating metrics and recommended next action.
- model artifact creation + version registration in DB.
- optional MLflow registry registration/promotion.

### 5.6 Security and limits

Implemented controls:
- CORS allowlist + trusted host validation.
- request body size limits.
- in-memory fixed-window rate limiting (global and write-specific).
- websocket message rate limit + max concurrent WS per host.
- JWT auth endpoints and dependencies for protected auth routes.

Current security note:
- Most non-auth business endpoints are not currently JWT-protected.
- Production ingress relies on Caddy TLS + headers; use JWT/auth hardening for API surface protection.

## 6. Frontend Architecture

### 6.1 Routing

Defined in `frontend/src/routes.tsx`.

Public routes:
- `/login`
- `/register`

Protected shell:
- `/dashboard`
- `/translate`
- `/dictionary`
- `/training`
- `/settings`
- `/profile`

### 6.2 Core live translation flow

Main page: `frontend/src/pages/TranslatePage.tsx`

Flow:
- camera stream (`useCamera`) + MediaPipe (`useMediaPipe`).
- serialize landmarks and send over WS (`useWebSocket`) to `/api/v1/translate/stream`.
- receive prediction/confidence/alternatives/sentence buffer.
- update Zustand translate store.
- optional speech synthesis.
- unknown-sign prompt logic with cooldown and pre-roll capture.

### 6.3 Training and dictionary UI tracks

There is complete V2 UI code for:
- training wizard and validation (`TrainPage`, `TrainingWizard`)
- dictionary graph/cards/detail/backlinks (`DictionaryPage`)
- dashboard with live API metrics (`DashboardPage`)

These are currently not mounted by default routes.

### 6.4 State stores

- `authStore` for JWT + user session persistence.
- `translateStore` for live translation data.
- `trainingStore` for session/progress/pending clip.
- `labelingStore` for unlabeled video workflow.
- `settingsStore` for local translation preferences.

### 6.5 PWA and offline behavior

- Manifest in `frontend/public/manifest.json`.
- Service worker in `frontend/public/sw.js`.
- Cache strategy:
  - cache-first for same-origin static assets.
  - stale-while-revalidate for dictionary/sign/stats category endpoints.

## 7. Environment Variables

Primary env file at repo root: `.env` (template: `.env.example`).

Key backend vars:
- `ENV`, `DATABASE_URL`, `REDIS_URL`
- search backend: `SEARCH_BACKEND`, `ELASTICSEARCH_URL`, `ELASTICSEARCH_INDEX`, `ELASTICSEARCH_TIMEOUT_MS`
- search lifecycle/resilience: `ELASTICSEARCH_REINDEX_ON_STARTUP`, `ELASTICSEARCH_FAIL_OPEN`, `ELASTICSEARCH_VERIFY_CERTS`
- `MODEL_DIR`, `VIDEO_DIR`, `EXPORT_DIR`
- `CORS_ORIGINS`, `TRUSTED_HOSTS`, `ENABLE_DOCS`
- rate/WS limits: `RATE_LIMIT_PER_MINUTE`, `WRITE_RATE_LIMIT_PER_MINUTE`, `WS_MESSAGES_PER_MINUTE`, `WS_MAX_CONNECTIONS_PER_IP`
- translation tuning: `TRANSLATE_*`
- TorchServe: `USE_TORCHSERVE`, `TORCHSERVE_URL`, `TORCHSERVE_TIMEOUT_MS`
- canary/shadow: `CANARY_*`, `SHADOW_*`
- metrics/drift: `INFERENCE_METRICS_ENABLED`, `DRIFT_*`
- active learning: `ACTIVE_LEARNING_*`
- MLflow registry: `MLFLOW_*`
- auth: `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`

Frontend:
- `VITE_DEV_PROXY_TARGET` for Vite `/api` dev proxy target.
- Optional absolute overrides: `VITE_API_URL`, `VITE_WS_URL`.

Important note:
- Frontend has two client bases (`src/lib/api.ts` and `src/api/client.ts`) with different URL assumptions.
- Prefer same-origin `/api/*` with Vite proxy in dev and reverse proxy in prod.

## 8. Running the Project

### 8.1 Docker dev (recommended)

From repo root:

```bash
cp .env.example .env
docker compose up --build
```

Services:
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- Swagger (if enabled): `http://localhost:8000/docs`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`
- Elasticsearch: `http://localhost:9200`
- MLflow: `http://localhost:5001`
- TorchServe: `http://localhost:8080` (inference), `8081` (management), `8082` (metrics)

### 8.2 Compose profile overrides

- Apple Silicon: `docker-compose.arm64.yml`
- CPU-only portability: `docker-compose.cpu.yml`
- NVIDIA GPU: `docker-compose.gpu.yml`
- production-like stack with Caddy: `docker-compose.prod.yml`

### 8.3 Native dev (split)

Backend:

```bash
cd backend
python3 -m pip install -e .[dev]
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Vite runs on port 3000 in this repo configuration.

## 9. Data and Assets

Runtime data directories (default in containers):
- `backend/data/models`
- `backend/data/videos`
- `backend/data/exports`

Dataset/bootstrap helpers:
- `scripts/seed_data.py`
- `scripts/download_dataset.py`

Example:

```bash
python scripts/download_dataset.py --dataset wlasl --max-signs 100 --clips-per-sign 20 --dry-run
```

## 10. Testing

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

Test suites include API behavior, ML modules, service logic, and UI components.

## 11. Deployment Notes

Production-like compose uses Caddy (`docker-compose.prod.yml` + `Caddyfile`):
- reverse proxy frontend + backend under one origin
- strict security headers
- docs disabled by default in prod (`ENABLE_DOCS=false`)

Before production use:
- set strong `JWT_SECRET_KEY`
- set non-empty `POSTGRES_PASSWORD`, `REDIS_PASSWORD`
- configure domain and review access-control strategy (JWT, network ACL, or external IdP)
- review CORS and trusted hosts

## 12. Known Gaps / Technical Debt

- Router currently mounts placeholder dictionary/training/settings pages instead of advanced V2 pages.
- `/train` navigation appears in some components while actual route is `/training`.
- Health endpoint is `/healthz`; any scripts using `/health` are outdated.
- Frontend API base handling should be consolidated to one strategy.
- Auth is not yet enforced on most business endpoints.

## 13. Agent Working Rules for this Repo

When editing SignFlow:
- Keep API schema changes backward-compatible where possible.
- If modifying routes, update matching frontend clients and docs in same change.
- If modifying model artifacts/metadata, update both inference loading and training export paths.
- Prefer explicit migration files for schema evolution; avoid relying only on runtime patches.
- For frontend, verify route wiring and navigation links together.
- For any change touching env vars, update both `.env.example` and documentation.
