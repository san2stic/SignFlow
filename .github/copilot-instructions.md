# Copilot Instructions for SignFlow

## Big picture (read this first)
- SignFlow is a full-stack app: `backend/` (FastAPI + ML + WS) and `frontend/` (React/Vite + MediaPipe + Zustand).
- Core runtime flow: camera landmarks in `frontend/src/pages/TranslatePage.tsx` -> `useWebSocket` -> `WS /api/v1/translate/stream` -> backend inference pipeline in `backend/app/ml/pipeline.py`.
- Training flow crosses API + service + async worker: `backend/app/api/training.py` -> `backend/app/services/training_service.py` -> Celery task `backend/app/tasks/training_tasks.py`.
- Backend boot behavior in `backend/app/main.py` is important: imports all models, runs `Base.metadata.create_all`, applies runtime SQLite schema patches, then preloads translation pipeline.

## Service boundaries and integration points
- API routes are mounted with prefix from `Settings.api_v1_prefix` (default `/api/v1`) in `backend/app/main.py`.
- Frontend expects same-origin `/api/*` by default; Vite dev proxy is configured in `frontend/vite.config.ts` (`VITE_DEV_PROXY_TARGET`, default `http://localhost:8000`).
- WebSocket URL is built in `frontend/src/hooks/useWebSocket.ts` as `${WS_BASE}/api/v1/...`; avoid hardcoding host/protocol in features.
- Search backend is switchable (`sql` vs `elasticsearch`) via env (`SEARCH_BACKEND`); startup may reindex depending on env flags.
- Optional integrations are feature-flagged: TorchServe, MLflow registry, drift/active-learning, canary/shadow routing.

## Project-specific conventions
- Prefer updating existing typed clients (`frontend/src/api/*`, `frontend/src/lib/api.ts`) instead of ad-hoc `fetch` calls.
- Keep API changes backward-compatible; if an endpoint contract changes, update frontend callers and docs in the same PR.
- Route wiring lives in `frontend/src/routes.tsx`; verify both `/training` and `/train` compatibility when touching training navigation.
- Use env-driven configuration (`backend/app/config.py` + `.env.example`); when adding/changing env vars, update `.env.example` and docs together.
- Health checks: canonical endpoint is `/healthz`; `/health` exists as legacy alias.

## Dev workflows that matter
- Full stack (recommended): `docker compose up --build` from repo root.
- Native backend: `cd backend && python3 -m pip install -e .[dev] && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`.
- Native frontend: `cd frontend && npm install && npm run dev` (repo Vite config uses port `3000`).
- Backend tests: `cd backend && python3 -m pytest -q`.
- Frontend checks: `cd frontend && npm run test -- --run && npm run build`.

## Agent guardrails for edits
- Do not replace Alembic migrations with runtime-only schema changes; runtime patching in `main.py` is compatibility glue, not migration strategy.
- For training/inference changes, keep model artifact paths consistent with `MODEL_DIR` and `app/utils/model_artifacts.py` usage.
- For frontend translation UX changes, preserve the existing cadence controls (frame throttling + WS reconnect/backoff) in `TranslatePage` and `useWebSocket`.
- Preserve production security defaults: trusted hosts, request-size limits, and rate-limit dependencies in API routers.
- Never commit secrets from `.env`; treat `.env.example` as the only template source.