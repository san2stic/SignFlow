# SignFlow Threat Model (VPS Pre-Deployment)

Date: 2026-02-13
Scope: `/Users/bastienjavaux/Library/Mobile Documents/com~apple~CloudDocs/SignFlow`

## Assumptions
- Deployment target is a public internet-facing VPS behind Caddy (`80/443`).
- Current API has no authentication/authorization layer by design.
- Video payloads and training/model operations are sensitive (integrity + availability critical).
- Single-tenant or trusted-user deployment was not explicitly confirmed.

## In-Scope Runtime Components
- Reverse proxy: Caddy (`Caddyfile`).
- Backend API: FastAPI + WebSocket (`backend/app/api/*`, `backend/app/main.py`).
- ML/training workers: in-process thread or Celery worker (`backend/app/services/training_service.py`, `backend/app/celery_app.py`).
- Data stores: PostgreSQL + Redis in Docker Compose (`docker-compose.prod.yml`).
- File storage: local `/app/data` videos/models/exports.
- Frontend static app (Nginx serving built assets).

Out of scope: CI workflows, tests, local dev scripts not exposed in production runtime.

## Trust Boundaries
1. Internet client -> Caddy (`HTTP(S)`): external, untrusted.
2. Caddy -> FastAPI backend (internal Docker network): trusted transport, no app auth boundary.
3. Backend -> Postgres/Redis/filesystem: privileged internal boundary.
4. Backend -> ML execution (ffmpeg, MediaPipe, PyTorch): high CPU/memory boundary with untrusted user inputs.

## Assets
- Model integrity (active model selection, model artifacts).
- Sign dictionary integrity (CRUD/import/export).
- Video corpus confidentiality/integrity (recordings + landmarks).
- Service availability (CPU, memory, disk, worker queues).
- Credentials/secrets (DB/Redis credentials, env config).

## Primary Entry Points
- REST: `/api/v1/signs`, `/api/v1/media`, `/api/v1/training`, `/api/v1/models`, `/api/v1/dictionary`, `/api/v1/videos`.
- WebSocket: `/api/v1/translate/stream`, `/api/v1/training/sessions/{id}/live`.
- File uploads: sign video upload + dictionary import ZIP.

## Prioritized Threats

### T1 - Unauthenticated control of critical operations (High)
- Abuse path: attacker calls model activation/deploy/training/sign CRUD/delete endpoints without credentials.
- Impact: model poisoning/sabotage, data deletion, malicious retraining, integrity loss.
- Evidence: no auth dependency in routes (`backend/app/api/models.py`, `backend/app/api/training.py`, `backend/app/api/signs.py`, `backend/app/api/media.py`, `backend/app/api/dictionary.py`, `backend/app/api/videos.py`).

### T2 - Resource exhaustion via expensive endpoints and WebSockets (High)
- Abuse path: flood training creation, upload heavy videos, open many WS streams, send high-rate frames.
- Impact: CPU starvation, queue backlog, API unavailability.
- Evidence: limited in-memory REST rate limiter only, no WS throttling/auth/quota (`backend/app/api/deps.py`, `backend/app/api/translate.py`, `backend/app/api/training.py`).

### T3 - Unbounded dictionary import (High)
- Abuse path: upload oversized ZIP / many files (or decompression bomb-like content) to import endpoint.
- Impact: memory/disk exhaustion, DB bloat, service crash.
- Evidence: full archive read into memory and broad parsing with no cap (`backend/app/services/dictionary_service.py:107-149`).

### T4 - Weak/default infrastructure credentials in prod compose (Medium)
- Abuse path: default DB credentials reused in production or leaked through ops channels.
- Impact: DB compromise if network boundary is bypassed or misconfigured.
- Evidence: static credentials in compose (`docker-compose.prod.yml:30`, `docker-compose.prod.yml:47`, `docker-compose.prod.yml:63`).

### T5 - Container hardening gaps (Medium)
- Abuse path: app compromise leads to root process inside container and easier post-exploitation.
- Impact: higher blast radius in container escape/misconfig scenarios.
- Evidence: no non-root `USER` in backend/frontend images (`backend/Dockerfile`, `frontend/Dockerfile.prod`).

### T6 - Unsafe model deserialization surface (Medium, conditional)
- Abuse path: loading malicious `.pt` checkpoint through `torch.load(..., weights_only=False)` if artifact path is attacker-controlled.
- Impact: potential arbitrary code execution at model-load time.
- Evidence: `backend/app/ml/fewshot.py:105`, `backend/app/ml/trainer.py:648`.
- Condition: becomes high if attacker can influence model files/paths.

### T7 - Information disclosure of internal file paths (Medium)
- Abuse path: enumerate filesystem layout through video labeling endpoints.
- Impact: aids targeted attacks and local path discovery.
- Evidence: API returns `file_path` and `landmarks_path` (`backend/app/api/videos.py:28-29`, `backend/app/api/videos.py:52-53`).

### T8 - Public API documentation exposure (Low)
- Abuse path: attacker discovers attack surface quickly via docs/openapi.
- Impact: lowers recon effort.
- Evidence: Caddy routes `/docs*` and `/openapi.json` publicly (`Caddyfile:4`).

## Existing Mitigations Observed
- Basic upload size and duration validation for sign video uploads (`backend/app/services/media_service.py`).
- Basic per-IP per-route fixed-window limiter for some REST endpoints (`backend/app/api/deps.py`).
- Caddy reverse proxy fronting backend; DB/Redis not directly published to host ports in compose.

## Immediate Mitigation Plan (Before Public VPS)
1. Add authentication + authorization middleware/dependencies for all mutating and sensitive read endpoints.
2. Add quota and concurrency limits for WS + training jobs; enforce per-user/API-key budgets.
3. Add hard upload caps and archive safeguards on dictionary import (max bytes, max files, max extracted bytes).
4. Rotate all default credentials; move secrets to env/secret manager; block weak defaults at startup.
5. Run containers as non-root, drop capabilities, set read-only FS where possible.
6. Restrict `torch.load` usage to trusted signed artifacts; prefer safer formats/verification checks.
7. Remove filesystem paths from API responses; return opaque media IDs/URLs only.
8. Protect `/docs` and `/openapi.json` behind auth or disable in production.

## Automated Scan Notes
- `npm audit` reports moderate dev-chain issues around Vite/Vitest/esbuild in frontend toolchain.
- `bandit` flags `torch.load(..., weights_only=False)` and minor hygiene findings.
- Python dependency CVE scan via `pip-audit` could not complete in this environment due local venv bootstrap failure.
