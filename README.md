# SignFlow

Plateforme mobile-first de traduction LSFB en temps réel, entraînement few-shot continu, et dictionnaire interconnecté style Obsidian.

## Scope V1

- Cible: **LSFB** (Langue des Signes de Belgique Francophone)
- Focus produit: **Translate + Train + Dictionary + Dashboard**
- Déploiement: **local Docker** + **stack prod reverse-proxy Caddy**

## Fonctionnalités livrées

- Backend FastAPI v1 (REST + WebSocket traduction live)
- Pipeline ML landmarks -> Transformer -> post-processing
- Few-shot training avec versioning de modèles, seuil de déploiement et activation
- CRUD signes + upload vidéos + extraction landmarks backend
- Video Labeling:
  - interface de labellisation assistée par ML
  - suggestions intelligentes basées sur similarité cosinus
  - labellisation groupée de vidéos similaires
  - recherche et création de signes inline
- Dictionary:
  - graphe relationnel
  - notes markdown + wikilinks `[[...]]`
  - backlinks API
  - import/export `json`, `markdown`, `obsidian-vault`
- Dashboard:
  - KPIs
  - accuracy history
  - signs per category
  - activation/export modèle
- Frontend PWA:
  - manifest + service worker
  - fallback offline sur données dictionary en cache

## Démarrage local (dev)

```bash
cp .env.example .env
docker compose up --build
```

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

## Démarrage production-like (Caddy)

```bash
cp .env.example .env
# définir SIGNFLOW_DOMAIN + credentials DB/Redis + auth Caddy dans .env
# générer le hash Caddy:
# docker run --rm caddy:2-alpine caddy hash-password --plaintext 'mot-de-passe-fort'
docker compose -f docker-compose.prod.yml up --build
```

- Entrée unique via Caddy: `http://localhost` (ou votre domaine)
- API routée via `/api/*`
- Accès protégé par HTTP Basic Auth (Caddy)
- `/docs` et `/openapi.json` désactivés par défaut en production

## Setup dataset LSFB / WLASL

### Seed initial (10 signes LSFB)

```bash
python scripts/seed_data.py
```

### Bootstrap WLASL subset

```bash
python scripts/download_dataset.py \
  --dataset wlasl \
  --max-signs 100 \
  --clips-per-sign 20 \
  --dry-run
```

Téléchargement clips:

```bash
python scripts/download_dataset.py \
  --dataset wlasl \
  --max-signs 100 \
  --clips-per-sign 20 \
  --download-videos \
  --skip-existing
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

## Smoke E2E manuel (release)

1. Ouvrir `Train`, créer/choisir un signe, enregistrer au moins 5 clips valides.
2. Lancer l’entraînement few-shot et attendre `completed`.
3. Valider step 4 avec live check WS, puis déployer le modèle.
4. Ouvrir `Translate` et confirmer détection live du signe.
5. Ouvrir `Dictionary`:
   - vérifier détail/backlinks
   - tester export/import zip
6. Ouvrir `Dashboard`:
   - vérifier courbe accuracy
   - vérifier répartition catégories

## API v1 (résumé)

- `WS /api/v1/translate/stream`
- `GET/POST/PUT/DELETE /api/v1/signs`
- `GET /api/v1/signs/{sign_id}/backlinks`
- `POST /api/v1/signs/{sign_id}/videos`
- `GET /api/v1/signs/{sign_id}/videos`
- `GET /api/v1/videos/unlabeled`
- `PATCH /api/v1/videos/{video_id}/label`
- `POST /api/v1/videos/{video_id}/suggestions`
- `PATCH /api/v1/videos/bulk-label`
- `GET/POST /api/v1/training/sessions`
- `POST /api/v1/training/sessions/{session_id}/deploy`
- `WS /api/v1/training/sessions/{session_id}/live`
- `GET /api/v1/models`
- `POST /api/v1/models/{model_id}/activate`
- `GET /api/v1/models/{model_id}/export`
- `GET /api/v1/dictionary/graph`
- `GET /api/v1/dictionary/search`
- `POST /api/v1/dictionary/export`
- `POST /api/v1/dictionary/import`
- `GET /api/v1/stats/overview`
- `GET /api/v1/stats/accuracy-history`
- `GET /api/v1/stats/signs-per-category`
