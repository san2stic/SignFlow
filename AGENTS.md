# 🤟 SIGNFLOW — Mega-Prompt de Développement

## Système de Traduction de Langue des Signes en Temps Réel avec Apprentissage Continu

> **Prompt destiné à** : Un LLM de développement (Claude, GPT-4, Codex) pour générer l'architecture complète, le code et les instructions de déploiement du projet SignFlow.

---

## 🎯 CONTEXTE & MISSION

```
Tu es un ingénieur full-stack senior spécialisé en Machine Learning appliqué à la vision par ordinateur. 
Tu dois concevoir et développer "SignFlow" — une plateforme web complète de traduction 
de langue des signes en temps réel par vidéo, avec les capacités suivantes :

1. TRADUCTION EN TEMPS RÉEL : Capturer le flux vidéo de la caméra, détecter les gestes 
   de langue des signes et les traduire en texte/audio instantanément.
   
2. ENTRAÎNEMENT DE NOUVEAUX SIGNES : Permettre à l'utilisateur d'enregistrer de nouveaux 
   signes via sa caméra, de les labelliser, et de fine-tuner le modèle en temps réel 
   (few-shot learning / transfer learning).
   
3. DICTIONNAIRE INTERACTIF : Un wiki/dictionnaire style Obsidian avec graphe de relations 
   entre les signes, vidéos de référence, métadonnées, tags, et navigation par liens bidirectionnels.

4. API REST COMPLÈTE : Endpoints pour toutes les opérations (traduction, entraînement, 
   CRUD dictionnaire, export/import).

5. MOBILE-FIRST : L'interface de traduction ET d'entraînement doit être parfaitement 
   utilisable sur mobile (responsive, touch-optimized, caméra native).
```

---

## 📐 ARCHITECTURE TECHNIQUE

### Stack Technologique

```yaml
# BACKEND
runtime: Python 3.11+
framework: FastAPI (async, WebSocket natif)
ml_framework: PyTorch 2.x + TorchVision
pose_estimation: MediaPipe Holistic (hands + pose + face)
model_base: Transformer léger custom (ou fine-tune d'un modèle pré-entraîné type Video Swin Transformer)
database: SQLite (dev) → PostgreSQL (prod)
orm: SQLAlchemy + Alembic (migrations)
file_storage: Local filesystem structuré (dev) → S3-compatible (prod)
task_queue: Celery + Redis (pour entraînement asynchrone)
websocket: FastAPI WebSocket natif

# FRONTEND
framework: React 18 + TypeScript
build: Vite
ui: Tailwind CSS + Radix UI (accessible, mobile-first)
state: Zustand (léger, performant)
video: MediaPipe JS SDK + WebRTC (getUserMedia)
graph_viz: D3.js ou react-force-graph (pour le dictionnaire style Obsidian)
pwa: Service Worker + manifest.json (installable sur mobile)
markdown: MDX ou react-markdown (pour les notes du dictionnaire)

# INFRA
containerization: Docker + docker-compose
reverse_proxy: Caddy (auto-HTTPS)
ci_cd: GitHub Actions
monitoring: Prometheus + Grafana (optionnel v2)
```

### Architecture Globale

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT (React PWA)                     │
│                                                           │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ TRANSLATE │  │   TRAIN      │  │   DICTIONARY       │  │
│  │ (Live)    │  │   (Record)   │  │   (Obsidian-like)  │  │
│  │           │  │              │  │                     │  │
│  │ Camera →  │  │ Camera →     │  │ Graph View ←→      │  │
│  │ MediaPipe │  │ Record clips │  │ Card View ←→       │  │
│  │ → WS →   │  │ → Label →    │  │ Video Player       │  │
│  │ Text/Audio│  │ Upload → API │  │ Markdown Notes      │  │
│  └──────────┘  └──────────────┘  └────────────────────┘  │
│         │              │                    │              │
│         │    WebSocket │        REST API    │              │
└─────────┼──────────────┼────────────────────┼─────────────┘
          │              │                    │
          ▼              ▼                    ▼
┌─────────────────────────────────────────────────────────┐
│                   API GATEWAY (FastAPI)                   │
│                                                           │
│  /api/v1/translate    (WebSocket - temps réel)            │
│  /api/v1/signs        (CRUD - dictionnaire)               │
│  /api/v1/training     (POST sessions, GET status)         │
│  /api/v1/dictionary   (search, graph, export/import)      │
│  /api/v1/media        (upload/download vidéos)            │
│  /api/v1/models       (versions, switch, rollback)        │
└──────────────┬──────────────────────┬─────────────────────┘
               │                      │
    ┌──────────▼──────────┐  ┌───────▼────────┐
    │   ML PIPELINE       │  │   DATABASE     │
    │                     │  │                │
    │  MediaPipe →        │  │  PostgreSQL    │
    │  Feature Extract →  │  │  ├─ signs     │
    │  Transformer →      │  │  ├─ videos    │
    │  Classification     │  │  ├─ training  │
    │                     │  │  ├─ models    │
    │  Fine-Tune Engine   │  │  └─ graph     │
    │  (few-shot)         │  │                │
    └─────────────────────┘  └────────────────┘
```

---

## 🧠 MODULE 1 : MODÈLE ML — SignFlow Model

### 1.1 Pipeline de Reconnaissance

```
Implémente le pipeline ML suivant :

ÉTAPE 1 — EXTRACTION DE FEATURES (MediaPipe Holistic)
- Utilise MediaPipe Holistic pour extraire en temps réel :
  • 21 landmarks par main (x, y, z) × 2 mains = 126 valeurs
  • 33 landmarks du corps (pose) = 99 valeurs  
  • 468 landmarks du visage = 1404 valeurs (optionnel, réduire via PCA à ~50)
- Normalise les coordonnées par rapport au centre du corps (hip_center)
- Applique un augmentation temporelle : fenêtre glissante de N frames (défaut: 30 frames = 1 sec à 30fps)
- Output : Tensor de shape [batch, seq_len, num_features]

ÉTAPE 2 — MODÈLE DE CLASSIFICATION
Architecture : Transformer Encoder léger
- Input embedding : Linear(num_features, d_model=256)
- Positional encoding : sinusoïdal
- Encoder : 4 couches, 8 heads, dim_feedforward=512, dropout=0.1
- Classification head : Linear(d_model, num_classes) avec pooling temporel (mean)
- Output : logits pour chaque signe du vocabulaire

Justification : Les Transformers capturent les dépendances temporelles longues 
mieux que les LSTM pour les séquences de landmarks, tout en restant parallélisables.

ÉTAPE 3 — POST-PROCESSING
- Softmax + threshold de confiance (défaut: 0.7)
- Lissage temporel : moyenne mobile sur les 3 dernières prédictions
- Détection de "silence" (pas de signe) via un classe spéciale [NONE]
- Buffer de mots pour construire des phrases
```

### 1.2 Entraînement & Few-Shot Learning

```
Implémente un système d'entraînement continu :

MODE 1 — ENTRAÎNEMENT INITIAL (base model)
- Dataset : un dataset public de langue des signes (WLASL, AUTSL, ou custom)
- Training classique : CrossEntropyLoss, AdamW, lr=3e-4, cosine annealing
- Validation split : 80/10/10
- Early stopping sur val_loss
- Sauvegarde du meilleur modèle comme "base model"

MODE 2 — FEW-SHOT FINE-TUNING (nouveaux signes)
Quand un utilisateur ajoute un nouveau signe :
1. L'utilisateur enregistre 5-20 clips vidéo du signe (3-5 secondes chacun)
2. MediaPipe extrait les landmarks de chaque clip
3. Data augmentation : 
   - Mirror horizontal (main gauche ↔ droite)
   - Jitter temporel (±5 frames)
   - Bruit gaussien sur les landmarks (σ=0.01)
   - Speed variation (0.8x à 1.2x)
4. Fine-tuning strategy :
   - Freeze les premières couches du Transformer (layers 0-2)
   - Ajouter un nouveau neurone à la couche de classification
   - Entraîner pendant 50-100 epochs avec lr=1e-4
   - Utiliser Focal Loss pour gérer le déséquilibre de classes
   - Prototypical Networks en fallback si < 5 exemples
5. Validation automatique : tester sur 20% des clips enregistrés
6. Si accuracy > 85% → déployer le nouveau modèle
7. Sinon → demander plus d'exemples à l'utilisateur

MODE 3 — APPRENTISSAGE ACTIF
- Pendant la traduction en temps réel, si confiance < 0.5 :
  → Proposer à l'utilisateur de labelliser le geste détecté
  → Ajouter automatiquement au dataset d'entraînement
  → Déclencher un fine-tuning incrémental en background (Celery task)
```

### 1.3 Gestion des Modèles

```
Implémente un système de versioning de modèles :

- Chaque entraînement produit un modèle versionné : model_v{N}.pt
- Métadonnées stockées en DB :
  {
    "version": "v12",
    "created_at": "2025-01-15T14:30:00Z",
    "num_classes": 247,
    "accuracy": 0.923,
    "training_samples": 12450,
    "new_signs_added": ["bonjour_v2", "merci"],
    "parent_version": "v11",
    "file_path": "/models/model_v12.pt",
    "file_size_mb": 45.2
  }
- Rollback possible vers n'importe quelle version
- A/B testing entre versions (optionnel v2)
- Export ONNX pour inférence optimisée côté client (optionnel)
```

---

## 🌐 MODULE 2 : API REST COMPLÈTE

### 2.1 Spécification des Endpoints

```yaml
# ═══════════════════════════════════════════
# TRADUCTION EN TEMPS RÉEL (WebSocket)
# ═══════════════════════════════════════════

WS /api/v1/translate/stream:
  description: "Stream WebSocket pour traduction temps réel"
  input: 
    type: binary frames (landmarks MediaPipe sérialisés en JSON)
    format: |
      {
        "timestamp": 1705312200.123,
        "frame_idx": 42,
        "hands": { "left": [[x,y,z], ...], "right": [[x,y,z], ...] },
        "pose": [[x,y,z], ...],
        "face": [[x,y,z], ...]  // optionnel
      }
  output:
    format: |
      {
        "prediction": "bonjour",
        "confidence": 0.94,
        "alternatives": [
          {"sign": "salut", "confidence": 0.78},
          {"sign": "hey", "confidence": 0.45}
        ],
        "sentence_buffer": "Bonjour comment",
        "is_sentence_complete": false
      }

# ═══════════════════════════════════════════
# GESTION DES SIGNES (CRUD)
# ═══════════════════════════════════════════

GET /api/v1/signs:
  description: "Liste tous les signes du dictionnaire"
  params:
    - search (string): recherche full-text
    - category (string): filtrer par catégorie
    - tag (string[]): filtrer par tags
    - sort (string): name | created_at | usage_count
    - page (int): pagination
    - per_page (int): défaut 20
  response: SignListResponse

GET /api/v1/signs/{sign_id}:
  description: "Détail d'un signe avec vidéos, notes, relations"
  response: SignDetailResponse

POST /api/v1/signs:
  description: "Créer un nouveau signe"
  body:
    name: string (requis)
    description: string (markdown)
    category: string
    tags: string[]
    related_signs: string[] (IDs pour le graph)
    variants: string[] (variantes régionales)
  response: SignDetailResponse

PUT /api/v1/signs/{sign_id}:
  description: "Modifier un signe existant"
  body: (mêmes champs que POST, tous optionnels)

DELETE /api/v1/signs/{sign_id}:
  description: "Supprimer un signe et ses médias associés"

# ═══════════════════════════════════════════
# MÉDIAS (Vidéos d'entraînement et référence)
# ═══════════════════════════════════════════

POST /api/v1/signs/{sign_id}/videos:
  description: "Upload une vidéo pour un signe"
  body: multipart/form-data
    file: video/webm ou video/mp4
    type: "training" | "reference" | "example"
    metadata: JSON (durée, fps, résolution)
  response: VideoResponse

GET /api/v1/signs/{sign_id}/videos:
  description: "Liste les vidéos d'un signe"

DELETE /api/v1/media/{video_id}:
  description: "Supprimer une vidéo"

GET /api/v1/media/{video_id}/stream:
  description: "Stream une vidéo (pour le lecteur)"

# ═══════════════════════════════════════════
# ENTRAÎNEMENT
# ═══════════════════════════════════════════

POST /api/v1/training/sessions:
  description: "Démarrer une session d'entraînement"
  body:
    sign_id: string (signe à entraîner)
    mode: "few-shot" | "full-retrain"
    config:
      epochs: int (défaut: 50)
      learning_rate: float (défaut: 1e-4)
      augmentation: boolean (défaut: true)
  response: TrainingSessionResponse

GET /api/v1/training/sessions/{session_id}:
  description: "Statut d'une session d'entraînement"
  response:
    status: "queued" | "preprocessing" | "training" | "validating" | "completed" | "failed"
    progress: float (0-100)
    current_epoch: int
    metrics:
      loss: float
      accuracy: float
      val_accuracy: float
    estimated_remaining: string (durée)

WS /api/v1/training/sessions/{session_id}/live:
  description: "WebSocket pour suivre l'entraînement en temps réel"
  output: métriques à chaque epoch

GET /api/v1/training/sessions:
  description: "Historique des sessions d'entraînement"

POST /api/v1/training/sessions/{session_id}/stop:
  description: "Arrêter un entraînement en cours"

# ═══════════════════════════════════════════
# MODÈLES
# ═══════════════════════════════════════════

GET /api/v1/models:
  description: "Liste des versions de modèles"

GET /api/v1/models/active:
  description: "Modèle actuellement en production"

POST /api/v1/models/{model_id}/activate:
  description: "Activer un modèle (rollback possible)"

GET /api/v1/models/{model_id}/export:
  description: "Exporter un modèle (format .pt ou .onnx)"

# ═══════════════════════════════════════════
# DICTIONNAIRE / GRAPHE (style Obsidian)
# ═══════════════════════════════════════════

GET /api/v1/dictionary/graph:
  description: "Données du graphe de relations entre signes"
  response:
    nodes: [{id, label, category, video_count, thumbnail_url}]
    edges: [{source, target, relation_type, weight}]

GET /api/v1/dictionary/search:
  description: "Recherche full-text dans le dictionnaire"
  params:
    q: string
    fields: name | description | tags | all

POST /api/v1/dictionary/export:
  description: "Exporter le dictionnaire complet"
  body:
    format: "json" | "markdown" | "obsidian-vault"
  response: fichier ZIP

POST /api/v1/dictionary/import:
  description: "Importer un dictionnaire"
  body: multipart/form-data (ZIP)

# ═══════════════════════════════════════════
# STATISTIQUES
# ═══════════════════════════════════════════

GET /api/v1/stats/overview:
  description: "Statistiques globales"
  response:
    total_signs: int
    total_videos: int
    model_accuracy: float
    total_translations: int
    most_used_signs: [{sign, count}]
    recent_activity: [{action, timestamp}]
```

### 2.2 Modèles de Données (Schemas Pydantic)

```python
# Génère les schemas Pydantic suivants avec validation complète :

class Sign(BaseModel):
    id: UUID
    name: str  # ex: "bonjour"
    slug: str  # ex: "bonjour" (URL-safe)
    description: Optional[str]  # Markdown
    category: Optional[str]  # ex: "salutations"
    tags: List[str]  # ex: ["courant", "formel"]
    variants: List[str]  # variantes régionales
    related_signs: List[UUID]  # liens bidirectionnels (graphe)
    video_count: int
    training_sample_count: int
    accuracy: Optional[float]  # précision du modèle sur ce signe
    usage_count: int  # combien de fois traduit
    notes: Optional[str]  # Notes markdown style Obsidian avec [[liens]]
    created_at: datetime
    updated_at: datetime

class Video(BaseModel):
    id: UUID
    sign_id: UUID
    file_path: str
    thumbnail_path: str
    duration_ms: int
    fps: int
    resolution: str  # "640x480"
    type: Literal["training", "reference", "example"]
    landmarks_extracted: bool
    landmarks_path: Optional[str]  # fichier .npy des landmarks
    created_at: datetime

class TrainingSession(BaseModel):
    id: UUID
    sign_id: Optional[UUID]  # null si full retrain
    mode: Literal["few-shot", "full-retrain"]
    status: Literal["queued", "preprocessing", "training", "validating", "completed", "failed"]
    progress: float  # 0-100
    config: TrainingConfig
    metrics: Optional[TrainingMetrics]
    model_version_produced: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]

class ModelVersion(BaseModel):
    id: UUID
    version: str  # "v12"
    is_active: bool
    num_classes: int
    accuracy: float
    training_session_id: UUID
    file_path: str
    file_size_mb: float
    created_at: datetime
    parent_version: Optional[str]
```

---

## 💻 MODULE 3 : INTERFACE WEB — FRONTEND

### 3.1 Architecture des Pages

```
Implémente les pages suivantes dans une application React + TypeScript + Tailwind :

NAVIGATION : Bottom tab bar (mobile) / Sidebar (desktop)
  📹 Translate    — Traduction en temps réel
  🎯 Train        — Enregistrer & entraîner de nouveaux signes  
  📖 Dictionary   — Dictionnaire interactif (style Obsidian)
  📊 Dashboard    — Statistiques et gestion des modèles
  ⚙️ Settings     — Configuration
```

### 3.2 Page TRANSLATE (Traduction Temps Réel)

```
DESIGN : Interface minimaliste, la vidéo est le héros.

LAYOUT MOBILE (portrait) :
┌──────────────────────────┐
│  ┌────────────────────┐  │
│  │                    │  │
│  │   FLUX CAMÉRA      │  │  ← 60% de l'écran
│  │   (avec overlay    │  │
│  │    des landmarks   │  │
│  │    MediaPipe)      │  │
│  │                    │  │
│  └────────────────────┘  │
│                          │
│  ┌────────────────────┐  │
│  │ 🟢 "Bonjour"       │  │  ← Mot actuel détecté (gros, animé)
│  │ Confiance: 94%     │  │
│  └────────────────────┘  │
│                          │
│  ┌────────────────────┐  │
│  │ Bonjour, comment   │  │  ← Buffer de phrase en construction
│  │ allez-vous ___     │  │
│  └────────────────────┘  │
│                          │
│  [🔊 Lire] [📋 Copier]  │  ← Actions
│  [🗑️ Reset] [⚙️ Config] │
│                          │
│  ──────────────────────  │
│  📹  🎯  📖  📊  ⚙️     │  ← Bottom nav
└──────────────────────────┘

FONCTIONNALITÉS :
- Toggle caméra front/back
- Overlay des landmarks en temps réel (MediaPipe JS)
- Affichage du mot avec animation de "confiance" (barre de couleur)
- Synthèse vocale (Web Speech API) du texte traduit
- Mode "épeler" : lettre par lettre (alphabet dactylologique)
- Mode "phrases" : accumulation de mots avec ponctuation auto
- Historique des traductions récentes (scroll down)
- Si confiance < 50% : popup "Signe inconnu — voulez-vous l'ajouter au dictionnaire ?"
  → Redirige vers la page Train avec le clip pré-enregistré
```

### 3.3 Page TRAIN (Entraînement de Nouveaux Signes)

```
DESIGN : Interface type "studio d'enregistrement" — guidée, step-by-step.

FLOW UTILISATEUR :

STEP 1 — NOMMER LE SIGNE
┌──────────────────────────┐
│  Nouveau Signe           │
│                          │
│  Nom : [___________]     │
│  Catégorie : [dropdown]  │
│  Tags : [chips input]    │
│  Description : [textarea]│
│  (supporte le Markdown)  │
│                          │
│  [Suivant →]             │
└──────────────────────────┘

STEP 2 — ENREGISTRER DES CLIPS
┌──────────────────────────┐
│  Enregistrez "Bonjour"   │
│  (minimum 5 clips)       │
│                          │
│  ┌────────────────────┐  │
│  │                    │  │
│  │   CAMÉRA LIVE      │  │
│  │   + guide visuel   │  │
│  │   (silhouette)     │  │
│  │                    │  │
│  └────────────────────┘  │
│                          │
│  [● REC] 3.2s           │  ← Gros bouton rouge, timer
│                          │
│  Clips enregistrés :     │
│  ┌───┐ ┌───┐ ┌───┐      │
│  │ ▶ │ │ ▶ │ │ ▶ │ ...  │  ← Thumbnails des clips
│  │ ✓ │ │ ✓ │ │ ✕ │      │  ← Qualité auto-détectée
│  └───┘ └───┘ └───┘      │
│  3/5 clips valides       │
│                          │
│  [← Retour] [Suivant →]  │
└──────────────────────────┘

- Chaque clip : 2-5 secondes
- Auto-détection de qualité :
  • Mains bien visibles ? (landmarks détectés > 80% des frames)
  • Éclairage suffisant ?
  • Geste centré dans le cadre ?
- Preview de chaque clip avec option de supprimer/refaire
- Indicateur de progression : "5/5 clips minimum"
- Bonus : option "miroir" pour varier les angles

STEP 3 — ENTRAÎNEMENT
┌──────────────────────────┐
│  Entraînement en cours   │
│  "Bonjour"               │
│                          │
│  ████████████░░░░ 72%    │  ← Progress bar animée
│                          │
│  Epoch: 36/50            │
│  Loss: 0.234 ↓           │
│  Accuracy: 89.2% ↑       │
│  Val Accuracy: 86.1% ↑   │
│                          │
│  ┌────────────────────┐  │
│  │ 📈 Graphe loss/acc │  │  ← Chart temps réel (Recharts)
│  │    en temps réel   │  │
│  └────────────────────┘  │
│                          │
│  Temps restant : ~45s    │
│                          │
│  [⏹ Arrêter]             │
└──────────────────────────┘

STEP 4 — VALIDATION
┌──────────────────────────┐
│  ✅ Entraînement terminé │
│                          │
│  Accuracy finale : 91.3% │
│  Modèle : v13            │
│                          │
│  Test en direct :        │
│  ┌────────────────────┐  │
│  │ CAMÉRA : testez    │  │
│  │ le signe maintenant│  │
│  │                    │  │
│  │ Résultat : ✅      │  │
│  │ "Bonjour" (93%)   │  │
│  └────────────────────┘  │
│                          │
│  [✓ Valider & Déployer]  │
│  [↻ Plus d'exemples]     │
│  [✕ Annuler]             │
└──────────────────────────┘
```

### 3.4 Page DICTIONARY (Style Obsidian)

```
DESIGN : Interface de knowledge base interconnectée, inspirée d'Obsidian.

DEUX VUES PRINCIPALES (toggle) :

═══ VUE GRAPHE ═══
┌──────────────────────────┐
│  🔍 [recherche...]  [+]  │
│  ──────────────────────  │
│                          │
│     (salut)──(bonjour)   │
│        \      / |        │
│      (hey)   (merci)     │  ← Graphe interactif D3.js
│              |    \      │     Zoom, pan, drag
│          (svp)  (pardon) │     Couleur par catégorie
│           |              │     Taille par usage_count
│        (excusez)         │
│                          │
│  ──────────────────────  │
│  Catégories :            │
│  [Salutations] [Émotions]│
│  [Questions] [Actions]   │
└──────────────────────────┘

- Clic sur un nœud → ouvre le détail du signe
- Drag pour réorganiser
- Filtres par catégorie, tags
- Zoom sémantique : zoom in = plus de détails, zoom out = clusters

═══ VUE LISTE / CARDS ═══
┌──────────────────────────┐
│  🔍 [recherche...]  [+]  │
│  Filtres: [cat▼] [tags▼] │
│  ──────────────────────  │
│                          │
│  ┌────────────────────┐  │
│  │ 🎬 Bonjour         │  │
│  │ #salutation #formel │  │
│  │ 12 vidéos │ 94% acc │  │
│  │ Liens: salut, hey   │  │
│  └────────────────────┘  │
│                          │
│  ┌────────────────────┐  │
│  │ 🎬 Merci           │  │
│  │ #politesse          │  │
│  │ 8 vidéos │ 91% acc  │  │
│  │ Liens: svp, pardon  │  │
│  └────────────────────┘  │
│                          │
│  ... (infinite scroll)   │
└──────────────────────────┘

═══ VUE DÉTAIL D'UN SIGNE (page) ═══
┌──────────────────────────┐
│  ← Retour                │
│                          │
│  # Bonjour               │
│  #salutation #formel     │
│  ──────────────────────  │
│                          │
│  ┌────────────────────┐  │
│  │ ▶ Vidéo de         │  │  ← Vidéo de référence principale
│  │   référence         │  │
│  └────────────────────┘  │
│                          │
│  ## Description          │
│  Le signe "bonjour" se   │
│  fait en portant la main │
│  ouverte au front...     │
│                          │
│  ## Variantes            │
│  - Bonjour formel        │
│  - Bonjour informel      │
│                          │
│  ## Signes liés          │
│  → [[Salut]] [[Hey]]     │  ← Liens cliquables (style Obsidian)
│  → [[Au revoir]]         │
│                          │
│  ## Vidéos (12)          │
│  ┌───┐ ┌───┐ ┌───┐      │
│  │ ▶ │ │ ▶ │ │ ▶ │ ...  │  ← Grille de thumbnails
│  └───┘ └───┘ └───┘      │
│                          │
│  ## Notes                │
│  (éditeur markdown)      │
│  Ce signe est souvent    │
│  confondu avec [[Salut]] │
│  mais la position de la  │
│  main diffère...         │
│                          │
│  ## Statistiques         │
│  Précision modèle: 94%  │
│  Utilisé 234 fois        │
│  Ajouté le 15/01/2025   │
│                          │
│  [✏️ Éditer] [🗑️ Suppr] │
│  [📤 Exporter]           │
└──────────────────────────┘

FONCTIONNALITÉS STYLE OBSIDIAN :
- Liens bidirectionnels : [[nom_du_signe]] dans les notes crée un lien
- Backlinks : afficher "Signes qui mentionnent celui-ci"
- Tags : système de tags avec vue par tag
- Recherche full-text dans noms, descriptions, notes
- Export au format Obsidian (dossier .md + attachments)
- Import depuis un vault Obsidian
```

### 3.5 Page DASHBOARD

```
LAYOUT :
┌──────────────────────────┐
│  📊 Dashboard            │
│  ──────────────────────  │
│                          │
│  ┌─────┐ ┌─────┐ ┌────┐ │
│  │ 247 │ │ 91% │ │ 1.2k│ │  ← KPI cards
│  │signs│ │ acc │ │trans│ │
│  └─────┘ └─────┘ └────┘ │
│                          │
│  Modèle actif : v13      │
│  Dernière MàJ : il y a 2h│
│                          │
│  [📈 Accuracy over time] │  ← Graphe
│  [📊 Signs per category] │  ← Bar chart
│  [🔄 Recent trainings]   │  ← Liste
│                          │
│  Gestion des modèles :   │
│  v13 ✅ (actif) - 91.3%  │
│  v12 - 89.7%    [activer]│
│  v11 - 88.2%    [activer]│
│                          │
│  [📤 Export Dict]         │
│  [📥 Import Dict]         │
│  [🗃️ Export Modèle]      │
└──────────────────────────┘
```

### 3.6 Directives UI/UX Mobile-First

```
PRINCIPES DE DESIGN :

1. MOBILE-FIRST IMPÉRATIF
   - Touch targets minimum 44×44px
   - Bottom navigation (pas de hamburger menu)
   - Swipe gestures pour navigation entre tabs
   - Pas de hover-dependent interactions
   - Font-size minimum 16px (éviter le zoom iOS)

2. PERFORMANCE CAMÉRA
   - Utiliser getUserMedia avec constraints optimisées :
     { video: { facingMode: "user", width: 640, height: 480, frameRate: 30 } }
   - MediaPipe en Web Worker si possible
   - Canvas overlay pour les landmarks (pas de DOM manipulation)
   - RequestAnimationFrame pour le rendering
   - Throttle les envois WebSocket à 10-15 fps (pas besoin de 30)

3. PWA (Progressive Web App)
   - manifest.json avec icônes, splash screens
   - Service Worker pour cache des assets
   - Mode offline : accès au dictionnaire même sans connexion
   - "Add to Home Screen" prompt

4. DESIGN SYSTEM
   - Couleurs :
     • Primary: #6366F1 (indigo — accessible, moderne)
     • Secondary: #10B981 (emerald — succès, validation)  
     • Accent: #F59E0B (amber — attention, en cours)
     • Background: #0F172A (slate-900 — dark mode par défaut)
     • Surface: #1E293B (slate-800)
     • Text: #F8FAFC (slate-50)
   - Typography : 
     • Headings: "Plus Jakarta Sans" (distinctive, moderne)
     • Body: "Inter" (lisibilité optimale)
     • Mono: "JetBrains Mono" (stats, code)
   - Border radius: 12px (cards), 8px (buttons), 9999px (pills)
   - Animations : Framer Motion, transitions douces 200-300ms
   - Dark mode par défaut (environnement caméra = souvent sombre)

5. ACCESSIBILITÉ
   - Labels ARIA sur tous les contrôles vidéo
   - Contraste WCAG AA minimum
   - Navigation clavier complète
   - Screen reader friendly (ironie intentionnelle ET nécessaire)
```

---

## 🐳 MODULE 4 : INFRASTRUCTURE & DÉPLOIEMENT

### 4.1 Structure du Projet

```
signflow/
├── docker-compose.yml
├── docker-compose.dev.yml
├── .env.example
├── README.md
│
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── alembic.ini
│   ├── alembic/
│   │   └── versions/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app factory
│   │   ├── config.py            # Pydantic Settings
│   │   ├── database.py          # SQLAlchemy setup
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── router.py        # Main router
│   │   │   ├── deps.py          # Dependencies (DB session, auth...)
│   │   │   ├── translate.py     # WebSocket translation
│   │   │   ├── signs.py         # CRUD signs
│   │   │   ├── media.py         # Video upload/stream
│   │   │   ├── training.py      # Training sessions
│   │   │   ├── models.py        # Model management
│   │   │   ├── dictionary.py    # Graph, search, export/import
│   │   │   └── stats.py         # Statistics
│   │   │
│   │   ├── models/              # SQLAlchemy models
│   │   │   ├── __init__.py
│   │   │   ├── sign.py
│   │   │   ├── video.py
│   │   │   ├── training.py
│   │   │   └── model_version.py
│   │   │
│   │   ├── schemas/             # Pydantic schemas
│   │   │   ├── __init__.py
│   │   │   ├── sign.py
│   │   │   ├── video.py
│   │   │   ├── training.py
│   │   │   └── model_version.py
│   │   │
│   │   ├── ml/                  # Machine Learning
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py      # Main inference pipeline
│   │   │   ├── model.py         # Transformer architecture
│   │   │   ├── features.py      # Feature extraction (MediaPipe)
│   │   │   ├── trainer.py       # Training loop
│   │   │   ├── fewshot.py       # Few-shot fine-tuning
│   │   │   ├── augmentation.py  # Data augmentation
│   │   │   ├── dataset.py       # PyTorch Dataset
│   │   │   └── utils.py         # Helpers
│   │   │
│   │   ├── services/            # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── sign_service.py
│   │   │   ├── training_service.py
│   │   │   ├── model_service.py
│   │   │   ├── dictionary_service.py
│   │   │   └── media_service.py
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── markdown.py      # Parse [[liens]] style Obsidian
│   │       ├── export.py        # Export Obsidian vault
│   │       └── video.py         # Video processing helpers
│   │
│   ├── tests/
│   │   ├── test_api/
│   │   ├── test_ml/
│   │   └── test_services/
│   │
│   └── data/
│       ├── models/              # Trained model files (.pt)
│       ├── videos/              # Uploaded videos
│       │   ├── training/
│       │   ├── reference/
│       │   └── thumbnails/
│       └── exports/             # Temporary export files
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── index.html
│   ├── public/
│   │   ├── manifest.json
│   │   ├── sw.js
│   │   └── icons/
│   │
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── routes.tsx
│       │
│       ├── components/
│       │   ├── layout/
│       │   │   ├── BottomNav.tsx
│       │   │   ├── Sidebar.tsx
│       │   │   └── PageShell.tsx
│       │   ├── camera/
│       │   │   ├── CameraFeed.tsx        # Composant caméra réutilisable
│       │   │   ├── LandmarkOverlay.tsx   # Overlay MediaPipe
│       │   │   └── RecordButton.tsx
│       │   ├── dictionary/
│       │   │   ├── GraphView.tsx         # D3.js force graph
│       │   │   ├── SignCard.tsx
│       │   │   ├── SignDetail.tsx
│       │   │   ├── MarkdownEditor.tsx
│       │   │   └── BacklinksPanel.tsx
│       │   ├── training/
│       │   │   ├── TrainingWizard.tsx    # Steps 1-4
│       │   │   ├── ClipRecorder.tsx
│       │   │   ├── TrainingProgress.tsx
│       │   │   └── ValidationTest.tsx
│       │   └── common/
│       │       ├── VideoPlayer.tsx
│       │       ├── ConfidenceBadge.tsx
│       │       ├── SearchBar.tsx
│       │       └── TagInput.tsx
│       │
│       ├── hooks/
│       │   ├── useCamera.ts
│       │   ├── useMediaPipe.ts
│       │   ├── useWebSocket.ts
│       │   ├── useTraining.ts
│       │   └── useDictionary.ts
│       │
│       ├── stores/
│       │   ├── translateStore.ts
│       │   ├── trainingStore.ts
│       │   ├── dictionaryStore.ts
│       │   └── settingsStore.ts
│       │
│       ├── api/
│       │   ├── client.ts            # Axios/fetch wrapper
│       │   ├── signs.ts
│       │   ├── training.ts
│       │   ├── models.ts
│       │   └── dictionary.ts
│       │
│       ├── lib/
│       │   ├── mediapipe.ts         # MediaPipe setup & helpers
│       │   ├── landmarks.ts         # Landmark processing
│       │   └── speech.ts            # Web Speech API wrapper
│       │
│       └── styles/
│           └── globals.css
│
└── scripts/
    ├── setup.sh                 # Installation initiale
    ├── seed_data.py             # Données de démo
    └── download_dataset.py      # Télécharger dataset public
```

### 4.2 Docker Compose

```yaml
# Génère un docker-compose.yml avec les services suivants :

services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes:
      - ./backend/data:/app/data
      - ./backend/app:/app/app  # hot reload dev
    environment:
      - DATABASE_URL=postgresql://signflow:signflow@db:5432/signflow
      - REDIS_URL=redis://redis:6379/0
      - MODEL_DIR=/app/data/models
      - VIDEO_DIR=/app/data/videos
    depends_on: [db, redis]

  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    environment:
      - VITE_API_URL=http://localhost:8000
      - VITE_WS_URL=ws://localhost:8000

  db:
    image: postgres:16-alpine
    volumes: [pgdata:/var/lib/postgresql/data]
    environment:
      - POSTGRES_DB=signflow
      - POSTGRES_USER=signflow
      - POSTGRES_PASSWORD=signflow

  redis:
    image: redis:7-alpine

  celery_worker:
    build: ./backend
    command: celery -A app.celery_app worker -l info -Q training
    volumes: [./backend/data:/app/data]
    depends_on: [redis, db]

volumes:
  pgdata:
```

---

## 🚀 INSTRUCTIONS DE DÉVELOPPEMENT

```
ORDRE DE DÉVELOPPEMENT RECOMMANDÉ :

PHASE 1 — FONDATIONS (Semaine 1-2)
  1. Setup Docker + DB + FastAPI skeleton
  2. Modèles SQLAlchemy + migrations Alembic
  3. Schemas Pydantic
  4. CRUD API pour signs + videos
  5. Frontend : layout, routing, bottom nav

PHASE 2 — ML PIPELINE (Semaine 3-4)
  6. MediaPipe feature extraction (Python)
  7. Architecture du Transformer
  8. Dataset loader + training loop
  9. Inférence pipeline
  10. WebSocket translation endpoint

PHASE 3 — INTERFACES CORE (Semaine 5-6)
  11. Page Translate : caméra + MediaPipe JS + WebSocket
  12. Page Train : recording wizard (steps 1-4)
  13. Celery tasks pour entraînement async
  14. Training progress via WebSocket

PHASE 4 — DICTIONNAIRE (Semaine 7-8)
  15. Page Dictionary : vue liste/cards
  16. Vue graphe D3.js
  17. Système de liens [[bidirectionnels]]
  18. Éditeur Markdown
  19. Export/Import Obsidian

PHASE 5 — POLISH (Semaine 9-10)
  20. Dashboard + statistiques
  21. PWA : manifest, service worker, offline
  22. Gestion des modèles (versions, rollback)
  23. Tests (API + ML + E2E)
  24. Documentation API (Swagger auto + guide)

CONTRAINTES :
- Chaque fichier doit avoir des docstrings complètes
- Type hints partout (Python + TypeScript strict)
- Error handling robuste (pas de crash silencieux)
- Logging structuré (structlog)
- Les vidéos doivent être compressées avant stockage (ffmpeg)
- CORS configuré pour dev et prod
- Rate limiting sur les endpoints publics
- Validation des uploads (taille max, format, durée)
```

---

## 📝 NOTES ADDITIONNELLES POUR LE LLM

```
IMPORTANT — CONTRAINTES TECHNIQUES :

1. MediaPipe Holistic est la clé du système. Côté frontend (JS), il tourne 
   dans le navigateur pour l'overlay visuel. Côté backend (Python), il est 
   utilisé pour l'extraction de features des vidéos d'entraînement. Les deux 
   doivent produire des landmarks au même format.

2. Le WebSocket de traduction doit être PERFORMANT :
   - Le frontend envoie les landmarks (pas les images raw !)
   - Le backend fait l'inférence sur les landmarks
   - Latence cible : < 100ms par prédiction
   - Bande passante : ~2-5 KB par frame de landmarks

3. Le few-shot learning est LE differentiator du produit :
   - Un utilisateur doit pouvoir ajouter un nouveau signe en < 5 minutes
   - 5-10 clips suffisent pour une accuracy > 85%
   - L'entraînement doit prendre < 2 minutes sur CPU (pas de GPU requis)
   - Utiliser des techniques de meta-learning si possible (MAML, Prototypical)

4. Le dictionnaire style Obsidian est une FEATURE CLÉ :
   - Les notes supportent la syntaxe [[lien]] pour créer des relations
   - Le parser de markdown doit détecter les [[liens]] et les transformer 
     en vraies relations dans la DB (edges du graphe)
   - L'export Obsidian doit produire un vault fonctionnel :
     chaque signe = un fichier .md, les vidéos en attachments

5. SÉCURITÉ :
   - Les vidéos sont des données sensibles (biométrie)
   - Stockage local uniquement (pas de cloud par défaut)
   - Option de chiffrement des vidéos at rest
   - Pas d'authentification requise pour v1 (single-user local)
   - Préparer les hooks pour auth multi-user en v2

6. Ce projet doit être DEPLOYABLE FACILEMENT :
   - Un seul `docker-compose up` pour tout lancer
   - Script de setup qui télécharge les modèles MediaPipe
   - Données de démo avec 10-20 signes pré-entraînés
   - README détaillé avec screenshots
```

---

> **SignFlow** — Traduire les mains en mots, un geste à la fois. 🤟
