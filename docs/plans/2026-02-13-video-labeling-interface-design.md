# Video Labeling Interface - Design Document

**Date:** 2026-02-13
**Status:** Approved
**Author:** Claude (avec validation utilisateur)

## Résumé Exécutif

Interface de labellisation pour les 46 vidéos d'entraînement non labellisées actuellement dans `backend/data/videos/training/`. L'objectif est de permettre l'association rapide de vidéos à des signes via une interface hybride (grille + modal) avec smart labeling basé sur la similarité des landmarks ML.

**Problème résolu :** Actuellement, le modèle ne peut pas s'entraîner efficacement car les vidéos n'ont pas de labels (metadata `sign_id` manquant). Cette interface permettra de labelliser rapidement les 46 vidéos existantes et futures vidéos.

**Approche choisie :** Backend-first avec ML similarity (Approche 1) - exploite les landmarks déjà extraits pour calculer la similarité entre vidéos et suggérer des labels automatiquement.

---

## 1. Architecture Globale

### 1.1 Navigation & Structure

**Page `/train` avec système d'onglets :**

```
┌─────────────────────────────────────┐
│  New Sign Training                  │
│  ┌─────────┬──────────────┐        │
│  │ Record  │ Label Videos │  (46)  │
│  └─────────┴──────────────┘        │
│                                     │
│  [Contenu de l'onglet actif]       │
└─────────────────────────────────────┘
```

- **Onglet "Record"** : workflow existant (TrainingWizard)
- **Onglet "Label Videos"** : nouvelle interface de labellisation
- Badge numérique dynamique affichant le nombre de vidéos non labellisées
- Persistance de l'onglet actif (localStorage)

### 1.2 Composants Principaux

**Nouveaux composants à créer :**

| Composant | Responsabilité |
|-----------|----------------|
| `VideoLabelingInterface.tsx` | Conteneur principal de l'interface de labellisation |
| `VideoGrid.tsx` | Grille de vignettes vidéo avec filtres/tri |
| `LabelingModal.tsx` | Modal de labellisation individuelle avec lecteur vidéo |
| `SignSelector.tsx` | Sélecteur hybride de signes (recent/search/create) |
| `SuggestionView.tsx` | Affichage des vidéos similaires suggérées |

**Composants réutilisés :**
- `TagInput` (existant)
- Classes Tailwind (card, touch-btn, etc.)

### 1.3 Intégration dans `TrainPage.tsx`

```tsx
// frontend/src/pages/TrainPage.tsx
export function TrainPage() {
  const [activeTab, setActiveTab] = useState<'record' | 'label'>('record');
  const { unlabeledCount } = useLabelingStore();

  return (
    <PageShell>
      <Tabs value={activeTab} onChange={setActiveTab}>
        <Tab label="Record" value="record" />
        <Tab
          label="Label Videos"
          value="label"
          badge={unlabeledCount > 0 ? unlabeledCount : undefined}
        />
      </Tabs>

      {activeTab === 'record' && <TrainingWizard />}
      {activeTab === 'label' && <VideoLabelingInterface />}
    </PageShell>
  );
}
```

---

## 2. Backend Architecture

### 2.1 API Endpoints

**Nouveau fichier : `backend/app/api/videos.py`**

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/videos/unlabeled` | GET | Retourne toutes les vidéos sans `sign_id` avec metadata |
| `/videos/{video_id}/label` | PATCH | Associe une vidéo à un signe |
| `/videos/{video_id}/suggestions` | POST | Calcule et retourne les vidéos similaires |
| `/videos/bulk-label` | PATCH | Labellise plusieurs vidéos en une fois |

**Schémas de réponse :**

```python
# GET /videos/unlabeled
{
  "items": [
    {
      "id": "uuid",
      "file_path": "data/videos/training/xyz.mp4",
      "thumbnail_path": "data/videos/thumbnails/xyz.jpg",
      "duration_ms": 2500,
      "fps": 30,
      "landmarks_extracted": true,
      "landmarks_path": "data/videos/training/xyz_landmarks.npy",
      "created_at": "2026-02-13T10:00:00Z"
    }
  ],
  "total": 46
}

# POST /videos/{id}/suggestions
{
  "suggestions": [
    {
      "id": "uuid",
      "file_path": "...",
      "similarity_score": 0.92,
      "duration_ms": 2400,
      ...
    }
  ]
}
```

### 2.2 Modifications du Modèle de Données

**Migration Alembic nécessaire :**

```python
# backend/alembic/versions/xxxx_make_sign_id_nullable.py

def upgrade():
    op.alter_column(
        'videos',
        'sign_id',
        existing_type=sa.String(36),
        nullable=True  # Changement : False → True
    )

def downgrade():
    # Supprimer les vidéos non labellisées avant rollback
    op.execute("DELETE FROM videos WHERE sign_id IS NULL")
    op.alter_column(
        'videos',
        'sign_id',
        existing_type=sa.String(36),
        nullable=False
    )
```

**Modèle `Video` modifié :**

```python
# backend/app/models/video.py
class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    sign_id: Mapped[Optional[str]] = mapped_column(  # ← Changement ici
        String(36),
        ForeignKey("signs.id", ondelete="SET NULL"),
        nullable=True,  # ← Ajout
        index=True
    )
    # ... reste identique
```

### 2.3 Algorithme de Similarité

**Nouveau module : `backend/app/ml/similarity.py`**

```python
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from app.ml.dataset import temporal_resample
from app.ml.feature_engineering import compute_enriched_features

def compute_video_similarity(
    landmarks_path_a: str | Path,
    landmarks_path_b: str | Path
) -> float:
    """
    Calcule la similarité cosine entre deux vidéos.

    Pipeline :
    1. Charger landmarks (.npy) → [frames_a, 225]
    2. Temporal resampling → [64, 225]
    3. Feature engineering → [64, 469]
    4. Mean pooling → [469] (vecteur représentation)
    5. Cosine similarity entre vecteurs

    Returns:
        float: Score 0.0-1.0 (1.0 = identiques)
    """
    landmarks_a = np.load(landmarks_path_a)
    landmarks_b = np.load(landmarks_path_b)

    # Normalisation temporelle
    seq_a = temporal_resample(landmarks_a, target_len=64)
    seq_b = temporal_resample(landmarks_b, target_len=64)

    # Enrichissement features
    features_a = compute_enriched_features(seq_a)  # [64, 469]
    features_b = compute_enriched_features(seq_b)

    # Mean pooling → embeddings
    embedding_a = features_a.mean(axis=0)  # [469]
    embedding_b = features_b.mean(axis=0)

    # Cosine similarity
    similarity = cosine_similarity(
        embedding_a.reshape(1, -1),
        embedding_b.reshape(1, -1)
    )[0, 0]

    return float(np.clip(similarity, 0.0, 1.0))

def find_similar_videos(
    target_video_path: str | Path,
    candidate_videos: list[Path],
    threshold: float = 0.75,
    top_k: int = 5
) -> list[tuple[Path, float]]:
    """
    Trouve les K vidéos les plus similaires.

    Args:
        target_video_path: Chemin landmarks de la vidéo de référence
        candidate_videos: Liste des chemins landmarks candidats
        threshold: Seuil minimum de similarité (défaut: 0.75)
        top_k: Nombre maximum de suggestions (défaut: 5)

    Returns:
        Liste de (chemin, score) triée par similarité décroissante
    """
    results = []

    for candidate_path in candidate_videos:
        try:
            score = compute_video_similarity(target_video_path, candidate_path)
            if score >= threshold:
                results.append((candidate_path, score))
        except Exception as e:
            # Skip vidéos corrompues ou landmarks invalides
            logger.warning(f"Skipping {candidate_path}: {e}")
            continue

    # Trier par score décroissant et limiter à top_k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
```

**Endpoint de suggestions :**

```python
# backend/app/api/videos.py
@router.post("/videos/{video_id}/suggestions")
async def get_video_suggestions(
    video_id: str,
    db: Session = Depends(get_db),
    threshold: float = 0.75,
    top_k: int = 5
) -> dict:
    """Calcule les vidéos similaires basé sur landmarks."""

    # Récupérer la vidéo source
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video or not video.landmarks_extracted:
        raise HTTPException(400, "Video landmarks not available")

    # Récupérer toutes les vidéos non labellisées (sauf celle-ci)
    unlabeled = db.query(Video).filter(
        Video.sign_id.is_(None),
        Video.id != video_id,
        Video.landmarks_extracted == True
    ).all()

    if not unlabeled:
        return {"suggestions": []}

    # Calculer similarité
    from app.ml.similarity import find_similar_videos

    candidate_paths = [Path(v.landmarks_path) for v in unlabeled]
    similar = find_similar_videos(
        Path(video.landmarks_path),
        candidate_paths,
        threshold=threshold,
        top_k=top_k
    )

    # Mapper résultats
    suggestions = []
    for candidate_path, score in similar:
        candidate_video = next(
            v for v in unlabeled
            if v.landmarks_path == str(candidate_path)
        )
        suggestions.append({
            "id": candidate_video.id,
            "file_path": candidate_video.file_path,
            "thumbnail_path": candidate_video.thumbnail_path,
            "duration_ms": candidate_video.duration_ms,
            "similarity_score": round(score, 3),
            "landmarks_extracted": True
        })

    return {"suggestions": suggestions}
```

### 2.4 Gestion des Landmarks Manquants

**Stratégie :**
- Si `landmarks_extracted=False` : extraction synchrone au premier GET
- Timeout 60s max
- Frontend polling toutes les 2s

```python
# backend/app/api/videos.py
@router.get("/videos/unlabeled")
async def get_unlabeled_videos(
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
) -> dict:
    """Retourne vidéos non labellisées, déclenche extraction si nécessaire."""

    videos = db.query(Video).filter(Video.sign_id.is_(None)).all()

    # Identifier vidéos sans landmarks
    needs_extraction = [
        v for v in videos
        if not v.landmarks_extracted and Path(v.file_path).exists()
    ]

    # Déclencher extraction en arrière-plan
    if needs_extraction and background_tasks:
        for video in needs_extraction:
            background_tasks.add_task(extract_landmarks_sync, video.id, db)

    return {
        "items": [serialize_video(v) for v in videos],
        "total": len(videos)
    }
```

---

## 3. Frontend Interface

### 3.1 Vue Grille (`VideoGrid.tsx`)

**Layout responsive :**

```tsx
// Grille adaptative
// Mobile: 1 colonne
// Tablet: 2 colonnes
// Desktop: 4 colonnes

<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
  {videos.map(video => (
    <VideoCard
      key={video.id}
      video={video}
      onClick={() => onSelectVideo(video)}
    />
  ))}
</div>
```

**Carte vidéo individuelle :**

```tsx
interface VideoCardProps {
  video: UnlabeledVideo;
  onClick: () => void;
}

function VideoCard({ video, onClick }: VideoCardProps) {
  return (
    <div
      className="card cursor-pointer hover:border-blue-500 transition-all"
      onClick={onClick}
    >
      {/* Thumbnail ou première frame */}
      <div className="aspect-video bg-slate-800 relative">
        <img
          src={video.thumbnail_path || generateThumbnail(video.file_path)}
          alt="Video thumbnail"
          className="w-full h-full object-cover"
        />

        {/* Overlay durée */}
        <div className="absolute bottom-2 right-2 bg-black/70 px-2 py-1 rounded text-xs">
          {formatDuration(video.duration_ms)}
        </div>

        {/* Icône état landmarks */}
        <div className="absolute top-2 left-2">
          {video.landmarks_extracted ? (
            <span className="text-emerald-400">✓</span>
          ) : (
            <span className="text-yellow-400">⏳</span>
          )}
        </div>
      </div>

      {/* Metadata */}
      <div className="p-2 text-xs text-slate-400">
        {formatDate(video.created_at)}
      </div>
    </div>
  );
}
```

**Filtres et tri (header) :**

```tsx
<div className="flex gap-4 items-center mb-4">
  <select
    value={sortBy}
    onChange={(e) => setSortBy(e.target.value)}
    className="rounded-btn bg-slate-800 px-3 py-2"
  >
    <option value="date">Date (newest first)</option>
    <option value="duration">Duration</option>
    <option value="landmarks">Landmarks Ready First</option>
  </select>

  <div className="flex-1" />

  <span className="text-sm text-slate-400">
    {videos.length} videos to label
  </span>
</div>
```

**Empty state :**

```tsx
{videos.length === 0 && (
  <div className="flex flex-col items-center justify-center py-20 text-center">
    <div className="text-6xl mb-4">🎉</div>
    <h3 className="text-xl font-heading mb-2">All videos labeled!</h3>
    <p className="text-slate-400 mb-6">
      Ready to train your model with {totalLabeled} videos.
    </p>
    <button
      className="touch-btn bg-primary text-white"
      onClick={() => setActiveTab('record')}
    >
      Go to Training
    </button>
  </div>
)}
```

### 3.2 Modal de Labellisation (`LabelingModal.tsx`)

**Structure du modal :**

```tsx
interface LabelingModalProps {
  video: UnlabeledVideo;
  onClose: () => void;
  onLabeled: (videoId: string, signId: string) => void;
  onSkip: () => void;
}

function LabelingModal({ video, onClose, onLabeled, onSkip }: LabelingModalProps) {
  const [selectedSignId, setSelectedSignId] = useState<string | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="bg-slate-900 rounded-lg w-full max-w-3xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex justify-between items-center p-4 border-b border-slate-700">
          <h2 className="font-heading text-lg">Label Video</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white">
            ✕
          </button>
        </div>

        {/* Video Player (60% height) */}
        <div className="p-4 bg-black">
          <video
            src={video.file_path}
            controls
            autoPlay
            loop
            className="w-full aspect-video"
          />
          <div className="flex gap-4 mt-2 text-xs text-slate-400">
            <span>Duration: {formatDuration(video.duration_ms)}</span>
            <span>FPS: {video.fps}</span>
          </div>
        </div>

        {/* Sign Selection (40% height) */}
        <div className="p-4 space-y-3 max-h-[400px] overflow-y-auto">
          <SignSelector
            onSelectSign={setSelectedSignId}
            selectedSignId={selectedSignId}
            showCreateForm={showCreateForm}
            onToggleCreateForm={() => setShowCreateForm(!showCreateForm)}
          />
        </div>

        {/* Footer Actions */}
        <div className="flex gap-2 p-4 border-t border-slate-700">
          <button
            className="touch-btn bg-slate-700 text-white"
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            className="touch-btn bg-slate-600 text-white"
            onClick={onSkip}
          >
            Skip
          </button>
          <div className="flex-1" />
          <button
            className="touch-btn bg-primary text-white disabled:opacity-50"
            disabled={!selectedSignId}
            onClick={() => selectedSignId && onLabeled(video.id, selectedSignId)}
          >
            Save Label →
          </button>
        </div>
      </div>
    </div>
  );
}
```

**Keyboard shortcuts :**

```tsx
useEffect(() => {
  const handleKeyPress = (e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose();
    if (e.key === 'Enter' && selectedSignId) {
      onLabeled(video.id, selectedSignId);
    }
    if (e.key === ' ') {
      // Toggle play/pause
      const videoEl = document.querySelector('video');
      if (videoEl) {
        videoEl.paused ? videoEl.play() : videoEl.pause();
      }
    }
  };

  window.addEventListener('keydown', handleKeyPress);
  return () => window.removeEventListener('keydown', handleKeyPress);
}, [selectedSignId, video.id, onClose, onLabeled]);
```

### 3.3 Sign Selector (`SignSelector.tsx`)

**3 modes d'interaction :**

```tsx
interface SignSelectorProps {
  selectedSignId: string | null;
  onSelectSign: (signId: string) => void;
  showCreateForm: boolean;
  onToggleCreateForm: () => void;
}

function SignSelector({
  selectedSignId,
  onSelectSign,
  showCreateForm,
  onToggleCreateForm
}: SignSelectorProps) {
  const { recentSigns, searchSigns } = useSignsStore();
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState<Sign[]>([]);

  // Debounced search
  useEffect(() => {
    if (query.length < 2) {
      setSuggestions([]);
      return;
    }

    const timer = setTimeout(() => {
      searchSigns(query).then(setSuggestions);
    }, 300);

    return () => clearTimeout(timer);
  }, [query, searchSigns]);

  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium">Assign to Sign</label>

      {/* 1. Recent Signs (Quick Picks) */}
      {recentSigns.length > 0 && (
        <div>
          <p className="text-xs text-slate-400 mb-2">Recent</p>
          <div className="flex gap-2 flex-wrap">
            {recentSigns.slice(0, 5).map(sign => (
              <button
                key={sign.id}
                className={cn(
                  "touch-btn",
                  selectedSignId === sign.id
                    ? "bg-primary text-white"
                    : "bg-slate-700 text-slate-200"
                )}
                onClick={() => onSelectSign(sign.id)}
              >
                {sign.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 2. Autocomplete Search */}
      <div>
        <p className="text-xs text-slate-400 mb-2">Search</p>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search existing signs..."
          list="sign-suggestions"
          className="w-full rounded-btn border border-slate-700 bg-slate-800 px-3 py-2"
        />
        <datalist id="sign-suggestions">
          {suggestions.map(sign => (
            <option key={sign.id} value={sign.name} />
          ))}
        </datalist>

        {/* Résultats recherche */}
        {suggestions.length > 0 && (
          <div className="mt-2 space-y-1">
            {suggestions.map(sign => (
              <button
                key={sign.id}
                className="w-full text-left px-3 py-2 rounded hover:bg-slate-700 text-sm"
                onClick={() => {
                  onSelectSign(sign.id);
                  setQuery('');
                  setSuggestions([]);
                }}
              >
                {sign.name}
                {sign.category && (
                  <span className="text-xs text-slate-400 ml-2">
                    {sign.category}
                  </span>
                )}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* 3. Create New Sign */}
      <div>
        {!showCreateForm ? (
          <button
            className="text-sm text-blue-400 hover:underline"
            onClick={onToggleCreateForm}
          >
            + Create new sign
          </button>
        ) : (
          <CreateSignForm
            onCreated={(sign) => {
              onSelectSign(sign.id);
              onToggleCreateForm();
            }}
            onCancel={onToggleCreateForm}
          />
        )}
      </div>
    </div>
  );
}
```

**Formulaire de création inline :**

```tsx
interface CreateSignFormProps {
  onCreated: (sign: Sign) => void;
  onCancel: () => void;
}

function CreateSignForm({ onCreated, onCancel }: CreateSignFormProps) {
  const [name, setName] = useState('');
  const [category, setCategory] = useState('lsfb-v1');
  const [tags, setTags] = useState<string[]>(['lsfb', 'v1']);
  const [isCreating, setIsCreating] = useState(false);

  const handleCreate = async () => {
    if (!name.trim()) return;

    setIsCreating(true);
    try {
      const sign = await createSign({
        name: normalizeSignName(name),
        category,
        tags,
        description: `Custom sign: ${name}`,
        variants: [],
        related_signs: [],
        notes: ''
      });
      onCreated(sign);
    } catch (error) {
      // Gérer erreur (toast notification)
      console.error('Failed to create sign:', error);
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <div className="card p-3 space-y-2 border border-slate-600">
      <input
        type="text"
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Sign name (e.g., bonjour)"
        className="w-full rounded border border-slate-700 bg-slate-800 px-2 py-1 text-sm"
        autoFocus
      />

      <input
        type="text"
        value={category}
        onChange={(e) => setCategory(e.target.value)}
        placeholder="Category"
        className="w-full rounded border border-slate-700 bg-slate-800 px-2 py-1 text-sm"
      />

      <TagInput tags={tags} onChange={setTags} />

      <div className="flex gap-2 pt-2">
        <button
          className="touch-btn bg-slate-600 text-white text-sm"
          onClick={onCancel}
        >
          Cancel
        </button>
        <button
          className="touch-btn bg-primary text-white text-sm disabled:opacity-50"
          disabled={!name.trim() || isCreating}
          onClick={handleCreate}
        >
          {isCreating ? 'Creating...' : 'Create & Assign'}
        </button>
      </div>
    </div>
  );
}
```

### 3.4 Smart Labeling (Suggestions View)

**Flow après labellisation :**

```tsx
// Dans LabelingModal après succès du PATCH
const handleLabelSuccess = async (videoId: string, signId: string) => {
  // 1. Optimistic update
  removeVideoFromGrid(videoId);

  // 2. Calculer suggestions
  setIsLoadingSuggestions(true);
  try {
    const { suggestions } = await getSuggestions(videoId);

    if (suggestions.length > 0) {
      // Afficher suggestions
      setSuggestions(suggestions);
      setShowSuggestions(true);
    } else {
      // Pas de suggestions, fermer modal
      onClose();
    }
  } catch (error) {
    // Timeout ou erreur, continuer sans suggestions
    onClose();
  } finally {
    setIsLoadingSuggestions(false);
  }
};
```

**Vue des suggestions :**

```tsx
interface SuggestionViewProps {
  labeledVideo: UnlabeledVideo;
  assignedSign: Sign;
  suggestions: SuggestedVideo[];
  onApplyAll: (videoIds: string[]) => void;
  onReview: () => void;
  onSkip: () => void;
}

function SuggestionView({
  labeledVideo,
  assignedSign,
  suggestions,
  onApplyAll,
  onReview,
  onSkip
}: SuggestionViewProps) {
  const [selected, setSelected] = useState<Set<string>>(
    new Set(suggestions.map(s => s.id))
  );

  return (
    <div className="space-y-4">
      {/* Success Header */}
      <div className="bg-emerald-500/10 border border-emerald-500/40 rounded p-3">
        <p className="text-emerald-100 font-medium">
          ✓ Video labeled as: {assignedSign.name}
        </p>
      </div>

      {/* Suggestions Header */}
      <div className="flex items-center gap-2">
        <span className="text-lg">🤖</span>
        <h3 className="font-heading">Similar videos detected</h3>
      </div>

      {/* Grid de suggestions */}
      <div className="grid grid-cols-3 gap-3">
        {suggestions.map(video => (
          <SuggestionCard
            key={video.id}
            video={video}
            isSelected={selected.has(video.id)}
            onToggle={() => {
              setSelected(prev => {
                const next = new Set(prev);
                if (next.has(video.id)) {
                  next.delete(video.id);
                } else {
                  next.add(video.id);
                }
                return next;
              });
            }}
          />
        ))}
      </div>

      {/* Actions */}
      <div className="bg-slate-800/50 rounded p-3">
        <p className="text-sm text-slate-300 mb-3">
          Apply "{assignedSign.name}" to {selected.size} selected video(s)?
        </p>

        <div className="flex gap-2">
          <button
            className="touch-btn bg-slate-700 text-white text-sm"
            onClick={onSkip}
          >
            Skip suggestions
          </button>

          <button
            className="touch-btn bg-slate-600 text-white text-sm"
            onClick={onReview}
          >
            Review individually
          </button>

          <div className="flex-1" />

          <button
            className="touch-btn bg-primary text-white disabled:opacity-50"
            disabled={selected.size === 0}
            onClick={() => onApplyAll(Array.from(selected))}
          >
            Apply to {selected.size} video(s) →
          </button>
        </div>
      </div>
    </div>
  );
}
```

**Carte de suggestion individuelle :**

```tsx
interface SuggestionCardProps {
  video: SuggestedVideo;
  isSelected: boolean;
  onToggle: () => void;
}

function SuggestionCard({ video, isSelected, onToggle }: SuggestionCardProps) {
  const confidenceLevel = video.similarity_score >= 0.9
    ? 'high'
    : video.similarity_score >= 0.75
    ? 'medium'
    : 'low';

  const borderColor = {
    high: 'border-emerald-500',
    medium: 'border-yellow-500',
    low: 'border-slate-500'
  }[confidenceLevel];

  return (
    <div
      className={cn(
        "card cursor-pointer transition-all relative",
        isSelected ? borderColor : 'border-slate-700',
        isSelected && 'ring-2 ring-blue-500'
      )}
      onClick={onToggle}
    >
      {/* Checkbox */}
      <div className="absolute top-2 left-2 z-10">
        <input
          type="checkbox"
          checked={isSelected}
          onChange={onToggle}
          className="w-4 h-4"
        />
      </div>

      {/* Thumbnail */}
      <div className="aspect-video bg-slate-800">
        <img
          src={video.thumbnail_path}
          alt="Suggestion"
          className="w-full h-full object-cover"
        />
      </div>

      {/* Score */}
      <div className="p-2 text-center">
        <span className={cn(
          "text-sm font-mono",
          confidenceLevel === 'high' && 'text-emerald-400',
          confidenceLevel === 'medium' && 'text-yellow-400',
          confidenceLevel === 'low' && 'text-slate-400'
        )}>
          {Math.round(video.similarity_score * 100)}% match
        </span>
      </div>
    </div>
  );
}
```

---

## 4. State Management

### 4.1 Zustand Store

**Nouveau store : `frontend/src/stores/labelingStore.ts`**

```typescript
import { create } from 'zustand';
import {
  getUnlabeledVideos,
  labelVideo,
  getSuggestions,
  bulkLabelVideos,
  type UnlabeledVideo,
  type SuggestedVideo
} from '../api/videos';
import { listSigns, type Sign } from '../api/signs';

interface LabelingState {
  // Data
  unlabeledVideos: UnlabeledVideo[];
  recentSigns: Sign[];
  selectedVideo: UnlabeledVideo | null;
  suggestions: SuggestedVideo[];

  // UI State
  isLoading: boolean;
  isLoadingSuggestions: boolean;
  error: string | null;

  // Actions
  loadUnlabeledVideos: () => Promise<void>;
  selectVideo: (video: UnlabeledVideo | null) => void;
  labelVideo: (videoId: string, signId: string) => Promise<void>;
  applySuggestions: (videoIds: string[], signId: string) => Promise<void>;
  refreshAfterLabel: () => void;
  addToRecentSigns: (sign: Sign) => void;
}

export const useLabelingStore = create<LabelingState>((set, get) => ({
  // Initial state
  unlabeledVideos: [],
  recentSigns: [],
  selectedVideo: null,
  suggestions: [],
  isLoading: false,
  isLoadingSuggestions: false,
  error: null,

  // Load unlabeled videos
  loadUnlabeledVideos: async () => {
    set({ isLoading: true, error: null });
    try {
      const videos = await getUnlabeledVideos();
      set({ unlabeledVideos: videos, isLoading: false });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Failed to load videos',
        isLoading: false
      });
    }
  },

  // Select video for labeling
  selectVideo: (video) => {
    set({ selectedVideo: video, suggestions: [] });
  },

  // Label a video
  labelVideo: async (videoId: string, signId: string) => {
    try {
      // API call
      await labelVideo(videoId, signId);

      // Optimistic update
      set(state => ({
        unlabeledVideos: state.unlabeledVideos.filter(v => v.id !== videoId)
      }));

      // Fetch suggestions
      set({ isLoadingSuggestions: true });
      const { suggestions } = await getSuggestions(videoId);
      set({ suggestions, isLoadingSuggestions: false });

    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Failed to label video'
      });
      throw error;
    }
  },

  // Apply label to multiple videos
  applySuggestions: async (videoIds: string[], signId: string) => {
    try {
      await bulkLabelVideos(videoIds, signId);

      // Remove from unlabeled list
      set(state => ({
        unlabeledVideos: state.unlabeledVideos.filter(
          v => !videoIds.includes(v.id)
        ),
        suggestions: []
      }));
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Failed to apply labels'
      });
      throw error;
    }
  },

  // Refresh after labeling
  refreshAfterLabel: () => {
    set({ selectedVideo: null, suggestions: [] });
    get().loadUnlabeledVideos();
  },

  // Add to recent signs (max 10)
  addToRecentSigns: (sign: Sign) => {
    set(state => {
      const filtered = state.recentSigns.filter(s => s.id !== sign.id);
      return {
        recentSigns: [sign, ...filtered].slice(0, 10)
      };
    });
  }
}));
```

### 4.2 API Client

**Nouveau fichier : `frontend/src/api/videos.ts`**

```typescript
import { apiClient } from './client';

export interface UnlabeledVideo {
  id: string;
  file_path: string;
  thumbnail_path?: string;
  duration_ms: number;
  fps: number;
  resolution: string;
  landmarks_extracted: boolean;
  landmarks_path?: string;
  created_at: string;
}

export interface SuggestedVideo extends UnlabeledVideo {
  similarity_score: number;
}

export async function getUnlabeledVideos(): Promise<UnlabeledVideo[]> {
  const response = await apiClient.get<{ items: UnlabeledVideo[] }>(
    '/videos/unlabeled'
  );
  return response.data.items;
}

export async function labelVideo(
  videoId: string,
  signId: string
): Promise<void> {
  await apiClient.patch(`/videos/${videoId}/label`, { sign_id: signId });
}

export async function getSuggestions(
  videoId: string,
  threshold: number = 0.75
): Promise<{ suggestions: SuggestedVideo[] }> {
  const response = await apiClient.post<{ suggestions: SuggestedVideo[] }>(
    `/videos/${videoId}/suggestions`,
    { threshold }
  );
  return response.data;
}

export async function bulkLabelVideos(
  videoIds: string[],
  signId: string
): Promise<void> {
  await apiClient.patch('/videos/bulk-label', {
    video_ids: videoIds,
    sign_id: signId
  });
}
```

---

## 5. Gestion d'Erreurs & Cas Limites

### 5.1 Erreurs Backend

**1. Landmarks non extraits :**

```tsx
// Affichage dans VideoCard
{!video.landmarks_extracted && (
  <div className="absolute inset-0 bg-yellow-500/20 flex items-center justify-center">
    <div className="text-center">
      <div className="text-yellow-400 text-2xl mb-1">⏳</div>
      <p className="text-xs text-yellow-200">Extracting...</p>
    </div>
  </div>
)}

// Polling jusqu'à extraction complète
useEffect(() => {
  const needsExtraction = unlabeledVideos.some(v => !v.landmarks_extracted);

  if (!needsExtraction) return;

  const interval = setInterval(() => {
    loadUnlabeledVideos(); // Refresh
  }, 2000);

  return () => clearInterval(interval);
}, [unlabeledVideos]);
```

**2. Vidéo corrompue ou manquante :**

```tsx
// Dans LabelingModal
<video
  src={video.file_path}
  onError={(e) => {
    setVideoError('Video file corrupted or missing');
  }}
  controls
/>

{videoError && (
  <div className="bg-red-500/10 border border-red-500/40 rounded p-3 text-red-200">
    ❌ {videoError}
    <button
      className="ml-4 underline"
      onClick={onSkip}
    >
      Skip this video
    </button>
  </div>
)}
```

**3. Suggestions timeout :**

```tsx
// Timeout 5s pour calcul suggestions
const getSuggestionsWithTimeout = async (videoId: string) => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);

  try {
    const result = await getSuggestions(videoId, {
      signal: controller.signal
    });
    clearTimeout(timeout);
    return result;
  } catch (error) {
    if (error.name === 'AbortError') {
      // Timeout, continuer sans suggestions
      return { suggestions: [] };
    }
    throw error;
  }
};
```

**4. Aucune suggestion trouvée :**

```tsx
{suggestions.length === 0 && !isLoadingSuggestions && (
  <div className="text-center py-4 text-slate-400">
    No similar videos found (threshold: 75%)
    <p className="text-xs mt-1">Closing automatically...</p>
  </div>
)}

// Auto-close après 2s
useEffect(() => {
  if (suggestions.length === 0 && !isLoadingSuggestions) {
    const timer = setTimeout(onClose, 2000);
    return () => clearTimeout(timer);
  }
}, [suggestions, isLoadingSuggestions, onClose]);
```

### 5.2 Cas Limites

**Dernière vidéo non labellisée :**

```tsx
{unlabeledVideos.length === 1 && (
  <div className="bg-emerald-500/10 border border-emerald-500/40 rounded p-3 mb-4">
    <p className="text-emerald-100">
      🎉 Last video! You'll be done after this one.
    </p>
  </div>
)}
```

**Création de sign échoue (duplicate) :**

```typescript
try {
  const newSign = await createSign({ name, category, tags });
  onCreated(newSign);
} catch (error) {
  if (error.response?.status === 409) {
    // Sign existe déjà, le récupérer
    const existing = await listSigns(name);
    const match = existing.items.find(s =>
      s.name.toLowerCase() === name.toLowerCase()
    );
    if (match) {
      toast.info(`Using existing sign: ${match.name}`);
      onCreated(match);
      return;
    }
  }
  toast.error('Failed to create sign');
}
```

**Navigation rapide (UX) :**

```tsx
// Breadcrumb navigation
<div className="flex items-center gap-2 text-sm text-slate-400">
  <button
    onClick={goToPrevious}
    disabled={currentIndex === 0}
    className="disabled:opacity-30"
  >
    ← Previous
  </button>

  <span>
    Video {currentIndex + 1} of {total}
  </span>

  <button
    onClick={goToNext}
    disabled={currentIndex === total - 1}
    className="disabled:opacity-30"
  >
    Next →
  </button>
</div>

// Progress bar
<div className="w-full bg-slate-700 h-1 rounded-full overflow-hidden">
  <div
    className="bg-primary h-full transition-all"
    style={{ width: `${(labeledCount / totalCount) * 100}%` }}
  />
</div>
```

**Keyboard shortcuts globaux :**

```tsx
useEffect(() => {
  const handleKeyPress = (e: KeyboardEvent) => {
    // N = Next video
    if (e.key === 'n' && !isModalOpen) {
      selectNextVideo();
    }

    // P = Previous video
    if (e.key === 'p' && !isModalOpen) {
      selectPreviousVideo();
    }

    // Esc = Close modal
    if (e.key === 'Escape' && isModalOpen) {
      closeModal();
    }
  };

  window.addEventListener('keydown', handleKeyPress);
  return () => window.removeEventListener('keydown', handleKeyPress);
}, [isModalOpen, selectNextVideo, selectPreviousVideo, closeModal]);
```

---

## 6. Tests & Validation

### 6.1 Tests Backend

**Tests unitaires (`backend/tests/test_similarity.py`) :**

```python
import pytest
import numpy as np
from app.ml.similarity import compute_video_similarity, find_similar_videos

def test_identical_videos_have_high_similarity():
    """Deux vidéos identiques doivent avoir similarité ~1.0"""
    landmarks = np.random.rand(60, 225)

    # Sauvegarder temporairement
    temp_file = "/tmp/test_landmarks.npy"
    np.save(temp_file, landmarks)

    similarity = compute_video_similarity(temp_file, temp_file)
    assert similarity > 0.99

def test_different_videos_have_low_similarity():
    """Deux vidéos très différentes doivent avoir similarité faible"""
    landmarks_a = np.random.rand(60, 225)
    landmarks_b = np.random.rand(60, 225) * 10  # Très différent

    temp_a = "/tmp/test_a.npy"
    temp_b = "/tmp/test_b.npy"
    np.save(temp_a, landmarks_a)
    np.save(temp_b, landmarks_b)

    similarity = compute_video_similarity(temp_a, temp_b)
    assert similarity < 0.5

def test_find_similar_videos_filters_by_threshold():
    """find_similar_videos doit respecter le threshold"""
    # ... test implementation
```

**Tests d'intégration API (`backend/tests/test_videos_api.py`) :**

```python
def test_get_unlabeled_videos_returns_only_unlabeled(client, db):
    """GET /videos/unlabeled ne retourne que les vidéos sans sign_id"""
    # Setup: créer 5 vidéos labellisées + 3 non labellisées
    # ...

    response = client.get("/videos/unlabeled")
    assert response.status_code == 200
    assert len(response.json()["items"]) == 3

def test_label_video_updates_sign_id(client, db):
    """PATCH /videos/{id}/label met à jour le sign_id"""
    # ...

def test_suggestions_returns_similar_videos(client, db):
    """POST /videos/{id}/suggestions retourne vidéos similaires"""
    # ...
```

### 6.2 Tests Frontend

**Tests composants (`frontend/src/components/__tests__/VideoGrid.test.tsx`) :**

```tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { VideoGrid } from '../VideoGrid';

describe('VideoGrid', () => {
  it('renders video cards for each unlabeled video', () => {
    const videos = [
      { id: '1', file_path: '/v1.mp4', duration_ms: 2000, ... },
      { id: '2', file_path: '/v2.mp4', duration_ms: 3000, ... }
    ];

    render(<VideoGrid videos={videos} onSelectVideo={jest.fn()} />);

    expect(screen.getAllByRole('button')).toHaveLength(2);
  });

  it('calls onSelectVideo when card is clicked', () => {
    const handleSelect = jest.fn();
    const videos = [{ id: '1', ... }];

    render(<VideoGrid videos={videos} onSelectVideo={handleSelect} />);

    fireEvent.click(screen.getByRole('button'));
    expect(handleSelect).toHaveBeenCalledWith(videos[0]);
  });

  it('shows empty state when no videos', () => {
    render(<VideoGrid videos={[]} onSelectVideo={jest.fn()} />);

    expect(screen.getByText(/All videos labeled/i)).toBeInTheDocument();
  });
});
```

### 6.3 Checklist de Validation

**Avant merge :**

- [ ] Migration Alembic testée (upgrade + downgrade)
- [ ] API endpoints testés avec Postman/curl
- [ ] Similarité ML validée sur 5-10 paires de vidéos
- [ ] Interface responsive testée (mobile/tablet/desktop)
- [ ] Keyboard shortcuts fonctionnels
- [ ] Gestion d'erreurs testée (réseau coupé, timeouts)
- [ ] Performance : grille avec 100+ vidéos reste fluide
- [ ] Accessibilité : navigation clavier complète

---

## 7. Plan d'Implémentation

### Phase 1 : Backend (Priorité 1)
1. Migration Alembic pour rendre `sign_id` nullable
2. Créer `backend/app/ml/similarity.py`
3. Créer `backend/app/api/videos.py` avec 4 endpoints
4. Tests unitaires pour similarité ML
5. Tests API endpoints

**Estimation :** 3-4h

### Phase 2 : Frontend Core (Priorité 1)
1. Créer store `labelingStore.ts`
2. Créer API client `videos.ts`
3. Modifier `TrainPage.tsx` avec système d'onglets
4. Créer `VideoGrid.tsx` avec cards et filtres
5. Créer `LabelingModal.tsx` avec lecteur vidéo

**Estimation :** 3-4h

### Phase 3 : Smart Labeling (Priorité 2)
1. Créer `SignSelector.tsx` avec 3 modes
2. Créer `SuggestionView.tsx`
3. Intégrer calcul similarité dans flow
4. Implémenter bulk labeling
5. Ajouter optimistic updates

**Estimation :** 2-3h

### Phase 4 : Polish & Tests (Priorité 3)
1. Gestion d'erreurs complète
2. Keyboard shortcuts
3. Empty states & loading states
4. Tests frontend
5. Documentation utilisateur

**Estimation :** 2h

**Total estimé :** 10-13 heures

---

## 8. Considérations Futures

### Améliorations Possibles (Phase 2)

1. **Extraction landmarks en arrière-plan**
   - Worker async pour extraction batch
   - Progress bar pour extraction en cours

2. **Filtres avancés**
   - Par date range
   - Par qualité landmarks (détection confidence)
   - Par durée vidéo

3. **Analytics**
   - Temps moyen de labellisation par vidéo
   - Distribution des signes labellisés
   - Accuracy des suggestions acceptées

4. **Bulk operations**
   - Sélection multiple dans grille
   - Actions : "Delete", "Export", "Re-extract landmarks"

5. **Historique & Undo**
   - Annuler dernière labellisation
   - Historique des modifications

### Optimisations Performance

1. **Cache embeddings**
   - Pré-calculer embeddings à l'extraction landmarks
   - Sauvegarder dans DB ou fichier séparé
   - Accélère calcul similarité 10-100x

2. **Thumbnails optimisés**
   - Générer thumbnails à l'upload
   - WebP format pour taille réduite
   - Lazy loading avec IntersectionObserver

3. **Pagination**
   - Si >100 vidéos, paginer par 50
   - Ou scroll infini avec react-window

---

## 9. Dépendances & Prérequis

### Backend

**Packages Python (requirements.txt) :**
```txt
scikit-learn>=1.3.0  # Pour cosine_similarity
```

**Migration Alembic :**
```bash
cd backend
alembic revision -m "make_video_sign_id_nullable"
alembic upgrade head
```

### Frontend

**Packages NPM (package.json) :**
```json
{
  "dependencies": {
    // Déjà installés, aucun nouveau package requis
  }
}
```

### Infrastructure

- PostgreSQL 14+ (déjà en place)
- Espace disque : ~500MB pour 100 vidéos + landmarks + thumbnails

---

## 10. Métriques de Succès

**KPIs après implémentation :**

1. **Temps de labellisation** : <30s par vidéo en moyenne
2. **Taux d'acceptation suggestions** : >70% des suggestions acceptées
3. **Couverture** : 100% des vidéos existantes labellisées en <1h
4. **Accuracy suggestions** : >85% des suggestions sont correctes (validation manuelle)
5. **Performance** : Grille reste fluide (<100ms) avec 100+ vidéos

**Validation du succès :**
- [ ] 46 vidéos actuelles labellisées et utilisables pour entraînement
- [ ] Modèle peut s'entraîner avec >5 classes différentes
- [ ] Val_accuracy >70% après réentraînement
- [ ] Interface utilisable sans documentation (intuitive)

---

## Conclusion

Cette conception fournit une interface de labellisation complète, ergonomique et intelligente qui exploite les capacités ML existantes du projet SignFlow. L'approche hybride (grille + modal) combinée au smart labeling basé sur la similarité permettra de labelliser rapidement les 46 vidéos existantes et de faciliter le workflow pour les futures vidéos.

Le système est conçu pour être extensible (futures optimisations, analytics, bulk operations) tout en restant simple à implémenter dans un premier temps (10-13h estimées).

**Prochaine étape :** Créer le plan d'implémentation détaillé avec writing-plans skill.
