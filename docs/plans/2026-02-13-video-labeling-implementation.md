# Video Labeling Interface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implémenter une interface de labellisation hybride (grille + modal) avec smart labeling ML pour les 46 vidéos non labellisées actuelles.

**Architecture:** Backend-first avec calcul de similarité ML entre landmarks extraits. Frontend React avec onglets, grille responsive, modal de labellisation et suggestions automatiques. Store Zustand pour état global.

**Tech Stack:** FastAPI, SQLAlchemy, Alembic, scikit-learn, React 18, TypeScript, Zustand, Tailwind CSS

---

## Phase 1: Backend Foundation

### Task 1: Database Migration - Make sign_id Nullable

**Files:**
- Create: `backend/alembic/versions/xxxx_make_sign_id_nullable.py`
- Modify: `backend/app/models/video.py:21`

**Step 1: Create Alembic migration file**

```bash
cd backend
alembic revision -m "make video sign_id nullable for unlabeled videos"
```

Expected: New file created in `backend/alembic/versions/`

**Step 2: Write upgrade migration**

Dans le fichier créé :

```python
"""make video sign_id nullable for unlabeled videos

Revision ID: xxxx
Revises: yyyy
Create Date: 2026-02-13
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'xxxx'
down_revision = 'yyyy'  # Update with actual previous revision
branch_labels = None
depends_on = None

def upgrade():
    op.alter_column(
        'videos',
        'sign_id',
        existing_type=sa.String(36),
        nullable=True
    )

def downgrade():
    # Delete unlabeled videos before making column NOT NULL again
    op.execute("DELETE FROM videos WHERE sign_id IS NULL")
    op.alter_column(
        'videos',
        'sign_id',
        existing_type=sa.String(36),
        nullable=False
    )
```

**Step 3: Update Video model**

Dans `backend/app/models/video.py`:

```python
# Ligne 21, modifier :
# Avant:
sign_id: Mapped[str] = mapped_column(String(36), ForeignKey("signs.id", ondelete="CASCADE"), index=True)

# Après:
sign_id: Mapped[Optional[str]] = mapped_column(
    String(36),
    ForeignKey("signs.id", ondelete="SET NULL"),
    nullable=True,
    index=True
)
```

**Step 4: Run migration**

```bash
cd backend
alembic upgrade head
```

Expected: Migration applies successfully

**Step 5: Verify migration**

```bash
cd backend
python3 -c "
from app.database import SessionLocal
from app.models.video import Video

db = SessionLocal()
# Test que sign_id peut être NULL
video_count = db.query(Video).filter(Video.sign_id.is_(None)).count()
print(f'Unlabeled videos: {video_count}')
db.close()
"
```

Expected: Prints count without error

**Step 6: Commit**

```bash
git add backend/alembic/versions/*.py backend/app/models/video.py
git commit -m "feat(db): make video sign_id nullable for labeling support

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2: ML Similarity Module

**Files:**
- Create: `backend/app/ml/similarity.py`
- Create: `backend/tests/ml/test_similarity.py`

**Step 1: Write test for compute_video_similarity**

Créer `backend/tests/ml/test_similarity.py`:

```python
"""Tests for video similarity computation."""
import numpy as np
import pytest
from pathlib import Path
from app.ml.similarity import compute_video_similarity, find_similar_videos


def test_identical_videos_have_perfect_similarity(tmp_path):
    """Identical videos should have similarity ~1.0"""
    landmarks = np.random.rand(60, 225).astype(np.float32)

    path_a = tmp_path / "video_a.npy"
    path_b = tmp_path / "video_b.npy"

    np.save(path_a, landmarks)
    np.save(path_b, landmarks)

    similarity = compute_video_similarity(path_a, path_b)

    assert similarity > 0.99, f"Expected >0.99, got {similarity}"
    assert similarity <= 1.0


def test_different_videos_have_low_similarity(tmp_path):
    """Very different videos should have low similarity"""
    landmarks_a = np.random.rand(60, 225).astype(np.float32)
    landmarks_b = np.random.rand(60, 225).astype(np.float32) * 10

    path_a = tmp_path / "video_a.npy"
    path_b = tmp_path / "video_b.npy"

    np.save(path_a, landmarks_a)
    np.save(path_b, landmarks_b)

    similarity = compute_video_similarity(path_a, path_b)

    assert similarity < 0.5, f"Expected <0.5, got {similarity}"


def test_find_similar_videos_respects_threshold(tmp_path):
    """find_similar_videos should filter by threshold"""
    # Create target video
    target_landmarks = np.random.rand(60, 225).astype(np.float32)
    target_path = tmp_path / "target.npy"
    np.save(target_path, target_landmarks)

    # Create 3 candidates: identical, similar, different
    candidates = []

    # Identical (similarity ~1.0)
    identical_path = tmp_path / "identical.npy"
    np.save(identical_path, target_landmarks)
    candidates.append(identical_path)

    # Similar (similarity ~0.8)
    similar_landmarks = target_landmarks + np.random.rand(60, 225).astype(np.float32) * 0.1
    similar_path = tmp_path / "similar.npy"
    np.save(similar_path, similar_landmarks)
    candidates.append(similar_path)

    # Different (similarity <0.5)
    different_landmarks = np.random.rand(60, 225).astype(np.float32) * 5
    different_path = tmp_path / "different.npy"
    np.save(different_path, different_landmarks)
    candidates.append(different_path)

    # Find similar with threshold 0.75
    results = find_similar_videos(target_path, candidates, threshold=0.75, top_k=5)

    # Should return only identical and similar (not different)
    assert len(results) >= 1, "Should find at least identical video"
    assert all(score >= 0.75 for _, score in results), "All scores should be >= threshold"


def test_find_similar_videos_limits_top_k(tmp_path):
    """find_similar_videos should limit results to top_k"""
    target_landmarks = np.random.rand(60, 225).astype(np.float32)
    target_path = tmp_path / "target.npy"
    np.save(target_path, target_landmarks)

    # Create 10 similar candidates
    candidates = []
    for i in range(10):
        path = tmp_path / f"candidate_{i}.npy"
        np.save(path, target_landmarks + np.random.rand(60, 225).astype(np.float32) * 0.05)
        candidates.append(path)

    # Request only top 3
    results = find_similar_videos(target_path, candidates, threshold=0.7, top_k=3)

    assert len(results) <= 3, f"Expected max 3 results, got {len(results)}"
```

**Step 2: Run tests to verify they fail**

```bash
cd backend
pytest tests/ml/test_similarity.py -v
```

Expected: FAIL with "No module named 'app.ml.similarity'"

**Step 3: Create similarity.py with minimal implementation**

Créer `backend/app/ml/similarity.py`:

```python
"""Video similarity computation using ML embeddings."""
from __future__ import annotations

import numpy as np
import structlog
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from app.ml.dataset import temporal_resample
from app.ml.feature_engineering import compute_enriched_features

logger = structlog.get_logger(__name__)


def compute_video_similarity(
    landmarks_path_a: str | Path,
    landmarks_path_b: str | Path
) -> float:
    """
    Compute cosine similarity between two videos based on landmarks.

    Pipeline:
    1. Load landmarks (.npy) → [frames, 225]
    2. Temporal resampling → [64, 225]
    3. Feature engineering → [64, 469]
    4. Mean pooling → [469] (embedding vector)
    5. Cosine similarity between vectors

    Args:
        landmarks_path_a: Path to first video landmarks .npy file
        landmarks_path_b: Path to second video landmarks .npy file

    Returns:
        float: Similarity score 0.0-1.0 (1.0 = identical)
    """
    # Load landmarks
    landmarks_a = np.load(landmarks_path_a)
    landmarks_b = np.load(landmarks_path_b)

    # Temporal normalization
    seq_a = temporal_resample(landmarks_a, target_len=64)
    seq_b = temporal_resample(landmarks_b, target_len=64)

    # Feature enrichment
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
    Find the K most similar videos to a target.

    Args:
        target_video_path: Landmarks path of reference video
        candidate_videos: List of candidate landmarks paths
        threshold: Minimum similarity threshold (default: 0.75)
        top_k: Maximum number of suggestions (default: 5)

    Returns:
        List of (path, score) tuples sorted by descending similarity
    """
    results = []

    for candidate_path in candidate_videos:
        try:
            score = compute_video_similarity(target_video_path, candidate_path)
            if score >= threshold:
                results.append((candidate_path, score))
        except Exception as e:
            # Skip corrupted videos or invalid landmarks
            logger.warning(
                "similarity_computation_failed",
                candidate=str(candidate_path),
                error=str(e)
            )
            continue

    # Sort by descending score and limit to top_k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
```

**Step 4: Run tests to verify they pass**

```bash
cd backend
pytest tests/ml/test_similarity.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add backend/app/ml/similarity.py backend/tests/ml/test_similarity.py
git commit -m "feat(ml): add video similarity computation using embeddings

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Videos API Endpoints

**Files:**
- Create: `backend/app/api/videos.py`
- Modify: `backend/app/api/router.py:15`
- Create: `backend/tests/api/test_videos.py`

**Step 1: Write test for GET /videos/unlabeled**

Créer `backend/tests/api/test_videos.py`:

```python
"""Tests for videos API endpoints."""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database import SessionLocal
from app.models.video import Video
from app.models.sign import Sign


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db():
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture
def setup_videos(db):
    """Create test videos: 3 labeled, 2 unlabeled"""
    # Create a sign for labeled videos
    sign = Sign(
        id="test-sign-id",
        name="test_sign",
        slug="test-sign",
        category="test"
    )
    db.add(sign)

    # Labeled videos
    for i in range(3):
        video = Video(
            id=f"labeled-video-{i}",
            sign_id="test-sign-id",
            file_path=f"/test/labeled_{i}.mp4",
            landmarks_extracted=True,
            landmarks_path=f"/test/labeled_{i}_landmarks.npy"
        )
        db.add(video)

    # Unlabeled videos
    for i in range(2):
        video = Video(
            id=f"unlabeled-video-{i}",
            sign_id=None,
            file_path=f"/test/unlabeled_{i}.mp4",
            landmarks_extracted=True,
            landmarks_path=f"/test/unlabeled_{i}_landmarks.npy"
        )
        db.add(video)

    db.commit()

    yield

    # Cleanup
    db.query(Video).delete()
    db.query(Sign).delete()
    db.commit()


def test_get_unlabeled_videos_returns_only_unlabeled(client, setup_videos):
    """GET /videos/unlabeled should return only videos without sign_id"""
    response = client.get("/videos/unlabeled")

    assert response.status_code == 200
    data = response.json()

    assert "items" in data
    assert len(data["items"]) == 2

    # All returned videos should have sign_id = null
    for video in data["items"]:
        assert video["sign_id"] is None


def test_label_video_updates_sign_id(client, setup_videos, db):
    """PATCH /videos/{id}/label should update sign_id"""
    video_id = "unlabeled-video-0"
    sign_id = "test-sign-id"

    response = client.patch(
        f"/videos/{video_id}/label",
        json={"sign_id": sign_id}
    )

    assert response.status_code == 200

    # Verify in database
    video = db.query(Video).filter(Video.id == video_id).first()
    assert video.sign_id == sign_id


def test_bulk_label_updates_multiple_videos(client, setup_videos, db):
    """PATCH /videos/bulk-label should update multiple videos"""
    video_ids = ["unlabeled-video-0", "unlabeled-video-1"]
    sign_id = "test-sign-id"

    response = client.patch(
        "/videos/bulk-label",
        json={"video_ids": video_ids, "sign_id": sign_id}
    )

    assert response.status_code == 200

    # Verify all videos were updated
    for video_id in video_ids:
        video = db.query(Video).filter(Video.id == video_id).first()
        assert video.sign_id == sign_id
```

**Step 2: Run tests to verify they fail**

```bash
cd backend
pytest tests/api/test_videos.py -v
```

Expected: FAIL with "404 Not Found" (endpoints don't exist yet)

**Step 3: Create videos.py with endpoints**

Créer `backend/app/api/videos.py`:

```python
"""REST endpoints for video labeling operations."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pathlib import Path

from app.api.deps import get_db
from app.models.video import Video
from app.ml.similarity import find_similar_videos

router = APIRouter()


@router.get("/videos/unlabeled")
def get_unlabeled_videos(db: Session = Depends(get_db)) -> dict:
    """
    Get all videos without sign_id.

    Returns:
        dict: {
            "items": [list of video objects],
            "total": count
        }
    """
    videos = db.query(Video).filter(Video.sign_id.is_(None)).all()

    items = [
        {
            "id": v.id,
            "file_path": v.file_path,
            "thumbnail_path": v.thumbnail_path,
            "duration_ms": v.duration_ms,
            "fps": v.fps,
            "resolution": v.resolution,
            "landmarks_extracted": v.landmarks_extracted,
            "landmarks_path": v.landmarks_path,
            "created_at": v.created_at.isoformat() if v.created_at else None,
            "sign_id": v.sign_id
        }
        for v in videos
    ]

    return {"items": items, "total": len(items)}


@router.patch("/videos/{video_id}/label")
def label_video(
    video_id: str,
    payload: dict,
    db: Session = Depends(get_db)
) -> dict:
    """
    Associate a video with a sign.

    Args:
        video_id: Video UUID
        payload: {"sign_id": "sign-uuid"}

    Returns:
        dict: {"success": true, "video_id": str, "sign_id": str}
    """
    sign_id = payload.get("sign_id")
    if not sign_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="sign_id is required"
        )

    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video {video_id} not found"
        )

    video.sign_id = sign_id
    db.commit()

    return {"success": True, "video_id": video_id, "sign_id": sign_id}


@router.post("/videos/{video_id}/suggestions")
async def get_video_suggestions(
    video_id: str,
    db: Session = Depends(get_db),
    threshold: float = 0.75,
    top_k: int = 5
) -> dict:
    """
    Compute similar videos based on landmarks.

    Args:
        video_id: Reference video UUID
        threshold: Minimum similarity score (default: 0.75)
        top_k: Maximum suggestions (default: 5)

    Returns:
        dict: {"suggestions": [list of similar videos with scores]}
    """
    # Get source video
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video {video_id} not found"
        )

    if not video.landmarks_extracted or not video.landmarks_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Video landmarks not available"
        )

    # Get all unlabeled videos (except this one)
    unlabeled = db.query(Video).filter(
        Video.sign_id.is_(None),
        Video.id != video_id,
        Video.landmarks_extracted == True
    ).all()

    if not unlabeled:
        return {"suggestions": []}

    # Compute similarity
    candidate_paths = [Path(v.landmarks_path) for v in unlabeled]
    similar = find_similar_videos(
        Path(video.landmarks_path),
        candidate_paths,
        threshold=threshold,
        top_k=top_k
    )

    # Map results back to video objects
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


@router.patch("/videos/bulk-label")
def bulk_label_videos(
    payload: dict,
    db: Session = Depends(get_db)
) -> dict:
    """
    Label multiple videos with the same sign.

    Args:
        payload: {
            "video_ids": ["uuid1", "uuid2"],
            "sign_id": "sign-uuid"
        }

    Returns:
        dict: {"success": true, "count": int}
    """
    video_ids = payload.get("video_ids", [])
    sign_id = payload.get("sign_id")

    if not video_ids or not sign_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="video_ids and sign_id are required"
        )

    # Update all videos
    updated = db.query(Video).filter(
        Video.id.in_(video_ids)
    ).update({"sign_id": sign_id}, synchronize_session=False)

    db.commit()

    return {"success": True, "count": updated}
```

**Step 4: Register router in app**

Dans `backend/app/api/router.py`, ajouter:

```python
from app.api import videos

# Dans setup_router()
api_router.include_router(videos.router, tags=["videos"])
```

**Step 5: Run tests to verify they pass**

```bash
cd backend
pytest tests/api/test_videos.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add backend/app/api/videos.py backend/app/api/router.py backend/tests/api/test_videos.py
git commit -m "feat(api): add video labeling endpoints with ML suggestions

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: Frontend Foundation

### Task 4: API Client for Videos

**Files:**
- Create: `frontend/src/api/videos.ts`

**Step 1: Create videos API client**

Créer `frontend/src/api/videos.ts`:

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
  sign_id: null;
}

export interface SuggestedVideo extends Omit<UnlabeledVideo, 'sign_id'> {
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

**Step 2: Commit**

```bash
git add frontend/src/api/videos.ts
git commit -m "feat(frontend): add videos API client

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 5: Labeling Store (Zustand)

**Files:**
- Create: `frontend/src/stores/labelingStore.ts`

**Step 1: Create labeling store**

Créer `frontend/src/stores/labelingStore.ts`:

```typescript
import { create } from 'zustand';
import {
  getUnlabeledVideos,
  labelVideo as apiLabelVideo,
  getSuggestions,
  bulkLabelVideos as apiBulkLabel,
  type UnlabeledVideo,
  type SuggestedVideo
} from '../api/videos';
import { type Sign } from '../api/signs';

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
      await apiLabelVideo(videoId, signId);

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
      await apiBulkLabel(videoIds, signId);

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

**Step 2: Commit**

```bash
git add frontend/src/stores/labelingStore.ts
git commit -m "feat(frontend): add labeling store for state management

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 6: Tabs Component for TrainPage

**Files:**
- Create: `frontend/src/components/common/Tabs.tsx`
- Modify: `frontend/src/pages/TrainPage.tsx:1-28`

**Step 1: Create Tabs component**

Créer `frontend/src/components/common/Tabs.tsx`:

```typescript
import { type ReactNode } from 'react';
import { cn } from '../../lib/utils';

interface TabsProps {
  value: string;
  onChange: (value: string) => void;
  children: ReactNode;
}

interface TabProps {
  label: string;
  value: string;
  badge?: number;
}

export function Tabs({ value, onChange, children }: TabsProps): JSX.Element {
  return (
    <div className="border-b border-slate-700 mb-6">
      <nav className="flex gap-4">
        {children}
      </nav>
    </div>
  );
}

export function Tab({ label, value, badge }: TabProps): JSX.Element {
  const isActive = false; // Will be set by parent context

  return (
    <button
      className={cn(
        "px-4 py-3 text-sm font-medium border-b-2 transition-colors",
        isActive
          ? "border-primary text-white"
          : "border-transparent text-slate-400 hover:text-slate-200"
      )}
    >
      {label}
      {badge !== undefined && badge > 0 && (
        <span className="ml-2 px-2 py-0.5 rounded-full bg-primary/20 text-primary text-xs">
          {badge}
        </span>
      )}
    </button>
  );
}

// Context-aware implementation
interface TabsContextProps {
  value: string;
  onChange: (value: string) => void;
  children: ReactNode;
}

export function TabsWithContext({ value, onChange, children }: TabsContextProps): JSX.Element {
  return (
    <div className="border-b border-slate-700 mb-6">
      <nav className="flex gap-4">
        {Array.isArray(children) ? children.map((child: any) => {
          if (!child || child.type !== Tab) return child;

          const isActive = child.props.value === value;

          return (
            <button
              key={child.props.value}
              onClick={() => onChange(child.props.value)}
              className={cn(
                "px-4 py-3 text-sm font-medium border-b-2 transition-colors",
                isActive
                  ? "border-primary text-white"
                  : "border-transparent text-slate-400 hover:text-slate-200"
              )}
            >
              {child.props.label}
              {child.props.badge !== undefined && child.props.badge > 0 && (
                <span className="ml-2 px-2 py-0.5 rounded-full bg-primary/20 text-primary text-xs">
                  {child.props.badge}
                </span>
              )}
            </button>
          );
        }) : children}
      </nav>
    </div>
  );
}
```

**Step 2: Update TrainPage with tabs**

Modifier `frontend/src/pages/TrainPage.tsx`:

```typescript
import { useMemo, useState, useEffect } from "react";
import { useLocation } from "react-router-dom";

import { useCamera } from "../hooks/useCamera";
import { TrainingWizard } from "../components/training/TrainingWizard";
import { TabsWithContext, Tab } from "../components/common/Tabs";
import { useLabelingStore } from "../stores/labelingStore";

interface TrainPageLocationState {
  assignedSign?: {
    id: string;
    name: string;
    trainingSampleCount?: number;
    videoCount?: number;
  };
}

export function TrainPage(): JSX.Element {
  const location = useLocation();
  const { videoRef, attachVideoRef } = useCamera();
  const [activeTab, setActiveTab] = useState<'record' | 'label'>('record');

  const { unlabeledVideos, loadUnlabeledVideos } = useLabelingStore();

  const initialAssignedSign = useMemo(() => {
    const state = location.state as TrainPageLocationState | null;
    if (!state?.assignedSign?.id || !state.assignedSign.name) {
      return undefined;
    }
    return state.assignedSign;
  }, [location.state]);

  // Load unlabeled videos count
  useEffect(() => {
    loadUnlabeledVideos();
  }, [loadUnlabeledVideos]);

  return (
    <section className="space-y-4">
      <header>
        <h1 className="font-heading text-2xl">Sign Training</h1>
      </header>

      <TabsWithContext value={activeTab} onChange={(v) => setActiveTab(v as 'record' | 'label')}>
        <Tab label="Record" value="record" />
        <Tab label="Label Videos" value="label" badge={unlabeledVideos.length} />
      </TabsWithContext>

      {activeTab === 'record' && (
        <TrainingWizard
          videoRef={videoRef}
          cameraRef={attachVideoRef}
          initialAssignedSign={initialAssignedSign}
        />
      )}

      {activeTab === 'label' && (
        <div className="card p-4">
          <p className="text-slate-400">Video labeling interface coming soon...</p>
        </div>
      )}
    </section>
  );
}
```

**Step 3: Commit**

```bash
git add frontend/src/components/common/Tabs.tsx frontend/src/pages/TrainPage.tsx
git commit -m "feat(frontend): add tabs component and integrate with TrainPage

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 3: Video Grid & Labeling Modal

### Task 7: Video Grid Component

**Files:**
- Create: `frontend/src/components/labeling/VideoGrid.tsx`
- Create: `frontend/src/components/labeling/VideoCard.tsx`

**Step 1: Create VideoCard component**

Créer `frontend/src/components/labeling/VideoCard.tsx`:

```typescript
import { type UnlabeledVideo } from '../../api/videos';
import { cn } from '../../lib/utils';

interface VideoCardProps {
  video: UnlabeledVideo;
  onClick: () => void;
}

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;

  if (minutes > 0) {
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  }
  return `${seconds}s`;
}

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('fr-FR', {
    month: 'short',
    day: 'numeric'
  });
}

export function VideoCard({ video, onClick }: VideoCardProps): JSX.Element {
  return (
    <div
      className={cn(
        "card cursor-pointer transition-all",
        "hover:border-blue-500 hover:shadow-lg hover:scale-[1.02]"
      )}
      onClick={onClick}
    >
      {/* Thumbnail */}
      <div className="aspect-video bg-slate-800 relative overflow-hidden">
        {video.thumbnail_path ? (
          <img
            src={video.thumbnail_path}
            alt="Video thumbnail"
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <svg
              className="w-12 h-12 text-slate-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
          </div>
        )}

        {/* Duration overlay */}
        <div className="absolute bottom-2 right-2 bg-black/70 px-2 py-1 rounded text-xs font-mono">
          {formatDuration(video.duration_ms)}
        </div>

        {/* Landmarks status */}
        <div className="absolute top-2 left-2">
          {video.landmarks_extracted ? (
            <span className="text-emerald-400 text-lg" title="Landmarks extracted">
              ✓
            </span>
          ) : (
            <span className="text-yellow-400 text-lg" title="Extracting landmarks...">
              ⏳
            </span>
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

**Step 2: Create VideoGrid component**

Créer `frontend/src/components/labeling/VideoGrid.tsx`:

```typescript
import { useState } from 'react';
import { type UnlabeledVideo } from '../../api/videos';
import { VideoCard } from './VideoCard';

interface VideoGridProps {
  videos: UnlabeledVideo[];
  onSelectVideo: (video: UnlabeledVideo) => void;
  onGoToRecord?: () => void;
}

type SortOption = 'date' | 'duration' | 'landmarks';

export function VideoGrid({ videos, onSelectVideo, onGoToRecord }: VideoGridProps): JSX.Element {
  const [sortBy, setSortBy] = useState<SortOption>('date');

  const sortedVideos = [...videos].sort((a, b) => {
    switch (sortBy) {
      case 'date':
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      case 'duration':
        return a.duration_ms - b.duration_ms;
      case 'landmarks':
        return (b.landmarks_extracted ? 1 : 0) - (a.landmarks_extracted ? 1 : 0);
      default:
        return 0;
    }
  });

  if (videos.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <div className="text-6xl mb-4">🎉</div>
        <h3 className="text-xl font-heading mb-2">All videos labeled!</h3>
        <p className="text-slate-400 mb-6">
          Ready to train your model with labeled data.
        </p>
        {onGoToRecord && (
          <button
            className="touch-btn bg-primary text-white"
            onClick={onGoToRecord}
          >
            Go to Training
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header with filters */}
      <div className="flex gap-4 items-center">
        <label className="text-sm text-slate-400">
          Sort by:
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as SortOption)}
            className="ml-2 rounded-btn bg-slate-800 border border-slate-700 px-3 py-1.5 text-sm"
          >
            <option value="date">Date (newest first)</option>
            <option value="duration">Duration</option>
            <option value="landmarks">Landmarks Ready First</option>
          </select>
        </label>

        <div className="flex-1" />

        <span className="text-sm text-slate-400">
          {videos.length} video{videos.length !== 1 ? 's' : ''} to label
        </span>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {sortedVideos.map(video => (
          <VideoCard
            key={video.id}
            video={video}
            onClick={() => onSelectVideo(video)}
          />
        ))}
      </div>
    </div>
  );
}
```

**Step 3: Commit**

```bash
git add frontend/src/components/labeling/VideoGrid.tsx frontend/src/components/labeling/VideoCard.tsx
git commit -m "feat(frontend): add video grid with cards and sorting

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 8: Labeling Modal Component

**Files:**
- Create: `frontend/src/components/labeling/LabelingModal.tsx`

**Step 1: Create LabelingModal component**

Créer `frontend/src/components/labeling/LabelingModal.tsx`:

```typescript
import { useEffect, useState } from 'react';
import { type UnlabeledVideo } from '../../api/videos';
import { cn } from '../../lib/utils';

interface LabelingModalProps {
  video: UnlabeledVideo;
  onClose: () => void;
  onLabeled: (videoId: string, signId: string) => void;
  onSkip: () => void;
}

function formatDuration(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`;
}

export function LabelingModal({
  video,
  onClose,
  onLabeled,
  onSkip
}: LabelingModalProps): JSX.Element {
  const [selectedSignId, setSelectedSignId] = useState<string | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
      if (e.key === 'Enter' && selectedSignId) {
        onLabeled(video.id, selectedSignId);
      }
      if (e.key === ' ') {
        e.preventDefault();
        const videoEl = document.querySelector<HTMLVideoElement>('video');
        if (videoEl) {
          videoEl.paused ? videoEl.play() : videoEl.pause();
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [selectedSignId, video.id, onClose, onLabeled]);

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900 rounded-lg w-full max-w-3xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex justify-between items-center p-4 border-b border-slate-700">
          <h2 className="font-heading text-lg">Label Video</h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors text-xl"
          >
            ✕
          </button>
        </div>

        {/* Video Player */}
        <div className="p-4 bg-black">
          <video
            src={video.file_path}
            controls
            autoPlay
            loop
            className="w-full aspect-video"
            onError={() => setVideoError('Video file corrupted or missing')}
          />
          <div className="flex gap-4 mt-2 text-xs text-slate-400">
            <span>Duration: {formatDuration(video.duration_ms)}</span>
            <span>FPS: {video.fps}</span>
            <span>Resolution: {video.resolution}</span>
          </div>

          {videoError && (
            <div className="mt-2 bg-red-500/10 border border-red-500/40 rounded p-3 text-red-200 text-sm">
              ❌ {videoError}
              <button
                className="ml-4 underline"
                onClick={onSkip}
              >
                Skip this video
              </button>
            </div>
          )}
        </div>

        {/* Sign Selection Area */}
        <div className="p-4 space-y-3 flex-1 overflow-y-auto">
          <label className="block text-sm font-medium">Assign to Sign</label>

          <div className="card p-4 bg-slate-800/50">
            <p className="text-sm text-slate-400 text-center">
              Sign selector component will be added in next task
            </p>

            {/* Temporary input for testing */}
            <input
              type="text"
              placeholder="Enter sign ID temporarily..."
              className="mt-3 w-full rounded-btn border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
              onChange={(e) => setSelectedSignId(e.target.value || null)}
            />
          </div>

          <div className="text-xs text-slate-400">
            <p>Keyboard shortcuts:</p>
            <ul className="mt-1 space-y-1">
              <li>• <kbd className="px-1 bg-slate-800 rounded">Esc</kbd> - Close</li>
              <li>• <kbd className="px-1 bg-slate-800 rounded">Enter</kbd> - Save (when sign selected)</li>
              <li>• <kbd className="px-1 bg-slate-800 rounded">Space</kbd> - Play/Pause</li>
            </ul>
          </div>
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
            className={cn(
              "touch-btn bg-primary text-white",
              !selectedSignId && "opacity-50 cursor-not-allowed"
            )}
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

**Step 2: Commit**

```bash
git add frontend/src/components/labeling/LabelingModal.tsx
git commit -m "feat(frontend): add labeling modal with video player

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 9: Video Labeling Interface Integration

**Files:**
- Create: `frontend/src/components/labeling/VideoLabelingInterface.tsx`
- Modify: `frontend/src/pages/TrainPage.tsx:44-48`

**Step 1: Create VideoLabelingInterface**

Créer `frontend/src/components/labeling/VideoLabelingInterface.tsx`:

```typescript
import { useEffect } from 'react';
import { useLabelingStore } from '../../stores/labelingStore';
import { VideoGrid } from './VideoGrid';
import { LabelingModal } from './LabelingModal';

interface VideoLabelingInterfaceProps {
  onGoToRecord?: () => void;
}

export function VideoLabelingInterface({ onGoToRecord }: VideoLabelingInterfaceProps): JSX.Element {
  const {
    unlabeledVideos,
    selectedVideo,
    isLoading,
    error,
    loadUnlabeledVideos,
    selectVideo,
    labelVideo,
    refreshAfterLabel
  } = useLabelingStore();

  useEffect(() => {
    loadUnlabeledVideos();
  }, [loadUnlabeledVideos]);

  const handleLabeled = async (videoId: string, signId: string) => {
    try {
      await labelVideo(videoId, signId);
      // Close modal and refresh
      selectVideo(null);
      refreshAfterLabel();
    } catch (error) {
      console.error('Failed to label video:', error);
    }
  };

  const handleSkip = () => {
    selectVideo(null);
  };

  if (isLoading) {
    return (
      <div className="card p-8 text-center">
        <div className="text-slate-400">Loading videos...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-4 bg-red-500/10 border-red-500/40">
        <p className="text-red-200">❌ {error}</p>
        <button
          className="mt-3 touch-btn bg-slate-700 text-white"
          onClick={() => loadUnlabeledVideos()}
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <>
      <VideoGrid
        videos={unlabeledVideos}
        onSelectVideo={selectVideo}
        onGoToRecord={onGoToRecord}
      />

      {selectedVideo && (
        <LabelingModal
          video={selectedVideo}
          onClose={() => selectVideo(null)}
          onLabeled={handleLabeled}
          onSkip={handleSkip}
        />
      )}
    </>
  );
}
```

**Step 2: Update TrainPage to use VideoLabelingInterface**

Modifier `frontend/src/pages/TrainPage.tsx`:

```typescript
// Importer le composant
import { VideoLabelingInterface } from "../components/labeling/VideoLabelingInterface";

// Remplacer le placeholder (ligne ~44-48) par :
{activeTab === 'label' && (
  <VideoLabelingInterface onGoToRecord={() => setActiveTab('record')} />
)}
```

**Step 3: Test l'interface**

```bash
cd frontend
npm run dev
```

Vérifier :
- [ ] Onglet "Label Videos" s'affiche avec badge
- [ ] Grille de vidéos charge correctement
- [ ] Click sur vidéo ouvre modal
- [ ] Modal affiche vidéo et contrôles

**Step 4: Commit**

```bash
git add frontend/src/components/labeling/VideoLabelingInterface.tsx frontend/src/pages/TrainPage.tsx
git commit -m "feat(frontend): integrate video labeling interface with TrainPage

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 4: Sign Selector & Smart Labeling

### Task 10: Sign Selector Component

**Files:**
- Create: `frontend/src/components/labeling/SignSelector.tsx`
- Create: `frontend/src/components/labeling/CreateSignForm.tsx`

**Step 1: Create CreateSignForm**

Créer `frontend/src/components/labeling/CreateSignForm.tsx`:

```typescript
import { useState } from 'react';
import { createSign, type Sign } from '../../api/signs';
import { TagInput } from '../common/TagInput';

interface CreateSignFormProps {
  onCreated: (sign: Sign) => void;
  onCancel: () => void;
}

function normalizeSignName(rawName: string): string {
  const compact = rawName.trim().replace(/\s+/g, '_').toLowerCase();
  if (!compact) return '';
  return compact.startsWith('lsfb_') ? compact : `lsfb_${compact}`;
}

export function CreateSignForm({ onCreated, onCancel }: CreateSignFormProps): JSX.Element {
  const [name, setName] = useState('');
  const [category, setCategory] = useState('lsfb-v1');
  const [tags, setTags] = useState<string[]>(['lsfb', 'v1']);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCreate = async () => {
    if (!name.trim()) {
      setError('Sign name is required');
      return;
    }

    setIsCreating(true);
    setError(null);

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
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create sign');
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <div className="card p-3 space-y-3 border border-slate-600">
      <input
        type="text"
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Sign name (e.g., bonjour)"
        className="w-full rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
        autoFocus
      />

      <input
        type="text"
        value={category}
        onChange={(e) => setCategory(e.target.value)}
        placeholder="Category"
        className="w-full rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
      />

      <TagInput tags={tags} onChange={setTags} />

      {error && (
        <p className="text-sm text-red-400">{error}</p>
      )}

      <div className="flex gap-2 pt-2">
        <button
          className="touch-btn bg-slate-600 text-white text-sm flex-1"
          onClick={onCancel}
          disabled={isCreating}
        >
          Cancel
        </button>
        <button
          className="touch-btn bg-primary text-white text-sm flex-1 disabled:opacity-50"
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

**Step 2: Create SignSelector**

Créer `frontend/src/components/labeling/SignSelector.tsx`:

```typescript
import { useState, useEffect } from 'react';
import { listSigns, type Sign } from '../../api/signs';
import { useLabelingStore } from '../../stores/labelingStore';
import { CreateSignForm } from './CreateSignForm';
import { cn } from '../../lib/utils';

interface SignSelectorProps {
  selectedSignId: string | null;
  onSelectSign: (signId: string) => void;
}

export function SignSelector({ selectedSignId, onSelectSign }: SignSelectorProps): JSX.Element {
  const { recentSigns, addToRecentSigns } = useLabelingStore();
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState<Sign[]>([]);
  const [showCreateForm, setShowCreateForm] = useState(false);

  // Debounced search
  useEffect(() => {
    if (query.length < 2) {
      setSuggestions([]);
      return;
    }

    const timer = setTimeout(() => {
      listSigns(query).then(response => {
        setSuggestions(response.items.slice(0, 10));
      }).catch(() => {
        setSuggestions([]);
      });
    }, 300);

    return () => clearTimeout(timer);
  }, [query]);

  const handleSignCreated = (sign: Sign) => {
    addToRecentSigns(sign);
    onSelectSign(sign.id);
    setShowCreateForm(false);
  };

  const handleSelectExisting = (sign: Sign) => {
    addToRecentSigns(sign);
    onSelectSign(sign.id);
    setQuery('');
    setSuggestions([]);
  };

  return (
    <div className="space-y-4">
      <label className="block text-sm font-medium mb-2">Assign to Sign</label>

      {/* 1. Recent Signs (Quick Picks) */}
      {recentSigns.length > 0 && (
        <div>
          <p className="text-xs text-slate-400 mb-2">Recent</p>
          <div className="flex gap-2 flex-wrap">
            {recentSigns.slice(0, 5).map(sign => (
              <button
                key={sign.id}
                className={cn(
                  "touch-btn text-sm",
                  selectedSignId === sign.id
                    ? "bg-primary text-white"
                    : "bg-slate-700 text-slate-200"
                )}
                onClick={() => handleSelectExisting(sign)}
              >
                {sign.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 2. Autocomplete Search */}
      <div>
        <p className="text-xs text-slate-400 mb-2">Search Existing</p>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Type to search signs..."
          className="w-full rounded-btn border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
        />

        {suggestions.length > 0 && (
          <div className="mt-2 space-y-1 max-h-48 overflow-y-auto">
            {suggestions.map(sign => (
              <button
                key={sign.id}
                className="w-full text-left px-3 py-2 rounded hover:bg-slate-700 text-sm transition-colors"
                onClick={() => handleSelectExisting(sign)}
              >
                <span className="font-medium">{sign.name}</span>
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
            onClick={() => setShowCreateForm(true)}
          >
            + Create new sign
          </button>
        ) : (
          <CreateSignForm
            onCreated={handleSignCreated}
            onCancel={() => setShowCreateForm(false)}
          />
        )}
      </div>
    </div>
  );
}
```

**Step 3: Integrate SignSelector into LabelingModal**

Modifier `frontend/src/components/labeling/LabelingModal.tsx`:

```typescript
// Ajouter import
import { SignSelector } from './SignSelector';

// Remplacer la section "Sign Selection Area" (ligne ~80-95) par :
<div className="p-4 space-y-3 flex-1 overflow-y-auto">
  <SignSelector
    selectedSignId={selectedSignId}
    onSelectSign={setSelectedSignId}
  />

  <div className="text-xs text-slate-400">
    <p>Keyboard shortcuts:</p>
    <ul className="mt-1 space-y-1">
      <li>• <kbd className="px-1 bg-slate-800 rounded">Esc</kbd> - Close</li>
      <li>• <kbd className="px-1 bg-slate-800 rounded">Enter</kbd> - Save (when sign selected)</li>
      <li>• <kbd className="px-1 bg-slate-800 rounded">Space</kbd> - Play/Pause</li>
    </ul>
  </div>
</div>
```

**Step 4: Test sign selection**

```bash
cd frontend
npm run dev
```

Vérifier :
- [ ] Recent signs s'affichent
- [ ] Autocomplete fonctionne
- [ ] Création de sign fonctionne
- [ ] Sélection met à jour l'état

**Step 5: Commit**

```bash
git add frontend/src/components/labeling/SignSelector.tsx frontend/src/components/labeling/CreateSignForm.tsx frontend/src/components/labeling/LabelingModal.tsx
git commit -m "feat(frontend): add sign selector with search and creation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 11: Smart Labeling (Suggestions)

**Files:**
- Create: `frontend/src/components/labeling/SuggestionView.tsx`
- Create: `frontend/src/components/labeling/SuggestionCard.tsx`
- Modify: `frontend/src/components/labeling/VideoLabelingInterface.tsx:22-33`

**Step 1: Create SuggestionCard**

Créer `frontend/src/components/labeling/SuggestionCard.tsx`:

```typescript
import { type SuggestedVideo } from '../../api/videos';
import { cn } from '../../lib/utils';

interface SuggestionCardProps {
  video: SuggestedVideo;
  isSelected: boolean;
  onToggle: () => void;
}

export function SuggestionCard({ video, isSelected, onToggle }: SuggestionCardProps): JSX.Element {
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

  const scoreColor = {
    high: 'text-emerald-400',
    medium: 'text-yellow-400',
    low: 'text-slate-400'
  }[confidenceLevel];

  return (
    <div
      className={cn(
        "card cursor-pointer transition-all relative",
        isSelected ? `${borderColor} ring-2 ring-blue-500` : 'border-slate-700'
      )}
      onClick={onToggle}
    >
      {/* Checkbox */}
      <div className="absolute top-2 left-2 z-10">
        <input
          type="checkbox"
          checked={isSelected}
          onChange={onToggle}
          className="w-4 h-4 cursor-pointer"
          onClick={(e) => e.stopPropagation()}
        />
      </div>

      {/* Thumbnail */}
      <div className="aspect-video bg-slate-800">
        {video.thumbnail_path ? (
          <img
            src={video.thumbnail_path}
            alt="Suggestion"
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <svg className="w-10 h-10 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </div>
        )}
      </div>

      {/* Similarity Score */}
      <div className="p-2 text-center">
        <span className={cn("text-sm font-mono font-medium", scoreColor)}>
          {Math.round(video.similarity_score * 100)}% match
        </span>
      </div>
    </div>
  );
}
```

**Step 2: Create SuggestionView**

Créer `frontend/src/components/labeling/SuggestionView.tsx`:

```typescript
import { useState } from 'react';
import { type SuggestedVideo } from '../../api/videos';
import { type Sign } from '../../api/signs';
import { SuggestionCard } from './SuggestionCard';
import { cn } from '../../lib/utils';

interface SuggestionViewProps {
  assignedSign: Sign;
  suggestions: SuggestedVideo[];
  onApplyAll: (videoIds: string[]) => Promise<void>;
  onSkip: () => void;
}

export function SuggestionView({
  assignedSign,
  suggestions,
  onApplyAll,
  onSkip
}: SuggestionViewProps): JSX.Element {
  const [selected, setSelected] = useState<Set<string>>(
    new Set(suggestions.map(s => s.id))
  );
  const [isApplying, setIsApplying] = useState(false);

  const handleToggle = (videoId: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(videoId)) {
        next.delete(videoId);
      } else {
        next.add(videoId);
      }
      return next;
    });
  };

  const handleApply = async () => {
    if (selected.size === 0) return;

    setIsApplying(true);
    try {
      await onApplyAll(Array.from(selected));
    } catch (error) {
      console.error('Failed to apply suggestions:', error);
    } finally {
      setIsApplying(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Success Header */}
      <div className="bg-emerald-500/10 border border-emerald-500/40 rounded-btn p-3">
        <p className="text-emerald-100 font-medium">
          ✓ Video labeled as: <span className="font-mono">{assignedSign.name}</span>
        </p>
      </div>

      {/* Suggestions Header */}
      <div className="flex items-center gap-2">
        <span className="text-lg">🤖</span>
        <h3 className="font-heading text-lg">Similar videos detected</h3>
      </div>

      {/* Grid de suggestions */}
      <div className="grid grid-cols-3 gap-3">
        {suggestions.map(video => (
          <SuggestionCard
            key={video.id}
            video={video}
            isSelected={selected.has(video.id)}
            onToggle={() => handleToggle(video.id)}
          />
        ))}
      </div>

      {/* Actions */}
      <div className="bg-slate-800/50 rounded-btn p-4">
        <p className="text-sm text-slate-300 mb-3">
          Apply "<span className="font-mono">{assignedSign.name}</span>" to {selected.size} selected video(s)?
        </p>

        <div className="flex gap-2">
          <button
            className="touch-btn bg-slate-700 text-white text-sm"
            onClick={onSkip}
            disabled={isApplying}
          >
            Skip suggestions
          </button>

          <div className="flex-1" />

          <button
            className={cn(
              "touch-btn bg-primary text-white",
              (selected.size === 0 || isApplying) && "opacity-50 cursor-not-allowed"
            )}
            disabled={selected.size === 0 || isApplying}
            onClick={handleApply}
          >
            {isApplying ? 'Applying...' : `Apply to ${selected.size} video(s) →`}
          </button>
        </div>
      </div>
    </div>
  );
}
```

**Step 3: Integrate suggestions into VideoLabelingInterface**

Modifier `frontend/src/components/labeling/VideoLabelingInterface.tsx`:

```typescript
// Ajouter imports
import { SuggestionView } from './SuggestionView';
import { listSigns } from '../../api/signs';
import { useState } from 'react';

// Dans le composant, ajouter état local
const [showSuggestions, setShowSuggestions] = useState(false);
const [labeledSign, setLabeledSign] = useState<any>(null);

// Modifier handleLabeled
const handleLabeled = async (videoId: string, signId: string) => {
  try {
    // Get sign details for display
    const signsResponse = await listSigns('');
    const sign = signsResponse.items.find(s => s.id === signId);

    await labelVideo(videoId, signId);

    // Check if suggestions were loaded
    if (suggestions.length > 0 && sign) {
      setLabeledSign(sign);
      setShowSuggestions(true);
    } else {
      // No suggestions, close modal
      selectVideo(null);
      refreshAfterLabel();
    }
  } catch (error) {
    console.error('Failed to label video:', error);
  }
};

// Ajouter handler pour suggestions
const handleApplySuggestions = async (videoIds: string[]) => {
  if (!labeledSign) return;

  try {
    await applySuggestions(videoIds, labeledSign.id);
    setShowSuggestions(false);
    selectVideo(null);
    refreshAfterLabel();
  } catch (error) {
    console.error('Failed to apply suggestions:', error);
  }
};

const handleSkipSuggestions = () => {
  setShowSuggestions(false);
  selectVideo(null);
  refreshAfterLabel();
};

// Modifier le rendu du modal
{selectedVideo && !showSuggestions && (
  <LabelingModal
    video={selectedVideo}
    onClose={() => selectVideo(null)}
    onLabeled={handleLabeled}
    onSkip={handleSkip}
  />
)}

{showSuggestions && labeledSign && (
  <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
    <div className="bg-slate-900 rounded-lg w-full max-w-5xl max-h-[90vh] overflow-y-auto p-6">
      <SuggestionView
        assignedSign={labeledSign}
        suggestions={suggestions}
        onApplyAll={handleApplySuggestions}
        onSkip={handleSkipSuggestions}
      />
    </div>
  </div>
)}
```

**Step 4: Test smart labeling**

```bash
cd frontend
npm run dev
```

Vérifier :
- [ ] Après labellisation, suggestions s'affichent
- [ ] Sélection/désélection des suggestions fonctionne
- [ ] "Apply" labellise toutes les vidéos sélectionnées
- [ ] "Skip" ferme sans appliquer

**Step 5: Commit**

```bash
git add frontend/src/components/labeling/SuggestionView.tsx frontend/src/components/labeling/SuggestionCard.tsx frontend/src/components/labeling/VideoLabelingInterface.tsx
git commit -m "feat(frontend): add smart labeling with ML-based suggestions

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 5: Error Handling & Polish

### Task 12: Loading States & Error Handling

**Files:**
- Modify: `frontend/src/components/labeling/VideoLabelingInterface.tsx:1-50`
- Modify: `frontend/src/stores/labelingStore.ts:50-80`

**Step 1: Add loading state for suggestions in modal**

Dans `LabelingModal.tsx`, après labellisation :

```typescript
// Ajouter état
const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false);

// Afficher pendant chargement suggestions
{isLoadingSuggestions && (
  <div className="text-center py-4">
    <div className="animate-spin w-8 h-8 border-4 border-primary border-t-transparent rounded-full mx-auto mb-2" />
    <p className="text-sm text-slate-400">Computing similar videos...</p>
  </div>
)}
```

**Step 2: Add timeout handling for suggestions**

Dans `labelingStore.ts`, modifier `labelVideo`:

```typescript
labelVideo: async (videoId: string, signId: string) => {
  try {
    await apiLabelVideo(videoId, signId);

    set(state => ({
      unlabeledVideos: state.unlabeledVideos.filter(v => v.id !== videoId)
    }));

    // Fetch suggestions with timeout
    set({ isLoadingSuggestions: true });

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);

    try {
      const { suggestions } = await getSuggestions(videoId);
      clearTimeout(timeout);
      set({ suggestions, isLoadingSuggestions: false });
    } catch (error) {
      if (error.name === 'AbortError') {
        // Timeout, continue without suggestions
        set({ suggestions: [], isLoadingSuggestions: false });
      } else {
        throw error;
      }
    }

  } catch (error) {
    set({
      error: error instanceof Error ? error.message : 'Failed to label video'
    });
    throw error;
  }
}
```

**Step 3: Add retry mechanism**

Dans `VideoLabelingInterface.tsx`:

```typescript
const [retryCount, setRetryCount] = useState(0);

const handleRetry = () => {
  setRetryCount(prev => prev + 1);
  loadUnlabeledVideos();
};

// Dans le rendu
{error && (
  <div className="card p-4 bg-red-500/10 border-red-500/40">
    <p className="text-red-200">❌ {error}</p>
    <button
      className="mt-3 touch-btn bg-slate-700 text-white"
      onClick={handleRetry}
    >
      Retry {retryCount > 0 && `(${retryCount})`}
    </button>
  </div>
)}
```

**Step 4: Commit**

```bash
git add frontend/src/components/labeling/VideoLabelingInterface.tsx frontend/src/components/labeling/LabelingModal.tsx frontend/src/stores/labelingStore.ts
git commit -m "feat(frontend): add loading states and error handling

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 13: Final Testing & Documentation

**Files:**
- Create: `docs/video-labeling-user-guide.md`

**Step 1: Manual testing checklist**

Tester manuellement :

- [ ] Migration DB appliquée correctement
- [ ] API endpoints retournent données correctes
- [ ] Similarité ML calcule scores cohérents
- [ ] Interface charge vidéos non labellisées
- [ ] Grille affiche cards avec thumbnails
- [ ] Modal s'ouvre au click
- [ ] Vidéo se lit dans modal
- [ ] Sign selector fonctionne (recent/search/create)
- [ ] Labellisation met à jour DB
- [ ] Suggestions s'affichent après labellisation
- [ ] Bulk label applique à plusieurs vidéos
- [ ] Grille met à jour après labellisation
- [ ] Badge compte reflète vidéos restantes
- [ ] Keyboard shortcuts fonctionnent
- [ ] Error handling affiche messages appropriés

**Step 2: Create user guide**

Créer `docs/video-labeling-user-guide.md`:

```markdown
# Video Labeling User Guide

## Overview

The video labeling interface allows you to quickly assign signs to unlabeled training videos. It features smart labeling powered by ML similarity to accelerate the labeling process.

## Accessing the Interface

1. Navigate to `/train` page
2. Click on "Label Videos" tab
3. Badge shows number of unlabeled videos

## Labeling a Video

### Quick Steps
1. Click on a video card in the grid
2. Modal opens with video player
3. Select or create a sign
4. Click "Save Label"
5. Review ML suggestions (if any)
6. Apply suggestions or skip

### Sign Selection Methods

**Recent Signs (Quick Pick)**
- Last 5 signs you've used
- Click to instantly assign

**Search Existing**
- Type to search all signs
- Auto-complete suggestions
- Select from results

**Create New**
- Click "+ Create new sign"
- Fill in name, category, tags
- Auto-assigns to video

## Smart Labeling

After labeling a video, the system computes similarity with other unlabeled videos:

- **Green border (>90%)**: High confidence match
- **Yellow border (75-90%)**: Medium confidence
- **Not shown (<75%)**: Low similarity

**Actions:**
- Select/deselect videos
- Click "Apply" to label all selected
- Click "Skip" to continue manually

## Keyboard Shortcuts

- `Esc` - Close modal
- `Enter` - Save label (when sign selected)
- `Space` - Play/Pause video
- `n` - Next video (in grid)
- `p` - Previous video (in grid)

## Tips for Efficient Labeling

1. **Use smart suggestions** - Accept high-confidence matches to save time
2. **Recent signs** - Frequently used signs appear at top for quick access
3. **Batch similar videos** - Record multiple examples of same sign together
4. **Check quality** - Verify video plays correctly before labeling

## Troubleshooting

**Video doesn't load**
- Skip the video and report issue
- Video file may be corrupted

**No suggestions appear**
- Landmarks may not be extracted yet
- No similar videos above 75% threshold

**Landmarks extracting slowly**
- Check "⏳" icon on video cards
- Refresh page after 1-2 minutes

## Next Steps

After labeling:
1. Go to "Record" tab
2. Start training with labeled data
3. Monitor training progress
```

**Step 3: Commit final docs**

```bash
git add docs/video-labeling-user-guide.md
git commit -m "docs: add video labeling user guide

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Completion Checklist

### Backend
- [x] Migration: sign_id nullable
- [x] ML similarity module
- [x] API endpoints (4)
- [x] Tests (similarity + API)

### Frontend
- [x] API client
- [x] Zustand store
- [x] Tabs component
- [x] Video grid
- [x] Video cards
- [x] Labeling modal
- [x] Sign selector
- [x] Suggestion view
- [x] Smart labeling flow
- [x] Error handling
- [x] Loading states

### Documentation
- [x] User guide
- [x] Design document
- [x] Implementation plan

### Testing
- [ ] Backend integration tests pass
- [ ] Manual testing completed
- [ ] Performance validated (grille fluide avec 100+ vidéos)

---

## Estimated Total Time

- **Phase 1 (Backend):** 3-4h
- **Phase 2 (Frontend Foundation):** 2-3h
- **Phase 3 (Grid & Modal):** 2-3h
- **Phase 4 (Smart Labeling):** 2-3h
- **Phase 5 (Polish):** 1-2h

**Total:** 10-15 heures

---

## Success Metrics

Après implémentation complète :

1. ✅ 46 vidéos actuelles labellisées en <1h
2. ✅ Taux d'acceptation suggestions >70%
3. ✅ Temps moyen labellisation <30s/vidéo
4. ✅ Interface responsive (mobile/tablet/desktop)
5. ✅ Performance fluide (<100ms) avec 100+ vidéos

---

## Notes for Implementation

**Dependencies:**
- Backend : `scikit-learn>=1.3.0` (add to requirements.txt)
- Frontend : No new dependencies needed

**Database:**
- Run migration before starting: `alembic upgrade head`
- Backup DB before migration

**Testing:**
- Run backend tests: `pytest backend/tests/ -v`
- Manual test frontend: `npm run dev`

**Deployment:**
- Deploy backend first (migration + API)
- Deploy frontend after (no breaking changes)

**Rollback Plan:**
- Migration downgrade: `alembic downgrade -1`
- Revert git commits in reverse order
