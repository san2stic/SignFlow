# Video Labeling Interface

## Overview

The video labeling interface allows users to efficiently label unlabeled training videos with sign names using ML-powered suggestions. This feature streamlines the process of building training datasets by intelligently grouping similar videos together.

## Features

### 1. Video Grid
- Displays all unlabeled videos in a responsive grid layout
- Sorting options:
  - Newest/oldest (by creation date)
  - Longest/shortest (by duration)
- Video metadata display:
  - Duration (mm:ss format)
  - Resolution (width x height)
  - Landmarks extraction status
- Badge counter showing total unlabeled videos
- Empty state message when all videos are labeled

### 2. Labeling Modal
- Full-featured video player with standard controls (play/pause, seek, volume)
- Video metadata panel showing file information
- Integrated sign selector component
- Label button with validation
- Keyboard support (ESC to close)
- Loading states during labeling operations
- Error handling with user-friendly messages

### 3. Sign Selector
- Recent signs list (last 10 used signs for quick access)
- Debounced search (300ms delay for performance)
- Search across existing signs in database
- Inline sign creation for new labels
- Auto-selection of newly created signs
- Visual distinction between recent and search results
- Loading states during search operations

### 4. Smart Labeling Suggestions
- ML-powered similarity detection using cosine similarity
- Default threshold: 75% similarity
- Configurable top-K results (default: 5 suggestions)
- Similarity percentage badges on suggestion cards
- Video preview thumbnails for each suggestion
- Bulk selection/deselection of suggestions
- One-click bulk labeling of similar videos
- Skip option to dismiss suggestions

## Usage Workflow

### Basic Labeling
1. Navigate to `/train` page
2. Switch to "Label Videos" tab
3. Click a video card to open labeling modal
4. Select an existing sign or create a new one
5. Click "Label Video" button
6. Video is immediately removed from grid (optimistic update)

### Smart Labeling with Suggestions
1. Follow basic labeling steps 1-5
2. After labeling, suggestions modal appears automatically (if similar videos found)
3. Review suggested videos and their similarity scores
4. Select/deselect videos you want to label with the same sign
5. Click "Label Selected" to bulk label
6. Or click "Skip" to dismiss suggestions
7. Grid updates to reflect newly labeled videos

### Edge Cases
- If no unlabeled videos exist, an empty state message is shown
- If no similar videos are found (below 75% threshold), suggestions modal doesn't appear
- If video has no landmarks extracted, it cannot be used for similarity computation
- ESC key closes all modals

## Architecture

### Backend Components

#### Database Schema
```sql
-- videos table has nullable sign_id foreign key
ALTER TABLE videos
  ADD COLUMN sign_id UUID REFERENCES signs(id) ON DELETE SET NULL;
```

#### API Endpoints

**GET /api/v1/videos/unlabeled**
- Returns all videos where `sign_id IS NULL`
- Response: `{ items: VideoResponse[], total: number }`

**PATCH /api/v1/videos/{video_id}/label**
- Updates `sign_id` for a single video
- Request: `{ sign_id: string }`
- Response: `VideoResponse`

**POST /api/v1/videos/{video_id}/suggestions**
- Computes ML similarity against unlabeled videos
- Query params: `threshold` (default: 0.75), `top_k` (default: 5)
- Response: `{ target_video_id: string, suggestions: SuggestionResponse[] }`

**PATCH /api/v1/videos/bulk-label**
- Bulk updates `sign_id` for multiple videos
- Request: `{ video_ids: string[], sign_id: string }`
- Response: `{ updated_count: number }`

#### ML Similarity Module

File: `backend/app/ml/similarity.py`

**Pipeline:**
1. Load landmarks from `.npy` files → `[frames, 225]` (pose + hands)
2. Temporal resample to fixed length → `[64, 225]`
3. Compute enriched features → `[64, 469]` (velocities, distances, angles, hand shape)
4. Mean pooling to single embedding → `[469]`
5. Cosine similarity between embeddings → `float [0.0, 1.0]`

**Functions:**
- `compute_video_similarity(path_a, path_b) -> float`
- `find_similar_videos(target, candidates, threshold, top_k) -> list[tuple[Path, float]]`

### Frontend Components

#### State Management

File: `frontend/src/stores/labelingStore.ts`

Zustand store managing:
- Unlabeled videos list
- Loading states
- Error states
- CRUD operations (fetch, label, bulk label)
- Optimistic updates

#### Components

**VideoGrid** (`frontend/src/components/labeling/VideoGrid.tsx`)
- Grid layout with responsive columns
- Sort controls
- Click handler to open labeling modal
- Empty state rendering

**LabelingModal** (`frontend/src/components/labeling/LabelingModal.tsx`)
- Video player integration
- Sign selector integration
- Label button with validation
- Modal overlay with backdrop
- Keyboard event handling

**SignSelector** (`frontend/src/components/labeling/SignSelector.tsx`)
- Recent signs display
- Search input with debouncing
- Create new sign inline
- Loading and error states

**SuggestionView** (`frontend/src/components/labeling/SuggestionView.tsx`)
- Similar videos grid
- Similarity score badges
- Bulk selection checkboxes
- Action buttons (label/skip)

**SuggestionCard** (`frontend/src/components/labeling/SuggestionCard.tsx`)
- Individual suggestion display
- Video thumbnail
- Similarity percentage
- Checkbox for bulk selection

#### API Client

File: `frontend/src/lib/api/videos.ts`

Type-safe wrappers for all video endpoints:
- `getUnlabeledVideos()`
- `labelVideo(videoId, signId)`
- `getSuggestions(videoId, threshold?, topK?)`
- `bulkLabelVideos(videoIds, signId)`

## Technical Details

### Similarity Computation

The similarity algorithm uses the same feature engineering pipeline as the training process:

1. **Temporal Normalization**: Videos of different lengths are resampled to 64 frames using linear interpolation
2. **Feature Enrichment**: Raw landmarks (225-dim) are enhanced to 469-dim features:
   - Velocities (first-order derivatives)
   - Accelerations (second-order derivatives)
   - Inter-landmark distances (hand openness, arm extension)
   - Angular features (joint angles)
   - Hand shape descriptors
3. **Embedding**: Mean pooling across time dimension creates a single 469-dim vector per video
4. **Similarity**: Cosine similarity measures the angle between embedding vectors (1.0 = identical direction)

### Performance Optimizations

- **Debounced Search**: 300ms delay prevents excessive API calls during typing
- **Optimistic Updates**: Videos removed from grid immediately, before API confirms
- **Lazy Loading**: Suggestions computed only after labeling, not on grid load
- **Batch Operations**: Bulk labeling uses single SQL UPDATE with WHERE IN clause
- **Error Recovery**: Failed operations show retry buttons without losing user context

### Error Handling

All operations include comprehensive error handling:

- **Network Errors**: Retry buttons with error messages
- **Validation Errors**: Inline validation (e.g., sign must be selected)
- **Missing Landmarks**: Clear error message explaining why suggestions unavailable
- **Empty Results**: Graceful handling with informative messages

## Testing

### Backend Tests

File: `backend/tests/ml/test_similarity.py`
- Test identical videos return 1.0 similarity
- Test different videos return low similarity
- Test threshold filtering works correctly
- Test top-K limiting works correctly

File: `backend/tests/test_api/test_videos.py`
- Test GET /unlabeled returns only unlabeled videos
- Test PATCH /label updates sign_id correctly
- Test PATCH /bulk-label updates multiple videos

### Manual Testing Checklist

- [ ] Navigate to `/train` page
- [ ] Switch to "Label Videos" tab
- [ ] Verify unlabeled videos load correctly
- [ ] Check badge count matches number of videos
- [ ] Click a video card
- [ ] Modal opens with video player
- [ ] Video plays correctly
- [ ] Metadata displays properly
- [ ] Recent signs appear (if any)
- [ ] Search finds existing signs
- [ ] Create new sign works
- [ ] Sign selection highlights properly
- [ ] Label button enables when sign selected
- [ ] Labeling succeeds
- [ ] Video removed from grid
- [ ] Badge count decrements
- [ ] Suggestions modal appears after labeling
- [ ] Similarity percentages show correctly
- [ ] Can select/deselect suggestions
- [ ] Bulk labeling works
- [ ] Skip dismisses suggestions
- [ ] Stop backend, verify error messages
- [ ] Retry buttons work
- [ ] No crashes on errors
- [ ] Empty state shows when all labeled
- [ ] No suggestions scenario works
- [ ] ESC key closes modals

## Future Enhancements

### User Experience
- Video preview on hover over grid cards
- Keyboard shortcuts for faster labeling (e.g., number keys to select recent signs)
- Bulk selection mode for labeling multiple videos at once
- Undo labeling action (mark as unlabeled again)
- Progress tracking (X of Y videos labeled)

### Performance
- Pagination for large unlabeled video sets
- Virtual scrolling for video grid
- Parallel similarity computation using worker threads
- Pre-compute embeddings on video upload

### Features
- Adjustable similarity threshold in UI
- Filter videos by duration, resolution, or quality
- Export labeled dataset as ZIP
- Import pre-labeled videos from CSV
- Confidence scores for ML suggestions
- Video playback speed controls
- Frame-by-frame stepping for detailed review

### Integration
- Auto-label mode (automatically apply high-confidence suggestions)
- Feedback loop (user corrections improve similarity model)
- Active learning (suggest most informative videos to label next)
- Multi-user labeling with conflict resolution

## Troubleshooting

### Suggestions Not Appearing
- Ensure target video has landmarks extracted
- Check that other unlabeled videos have landmarks
- Verify similarity scores aren't all below threshold (default: 0.75)
- Check browser console for API errors

### Slow Similarity Computation
- Reduce `top_k` parameter (fewer candidates to evaluate)
- Increase `threshold` (fewer results to compute)
- Ensure landmarks are properly extracted (invalid landmarks slow down computation)

### Videos Not Updating After Labeling
- Check browser console for API errors
- Verify database connection is active
- Ensure sign_id being used exists in signs table
- Try refreshing the page to clear stale cache

## API Examples

### Get Unlabeled Videos
```bash
curl http://localhost:8000/api/v1/videos/unlabeled
```

Response:
```json
{
  "items": [
    {
      "id": "uuid-1",
      "sign_id": null,
      "file_path": "/data/videos/video1.mp4",
      "landmarks_path": "/data/landmarks/video1.npy",
      "landmarks_extracted": true,
      "duration_ms": 2500,
      "fps": 30
    }
  ],
  "total": 1
}
```

### Label a Video
```bash
curl -X PATCH http://localhost:8000/api/v1/videos/uuid-1/label \
  -H "Content-Type: application/json" \
  -d '{"sign_id": "sign-uuid-123"}'
```

### Get Suggestions
```bash
curl -X POST http://localhost:8000/api/v1/videos/uuid-1/suggestions \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.75, "top_k": 5}'
```

Response:
```json
{
  "target_video_id": "uuid-1",
  "suggestions": [
    {
      "video_id": "uuid-2",
      "similarity_score": 0.89,
      "file_path": "/data/videos/video2.mp4",
      "landmarks_path": "/data/landmarks/video2.npy"
    }
  ]
}
```

### Bulk Label Videos
```bash
curl -X PATCH http://localhost:8000/api/v1/videos/bulk-label \
  -H "Content-Type: application/json" \
  -d '{"video_ids": ["uuid-2", "uuid-3"], "sign_id": "sign-uuid-123"}'
```

Response:
```json
{
  "updated_count": 2
}
```

## Implementation Statistics

- **Backend**: 4 API endpoints, 1 ML module, 2 test files
- **Frontend**: 6 components, 1 store, 1 API client
- **Database**: 1 migration (nullable sign_id column)
- **Lines of Code**: ~1500 (backend + frontend)
- **Test Coverage**: 7 tests (100% passing)
- **Development Time**: 13 tasks across full-stack implementation
