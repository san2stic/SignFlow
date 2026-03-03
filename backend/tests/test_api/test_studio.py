"""Tests for Studio API — annotation sessions, video annotations, export.

Covers:
  - GET /studio/sessions — liste vide, avec données, filtrage statut
  - POST /studio/sessions — création session
  - GET /studio/sessions/{id} — avec stats
  - PATCH /studio/sessions/{id} — modification
  - DELETE /studio/sessions/{id} — suppression + cascade
  - POST /studio/videos/{id}/annotations — création annotation
  - GET /studio/videos/{id}/annotations — liste, filtre par session
  - PUT /studio/annotations/{id} — mise à jour
  - DELETE /studio/annotations/{id} — suppression
  - POST /studio/videos/{id}/annotations/bulk — import en masse
  - GET /studio/sessions/{id}/export?format=json — export JSON
  - GET /studio/sessions/{id}/export?format=csv — export CSV
  - GET /studio/stats — statistiques globales
  - 404 sur ressources inexistantes
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.main import app
from app.models.video import Video
from app.models.annotation import AnnotationSession, VideoAnnotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_video_in_db(sign_id: str | None = None) -> str:
    """Crée directement une vidéo en DB (sans upload réel) et retourne son id."""
    video_id = str(uuid4())
    with SessionLocal() as db:
        video = Video(
            id=video_id,
            sign_id=sign_id,
            file_path=f"/data/videos/{video_id}.mp4",
            type="annotation",
            duration_ms=5000,
            fps=30,
            resolution="640x480",
            landmarks_extracted=False,
            detection_rate=0.0,
            quality_score=0.0,
        )
        db.add(video)
        db.commit()
    return video_id


def _create_session_via_api(client: TestClient, name: str = "Test Session") -> dict:
    """Crée une session via l'API et retourne le JSON de réponse."""
    res = client.post(
        "/api/v1/studio/sessions",
        json={"name": name, "description": "Test description", "status": "active"},
    )
    assert res.status_code == 201, res.text
    return res.json()


# ---------------------------------------------------------------------------
# Tests GET /studio/sessions
# ---------------------------------------------------------------------------


def test_list_sessions_empty() -> None:
    """GET /studio/sessions doit retourner une liste vide si aucune session n'existe."""
    with TestClient(app) as client:
        res = client.get("/api/v1/studio/sessions")
    assert res.status_code == 200
    assert res.json() == []


def test_list_sessions_with_data() -> None:
    """GET /studio/sessions doit retourner les sessions créées."""
    with TestClient(app) as client:
        _create_session_via_api(client, "Session Alpha")
        _create_session_via_api(client, "Session Beta")
        res = client.get("/api/v1/studio/sessions")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 2
    names = {s["name"] for s in data}
    assert "Session Alpha" in names
    assert "Session Beta" in names


def test_list_sessions_with_stats_fields() -> None:
    """Chaque session dans la liste doit inclure les champs stats."""
    with TestClient(app) as client:
        _create_session_via_api(client, "Stats Session")
        res = client.get("/api/v1/studio/sessions")
    data = res.json()
    assert len(data) >= 1
    session = data[0]
    for field in ("video_count", "annotation_count", "verified_count", "coverage_percent"):
        assert field in session, f"Champ manquant: {field}"


def test_list_sessions_filter_by_status() -> None:
    """GET /studio/sessions?status=active doit filtrer par statut."""
    with TestClient(app) as client:
        _create_session_via_api(client, "Active Session")
        client.post(
            "/api/v1/studio/sessions",
            json={"name": "Archived Session", "status": "archived"},
        )
        res = client.get("/api/v1/studio/sessions", params={"status": "active"})
    assert res.status_code == 200
    data = res.json()
    assert all(s["status"] == "active" for s in data)


# ---------------------------------------------------------------------------
# Tests POST /studio/sessions
# ---------------------------------------------------------------------------


def test_create_session_returns_201() -> None:
    """POST /studio/sessions doit créer une session et retourner 201."""
    with TestClient(app) as client:
        res = client.post(
            "/api/v1/studio/sessions",
            json={"name": "Nouvelle Session", "description": "Desc", "status": "active"},
        )
    assert res.status_code == 201
    data = res.json()
    assert data["name"] == "Nouvelle Session"
    assert data["description"] == "Desc"
    assert data["status"] == "active"
    assert "id" in data
    assert "created_at" in data


def test_create_session_name_required() -> None:
    """POST /studio/sessions sans nom doit retourner 422."""
    with TestClient(app) as client:
        res = client.post("/api/v1/studio/sessions", json={"description": "No name"})
    assert res.status_code == 422


def test_create_session_invalid_status() -> None:
    """POST /studio/sessions avec un statut invalide doit retourner 422."""
    with TestClient(app) as client:
        res = client.post(
            "/api/v1/studio/sessions",
            json={"name": "Bad Status", "status": "invalid_value"},
        )
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# Tests GET /studio/sessions/{id}
# ---------------------------------------------------------------------------


def test_get_session_by_id() -> None:
    """GET /studio/sessions/{id} doit retourner les détails avec stats."""
    with TestClient(app) as client:
        created = _create_session_via_api(client, "Get By ID")
        session_id = created["id"]
        res = client.get(f"/api/v1/studio/sessions/{session_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == session_id
    assert data["name"] == "Get By ID"
    assert "video_count" in data
    assert "annotation_count" in data


def test_get_session_not_found() -> None:
    """GET /studio/sessions/99999 doit retourner 404."""
    with TestClient(app) as client:
        res = client.get("/api/v1/studio/sessions/99999")
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Tests PATCH /studio/sessions/{id}
# ---------------------------------------------------------------------------


def test_patch_session_name() -> None:
    """PATCH /studio/sessions/{id} doit mettre à jour le nom."""
    with TestClient(app) as client:
        created = _create_session_via_api(client, "Old Name")
        session_id = created["id"]
        res = client.patch(
            f"/api/v1/studio/sessions/{session_id}",
            json={"name": "New Name"},
        )
    assert res.status_code == 200
    assert res.json()["name"] == "New Name"


def test_patch_session_status() -> None:
    """PATCH /studio/sessions/{id} doit permettre de changer le statut."""
    with TestClient(app) as client:
        created = _create_session_via_api(client, "Status Session")
        session_id = created["id"]
        res = client.patch(
            f"/api/v1/studio/sessions/{session_id}",
            json={"status": "completed"},
        )
    assert res.status_code == 200
    assert res.json()["status"] == "completed"


def test_patch_session_not_found() -> None:
    """PATCH /studio/sessions/99999 doit retourner 404."""
    with TestClient(app) as client:
        res = client.patch(
            "/api/v1/studio/sessions/99999",
            json={"name": "Ghost"},
        )
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Tests DELETE /studio/sessions/{id}
# ---------------------------------------------------------------------------


def test_delete_session() -> None:
    """DELETE /studio/sessions/{id} doit supprimer la session et retourner 204."""
    with TestClient(app) as client:
        created = _create_session_via_api(client, "Session à supprimer")
        session_id = created["id"]
        res = client.delete(f"/api/v1/studio/sessions/{session_id}")
        assert res.status_code == 204
        # Vérifier que la session est bien supprimée
        get_res = client.get(f"/api/v1/studio/sessions/{session_id}")
        assert get_res.status_code == 404


def test_delete_session_cascade_annotations() -> None:
    """Supprimer une session doit aussi supprimer ses annotations (cascade)."""
    video_id = _create_video_in_db()
    with TestClient(app) as client:
        session = _create_session_via_api(client, "Cascade Test")
        session_id = session["id"]
        # Créer une annotation
        ann_res = client.post(
            f"/api/v1/studio/videos/{video_id}/annotations",
            params={"session_id": session_id},
            json={
                "sign_label": "BONJOUR",
                "start_frame": 0,
                "end_frame": 30,
                "start_time_ms": 0.0,
                "end_time_ms": 1000.0,
            },
        )
        assert ann_res.status_code == 201
        annotation_id = ann_res.json()["id"]
        # Supprimer la session
        client.delete(f"/api/v1/studio/sessions/{session_id}")
        # Les annotations doivent être supprimées en cascade
        del_ann_res = client.delete(f"/api/v1/studio/annotations/{annotation_id}")
        # 404 car déjà supprimé en cascade
        assert del_ann_res.status_code == 404


def test_delete_session_not_found() -> None:
    """DELETE /studio/sessions/99999 doit retourner 404."""
    with TestClient(app) as client:
        res = client.delete("/api/v1/studio/sessions/99999")
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Tests POST /studio/videos/{id}/annotations
# ---------------------------------------------------------------------------


def test_create_annotation() -> None:
    """POST /studio/videos/{id}/annotations doit créer une annotation."""
    video_id = _create_video_in_db()
    with TestClient(app) as client:
        session = _create_session_via_api(client, "Annotation Session")
        session_id = session["id"]
        res = client.post(
            f"/api/v1/studio/videos/{video_id}/annotations",
            params={"session_id": session_id},
            json={
                "sign_label": "BONJOUR",
                "start_frame": 0,
                "end_frame": 30,
                "start_time_ms": 0.0,
                "end_time_ms": 1000.0,
                "confidence": 0.9,
                "is_verified": False,
            },
        )
    assert res.status_code == 201
    data = res.json()
    assert data["sign_label"] == "BONJOUR"
    assert data["video_id"] == video_id
    assert data["session_id"] == session_id
    assert data["confidence"] == pytest.approx(0.9)


def test_create_annotation_video_not_found() -> None:
    """POST /studio/videos/nonexistent/annotations doit retourner 404."""
    with TestClient(app) as client:
        session = _create_session_via_api(client, "Test Session")
        session_id = session["id"]
        res = client.post(
            "/api/v1/studio/videos/nonexistent/annotations",
            params={"session_id": session_id},
            json={
                "sign_label": "TEST",
                "start_frame": 0,
                "end_frame": 10,
                "start_time_ms": 0.0,
                "end_time_ms": 333.0,
            },
        )
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Tests GET /studio/videos/{id}/annotations
# ---------------------------------------------------------------------------


def test_list_annotations_for_video() -> None:
    """GET /studio/videos/{id}/annotations doit lister les annotations d'une vidéo."""
    video_id = _create_video_in_db()
    with TestClient(app) as client:
        session = _create_session_via_api(client, "List Annotations")
        session_id = session["id"]
        # Créer deux annotations
        for label in ("BONJOUR", "AU_REVOIR"):
            client.post(
                f"/api/v1/studio/videos/{video_id}/annotations",
                params={"session_id": session_id},
                json={
                    "sign_label": label,
                    "start_frame": 0,
                    "end_frame": 30,
                    "start_time_ms": 0.0,
                    "end_time_ms": 1000.0,
                },
            )
        res = client.get(f"/api/v1/studio/videos/{video_id}/annotations")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 2
    labels = {a["sign_label"] for a in data}
    assert "BONJOUR" in labels
    assert "AU_REVOIR" in labels


def test_list_annotations_filter_by_session() -> None:
    """GET /studio/videos/{id}/annotations?session_id=X ne retourne que les annotations de X."""
    video_id = _create_video_in_db()
    with TestClient(app) as client:
        s1 = _create_session_via_api(client, "Session One")
        s2 = _create_session_via_api(client, "Session Two")
        for sid, label in [(s1["id"], "SIGN_S1"), (s2["id"], "SIGN_S2")]:
            client.post(
                f"/api/v1/studio/videos/{video_id}/annotations",
                params={"session_id": sid},
                json={
                    "sign_label": label,
                    "start_frame": 0,
                    "end_frame": 30,
                    "start_time_ms": 0.0,
                    "end_time_ms": 1000.0,
                },
            )
        res = client.get(
            f"/api/v1/studio/videos/{video_id}/annotations",
            params={"session_id": s1["id"]},
        )
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["sign_label"] == "SIGN_S1"


# ---------------------------------------------------------------------------
# Tests PUT /studio/annotations/{id}
# ---------------------------------------------------------------------------


def test_update_annotation() -> None:
    """PUT /studio/annotations/{id} doit mettre à jour l'annotation."""
    video_id = _create_video_in_db()
    with TestClient(app) as client:
        session = _create_session_via_api(client, "Update Test")
        session_id = session["id"]
        ann_res = client.post(
            f"/api/v1/studio/videos/{video_id}/annotations",
            params={"session_id": session_id},
            json={
                "sign_label": "ORIGINAL",
                "start_frame": 0,
                "end_frame": 30,
                "start_time_ms": 0.0,
                "end_time_ms": 1000.0,
                "is_verified": False,
            },
        )
        annotation_id = ann_res.json()["id"]
        res = client.put(
            f"/api/v1/studio/annotations/{annotation_id}",
            json={"sign_label": "UPDATED", "is_verified": True},
        )
    assert res.status_code == 200
    data = res.json()
    assert data["sign_label"] == "UPDATED"
    assert data["is_verified"] is True


def test_update_annotation_not_found() -> None:
    """PUT /studio/annotations/99999 doit retourner 404."""
    with TestClient(app) as client:
        res = client.put(
            "/api/v1/studio/annotations/99999",
            json={"sign_label": "GHOST"},
        )
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Tests DELETE /studio/annotations/{id}
# ---------------------------------------------------------------------------


def test_delete_annotation() -> None:
    """DELETE /studio/annotations/{id} doit supprimer l'annotation."""
    video_id = _create_video_in_db()
    with TestClient(app) as client:
        session = _create_session_via_api(client, "Delete Annotation Test")
        session_id = session["id"]
        ann_res = client.post(
            f"/api/v1/studio/videos/{video_id}/annotations",
            params={"session_id": session_id},
            json={
                "sign_label": "TO_DELETE",
                "start_frame": 0,
                "end_frame": 10,
                "start_time_ms": 0.0,
                "end_time_ms": 333.0,
            },
        )
        annotation_id = ann_res.json()["id"]
        del_res = client.delete(f"/api/v1/studio/annotations/{annotation_id}")
        assert del_res.status_code == 204
        # Vérifier la suppression
        del_again = client.delete(f"/api/v1/studio/annotations/{annotation_id}")
        assert del_again.status_code == 404


# ---------------------------------------------------------------------------
# Tests POST /studio/videos/{id}/annotations/bulk
# ---------------------------------------------------------------------------


def test_bulk_create_annotations() -> None:
    """POST /studio/videos/{id}/annotations/bulk doit créer plusieurs annotations."""
    video_id = _create_video_in_db()
    with TestClient(app) as client:
        session = _create_session_via_api(client, "Bulk Import Test")
        session_id = session["id"]
        res = client.post(
            f"/api/v1/studio/videos/{video_id}/annotations/bulk",
            params={"session_id": session_id},
            json={
                "annotations": [
                    {
                        "sign_label": "SIGNE_1",
                        "start_frame": 0,
                        "end_frame": 30,
                        "start_time_ms": 0.0,
                        "end_time_ms": 1000.0,
                    },
                    {
                        "sign_label": "SIGNE_2",
                        "start_frame": 31,
                        "end_frame": 60,
                        "start_time_ms": 1033.0,
                        "end_time_ms": 2000.0,
                    },
                    {
                        "sign_label": "SIGNE_3",
                        "start_frame": 61,
                        "end_frame": 90,
                        "start_time_ms": 2033.0,
                        "end_time_ms": 3000.0,
                    },
                ]
            },
        )
    assert res.status_code == 201
    data = res.json()
    assert len(data) == 3
    labels = [a["sign_label"] for a in data]
    assert "SIGNE_1" in labels
    assert "SIGNE_3" in labels


def test_bulk_create_annotations_video_not_found() -> None:
    """Bulk import sur vidéo inexistante doit retourner 404."""
    with TestClient(app) as client:
        session = _create_session_via_api(client, "Bulk 404 Test")
        session_id = session["id"]
        res = client.post(
            "/api/v1/studio/videos/nonexistent/annotations/bulk",
            params={"session_id": session_id},
            json={"annotations": []},
        )
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Tests GET /studio/sessions/{id}/export
# ---------------------------------------------------------------------------


def test_export_session_json() -> None:
    """GET /studio/sessions/{id}/export?format=json doit retourner un JSON valide."""
    video_id = _create_video_in_db()
    with TestClient(app) as client:
        session = _create_session_via_api(client, "Export JSON Test")
        session_id = session["id"]
        client.post(
            f"/api/v1/studio/videos/{video_id}/annotations",
            params={"session_id": session_id},
            json={
                "sign_label": "EXPORT_SIGN",
                "start_frame": 0,
                "end_frame": 30,
                "start_time_ms": 0.0,
                "end_time_ms": 1000.0,
            },
        )
        res = client.get(
            f"/api/v1/studio/sessions/{session_id}/export",
            params={"format": "json"},
        )
    assert res.status_code == 200
    assert "application/json" in res.headers["content-type"]
    data = res.json()
    assert "session" in data
    assert "annotations" in data
    assert len(data["annotations"]) >= 1


def test_export_session_csv() -> None:
    """GET /studio/sessions/{id}/export?format=csv doit retourner un CSV valide."""
    video_id = _create_video_in_db()
    with TestClient(app) as client:
        session = _create_session_via_api(client, "Export CSV Test")
        session_id = session["id"]
        client.post(
            f"/api/v1/studio/videos/{video_id}/annotations",
            params={"session_id": session_id},
            json={
                "sign_label": "CSV_SIGN",
                "start_frame": 0,
                "end_frame": 30,
                "start_time_ms": 0.0,
                "end_time_ms": 1000.0,
            },
        )
        res = client.get(
            f"/api/v1/studio/sessions/{session_id}/export",
            params={"format": "csv"},
        )
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    content = res.text
    assert "sign_label" in content
    assert "CSV_SIGN" in content


def test_export_session_empty() -> None:
    """Export d'une session vide doit retourner 200 avec liste vide."""
    with TestClient(app) as client:
        session = _create_session_via_api(client, "Empty Export")
        session_id = session["id"]
        res = client.get(
            f"/api/v1/studio/sessions/{session_id}/export",
            params={"format": "json"},
        )
    assert res.status_code == 200
    data = res.json()
    assert data["annotations"] == []


def test_export_session_not_found() -> None:
    """Export d'une session inexistante doit retourner 404."""
    with TestClient(app) as client:
        res = client.get(
            "/api/v1/studio/sessions/99999/export",
            params={"format": "json"},
        )
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Tests GET /studio/stats
# ---------------------------------------------------------------------------


def test_get_studio_stats_empty() -> None:
    """GET /studio/stats sur état vide doit retourner des compteurs à zéro."""
    with TestClient(app) as client:
        res = client.get("/api/v1/studio/stats")
    assert res.status_code == 200
    data = res.json()
    for field in (
        "total_sessions",
        "active_sessions",
        "total_videos_annotated",
        "total_annotations",
        "verified_annotations",
        "total_grammar_annotations",
    ):
        assert field in data, f"Champ stats manquant: {field}"
        assert data[field] == 0


def test_get_studio_stats_with_data() -> None:
    """GET /studio/stats doit refléter les données créées."""
    video_id = _create_video_in_db()
    with TestClient(app) as client:
        session = _create_session_via_api(client, "Stats Data Test")
        session_id = session["id"]
        # Créer deux annotations : une vérifiée, une non vérifiée
        client.post(
            f"/api/v1/studio/videos/{video_id}/annotations",
            params={"session_id": session_id},
            json={
                "sign_label": "VERIFIED",
                "start_frame": 0,
                "end_frame": 30,
                "start_time_ms": 0.0,
                "end_time_ms": 1000.0,
                "is_verified": True,
            },
        )
        client.post(
            f"/api/v1/studio/videos/{video_id}/annotations",
            params={"session_id": session_id},
            json={
                "sign_label": "UNVERIFIED",
                "start_frame": 31,
                "end_frame": 60,
                "start_time_ms": 1033.0,
                "end_time_ms": 2000.0,
                "is_verified": False,
            },
        )
        res = client.get("/api/v1/studio/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["total_sessions"] >= 1
    assert data["active_sessions"] >= 1
    assert data["total_annotations"] >= 2
    assert data["verified_annotations"] >= 1
    assert data["total_videos_annotated"] >= 1
