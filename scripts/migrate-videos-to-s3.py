#!/usr/bin/env python3
"""
migrate-videos-to-s3.py — Migration one-shot des vidéos locales vers MinIO S3.

À exécuter UNE SEULE FOIS avant d'activer USE_S3_STORAGE=true.
Les vidéos existantes sur le filesystem local sont uploadées vers MinIO,
et les entrées DB sont mises à jour avec les clés S3.

Pré-requis :
    - MinIO doit être démarré (docker compose up minio minio_setup)
    - USE_S3_STORAGE doit être FALSE pendant la migration
    - Exécuter avec le venv Python du backend activé

Usage :
    # Tester sans rien modifier
    python scripts/migrate-videos-to-s3.py --dry-run \\
        --db-url "postgresql+psycopg://signflow:PASSWORD@localhost:5432/signflow" \\
        --video-dir ./backend/data/videos \\
        --s3-endpoint http://localhost:9000 \\
        --s3-access-key signflow-admin \\
        --s3-secret-key SECRETPASSWORD \\
        --s3-bucket signflow-videos

    # Migration réelle (retirer --dry-run)
    python scripts/migrate-videos-to-s3.py \\
        --db-url "postgresql+psycopg://signflow:PASSWORD@localhost:5432/signflow" \\
        --video-dir ./backend/data/videos \\
        --s3-endpoint http://localhost:9000 \\
        --s3-access-key signflow-admin \\
        --s3-secret-key SECRETPASSWORD \\
        --s3-bucket signflow-videos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ajouter le backend au sys.path pour importer les modèles ORM
SCRIPT_DIR = Path(__file__).parent.resolve()
BACKEND_DIR = SCRIPT_DIR.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


def is_s3_key(path: str) -> bool:
    """Retourne True si le chemin est déjà une clé S3 (pas un chemin absolu)."""
    return bool(path) and not path.startswith("/")


def get_video_type_from_path(path: str) -> str:
    """Extraire le type de vidéo depuis le chemin local."""
    parts = Path(path).parts
    for part in parts:
        if part in ("training", "reference", "example"):
            return part
    return "reference"


def migrate(args: argparse.Namespace) -> None:
    import boto3
    from botocore.exceptions import ClientError
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    mode_label = "[DRY RUN] " if args.dry_run else ""

    print(f"\n{'='*60}")
    print(f"  SignFlow — Migration vidéos vers MinIO S3")
    print(f"  {mode_label}Endpoint  : {args.s3_endpoint}")
    print(f"  {mode_label}Bucket    : {args.s3_bucket}")
    print(f"  Video dir : {args.video_dir}")
    print(f"  DB URL    : {args.db_url[:50]}...")
    print(f"{'='*60}\n")

    # ── Connexion S3 ──────────────────────────────────────────────────────
    s3 = boto3.client(
        "s3",
        endpoint_url=args.s3_endpoint,
        aws_access_key_id=args.s3_access_key,
        aws_secret_access_key=args.s3_secret_key,
        region_name="us-east-1",
    )

    # Vérifier que le bucket existe
    try:
        s3.head_bucket(Bucket=args.s3_bucket)
        print(f"✓ Bucket '{args.s3_bucket}' accessible\n")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code == "404":
            if args.dry_run:
                print(f"  [DRY RUN] Bucket '{args.s3_bucket}' n'existe pas (sera créé au déploiement)\n")
            else:
                print(f"✗ Bucket '{args.s3_bucket}' introuvable. Démarrer minio_setup d'abord.")
                sys.exit(1)
        else:
            print(f"✗ Erreur d'accès MinIO : {exc}")
            sys.exit(1)

    # ── Connexion DB ──────────────────────────────────────────────────────
    try:
        from app.models.video import Video
    except ImportError as exc:
        print(f"✗ Impossible d'importer les modèles ORM : {exc}")
        print("  Vérifier que le PYTHONPATH inclut le répertoire backend/")
        sys.exit(1)

    engine = create_engine(args.db_url)

    # ── Scan et migration ─────────────────────────────────────────────────
    total = migrated = skipped = errors = 0

    with Session(engine) as db:
        videos = db.scalars(select(Video)).all()
        total = len(videos)
        print(f"  {total} vidéo(s) trouvée(s) en base de données\n")

        for video in videos:
            vid_id = str(video.id)
            file_path = video.file_path or ""
            lm_path = video.landmarks_path or ""

            # ── Skip si déjà migré (clé S3) ──────────────────────────────
            if is_s3_key(file_path):
                print(f"  SKIP  {vid_id[:8]}... : déjà en S3 ({file_path})")
                skipped += 1
                continue

            local_path = Path(file_path)
            if not local_path.exists():
                print(f"  WARN  {vid_id[:8]}... : fichier manquant {local_path}")
                errors += 1
                continue

            video_type = get_video_type_from_path(file_path)
            s3_key = f"videos/{video_type}/{local_path.name}"

            # ── Upload vidéo ──────────────────────────────────────────────
            print(f"  UPLOAD {vid_id[:8]}... : {local_path.name} → {s3_key}", end="", flush=True)

            if not args.dry_run:
                try:
                    s3.upload_file(str(local_path), args.s3_bucket, s3_key)
                    video.file_path = s3_key
                    print(" ✓")
                except Exception as exc:
                    print(f" ✗ ({exc})")
                    errors += 1
                    continue
            else:
                size_mb = local_path.stat().st_size / (1024 * 1024)
                print(f" [{size_mb:.1f}MB — dry run]")

            # ── Upload landmarks ──────────────────────────────────────────
            if lm_path and not is_s3_key(lm_path):
                lm_local = Path(lm_path)
                if lm_local.exists():
                    lm_s3_key = f"videos/{video_type}/{lm_local.name}"
                    print(f"         landmarks → {lm_s3_key}", end="", flush=True)

                    if not args.dry_run:
                        try:
                            s3.upload_file(str(lm_local), args.s3_bucket, lm_s3_key)
                            video.landmarks_path = lm_s3_key
                            print(" ✓")
                        except Exception as exc:
                            print(f" ✗ ({exc})")
                    else:
                        print(" [dry run]")
                else:
                    print(f"         landmarks manquants : {lm_local}")

            if not args.dry_run:
                db.commit()

            migrated += 1

    # ── Résumé ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {mode_label}Résultat :")
    print(f"    Total   : {total}")
    print(f"    Migré   : {migrated}")
    print(f"    Skip    : {skipped} (déjà en S3)")
    print(f"    Erreurs : {errors}")

    if args.dry_run:
        print(f"\n  DRY RUN terminé. Relancer sans --dry-run pour migrer réellement.")
    elif errors == 0:
        print(f"\n  ✓ Migration terminée avec succès !")
        print(f"  Activer USE_S3_STORAGE=true dans .env.server et redémarrer les services.")
    else:
        print(f"\n  ⚠ Migration terminée avec {errors} erreur(s). Vérifier les logs ci-dessus.")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrer les vidéos SignFlow du filesystem local vers MinIO S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db-url",
        required=True,
        help="URL de connexion SQLAlchemy (ex: postgresql+psycopg://user:pass@localhost:5432/signflow)",
    )
    parser.add_argument(
        "--video-dir",
        default="./backend/data/videos",
        help="Répertoire local des vidéos (défaut: ./backend/data/videos)",
    )
    parser.add_argument(
        "--s3-endpoint",
        default="http://localhost:9000",
        help="URL endpoint MinIO (défaut: http://localhost:9000)",
    )
    parser.add_argument(
        "--s3-access-key",
        required=True,
        help="Access key MinIO (= MINIO_ROOT_USER)",
    )
    parser.add_argument(
        "--s3-secret-key",
        required=True,
        help="Secret key MinIO (= MINIO_ROOT_PASSWORD)",
    )
    parser.add_argument(
        "--s3-bucket",
        default="signflow-videos",
        help="Nom du bucket MinIO (défaut: signflow-videos)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simuler sans modifier la DB ni uploader vers S3",
    )
    args = parser.parse_args()
    migrate(args)


if __name__ == "__main__":
    main()
