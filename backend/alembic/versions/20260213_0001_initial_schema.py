"""Initial SignFlow schema.

Revision ID: 20260213_0001
Revises:
Create Date: 2026-02-13 14:10:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260213_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "signs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=120), nullable=False, unique=True),
        sa.Column("slug", sa.String(length=140), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(length=80), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("variants", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("usage_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("video_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("training_sample_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_signs_slug", "signs", ["slug"], unique=True)
    op.create_index("ix_signs_category", "signs", ["category"], unique=False)

    op.create_table(
        "training_sessions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("sign_id", sa.String(length=36), sa.ForeignKey("signs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
        sa.Column("config", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("model_version_produced", sa.String(length=24), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_training_sessions_status", "training_sessions", ["status"], unique=False)

    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("version", sa.String(length=24), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("num_classes", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("accuracy", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "training_session_id",
            sa.String(length=36),
            sa.ForeignKey("training_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("file_path", sa.String(length=512), nullable=False),
        sa.Column("file_size_mb", sa.Float(), nullable=False, server_default="0"),
        sa.Column("parent_version", sa.String(length=24), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_model_versions_is_active", "model_versions", ["is_active"], unique=False)
    op.create_index("ix_model_versions_version", "model_versions", ["version"], unique=True)

    op.create_table(
        "videos",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("sign_id", sa.String(length=36), sa.ForeignKey("signs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("file_path", sa.String(length=512), nullable=False),
        sa.Column("thumbnail_path", sa.String(length=512), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("fps", sa.Integer(), nullable=False, server_default="30"),
        sa.Column("resolution", sa.String(length=32), nullable=False, server_default="640x480"),
        sa.Column("type", sa.String(length=32), nullable=False, server_default="reference"),
        sa.Column("landmarks_extracted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("landmarks_path", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_videos_sign_id", "videos", ["sign_id"], unique=False)
    op.create_index("ix_videos_type", "videos", ["type"], unique=False)

    op.create_table(
        "sign_relations",
        sa.Column("source_sign_id", sa.String(length=36), sa.ForeignKey("signs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("target_sign_id", sa.String(length=36), sa.ForeignKey("signs.id", ondelete="CASCADE"), primary_key=True),
    )


def downgrade() -> None:
    op.drop_table("sign_relations")
    op.drop_index("ix_videos_type", table_name="videos")
    op.drop_index("ix_videos_sign_id", table_name="videos")
    op.drop_table("videos")
    op.drop_index("ix_model_versions_version", table_name="model_versions")
    op.drop_index("ix_model_versions_is_active", table_name="model_versions")
    op.drop_table("model_versions")
    op.drop_index("ix_training_sessions_status", table_name="training_sessions")
    op.drop_table("training_sessions")
    op.drop_index("ix_signs_category", table_name="signs")
    op.drop_index("ix_signs_slug", table_name="signs")
    op.drop_table("signs")
