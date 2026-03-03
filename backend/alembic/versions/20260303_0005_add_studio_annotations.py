"""add studio annotation tables

Revision ID: 20260303_0005
Revises: ffaa343c9619
Create Date: 2026-03-03 13:00:00.000000

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "20260303_0005"
down_revision = "ffaa343c9619"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- annotation_sessions ---
    op.create_table(
        "annotation_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_annotation_sessions_status", "annotation_sessions", ["status"])

    # --- annotation_session_videos (join table) ---
    op.create_table(
        "annotation_session_videos",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.String(36), nullable=False),
        sa.Column(
            "added_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"], ["annotation_sessions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_annotation_session_videos_session_id",
        "annotation_session_videos",
        ["session_id"],
    )
    op.create_index(
        "ix_annotation_session_videos_video_id",
        "annotation_session_videos",
        ["video_id"],
    )

    # --- video_annotations ---
    op.create_table(
        "video_annotations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("video_id", sa.String(36), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("sign_label", sa.String(120), nullable=False),
        sa.Column("start_frame", sa.Integer(), nullable=False),
        sa.Column("end_frame", sa.Integer(), nullable=False),
        sa.Column("start_time_ms", sa.Float(), nullable=False),
        sa.Column("end_time_ms", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("annotator_notes", sa.Text(), nullable=True),
        sa.Column("nmm_tags", sa.JSON().with_variant(JSONB, "postgresql"), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["session_id"], ["annotation_sessions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_video_annotations_video_id", "video_annotations", ["video_id"])
    op.create_index(
        "ix_video_annotations_session_id", "video_annotations", ["session_id"]
    )
    op.create_index(
        "ix_video_annotations_sign_label", "video_annotations", ["sign_label"]
    )

    # --- grammar_annotations ---
    op.create_table(
        "grammar_annotations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.String(36), nullable=False),
        sa.Column(
            "sign_sequence",
            sa.JSON().with_variant(JSONB, "postgresql"),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("french_translation", sa.Text(), nullable=False),
        sa.Column(
            "grammar_tags",
            sa.JSON().with_variant(JSONB, "postgresql"),
            nullable=True,
        ),
        sa.Column("annotator_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"], ["annotation_sessions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["annotator_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_grammar_annotations_session_id", "grammar_annotations", ["session_id"]
    )
    op.create_index(
        "ix_grammar_annotations_video_id", "grammar_annotations", ["video_id"]
    )
    op.create_index(
        "ix_grammar_annotations_annotator_id", "grammar_annotations", ["annotator_id"]
    )


def downgrade() -> None:
    op.drop_table("grammar_annotations")
    op.drop_table("video_annotations")
    op.drop_table("annotation_session_videos")
    op.drop_table("annotation_sessions")
