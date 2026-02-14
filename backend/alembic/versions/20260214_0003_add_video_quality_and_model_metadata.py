"""Add video quality metadata and model artifact metadata.

Revision ID: 20260214_0003
Revises: ffaa343c9619
Create Date: 2026-02-14 11:10:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260214_0003"
down_revision = "ffaa343c9619"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("videos", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("detection_rate", sa.Float(), nullable=False, server_default=sa.text("0"))
        )
        batch_op.add_column(
            sa.Column("quality_score", sa.Float(), nullable=False, server_default=sa.text("0"))
        )
        batch_op.add_column(
            sa.Column("is_trainable", sa.Boolean(), nullable=False, server_default=sa.text("true"))
        )
        batch_op.add_column(
            sa.Column("landmark_feature_dim", sa.Integer(), nullable=False, server_default="225")
        )
        batch_op.create_index("ix_videos_is_trainable", ["is_trainable"], unique=False)

    op.alter_column("videos", "detection_rate", server_default=None)
    op.alter_column("videos", "quality_score", server_default=None)
    op.alter_column("videos", "is_trainable", server_default=None)
    op.alter_column("videos", "landmark_feature_dim", server_default=None)

    with op.batch_alter_table("model_versions", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'"))
        )
    op.alter_column("model_versions", "metadata", server_default=None)


def downgrade() -> None:
    with op.batch_alter_table("model_versions", schema=None) as batch_op:
        batch_op.drop_column("metadata")

    with op.batch_alter_table("videos", schema=None) as batch_op:
        batch_op.drop_index("ix_videos_is_trainable")
        batch_op.drop_column("landmark_feature_dim")
        batch_op.drop_column("is_trainable")
        batch_op.drop_column("quality_score")
        batch_op.drop_column("detection_rate")
