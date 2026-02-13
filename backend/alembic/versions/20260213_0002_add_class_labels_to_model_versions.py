"""Add class_labels to model versions.

Revision ID: 20260213_0002
Revises: 20260213_0001
Create Date: 2026-02-13 14:12:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260213_0002"
down_revision = "20260213_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "model_versions",
        sa.Column("class_labels", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
    )
    op.alter_column("model_versions", "class_labels", server_default=None)


def downgrade() -> None:
    op.drop_column("model_versions", "class_labels")
