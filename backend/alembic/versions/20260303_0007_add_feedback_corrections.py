"""add feedback_corrections table

Revision ID: 20260303_0007
Revises: 20260303_0006
Create Date: 2026-03-03 18:00:00.000000

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260303_0007"
down_revision = "20260303_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "feedback_corrections",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("predicted_sign", sa.String(200), nullable=False),
        sa.Column("corrected_sign", sa.String(200), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("landmarks_path", sa.String(500), nullable=True),
        sa.Column("session_id", sa.String(200), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_feedback_corrections_predicted_sign",
        "feedback_corrections",
        ["predicted_sign"],
    )
    op.create_index(
        "ix_feedback_corrections_corrected_sign",
        "feedback_corrections",
        ["corrected_sign"],
    )
    op.create_index(
        "ix_feedback_corrections_status",
        "feedback_corrections",
        ["status"],
    )
    op.create_index(
        "ix_feedback_corrections_session_id",
        "feedback_corrections",
        ["session_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_feedback_corrections_session_id", table_name="feedback_corrections")
    op.drop_index("ix_feedback_corrections_status", table_name="feedback_corrections")
    op.drop_index("ix_feedback_corrections_corrected_sign", table_name="feedback_corrections")
    op.drop_index("ix_feedback_corrections_predicted_sign", table_name="feedback_corrections")
    op.drop_table("feedback_corrections")
