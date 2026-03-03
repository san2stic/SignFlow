"""add deployment_history table

Revision ID: 20260303_0006
Revises: 20260303_0005
Create Date: 2026-03-03 16:00:00.000000

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260303_0006"
down_revision = "20260303_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "deployment_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("triggered_by", sa.String(20), nullable=False, server_default="auto"),
        sa.Column("commit_hash", sa.String(40), nullable=True),
        sa.Column("previous_commit_hash", sa.String(40), nullable=True),
        sa.Column("commit_message", sa.Text(), nullable=True),
        sa.Column("commit_author", sa.String(200), nullable=True),
        sa.Column("committed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("build_log", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("build_duration_s", sa.Float(), nullable=True),
        sa.Column("deploy_duration_s", sa.Float(), nullable=True),
        sa.Column("total_duration_s", sa.Float(), nullable=True),
        sa.Column("rollback_of_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["rollback_of_id"],
            ["deployment_history.id"],
            name="fk_deployment_history_rollback_of_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_deployment_history_status",
        "deployment_history",
        ["status"],
    )
    op.create_index(
        "ix_deployment_history_created_at",
        "deployment_history",
        ["created_at"],
    )
    op.create_index(
        "ix_deployment_history_triggered_by",
        "deployment_history",
        ["triggered_by"],
    )


def downgrade() -> None:
    op.drop_index("ix_deployment_history_triggered_by", table_name="deployment_history")
    op.drop_index("ix_deployment_history_created_at", table_name="deployment_history")
    op.drop_index("ix_deployment_history_status", table_name="deployment_history")
    op.drop_table("deployment_history")
