"""make video sign_id nullable for unlabeled videos

Revision ID: ffaa343c9619
Revises: 20260213_0002
Create Date: 2026-02-13 18:44:33.030572
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ffaa343c9619'
down_revision = '20260213_0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # SQLite requires batch operations for ALTER COLUMN
    with op.batch_alter_table('videos', schema=None) as batch_op:
        batch_op.alter_column(
            'sign_id',
            existing_type=sa.String(36),
            nullable=True
        )


def downgrade() -> None:
    # Delete unlabeled videos before making column NOT NULL again
    op.execute("DELETE FROM videos WHERE sign_id IS NULL")

    # SQLite requires batch operations for ALTER COLUMN
    with op.batch_alter_table('videos', schema=None) as batch_op:
        batch_op.alter_column(
            'sign_id',
            existing_type=sa.String(36),
            nullable=False
        )
