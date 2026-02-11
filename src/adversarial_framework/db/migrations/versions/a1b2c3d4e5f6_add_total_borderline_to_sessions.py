"""Add total_borderline column to sessions table

Revision ID: a1b2c3d4e5f6
Revises: 5ae5f7245643
Create Date: 2026-02-02 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "5ae5f7245643"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column("total_borderline", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("sessions", "total_borderline")
