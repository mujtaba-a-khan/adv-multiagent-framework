"""Add defenses_applied to playground_messages

Revision ID: h8c9d0e1f2g3
Revises: 3cac73e583ec
Create Date: 2026-02-10 14:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "h8c9d0e1f2g3"
down_revision: str | None = "3cac73e583ec"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "playground_messages",
        sa.Column(
            "defenses_applied",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("playground_messages", "defenses_applied")
