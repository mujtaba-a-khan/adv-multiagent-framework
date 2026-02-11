"""Add attacker_reasoning column to turns table"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e5f6a7b8c9d0"
down_revision: str | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "turns",
        sa.Column("attacker_reasoning", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("turns", "attacker_reasoning")
