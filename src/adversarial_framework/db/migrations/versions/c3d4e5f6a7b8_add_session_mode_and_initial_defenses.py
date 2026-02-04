"""Add session_mode and initial_defenses columns to sessions table."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column(
            "session_mode",
            sa.String(32),
            nullable=False,
            server_default="attack",
        ),
    )
    op.add_column(
        "sessions",
        sa.Column(
            "initial_defenses",
            JSON(),
            nullable=False,
            server_default="[]",
        ),
    )


def downgrade() -> None:
    op.drop_column("sessions", "initial_defenses")
    op.drop_column("sessions", "session_mode")
