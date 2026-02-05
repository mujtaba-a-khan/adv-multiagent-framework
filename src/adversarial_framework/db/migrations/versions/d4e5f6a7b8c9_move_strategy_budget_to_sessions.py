"""Move strategy_name, strategy_params, max_turns, max_cost_usd to sessions table.

Enables per-session strategy selection so one experiment can test multiple
attack strategies and compare results.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add strategy and budget columns to sessions table
    op.add_column(
        "sessions",
        sa.Column("strategy_name", sa.String(64), nullable=False, server_default="pair"),
    )
    op.add_column(
        "sessions",
        sa.Column("strategy_params", JSON, nullable=False, server_default="{}"),
    )
    op.add_column(
        "sessions",
        sa.Column("max_turns", sa.Integer, nullable=False, server_default="20"),
    )
    op.add_column(
        "sessions",
        sa.Column("max_cost_usd", sa.Float, nullable=False, server_default="10.0"),
    )

    # Backfill existing sessions from their parent experiment
    op.execute(
        """
        UPDATE sessions
        SET strategy_name = e.strategy_name,
            strategy_params = e.strategy_params,
            max_turns = e.max_turns,
            max_cost_usd = e.max_cost_usd
        FROM experiments e
        WHERE sessions.experiment_id = e.id
        """
    )


def downgrade() -> None:
    op.drop_column("sessions", "max_cost_usd")
    op.drop_column("sessions", "max_turns")
    op.drop_column("sessions", "strategy_params")
    op.drop_column("sessions", "strategy_name")
