"""Add finetuning_jobs table"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON, UUID

# revision identifiers, used by Alembic.
revision: str = 'f6a7b8c9d0e1'
down_revision: Union[str, None] = 'e5f6a7b8c9d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'finetuning_jobs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('job_type', sa.String(32), nullable=False),
        sa.Column('source_model', sa.String(255), nullable=False),
        sa.Column('output_model_name', sa.String(255), nullable=False),
        sa.Column('config', JSON, server_default='{}'),
        sa.Column('status', sa.String(32), server_default='pending'),
        sa.Column('progress_pct', sa.Float, server_default='0.0'),
        sa.Column('current_step', sa.String(255), nullable=True),
        sa.Column('logs', JSON, server_default='[]'),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('peak_memory_gb', sa.Float, nullable=True),
        sa.Column('total_duration_seconds', sa.Float, nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            'completed_at', sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table('finetuning_jobs')
