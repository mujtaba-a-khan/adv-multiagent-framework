"""Add abliteration_prompts table"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = 'g7b8c9d0e1f2'
down_revision: Union[str, None] = 'f6a7b8c9d0e1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'abliteration_prompts',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('text', sa.Text, nullable=False),
        sa.Column('category', sa.String(16), nullable=False),
        sa.Column('source', sa.String(32), server_default='manual'),
        sa.Column('status', sa.String(16), server_default='active'),
        sa.Column(
            'experiment_id',
            UUID(as_uuid=True),
            sa.ForeignKey('experiments.id', ondelete='SET NULL'),
            nullable=True,
        ),
        sa.Column(
            'session_id',
            UUID(as_uuid=True),
            sa.ForeignKey('sessions.id', ondelete='SET NULL'),
            nullable=True,
        ),
        sa.Column(
            'pair_id',
            UUID(as_uuid=True),
            sa.ForeignKey('abliteration_prompts.id', ondelete='SET NULL'),
            nullable=True,
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        'ix_abliteration_prompts_category',
        'abliteration_prompts',
        ['category'],
    )
    op.create_index(
        'ix_abliteration_prompts_status',
        'abliteration_prompts',
        ['status'],
    )


def downgrade() -> None:
    op.drop_index('ix_abliteration_prompts_status')
    op.drop_index('ix_abliteration_prompts_category')
    op.drop_table('abliteration_prompts')
