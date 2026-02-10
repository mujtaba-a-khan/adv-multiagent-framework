"""Add playground tables

Revision ID: 3cac73e583ec
Revises: g7b8c9d0e1f2
Create Date: 2026-02-10 00:46:20.660316
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '3cac73e583ec'
down_revision: Union[str, None] = 'g7b8c9d0e1f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('playground_conversations',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('title', sa.String(length=255), nullable=False),
    sa.Column('target_model', sa.String(length=128), nullable=False),
    sa.Column('target_provider', sa.String(length=64), nullable=False),
    sa.Column('system_prompt', sa.Text(), nullable=True),
    sa.Column('analyzer_model', sa.String(length=128), nullable=False),
    sa.Column('active_defenses', postgresql.JSON(astext_type=sa.Text()), nullable=False),
    sa.Column('total_messages', sa.Integer(), nullable=False),
    sa.Column('total_jailbreaks', sa.Integer(), nullable=False),
    sa.Column('total_blocked', sa.Integer(), nullable=False),
    sa.Column('total_target_tokens', sa.Integer(), nullable=False),
    sa.Column('total_analyzer_tokens', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('playground_messages',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('conversation_id', sa.UUID(), nullable=False),
    sa.Column('message_number', sa.Integer(), nullable=False),
    sa.Column('user_prompt', sa.Text(), nullable=False),
    sa.Column('target_response', sa.Text(), nullable=False),
    sa.Column('raw_target_response', sa.Text(), nullable=True),
    sa.Column('target_blocked', sa.Boolean(), nullable=False),
    sa.Column('blocked_by_defense', sa.String(length=64), nullable=True),
    sa.Column('block_reason', sa.Text(), nullable=True),
    sa.Column('judge_verdict', sa.String(length=32), nullable=False),
    sa.Column('judge_confidence', sa.Float(), nullable=False),
    sa.Column('severity_score', sa.Float(), nullable=True),
    sa.Column('specificity_score', sa.Float(), nullable=True),
    sa.Column('vulnerability_category', sa.String(length=64), nullable=True),
    sa.Column('attack_technique', sa.String(length=64), nullable=True),
    sa.Column('target_tokens', sa.Integer(), nullable=False),
    sa.Column('analyzer_tokens', sa.Integer(), nullable=False),
    sa.Column('target_latency_ms', sa.Float(), nullable=False),
    sa.Column('analyzer_latency_ms', sa.Float(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['conversation_id'], ['playground_conversations.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('playground_messages')
    op.drop_table('playground_conversations')
