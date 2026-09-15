"""add_feedback_review_fields

Revision ID: 003_feedback_review
Revises: 002
Create Date: 2026-09-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003_feedback_review'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add review fields to feedback table
    op.add_column('feedback', sa.Column('review_status', sa.String(length=20), server_default='pending', nullable=False))
    op.add_column('feedback', sa.Column('final_label', sa.String(length=30), nullable=True))
    op.add_column('feedback', sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('feedback', sa.Column('reviewed_by_id', sa.Integer(), nullable=True))

    op.create_index(op.f('ix_feedback_review_status'), 'feedback', ['review_status'], unique=False)
    op.create_index(op.f('ix_feedback_reviewed_by_id'), 'feedback', ['reviewed_by_id'], unique=False)
    op.create_foreign_key('feedback_reviewed_by_id_fkey', 'feedback', 'users', ['reviewed_by_id'], ['id'], ondelete='SET NULL')


def downgrade() -> None:
    op.drop_constraint('feedback_reviewed_by_id_fkey', 'feedback', type_='foreignkey')
    op.drop_index(op.f('ix_feedback_reviewed_by_id'), table_name='feedback')
    op.drop_index(op.f('ix_feedback_review_status'), table_name='feedback')

    op.drop_column('feedback', 'reviewed_by_id')
    op.drop_column('feedback', 'reviewed_at')
    op.drop_column('feedback', 'final_label')
    op.drop_column('feedback', 'review_status')
