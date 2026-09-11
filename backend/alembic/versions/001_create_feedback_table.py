"""create feedback table

Revision ID: 001_feedback
Revises: 
Create Date: 2026-09-11 17:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "001_feedback"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("feedback_id", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=20), nullable=False),
        sa.Column("predicted_emotion", sa.String(length=30), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("corrected_emotion", sa.String(length=30), nullable=True),
        sa.Column("image_path", sa.String(length=512), nullable=True),
        sa.Column("bounding_box", sa.JSON(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("model_version", sa.String(length=100), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_feedback_feedback_id", "feedback", ["feedback_id"], unique=True)
    op.create_index("ix_feedback_state", "feedback", ["state"], unique=False)
    op.create_index("ix_feedback_predicted_emotion", "feedback", ["predicted_emotion"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_feedback_predicted_emotion", table_name="feedback")
    op.drop_index("ix_feedback_state", table_name="feedback")
    op.drop_index("ix_feedback_feedback_id", table_name="feedback")
    op.drop_table("feedback")
