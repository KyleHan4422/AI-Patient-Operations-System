"""Conversations: the human-readable transcript of every chat.

The LangGraph checkpoint also stores each conversation, as serialised graph
state -- that copy is the model's memory, and LangGraph creates and versions
its own tables for it (scripts/setup_checkpointer.py; alembic/env.py ignores
them). These two tables are the copy people and tools read.

Only the columns Phase 2 uses. patient_id, status and ended_at arrive with the
phases that first need them (6 and 10).

Hand-reviewed from an autogenerate draft; nothing needed adding.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamp(name: str) -> sa.Column:
    return sa.Column(
        name, sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
    )


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("thread_id", sa.Uuid(), nullable=False),
        sa.Column("channel", sa.Text(), nullable=False),
        _timestamp("created_at"),
        _timestamp("last_message_at"),
        sa.CheckConstraint("channel IN ('web', 'voice')", name=op.f("ck_conversations_channel")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversations")),
        sa.UniqueConstraint("thread_id", name=op.f("uq_conversations_thread_id")),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("conversation_id", sa.BigInteger(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "meta",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        _timestamp("created_at"),
        sa.CheckConstraint("role IN ('user', 'assistant')", name=op.f("ck_messages_role")),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_messages_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_messages")),
    )
    op.create_index("ix_messages_conversation_id_id", "messages", ["conversation_id", "id"])


def downgrade() -> None:
    op.drop_index("ix_messages_conversation_id_id", table_name="messages")
    op.drop_table("messages")
    op.drop_table("conversations")
