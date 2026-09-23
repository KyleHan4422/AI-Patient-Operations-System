"""What the assistant looked up, and what it could not answer.

Two tables, both hanging off the assistant message of the turn that produced
them, both written in that turn's single transaction:

  tool_calls  one row per completed read-only lookup. The trace behind an
              answer: which tools, with what arguments, and how long each took.
              Only completed calls -- a failing tool ends the turn before
              anything is written, and gets its columns when Phase 8 makes a
              failure survivable.
  kb_gaps     one row per abstention. The byproduct that pays for saying "I
              don't have that on file": the question, why it could not be
              answered, and how close retrieval got. best_score and threshold
              are nullable because a turn can abstain before it ever searches.

ON DELETE CASCADE on both: deleting a conversation must not leave its trace
behind. Hand-reviewed from an autogenerate draft; what it emitted was correct,
and the CHECK on kb_gaps.reason is spelled out below as SQL a reviewer can
read rather than as a generated one-liner.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Kept in step with db/models.KB_GAP_REASONS and graph/grounding.GAP_REASONS by
# a test, so a reason cannot be added in one place and be unwritable here.
GAP_REASONS = (
    "no_passage_above_threshold",
    "passages_did_not_answer",
    "answer_without_citation",
    "fabricated_citation",
    "unsupported_figure",
    "no_answer",
)


def upgrade() -> None:
    op.create_table(
        "tool_calls",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("message_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "args",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("summary", sa.Text(), nullable=False),
        # Integer, not SmallInteger: a lookup that embeds a query through the
        # provider can outlast 32767ms once the SDK has retried.
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.id"],
            name=op.f("fk_tool_calls_message_id_messages"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tool_calls")),
    )
    # Every read of this table is "the calls behind this reply".
    op.create_index("ix_tool_calls_message_id", "tool_calls", ["message_id"], unique=False)

    op.create_table(
        "kb_gaps",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("message_id", sa.BigInteger(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("best_score", sa.Float(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column("nearest_heading", sa.Text(), nullable=True),
        sa.Column("nearest_source", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "reason IN (" + ", ".join(f"'{reason}'" for reason in GAP_REASONS) + ")",
            name=op.f("ck_kb_gaps_reason"),
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.id"],
            name=op.f("fk_kb_gaps_message_id_messages"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_kb_gaps")),
    )


def downgrade() -> None:
    op.drop_table("kb_gaps")
    op.drop_index("ix_tool_calls_message_id", table_name="tool_calls")
    op.drop_table("tool_calls")
