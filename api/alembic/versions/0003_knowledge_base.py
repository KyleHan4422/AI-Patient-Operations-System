"""Knowledge base: the clinic's prose, chunked and embedded.

Hand-reviewed from an autogenerate draft. What autogenerate did not do:
  - install the `vector` extension, without which the embedding column's type
    does not exist. Like btree_gist in 0001, it belongs here and not in
    db/init/: those scripts run only when the Postgres volume is first
    created, so they never reach CI's database or the test database;
  - import pgvector, although it happily emitted pgvector's type name.

No index on kb_chunks.embedding. At this corpus size (tens of chunks) an
exact scan is microseconds, and an approximate index would trade recall for
an unmeasurable speed-up. See README for the size at which that changes.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 1536


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "kb_documents",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("effective_date", sa.Date(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("chunker_version", sa.SmallInteger(), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "category IN ('policy', 'clinical', 'scheduling', 'general')",
            name=op.f("ck_kb_documents_category"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_kb_documents")),
        sa.UniqueConstraint("source_path", name=op.f("uq_kb_documents_source_path")),
    )
    op.create_table(
        "kb_chunks",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("document_id", sa.BigInteger(), nullable=False),
        sa.Column("ordinal", sa.SmallInteger(), nullable=False),
        sa.Column("heading_path", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("approx_tokens", sa.SmallInteger(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=EMBEDDING_DIM), nullable=False),
        sa.CheckConstraint("approx_tokens > 0", name=op.f("ck_kb_chunks_positive_tokens")),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["kb_documents.id"],
            name=op.f("fk_kb_chunks_document_id_kb_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_kb_chunks")),
        sa.UniqueConstraint(
            "document_id", "ordinal", name=op.f("uq_kb_chunks_document_id_ordinal")
        ),
    )


def downgrade() -> None:
    op.drop_table("kb_chunks")  # CASCADE on the FK is for deletes, not for DDL
    op.drop_table("kb_documents")
    op.execute("DROP EXTENSION IF EXISTS vector")
