"""The system of record.

Seven domain tables own every correctness guarantee; two conversation tables
record what was said, for people to read; two more record what the assistant
looked up and what it could not answer; two knowledge-base tables hold the
clinic's prose and its embeddings.

The two constraints that matter most are on `Appointment`:

    no_overlap        EXCLUDE -- one provider cannot hold two *booked*
                      appointments whose time ranges overlap. Postgres enforces
                      it, so no race between two application processes can get
                      around it.
    idempotency_key   UNIQUE -- the same booking request sent twice produces one
                      row; the second attempt finds the first instead of
                      creating another.

Conventions:
- Every instant is TIMESTAMPTZ. Schedules are local wall-clock TIME values in
  the clinic's timezone; turning them into instants is domain/availability.py's
  job, never the database's.
- Enumerations are TEXT + CHECK, not Postgres ENUM types: adding a value to an
  ENUM needs ALTER TYPE; a CHECK is replaced in one migration.
- Money is NUMERIC, never float.
- No relationship() attributes. Under asyncio an implicit lazy load is hidden
  IO that fails at runtime; every join is written out in db/repo.py.
- Constraint names come from NAMING_CONVENTION, so a migration can drop or
  alter a constraint by a name it knows in advance.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    Numeric,
    SmallInteger,
    Text,
    Time,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, ExcludeConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

SPECIALTIES = ("general", "hygienist", "orthodontist")
KB_CATEGORIES = ("policy", "clinical", "scheduling", "general")
APPOINTMENT_STATUSES = ("booked", "cancelled", "completed")
BOOKING_SOURCES = ("web", "voice", "staff")
CHANNELS = ("web", "voice")
MESSAGE_ROLES = ("user", "assistant")
# Why a turn ended in "I don't have that on file". graph/grounding.py decides
# which one applies and holds the same tuple; a test asserts they agree, so a
# reason cannot be invented on one side and be unwritable on the other.
KB_GAP_REASONS = (
    "no_passage_above_threshold",
    "passages_did_not_answer",
    "answer_without_citation",
    "fabricated_citation",
    "unsupported_figure",
    "no_answer",
)

# Dimensions of one embedding vector -- text-embedding-3-small's native size.
# Part of the schema, not configuration: a model of a different size needs a
# migration, and that friction is correct. Vectors from two different models
# are not comparable at all, so kb_documents also records which model produced
# them and retrieval filters on it.
EMBEDDING_DIM = 1536

# Tables LangGraph's checkpointer creates and versions itself (its setup() runs
# from `make migrate`). They share this database but not this metadata, so
# Alembic ignores exactly these names -- an explicit list, not "anything
# unknown", which would also hide a table a migration forgot to drop.
LANGGRAPH_TABLES: frozenset[str] = frozenset(
    {"checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"}
)


def _one_of(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# People and resources
# ---------------------------------------------------------------------------
class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    full_name: Mapped[str] = mapped_column(Text)
    # E.164 ("+12125550143"), normalised by domain/phone.py before it gets here,
    # so UNIQUE compares like with like.
    phone: Mapped[str] = mapped_column(Text, unique=True)
    dob: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = _created_at()


class Provider(Base):
    __tablename__ = "providers"
    __table_args__ = (CheckConstraint(_one_of("specialty", SPECIALTIES), name="specialty"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True)  # natural key for seed upserts
    specialty: Mapped[str] = mapped_column(Text)


class ProviderSchedule(Base):
    """A recurring working window, e.g. Monday 09:00-12:00 local time."""

    __tablename__ = "provider_schedules"
    __table_args__ = (
        # Python's date.weekday(): Monday=0 .. Sunday=6. NOT Postgres's
        # extract(dow), which starts at Sunday=0 -- mixing the two shifts every
        # schedule by a day.
        CheckConstraint("weekday BETWEEN 0 AND 6", name="weekday_range"),
        CheckConstraint("end_time > start_time", name="end_after_start"),
        UniqueConstraint("provider_id", "weekday", "start_time"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    provider_id: Mapped[int] = mapped_column(ForeignKey("providers.id"))
    weekday: Mapped[int] = mapped_column(SmallInteger)
    start_time: Mapped[time] = mapped_column(Time)  # local wall-clock time
    end_time: Mapped[time] = mapped_column(Time)


class ClinicClosure(Base):
    __tablename__ = "clinic_closures"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    closed_on: Mapped[date] = mapped_column(Date, unique=True)  # a clinic-local date
    reason: Mapped[str] = mapped_column(Text)


# ---------------------------------------------------------------------------
# Reference data behind the agents' read-only tools
# ---------------------------------------------------------------------------
class Procedure(Base):
    __tablename__ = "procedures"
    __table_args__ = (
        CheckConstraint("duration_min > 0", name="positive_duration"),
        CheckConstraint("price_max >= price_min", name="price_range"),
        CheckConstraint(_one_of("specialty", SPECIALTIES), name="specialty"),
    )

    code: Mapped[str] = mapped_column(Text, primary_key=True)  # "CLEANING", "CROWN"
    name: Mapped[str] = mapped_column(Text)
    duration_min: Mapped[int] = mapped_column(SmallInteger)
    # NULL = we have no price on file. The pricing tool must then say so, never
    # guess.
    price_min: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    price_max: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    # Which kind of provider performs it -- without this a crown could be
    # booked with a hygienist.
    specialty: Mapped[str] = mapped_column(Text)


class InsurancePlan(Base):
    __tablename__ = "insurance_plans"
    __table_args__ = (
        CheckConstraint("accepted OR NOT in_network", name="in_network_implies_accepted"),
        # Case-insensitive, exact. Deliberately not fuzzy: fuzzy matching is
        # exactly what would blur "Delta Dental PPO" into "Cigna DPPO".
        Index("uq_insurance_plans_lower_plan_name", func.lower(text("plan_name")), unique=True),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    plan_name: Mapped[str] = mapped_column(Text)
    accepted: Mapped[bool] = mapped_column(Boolean)
    in_network: Mapped[bool] = mapped_column(Boolean)
    notes: Mapped[str | None] = mapped_column(Text)


# ---------------------------------------------------------------------------
# Appointments -- where the correctness guarantees live
# ---------------------------------------------------------------------------
class Appointment(Base):
    __tablename__ = "appointments"
    __table_args__ = (
        CheckConstraint("end_at > start_at", name="end_after_start"),
        CheckConstraint(_one_of("status", APPOINTMENT_STATUSES), name="status"),
        CheckConstraint(_one_of("source", BOOKING_SOURCES), name="source"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    provider_id: Mapped[int] = mapped_column(ForeignKey("providers.id"))
    procedure_code: Mapped[str] = mapped_column(ForeignKey("procedures.code"))
    start_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(Text, server_default="booked")
    source: Mapped[str] = mapped_column(Text)
    # NOT NULL: every write path must say which request it is. A duplicate
    # request is then caught by the database, not by application logic.
    idempotency_key: Mapped[str] = mapped_column(Text, unique=True)
    external_ref: Mapped[str | None] = mapped_column(Text)  # the calendar system's id
    created_at: Mapped[datetime] = _created_at()
    # onupdate fires for updates made through the ORM only; a raw SQL UPDATE
    # leaves it unchanged. Nothing updates appointments yet -- a trigger would
    # be the fix once something does.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# Added after the class body because it needs real Column objects to build the
# range expression. Reads: "no two rows may have an equal provider_id AND
# overlapping [start_at, end_at) ranges -- counting only booked rows".
# tstzrange's default bounds are half-open, so 10:00-11:00 and 11:00-12:00 do
# not overlap. `provider_id WITH =` inside a GiST index needs btree_gist.
_appt = Appointment.__table__
_appt.append_constraint(
    ExcludeConstraint(
        (_appt.c.provider_id, "="),
        (func.tstzrange(_appt.c.start_at, _appt.c.end_at), "&&"),
        name="no_overlap",
        using="gist",
        where=text("status = 'booked'"),
    )
)


# ---------------------------------------------------------------------------
# Conversations -- the transcript, for people
# ---------------------------------------------------------------------------
# The LangGraph checkpoint also holds the conversation, as serialised graph
# state: that copy is the model's memory. These rows are the copy people read
# -- the chat history after a page reload, the /ops view, the eval runner.
class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (CheckConstraint(_one_of("channel", CHANNELS), name="channel"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    # Also the checkpoint key: the same id finds the model's memory and the
    # human-readable transcript. Minted by the server, never by a client.
    thread_id: Mapped[uuid.UUID] = mapped_column(Uuid, unique=True)
    channel: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()
    last_message_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(_one_of("role", MESSAGE_ROLES), name="role"),
        # Every read is "this conversation's messages, in order".
        Index("ix_messages_conversation_id_id", "conversation_id", "id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    # Per-turn diagnostics (request id; later model and token usage). Their
    # shape changes as the system grows -- one of the few places where
    # schemaless is the right call.
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = _created_at()


# ---------------------------------------------------------------------------
# What the assistant did, and what it could not answer
# ---------------------------------------------------------------------------
class ToolCall(Base):
    """One read-only lookup an agent made, as part of one turn.

    Written by the same transaction as the turn it belongs to, so a reply and
    the evidence behind it are never half saved. Only completed calls get a
    row: a tool that fails ends the turn, and nothing is written at all. When
    Phase 8 makes a failed call survivable, failure gets its columns.
    """

    __tablename__ = "tool_calls"
    __table_args__ = (Index("ix_tool_calls_message_id", "message_id"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    # The assistant message this call was made for. CASCADE, because a tool
    # call outliving the reply it produced is not evidence of anything.
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(Text)
    # What the model passed. Schemaless on purpose: every tool has different
    # arguments, and this is a trace, not a queryable dimension.
    args: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    summary: Mapped[str] = mapped_column(Text)  # one line: what came back
    # Integer, not SmallInteger: a search embeds the query through the
    # provider, and 30s of timeout plus the SDK's retries is past 32767ms.
    # Overflowing here would fail the insert of a turn that had already been
    # answered -- an error event, with nothing kept.
    latency_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = _created_at()


class KbGap(Base):
    """A question the clinic's records could not answer.

    The byproduct that pays for the abstention. Each row names a document the
    clinic should write, with how close retrieval got to it -- which is the
    difference between "we are missing a page on X" and "our page on X is
    worded nothing like the way patients ask".
    """

    __tablename__ = "kb_gaps"
    __table_args__ = (CheckConstraint(_one_of("reason", KB_GAP_REASONS), name="reason"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    question: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    # NULL when the turn never got as far as searching the documents.
    best_score: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)
    # Scores only mean something next to the model that produced them.
    embedding_model: Mapped[str] = mapped_column(Text)
    nearest_heading: Mapped[str | None] = mapped_column(Text)
    nearest_source: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()


# ---------------------------------------------------------------------------
# Knowledge base -- the narrative half of what the clinic knows
# ---------------------------------------------------------------------------
# The split is deliberate: exact facts (which plans are accepted, what a crown
# costs, when a provider works) live in the tables above and are read by exact
# queries. These two tables hold the prose -- policies, aftercare, what to
# expect -- which is what similarity search is actually good at.
class KbDocument(Base):
    """One markdown file under knowledge_base/, as last ingested."""

    __tablename__ = "kb_documents"
    __table_args__ = (CheckConstraint(_one_of("category", KB_CATEGORIES), name="category"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    # Path relative to the corpus directory ("aftercare.md") -- the natural key
    # ingestion upserts on, and stable across machines.
    source_path: Mapped[str] = mapped_column(Text, unique=True)
    title: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(Text)
    # Shown with every citation: a policy answer without a date is a rumour.
    effective_date: Mapped[date] = mapped_column(Date)
    # The three values ingestion compares to decide whether this document has
    # to be chunked and embedded again. The file's own hash is not enough: new
    # chunking rules or a new embedding model make the stored vectors stale
    # while the file itself is untouched.
    content_hash: Mapped[str] = mapped_column(Text)
    chunker_version: Mapped[int] = mapped_column(SmallInteger)
    embedding_model: Mapped[str] = mapped_column(Text)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class KbChunk(Base):
    """A retrievable passage: a section of a document, with its vector."""

    __tablename__ = "kb_chunks"
    __table_args__ = (
        CheckConstraint("approx_tokens > 0", name="positive_tokens"),
        # Also the index behind "this document's chunks, in order", which is
        # every read that is not a similarity search.
        UniqueConstraint("document_id", "ordinal"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    # CASCADE: re-ingesting a changed document deletes its chunks and writes
    # new ones, in one transaction. Chunk ids are therefore not stable across
    # ingests -- citations quote title + heading_path, never an id.
    document_id: Mapped[int] = mapped_column(ForeignKey("kb_documents.id", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(SmallInteger)
    # "Aftercare > After an Extraction > First 24 Hours" -- the citation a
    # patient can actually check, built by the chunker from the headings.
    heading_path: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    approx_tokens: Mapped[int] = mapped_column(SmallInteger)
    # No index, on purpose. At this corpus size an exact scan is microseconds,
    # and an approximate index (HNSW/IVFFlat) would trade recall for a speed-up
    # nobody can measure. README states the row count at which that changes.
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
