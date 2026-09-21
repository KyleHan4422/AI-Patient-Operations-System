"""Domain core: the seven tables of the system of record.

Only what Phase 1 uses. Conversations, the knowledge base, tool-call traces,
notifications and escalations each arrive in their own migration, in the phase
that first needs them.

Hand-reviewed from an autogenerate draft. What autogenerate did not do:
  - install btree_gist, without which the no_overlap constraint cannot exist;
  - state no_overlap as the SQL a reviewer should read, rather than as a
    constructor call.

Extensions are created here, not in db/init/: those scripts run only when the
Postgres volume is first initialised, so they would never reach an existing dev
volume, CI's fresh database, or the test database. Migrations run everywhere.

Revision ID: 0001
Revises:
Create Date: 2026-09-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
    )


def upgrade() -> None:
    # `provider_id WITH =` inside a GiST index needs btree_gist: plain GiST
    # has no equality operator class for integers.
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")

    # --- Reference data ----------------------------------------------------
    op.create_table(
        "providers",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("specialty", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "specialty IN ('general', 'hygienist', 'orthodontist')",
            name=op.f("ck_providers_specialty"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_providers")),
        sa.UniqueConstraint("name", name=op.f("uq_providers_name")),
    )
    op.create_table(
        "provider_schedules",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("provider_id", sa.BigInteger(), nullable=False),
        sa.Column("weekday", sa.SmallInteger(), nullable=False),
        sa.Column("start_time", sa.Time(), nullable=False),
        sa.Column("end_time", sa.Time(), nullable=False),
        sa.CheckConstraint(
            "weekday BETWEEN 0 AND 6", name=op.f("ck_provider_schedules_weekday_range")
        ),
        sa.CheckConstraint(
            "end_time > start_time", name=op.f("ck_provider_schedules_end_after_start")
        ),
        sa.ForeignKeyConstraint(
            ["provider_id"],
            ["providers.id"],
            name=op.f("fk_provider_schedules_provider_id_providers"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_provider_schedules")),
        sa.UniqueConstraint(
            "provider_id",
            "weekday",
            "start_time",
            name=op.f("uq_provider_schedules_provider_id_weekday_start_time"),
        ),
    )
    op.create_table(
        "clinic_closures",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("closed_on", sa.Date(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_clinic_closures")),
        sa.UniqueConstraint("closed_on", name=op.f("uq_clinic_closures_closed_on")),
    )
    op.create_table(
        "procedures",
        sa.Column("code", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("duration_min", sa.SmallInteger(), nullable=False),
        sa.Column("price_min", sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column("price_max", sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column("specialty", sa.Text(), nullable=False),
        sa.CheckConstraint("duration_min > 0", name=op.f("ck_procedures_positive_duration")),
        sa.CheckConstraint("price_max >= price_min", name=op.f("ck_procedures_price_range")),
        sa.CheckConstraint(
            "specialty IN ('general', 'hygienist', 'orthodontist')",
            name=op.f("ck_procedures_specialty"),
        ),
        sa.PrimaryKeyConstraint("code", name=op.f("pk_procedures")),
    )
    op.create_table(
        "insurance_plans",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("plan_name", sa.Text(), nullable=False),
        sa.Column("accepted", sa.Boolean(), nullable=False),
        sa.Column("in_network", sa.Boolean(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "accepted OR NOT in_network",
            name=op.f("ck_insurance_plans_in_network_implies_accepted"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_insurance_plans")),
    )
    op.create_index(
        "uq_insurance_plans_lower_plan_name",
        "insurance_plans",
        [sa.literal_column("lower(plan_name)")],
        unique=True,
    )

    # --- People ------------------------------------------------------------
    op.create_table(
        "patients",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("full_name", sa.Text(), nullable=False),
        sa.Column("phone", sa.Text(), nullable=False),
        sa.Column("dob", sa.Date(), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_patients")),
        sa.UniqueConstraint("phone", name=op.f("uq_patients_phone")),
    )

    # --- Appointments ------------------------------------------------------
    op.create_table(
        "appointments",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("patient_id", sa.BigInteger(), nullable=False),
        sa.Column("provider_id", sa.BigInteger(), nullable=False),
        sa.Column("procedure_code", sa.Text(), nullable=False),
        sa.Column("start_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.Text(), server_default="booked", nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column("external_ref", sa.Text(), nullable=True),
        _created_at(),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("end_at > start_at", name=op.f("ck_appointments_end_after_start")),
        sa.CheckConstraint(
            "status IN ('booked', 'cancelled', 'completed')", name=op.f("ck_appointments_status")
        ),
        sa.CheckConstraint(
            "source IN ('web', 'voice', 'staff')", name=op.f("ck_appointments_source")
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"], ["patients.id"], name=op.f("fk_appointments_patient_id_patients")
        ),
        sa.ForeignKeyConstraint(
            ["provider_id"], ["providers.id"], name=op.f("fk_appointments_provider_id_providers")
        ),
        sa.ForeignKeyConstraint(
            ["procedure_code"],
            ["procedures.code"],
            name=op.f("fk_appointments_procedure_code_procedures"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_appointments")),
        sa.UniqueConstraint("idempotency_key", name=op.f("uq_appointments_idempotency_key")),
    )

    # The database-level guarantee against double booking. Application code
    # that checks "is this slot free?" and then inserts has a race window
    # between the two steps; this constraint has none.
    #   - provider_id WITH =          same provider ...
    #   - tstzrange(...) WITH &&      ... and overlapping time ranges ...
    #   - WHERE status = 'booked'     ... counting only live bookings.
    # tstzrange defaults to half-open [start, end): back-to-back appointments
    # (10:00-11:00, 11:00-12:00) do not overlap.
    op.execute(
        """
        ALTER TABLE appointments ADD CONSTRAINT no_overlap
          EXCLUDE USING gist (provider_id WITH =, tstzrange(start_at, end_at) WITH &&)
          WHERE (status = 'booked')
        """
    )


def downgrade() -> None:
    op.drop_table("appointments")  # takes no_overlap with it
    op.drop_table("patients")
    op.drop_index("uq_insurance_plans_lower_plan_name", table_name="insurance_plans")
    op.drop_table("insurance_plans")
    op.drop_table("procedures")
    op.drop_table("clinic_closures")
    op.drop_table("provider_schedules")
    op.drop_table("providers")
    op.execute("DROP EXTENSION IF EXISTS btree_gist")
