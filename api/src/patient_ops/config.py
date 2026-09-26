"""Typed application configuration, loaded once from the environment.

Config is validated at import time, not at first use: a missing or malformed
setting must crash the process on boot, never at 3am on the first request that
happens to need it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from patient_ops.faults import FaultSpec, parse_fault_specs

# The single .env lives at the repository root, shared by the app, the ARQ
# worker and docker compose. Anchor to it absolutely: a relative ".env" would
# resolve against the current working directory, so `pytest` run from api/
# would silently fall back to defaults instead of failing loudly.
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",  # .env also carries POSTGRES_* vars consumed by compose
        case_sensitive=False,
    )

    # --- Application ------------------------------------------------------
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    api_port: int = 8001
    cors_origins: str = "http://localhost:3000"

    # --- Postgres: the system of record. Correctness lives here. ----------
    database_url: str = "postgresql://patient_ops:patient_ops@localhost:5433/patient_ops"
    db_pool_min_size: int = 1
    db_pool_max_size: int = 10
    db_connect_timeout_s: float = 5.0

    # --- Redis: the coordination layer. Degradable by design. -------------
    redis_url: str = "redis://localhost:6379/0"
    redis_connect_timeout_s: float = 2.0
    redis_socket_timeout_s: float = 2.0
    # After Redis fails, how long every coordination feature takes its
    # fallback straight away instead of waiting out another timeout. It is
    # also how long a recovered Redis goes unused -- a few seconds either way.
    redis_down_backoff_s: float = Field(default=5.0, ge=0, le=60)
    # R1 slot holds: how long an offered slot is kept for the patient who was
    # offered it. Long enough to read three options and answer; short enough
    # that an abandoned conversation does not hide a slot for long.
    hold_ttl_s: int = Field(default=120, ge=5, le=900)
    # R4 in-flight dedup: how long a claim lives if its owner dies mid-write,
    # and how long a duplicate request waits for the original's result.
    idempotency_inflight_ttl_s: int = Field(default=30, ge=1, le=300)
    idempotency_wait_s: float = Field(default=5.0, gt=0, le=60)
    # R3 circuit breaker around the calendar: consecutive failures that open
    # it, and how long it stays open before one probe is let through.
    breaker_failure_threshold: int = Field(default=5, ge=1, le=100)
    breaker_cooldown_s: float = Field(default=30.0, gt=0, le=600)
    # R5 rate limit on chat turns, per client: a token bucket. 20 turns in a
    # burst, then one every three seconds.
    rate_limit_enabled: bool = True
    rate_limit_capacity: int = Field(default=20, ge=1, le=10_000)
    rate_limit_refill_per_s: float = Field(default=0.33, gt=0, le=1_000)

    # --- Health -----------------------------------------------------------
    # Per-dependency probe budget. A health endpoint that can hang is worse
    # than one that fails: probes pile up behind it.
    health_probe_timeout_s: float = Field(default=1.0, gt=0, le=10)

    # --- Scheduling policy ---------------------------------------------------
    # Per-clinic rules, so configuration rather than code: a second clinic is
    # a different .env, not a code change.
    clinic_timezone: str = "America/New_York"
    slot_step_min: int = Field(default=30, gt=0, le=240)
    booking_min_lead_min: int = Field(default=120, ge=0)
    booking_max_horizon_days: int = Field(default=180, gt=0)
    # The number an emergency reply (guardrail G0) tells the patient to call.
    # 555-01xx is reserved for fictional use, like the seed data's numbers.
    clinic_phone: str = Field(default="(212) 555-0199", min_length=1)

    # --- Fault injection: dev and test only -----------------------------------
    # e.g. "book_appointment:timeout@1". Parsed at boot; refused in prod.
    fault_inject: str = ""

    # --- LLM ----------------------------------------------------------------
    # "fake" is a deterministic, offline model for tests, CI and key-less
    # demos; it is refused in prod, like FAULT_INJECT.
    llm_provider: Literal["openai", "fake"] = "openai"
    llm_model: str = "gpt-4.1-mini"
    # SecretStr: repr() prints '**********', so logging Settings cannot leak it.
    openai_api_key: SecretStr = SecretStr("")
    llm_timeout_s: float = Field(default=30.0, gt=0, le=120)
    # Transport-level retries with backoff, inside the SDK. The graph never
    # sees them -- semantic retries are graph edges, not this.
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # --- Knowledge base and retrieval ---------------------------------------
    # The corpus lives in the repository, next to the code that ingests it, so
    # a clone has the clinic's documents without a data dump.
    kb_dir: Path = REPO_ROOT / "knowledge_base"
    # Ignored when LLM_PROVIDER=fake, which brings its own offline embedder.
    # Changing this invalidates every stored vector: embeddings are only
    # comparable within one model, so `make ingest` re-embeds everything.
    embedding_model: str = "text-embedding-3-small"
    rag_top_k: int = Field(default=4, ge=1, le=20)
    # The abstention threshold, when set, overrides the value calibrated for
    # the embedding model in use (adapters/llm/embeddings.py). Empty = use the
    # calibrated one, which is what the evaluation report measured.
    rag_min_score: float | None = Field(default=None, ge=0.0, le=1.0)

    # --- Agents -------------------------------------------------------------
    # How many times the knowledge agent may call tools before the turn gives
    # up and abstains. A cost ceiling and a loop guard in one number: three is
    # enough for "search, search again in other words, answer".
    agent_max_tool_rounds: int = Field(default=3, ge=1, le=6)

    # --- Chat ---------------------------------------------------------------
    # How many past messages the model is shown per turn. The checkpoint keeps
    # the whole conversation; this only bounds what each LLM call costs.
    chat_history_max_messages: int = Field(default=20, ge=2, le=200)

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @field_validator("clinic_timezone")
    @classmethod
    def _known_timezone(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"unknown IANA timezone {v!r}") from None
        return v

    @field_validator("rag_min_score", mode="before")
    @classmethod
    def _blank_is_unset(cls, v: object) -> object:
        # `RAG_MIN_SCORE=` in .env means "not set", not "the empty string".
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("fault_inject")
    @classmethod
    def _parseable_faults(cls, v: str) -> str:
        parse_fault_specs(v)  # raises on a typo -- at boot, not mid-test
        return v.strip()

    @model_validator(mode="after")
    def _pool_bounds(self) -> Settings:
        if not 1 <= self.db_pool_min_size <= self.db_pool_max_size:
            raise ValueError("need 1 <= DB_POOL_MIN_SIZE <= DB_POOL_MAX_SIZE")
        return self

    @model_validator(mode="after")
    def _no_fault_injection_in_prod(self) -> Settings:
        # A startup error, not a line in a runbook: a production process that
        # deliberately fails bookings must be impossible to start.
        if self.app_env == "prod" and self.fault_inject:
            raise ValueError("FAULT_INJECT is refused when APP_ENV=prod")
        return self

    @model_validator(mode="after")
    def _real_llm_in_prod(self) -> Settings:
        # In dev a missing key still boots -- the chat endpoint answers 503
        # and says why. In prod it is a startup error.
        if self.app_env == "prod" and self.llm_provider == "fake":
            raise ValueError("LLM_PROVIDER=fake is refused when APP_ENV=prod")
        if self.app_env == "prod" and not self.llm_configured:
            raise ValueError("OPENAI_API_KEY is required when APP_ENV=prod")
        return self

    @property
    def llm_configured(self) -> bool:
        return self.llm_provider == "fake" or bool(self.openai_api_key.get_secret_value())

    @property
    def clinic_tz(self) -> ZoneInfo:
        return ZoneInfo(self.clinic_timezone)

    @property
    def fault_specs(self) -> tuple[FaultSpec, ...]:
        return parse_fault_specs(self.fault_inject)

    @property
    def sqlalchemy_url(self) -> str:
        """DATABASE_URL with the psycopg3 driver named for SQLAlchemy.

        .env keeps the plain libpq form (postgresql://...) that psql, psycopg
        and LangGraph all accept as-is; only SQLAlchemy needs the driver
        spelled out. create_async_engine picks psycopg's async mode from this
        same URL, and Alembic uses it synchronously.
        """
        scheme, sep, rest = self.database_url.partition("://")
        if sep and scheme in ("postgresql", "postgres"):
            return f"postgresql+psycopg://{rest}"
        return self.database_url

    @property
    def libpq_url(self) -> str:
        """DATABASE_URL as plain libpq (postgresql://...), whatever form it was given in.

        psycopg's own pool -- the checkpointer's -- rejects a SQLAlchemy
        driver suffix such as postgresql+psycopg://.
        """
        scheme, sep, rest = self.database_url.partition("://")
        if sep and scheme.split("+", 1)[0] in ("postgresql", "postgres"):
            return f"postgresql://{rest}"
        return self.database_url

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_dev(self) -> bool:
        return self.app_env == "dev"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton. Cached so the .env file is parsed exactly once."""
    return Settings()
