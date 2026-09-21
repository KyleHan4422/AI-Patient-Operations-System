"""Typed application configuration, loaded once from the environment.

Config is validated at import time, not at first use: a missing or malformed
setting must crash the process on boot, never at 3am on the first request that
happens to need it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    # --- Health -----------------------------------------------------------
    # Per-dependency probe budget. A health endpoint that can hang is worse
    # than one that fails: probes pile up behind it.
    health_probe_timeout_s: float = Field(default=1.0, gt=0, le=10)

    # --- LLM: unused until Phase 2 ----------------------------------------
    openai_api_key: str = ""

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

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
