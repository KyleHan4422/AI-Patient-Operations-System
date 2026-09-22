# Executable documentation: every command needed to run, test and inspect the
# system lives here. Onboarding docs go stale; a Makefile that CI also runs
# cannot.
#
#   make help        list every target

SHELL := /bin/bash
API   := api
WEB   := web
UV    := cd $(API) && uv run

# Read API_PORT from .env if present, else fall back.
API_PORT ?= $(shell grep -E '^API_PORT=' .env 2>/dev/null | cut -d= -f2)
API_PORT := $(if $(API_PORT),$(API_PORT),8001)
API_URL  := http://localhost:$(API_PORT)

.DEFAULT_GOAL := help

.PHONY: help env install up down restart logs ps wait nuke \
        migrate migration db-check db-reset seed slots psql ingest search calibrate \
        dev worker web health chat test test-unit lint fmt check-invariants check

## ---------------------------------------------------------------------------
## Setup
## ---------------------------------------------------------------------------
help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

env: ## Create .env from .env.example if missing
	@test -f .env || (cp .env.example .env && echo "created .env from .env.example")
	@test -f $(WEB)/.env.local || (cp $(WEB)/.env.local.example $(WEB)/.env.local \
	  && echo "created web/.env.local")

install: env ## Install Python and Node dependencies
	cd $(API) && uv sync --extra dev
	cd $(WEB) && npm install

## ---------------------------------------------------------------------------
## Infrastructure (Postgres + Redis)
## ---------------------------------------------------------------------------
up: env ## Start Postgres and Redis, wait until healthy
	docker compose up -d
	@$(MAKE) --no-print-directory wait

down: ## Stop containers, keep data
	docker compose down

restart: down up ## Restart containers

logs: ## Tail container logs
	docker compose logs -f

ps: ## Show container status
	docker compose ps

wait: ## Block until both containers report healthy
	@echo -n "waiting for containers to become healthy"
	@for i in $$(seq 1 60); do \
	  pg=$$(docker inspect --format '{{.State.Health.Status}}' patient-ops-postgres 2>/dev/null); \
	  rd=$$(docker inspect --format '{{.State.Health.Status}}' patient-ops-redis 2>/dev/null); \
	  if [ "$$pg" = "healthy" ] && [ "$$rd" = "healthy" ]; then echo " ok"; exit 0; fi; \
	  echo -n "."; sleep 1; \
	done; echo " TIMEOUT"; docker compose ps; exit 1

nuke: ## Destroy containers AND the Postgres volume (re-runs db/init scripts)
	docker compose down -v
	@echo "volume removed -- db/init/*.sql will re-run on the next 'make up'"

## ---------------------------------------------------------------------------
## Database: schema, demo data, inspection
## ---------------------------------------------------------------------------
migrate: ## Apply every migration, then create/upgrade LangGraph's checkpoint tables
	$(UV) alembic upgrade head
	$(UV) python scripts/setup_checkpointer.py

migration: ## Draft a migration from model changes: make migration m="add notifications"
	@test -n "$(m)" || (echo 'usage: make migration m="describe the change"'; exit 1)
	$(UV) alembic revision --autogenerate -m "$(m)"

db-check: ## Fail if the ORM models and the migrations disagree
	$(UV) alembic check

db-reset: ## Rebuild the schema from zero, reseed and re-ingest (DESTROYS dev data)
	$(UV) alembic downgrade base
	@# The checkpoints go too: conversations the transcript no longer has must
	@# not live on as model memory. setup_checkpointer recreates the tables.
	docker exec patient-ops-postgres sh -c 'psql -q -U "$$POSTGRES_USER" -d "$$POSTGRES_DB" \
	  -c "DROP TABLE IF EXISTS checkpoints, checkpoint_blobs, checkpoint_writes, checkpoint_migrations"'
	@$(MAKE) --no-print-directory migrate
	@$(MAKE) --no-print-directory seed
	@# A reset database is one you can demo from, so the knowledge base comes back too.
	@$(MAKE) --no-print-directory ingest

seed: ## Upsert the demo clinic (safe to re-run)
	$(UV) python scripts/seed.py

ingest: ## Load knowledge_base/*.md into Postgres (idempotent: unchanged files cost nothing)
	$(UV) python scripts/ingest.py

# The query travels as an environment variable, never spliced into the shell,
# so quotes and apostrophes ("what's the policy?") arrive intact.
search: export KB_QUERY = $(q)
search: ## Search the knowledge base by eye: make search q="how do I cancel" [k=4]
	@test -n "$$KB_QUERY" || (echo 'usage: make search q="your question" [k=4]'; exit 1)
	@cd $(API) && uv run python scripts/search.py "$$KB_QUERY" $(if $(k),--k $(k))

calibrate: ## Measure the abstention threshold: make calibrate [PROVIDER=fake]
	$(UV) python scripts/calibrate_threshold.py $(if $(PROVIDER),--provider $(PROVIDER))

slots: ## Show bookable slots: make slots p=CROWN [days=7] [from=2026-11-23]
	$(UV) python scripts/show_slots.py $(or $(p),CLEANING) --days $(or $(days),7) \
	  $(if $(from),--from $(from))

psql: ## Open psql inside the Postgres container
	docker exec -it patient-ops-postgres sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'

## ---------------------------------------------------------------------------
## Processes (run each in its own terminal)
## ---------------------------------------------------------------------------
dev: ## Run the FastAPI app with hot reload
	cd $(API) && uv run uvicorn patient_ops.main:app \
	  --reload --reload-dir src --host 0.0.0.0 --port $(API_PORT)

worker: ## Run the ARQ worker (non-critical path)
	cd $(API) && uv run arq patient_ops.worker.WorkerSettings

web: ## Run the Next.js dev server
	cd $(WEB) && npm run dev

## ---------------------------------------------------------------------------
## Inspection and checks
## ---------------------------------------------------------------------------
health: ## Print /health with its HTTP status code
	@code=$$(curl -s -o /tmp/patient-ops-health.json -w '%{http_code}' $(API_URL)/health) \
	  && echo "HTTP $$code" && cat /tmp/patient-ops-health.json | $(API)/.venv/bin/python -m json.tool

# The message travels as an environment variable, never spliced into the shell
# command, so quotes and apostrophes ("What's my name?") arrive intact.
chat: export CHAT_MESSAGE = $(m)
chat: export CHAT_THREAD = $(t)
chat: ## Send one chat turn, print the raw SSE stream: make chat m="hello" [t=<thread_id>]
	@test -n "$$CHAT_MESSAGE" || (echo 'usage: make chat m="your message" [t=<thread_id>]'; exit 1)
	@python3 -c 'import json, os; t = os.environ["CHAT_THREAD"]; \
	  print(json.dumps({"message": os.environ["CHAT_MESSAGE"], **({"thread_id": t} if t else {})}))' \
	  | curl -sN -X POST $(API_URL)/api/chat/turn -H 'Content-Type: application/json' --data-binary @-

test: ## Run the whole Python test suite (needs `make up`)
	$(UV) pytest

test-unit: ## Run only the tests that need no database (works with containers stopped)
	$(UV) pytest -m "not db"

lint: ## Lint Python and TypeScript
	$(UV) ruff check src tests scripts alembic
	cd $(WEB) && npx tsc --noEmit

fmt: ## Auto-format and auto-fix Python
	$(UV) ruff check --fix src tests scripts alembic
	$(UV) ruff format src tests scripts alembic

check-invariants: ## Assert agents/ contains no write operations
	$(UV) python scripts/check_invariants.py

check: lint check-invariants db-check test ## Everything CI runs
