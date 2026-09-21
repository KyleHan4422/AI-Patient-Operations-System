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
        dev worker web health test lint fmt check-invariants check

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

test: ## Run the Python test suite
	$(UV) pytest

lint: ## Lint Python and TypeScript
	$(UV) ruff check src tests scripts
	cd $(WEB) && npx tsc --noEmit

fmt: ## Auto-format and auto-fix Python
	$(UV) ruff check --fix src tests scripts
	$(UV) ruff format src tests scripts

check-invariants: ## Assert agents/ contains no write operations
	$(UV) python scripts/check_invariants.py

check: lint check-invariants test ## Everything CI runs
