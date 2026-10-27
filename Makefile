.PHONY: check test lint format benchmark api worker db-up db-down

check: lint test

test:
	pytest

lint:
	ruff check backend

format:
	ruff format backend
	ruff check --fix backend

benchmark:
	python -m backend.app.benchmark.cli

api:
	uvicorn backend.app.main:app --reload

worker:
	python scripts/gatekeeper_worker.py

db-up:
	docker compose up -d postgres

db-down:
	docker compose down
