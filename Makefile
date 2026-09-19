.PHONY: sync test lint fmt gen demo run report compare

sync:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

fmt:
	uv run ruff format src tests scripts

gen:
	uv run python scripts/make_emails.py

demo:
	uv run cascade demo

run:
	uv run cascade run --backend mock-jev

report:
	uv run cascade report $(RUN)

compare:
	uv run cascade compare $(A) $(B)
