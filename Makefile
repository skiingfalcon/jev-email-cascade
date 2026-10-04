.PHONY: sync test test-live lint fmt gen demo run run-laya bench-laya report compare

sync:
	uv sync

test:
	uv run pytest -q

# Needs laya-serve on LAYA_URL (laya-host); skipped otherwise.
test-live:
	uv run pytest -q -m live

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

run-laya:
	uv run cascade run --backend laya

bench-laya:
	uv run python scripts/bench_laya.py --label $(or $(LABEL),laya)

report:
	uv run cascade report $(RUN)

compare:
	uv run cascade compare $(A) $(B)
