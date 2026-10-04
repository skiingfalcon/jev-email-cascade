.PHONY: sync test test-live lint fmt gen demo run run-laya run-rune bench-laya report compare

sync:
	uv sync

test:
	uv run pytest -q

# Needs the local-decision-model servers on LAYA_URL / RUNE_URL; each is skipped if down.
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

run-rune:
	uv run cascade run --backend rune

bench-laya:
	uv run python scripts/bench_laya.py --backend $(or $(BACKEND),laya) --label $(or $(LABEL),$(or $(BACKEND),laya))

report:
	uv run cascade report $(RUN)

compare:
	uv run cascade compare $(A) $(B)
