.PHONY: setup manifest verify-raw bad-counts frame stability-frame model stability report test check all

setup:
	uv sync
	git config core.hooksPath .githooks

# sha256 of every raw file; raw files themselves never leave this machine.
manifest:
	cd data/raw && find . -type f ! -name '.*' | sort | xargs shasum -a 256 > ../manifest.sha256

# Re-hash local raw files; fails if they drifted from the committed manifest.
verify-raw: manifest
	git diff --exit-code data/manifest.sha256

bad-counts:
	PYTHONPATH=src uv run python -m crs.target data/raw > docs/bad_counts_raw.md

# Needs data/raw. frame -> data/interim/model_frame.parquet; model -> outputs/metrics.json + figure.
frame:
	PYTHONPATH=src uv run python -m crs.frame

# Every labelled vintage incl. gap years -> data/interim/stability_frame.parquet (not committed).
stability-frame:
	PYTHONPATH=src uv run python -m crs.frame --all

model:
	PYTHONPATH=src uv run python -m crs.model

# Needs both frames and outputs/metrics.json -> outputs/stability.json, app/data/agg/*.csv, figures.
stability:
	PYTHONPATH=src uv run python -m crs.stability

report:
	PYTHONPATH=src uv run python -m crs.report
	PYTHONPATH=src uv run python reports/overview/stability_overview.py

test:
	uv run pytest -q

CHECKS := leakage split repro numbers redline

# Runs every check (no early stop) and fails if any failed.
check: test
	@fail=0; for c in $(CHECKS); do uv run python -m checks.check_$$c || fail=1; done; exit $$fail

all: manifest bad-counts frame stability-frame model stability report check
