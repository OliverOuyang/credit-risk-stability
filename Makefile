.PHONY: setup manifest bad-counts test check all

setup:
	uv sync
	git config core.hooksPath .githooks

# sha256 of every raw file; raw files themselves never leave this machine.
manifest:
	cd data/raw && find . -type f ! -name '.*' | sort | xargs shasum -a 256 > ../manifest.sha256

bad-counts:
	PYTHONPATH=src uv run python -m crs.target data/raw > docs/bad_counts_raw.md

test:
	uv run pytest -q

# 2b adds checks/check_{leakage,split,repro,numbers,redline}.py here.
check: test

all: manifest bad-counts check
