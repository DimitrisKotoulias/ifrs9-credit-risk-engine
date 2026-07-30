.PHONY: setup setup-pinned data-download data pipeline report readme test lint lint-full lint-fix clean

PYTHON := python

# ── Setup ─────────────────────────────────────────────────────────────────────
# `setup` resolves the ranges in pyproject.toml (which now carry upper bounds).
# `setup-pinned` reproduces the exact versions the reported results were produced with;
# previously requirements.txt was installed by no target at all (Flaws.md finding N22).
setup:
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

setup-pinned:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e ".[dev]" --no-deps
	pre-commit install
	@echo "Setup complete. Place kaggle.json at %USERPROFILE%\\.kaggle\\kaggle.json before running make data-download."

# ── Data ──────────────────────────────────────────────────────────────────────
data-download:
	$(PYTHON) -m credit_risk.data.download

data:
	$(PYTHON) -m credit_risk.data.loader

# ── Full pipeline (Phases 1–9) ────────────────────────────────────────────────
pipeline:
	$(PYTHON) -m credit_risk.pipeline

# ── Report ────────────────────────────────────────────────────────────────────
report:
	$(PYTHON) reports/render_latex.py

# ── README Key Results table (regenerated from outputs/metrics.json) ─────────
readme:
	$(PYTHON) scripts/update_readme_metrics.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest

# ── Lint ──────────────────────────────────────────────────────────────────────
# `lint` is the blocking gate CI runs: pyflakes + syntax errors across the tree.
lint:
	ruff check --select F,E9 --ignore F821 src/ tests/ reports/ scripts/

# `lint-full` is advisory — style/annotation backlog, not wired into CI.
lint-full:
	ruff check src/ tests/ reports/ scripts/
	mypy src/

lint-fix:
	ruff check --fix src/ tests/ reports/ scripts/
	ruff format src/ tests/ reports/ scripts/

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf outputs/*.parquet outputs/*.pkl outputs/metrics.json
	rm -rf reports/figures/eda/* reports/figures/validation/*

