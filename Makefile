.PHONY: setup data-download data pipeline report readme test lint clean

PYTHON := python

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	$(PYTHON) -m pip install -e ".[dev]"
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
lint:
	ruff check src/ tests/
	black --check src/ tests/
	mypy src/

lint-fix:
	ruff check --fix src/ tests/
	black src/ tests/

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf outputs/*.parquet outputs/*.pkl outputs/metrics.json
	rm -rf reports/figures/eda/* reports/figures/validation/*

