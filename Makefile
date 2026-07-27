.PHONY: setup setup-pinned data-download data pipeline report readme test lint clean

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

