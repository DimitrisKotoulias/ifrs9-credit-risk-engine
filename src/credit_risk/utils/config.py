"""Pydantic-validated configuration loader from config/config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    source: str = "real"
    synthetic_n_loans: int = 50000
    kaggle_dataset: str = "wordsforthewise/lending-club"
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    accepted_file: str = "accepted_2007_to_2018Q4.csv.gz"
    rejected_file: str = "rejected_2007_to_2018Q4.csv.gz"

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        if v not in ("real", "synthetic"):
            raise ValueError(f"data.source must be 'real' or 'synthetic', got '{v}'")
        return v


class TargetConfig(BaseModel):
    bad_statuses: list[str]
    good_statuses: list[str]


class LeakageConfig(BaseModel):
    deny_list: list[str]
    allow_overrides: list[str] = Field(default_factory=list)


class SplitConfig(BaseModel):
    train_cutoff: str
    oot_cutoff: str
    holdout_frac: float = 0.20


class ScorecardConfig(BaseModel):
    pdo: float = 20.0
    base_score: float = 600.0
    base_odds: float = 50.0
    min_iv: float = 0.02
    max_iv: float = 0.50
    max_vif: float = 5.0


class LGDConfig(BaseModel):
    downturn_percentile: float = 90.0
    min_lgd: float = 0.0
    max_lgd: float = 1.0


class BaselConfig(BaseModel):
    pd_floor: float = 0.0003
    capital_ratio: float = 0.08
    maturity_adjustment: bool = False
    stress_rho: float = 0.15    # ASRF retail asset correlation for stress test
    stress_z: float = -2.0      # Vasicek systematic factor shock (severe recession)


class MacroScenario(BaseModel):
    weight: float
    macro_shock: float


class SICRConfigYaml(BaseModel):
    pd_multiplier: float = 2.5
    abs_threshold: float = 0.20
    dpd_backstop: int = 30


class IFRS9Config(BaseModel):
    sicr_pd_multiplier: float = 2.5
    sicr_abs_threshold: float = 0.20   # absolute lifetime-PD level that triggers SICR
    sicr_dpd_backstop: int = 30
    stage3_dpd: int = 90
    scenarios: dict[str, MacroScenario]
    macro_gamma: float = 0.8
    macro_unrate_lag: int = 2          # quarters to lag macro vs origination cohort
    macro_enforce_sign_priors: bool = True  # impose economic signs on scenario projection

    @property
    def sicr(self) -> SICRConfigYaml:
        """Provide nested SICR access for the pipeline."""
        return SICRConfigYaml(
            pd_multiplier=self.sicr_pd_multiplier,
            abs_threshold=self.sicr_abs_threshold,
            dpd_backstop=self.sicr_dpd_backstop,
        )


class EconCapConfig(BaseModel):
    """Monte Carlo economic-capital settings (ASRF loss distribution)."""
    n_simulations: int = 50000
    rho: float = 0.15          # asset correlation to the systematic factor
    es_alpha: float = 0.999    # VaR/ES confidence level
    seed: int = 42
    n_buckets: int = 50        # PD-ranked buckets for tractable simulation


class MacroTsConfig(BaseModel):
    """Macro time-series diagnostics (Granger / AIC lag / Johansen-VECM)."""
    max_lag: int = 4


class BusinessConfig(BaseModel):
    interest_revenue_rate: float = 0.12
    cost_per_approval: float = 50.0
    # Cut-off economics (used by the RAROC-hurdle decision rule, Phase 9)
    fee_income_rate: float = 0.01        # upfront fee income as fraction of EAD
    funding_cost_rate: float = 0.04      # cost of funds as fraction of EAD
    operating_cost_rate: float = 0.015   # servicing/opex as fraction of EAD
    cost_of_capital: float = 0.12        # required return on economic capital
    raroc_hurdle: float = 0.15           # cost-of-capital hurdle referenced in RAROC framing
    max_bad_rate: float = 0.15           # board risk-appetite ceiling on approved bad rate


class PathsConfig(BaseModel):
    outputs: str = "outputs"
    reports: str = "reports"
    figures: str = "reports/figures"
    models: str = "outputs"


class Config(BaseModel):
    random_seed: int = 42
    data: DataConfig
    target: TargetConfig
    leakage: LeakageConfig
    split: SplitConfig
    scorecard: ScorecardConfig
    lgd: LGDConfig
    basel: BaselConfig
    ifrs9: IFRS9Config
    business: BusinessConfig
    paths: PathsConfig
    econ_cap: EconCapConfig = Field(default_factory=EconCapConfig)
    macro_ts: MacroTsConfig = Field(default_factory=MacroTsConfig)


_DEFAULT_CONFIG_PATH = Path(__file__).parents[3] / "config" / "config.yaml"
_config_cache: Config | None = None


def load_config(path: Path | None = None) -> Config:
    """Load and validate YAML config. Results are cached after first call."""
    global _config_cache  # noqa: PLW0603
    if _config_cache is not None and path is None:
        return _config_cache

    config_path = path or _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Run from the project root or pass an explicit path."
        )

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = Config(**raw)

    if path is None:
        _config_cache = cfg
    return cfg
