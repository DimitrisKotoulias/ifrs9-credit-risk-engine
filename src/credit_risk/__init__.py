"""Credit Risk Modelling & IFRS 9 ECL Engine."""

from credit_risk.utils._compat import patch_optbinning_sklearn_compat

# Applied at import time so production runs and tests share one shim (the internal review log N22).
patch_optbinning_sklearn_compat()

__version__ = "0.1.0"
