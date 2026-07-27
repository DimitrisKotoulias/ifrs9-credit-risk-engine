"""Third-party compatibility shims, applied once at package import.

This lives in the package (not in ``tests/conftest.py``) so that production runs and the
test suite exercise exactly the same code path. It previously existed as two independent
copies, which meant a shim regression could turn the tests green while ``make pipeline``
failed, or vice versa (Flaws.md finding N22).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def patch_optbinning_sklearn_compat() -> bool:
    """Make optbinning 0.19 work with scikit-learn >= 1.6.

    sklearn renamed ``check_array(force_all_finite=...)`` to ``ensure_all_finite`` in
    1.6; optbinning 0.19 still passes the old name. This wraps ``check_array`` to accept
    either spelling, and re-points the already-imported references inside optbinning's
    own modules.

    Returns True when the patch was installed. Idempotent.
    """
    try:
        import sklearn.utils.validation as _skv  # noqa: PLC0415
    except ImportError:  # pragma: no cover - sklearn is a hard dependency
        return False

    if getattr(_skv.check_array, "_credit_risk_patched", False):
        return True

    _orig_check_array = _skv.check_array

    def _patched_check_array(*args: Any, **kwargs: Any) -> Any:
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _orig_check_array(*args, **kwargs)

    _patched_check_array._credit_risk_patched = True  # type: ignore[attr-defined]
    _skv.check_array = _patched_check_array

    # optbinning does `from sklearn.utils import check_array` at module import time, so
    # patching the origin is not enough — the already-bound names must be re-pointed too.
    for mod_name in (
        "optbinning.binning.metrics",
        "optbinning.binning.binning",
        "optbinning.binning.binning_process",
    ):
        try:
            import importlib  # noqa: PLC0415

            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, "check_array"):
            mod.check_array = _patched_check_array

    return True
