"""Credit Risk Modelling & IFRS 9 ECL Engine."""

# Monkeypatch sklearn.utils.validation for optbinning compatibility with sklearn >= 1.6
try:
    import sklearn.utils.validation
    _orig_check_array = sklearn.utils.validation.check_array
    def _patched_check_array(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _orig_check_array(*args, **kwargs)
    sklearn.utils.validation.check_array = _patched_check_array

    # Also patch direct imports inside optbinning modules
    try:
        import optbinning.binning.metrics  # noqa: F401
        optbinning.binning.metrics.check_array = _patched_check_array
    except Exception:
        pass
    try:
        import optbinning.binning.binning  # noqa: F401
        optbinning.binning.binning.check_array = _patched_check_array
    except Exception:
        pass
    try:
        import optbinning.binning.binning_process  # noqa: F401
        optbinning.binning.binning_process.check_array = _patched_check_array
    except Exception:
        pass
except Exception:
    pass

__version__ = "0.1.0"


