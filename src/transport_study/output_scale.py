"""Convert counts and log values for all methods and evaluators."""
from __future__ import annotations
import numpy as np


def library_sizes(counts) -> np.ndarray:
    x = np.asarray(counts, dtype=np.float64)
    if x.ndim != 2 or np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError("counts must be a finite nonnegative cell-by-gene matrix")
    return x.sum(axis=1)


def normalize_log1p(counts, target_sum: float = 10_000.0) -> np.ndarray:
    """Normalize each cell to the target sum. Then, apply log1p."""
    x = np.asarray(counts, dtype=np.float64)
    totals = library_sizes(x)[:, None]
    return np.log1p(np.divide(x * target_sum, totals, out=np.zeros_like(x), where=totals > 0))


def log1p_to_expected_counts(log_expression, query_library_sizes) -> np.ndarray:
    """Invert normalized log expression. Apply the query library-size rule."""
    z = np.asarray(log_expression, dtype=np.float64)
    libs = np.asarray(query_library_sizes, dtype=np.float64).reshape(-1, 1)
    if z.ndim != 2 or len(z) != len(libs) or np.any(~np.isfinite(z)) or np.any(libs < 0):
        raise ValueError("invalid log-expression or library sizes")
    abundance = np.maximum(np.expm1(z), 0.0)
    totals = abundance.sum(axis=1, keepdims=True)
    return np.divide(abundance * libs, totals, out=np.zeros_like(abundance), where=totals > 0)


def enforce_library_sizes(expected_counts, query_library_sizes) -> np.ndarray:
    x = np.maximum(np.asarray(expected_counts, dtype=np.float64), 0.0)
    libs = np.asarray(query_library_sizes, dtype=np.float64).reshape(-1, 1)
    totals = x.sum(axis=1, keepdims=True)
    return np.divide(x * libs, totals, out=np.zeros_like(x), where=totals > 0)


def synthetic_control_correct(q0_counts, perturbed_expected, noop_expected, *, return_diagnostics=False):
    """Add the predicted change to Q0. Clip the result and apply each Q0 library size."""
    q0 = np.asarray(q0_counts, dtype=np.float64)
    pert = np.asarray(perturbed_expected, dtype=np.float64)
    noop = np.asarray(noop_expected, dtype=np.float64)
    if q0.shape != pert.shape or q0.shape != noop.shape:
        raise ValueError("synthetic-control matrices must have identical query-cell shape")
    anchored = q0 + (pert - noop)
    corrected = np.maximum(anchored, 0.0)
    result = enforce_library_sizes(corrected, library_sizes(q0))
    if not return_diagnostics:return result
    clipped=np.maximum(-anchored,0).sum()
    return result,{"clipped_mass":float(clipped),"clipped_fraction":float(clipped/max(np.abs(anchored).sum(),1e-12))}


def prepare_for_evaluation(expected_counts) -> np.ndarray:
    """Apply the common evaluation transform exactly one time."""
    return normalize_log1p(expected_counts, target_sum=10_000.0)
