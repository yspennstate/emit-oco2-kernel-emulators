"""Fitting/scoring contract for the published retrieval-weighted stack.

The fitting loss uses raw Y2/Y3 errors.  Only base Y4 is clipped before fitting;
total-flux projection is performed later by retrieval scoring.  These helpers
make the historical finite-threshold weights and precision policy explicit.
They do not replace the historical approximate NNLS/IRLS optimizer.
"""
from __future__ import annotations
from typing import Mapping
import numpy as np


def component_weights(transmission: np.ndarray, R: float, tau_over_scale: float,
                      flux_scale: float) -> dict[str, np.ndarray]:
    """Return the historical separable quadratic's fixed entry weights.

    ``inf`` is a distinct unweighted-component baseline, not a limiting value of
    J'_tau.  Nonpositive transmission is preserved as the historical fit's zero
    floor; such entries do not satisfy the retrieval-transfer theorem.
    """
    t = np.asarray(transmission, dtype=np.float64)
    if not np.all(np.isfinite(t)):
        raise ValueError("Transmission must be finite for a finite fitting objective")
    if not (0 < R <= 1) or not np.isfinite(flux_scale) or flux_scale <= 0:
        raise ValueError("Require 0 < R <= 1 and a finite positive training flux scale")
    if np.isnan(tau_over_scale) or tau_over_scale <= 0:
        raise ValueError("Threshold must be positive; +inf selects the unweighted baseline")
    tv = np.maximum(t, 0.0)
    if np.isposinf(tau_over_scale):
        cond_at, cond_s = np.ones_like(tv), np.ones_like(tv)
    else:
        tt = np.maximum(tv, tau_over_scale * flux_scale)
        cond_at, cond_s = 1.0 / tt ** 2, (tv / tt) ** 2
    return {"Y1": cond_at, "Y2": R * R * cond_at,
            "Y3": R * R * cond_at, "Y4": R ** 4 * cond_s}


def require_scoring_precision(arrays: Mapping[str, np.ndarray], *,
                              allow_precision_altered: bool = False) -> None:
    """Reject precision-altered validation AND test dumps unless explicitly allowed."""
    bad = [name for name, a in arrays.items()
           if np.asarray(a).dtype.kind != "f" or np.asarray(a).dtype.itemsize < 8]
    if bad and not allow_precision_altered:
        raise ValueError("Precision-altered prediction arrays (explicit opt-in required): "
                         + ", ".join(sorted(bad)))


def physical_domain(transmission: np.ndarray, albedo: np.ndarray, *,
                    rho: float, R: float, S: float) -> np.ndarray:
    """Full truth-domain mask for the exact-arithmetic transfer theorem.

    This is stronger than the s<=S-only legacy violation mask.  It does not
    certify floating-point forward/inverse arithmetic or repair archived scores.
    """
    if not (0 <= rho <= R <= 1 and 0 <= S and R * S < 1):
        raise ValueError("Require 0 <= rho <= R <= 1, S >= 0 and R*S < 1")
    t, s = np.broadcast_arrays(transmission, albedo)
    return np.isfinite(t) & np.isfinite(s) & (t > 0) & (s >= 0) & (s <= S)
