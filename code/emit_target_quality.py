"""Training-only target-quality sensitivity helpers (no imputation or refitting).

A row is admissible when all four spectra are finite and every band has positive
summed flux and albedo in [0, 1). Small *positive* flux is not removed at training
by a numerical threshold. Row removal preserves complete spectra for ordinary
PCA. It changes the training distribution, so a matched-size unfiltered arm is
provided. These policies do not certify the provenance of the original targets.
"""
from __future__ import annotations

import hashlib
from typing import Mapping
import numpy as np

COMPONENTS = ("Y1", "Y2", "Y3", "Y4")
POLICIES = ("raw", "admissible", "matched-unfiltered")


def target_arrays(ys: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    arrays = [np.asarray(ys[c], dtype=np.float64) for c in COMPONENTS]
    if arrays[0].ndim != 2 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Targets must be equally shaped [state, band] matrices")
    if not all(arrays[0].shape):
        raise ValueError("Target matrices must be nonempty")
    return arrays


def admissible_entries(ys: Mapping[str, np.ndarray]) -> np.ndarray:
    arrays = target_arrays(ys)
    with np.errstate(over="ignore", invalid="ignore"):
        flux = arrays[1] + arrays[2]
    return (np.logical_and.reduce([np.isfinite(a) for a in arrays])
            & np.isfinite(flux) & (flux > 0)
            & (arrays[3] >= 0) & (arrays[3] < 1))


def audit_targets(ys: Mapping[str, np.ndarray]) -> dict:
    arrays = target_arrays(ys)
    with np.errstate(over="ignore", invalid="ignore"):
        flux = arrays[1] + arrays[2]
    mask = admissible_entries(ys)
    s = arrays[3]
    return {
        "states": int(mask.shape[0]), "bands": int(mask.shape[1]),
        "entries": int(mask.size), "admissible_entries": int(mask.sum()),
        "admissible_rows": int(mask.all(axis=1).sum()),
        "nonfinite_component_entries": int((~np.logical_and.reduce(
            [np.isfinite(a) for a in arrays])).sum()),
        "nonpositive_flux_entries": int((np.isfinite(flux) & (flux <= 0)).sum()),
        "negative_albedo_entries": int((np.isfinite(s) & (s < 0)).sum()),
        "albedo_at_least_one_entries": int((np.isfinite(s) & (s >= 1)).sum()),
        "albedo_exactly_two_entries": int((s == 2).sum()),
        "definition": "finite components; Y2+Y3>0; 0<=Y4<1 at every retained band",
        "note": "Overlapping entry counts; exact value 2 alone does not prove a fill-code convention",
    }


def indices_digest(indices: np.ndarray) -> str:
    """Canonical hash of ordered int64 indices, independent of platform byte order."""
    a = np.asarray(indices)
    if a.ndim != 1 or a.dtype.kind not in "iu":
        raise ValueError("Indices must be a one-dimensional integer array")
    return hashlib.sha256(np.asarray(a, dtype="<i8").tobytes()).hexdigest()


def seeded_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return training, validation, test indices using the original campaign rule."""
    if n < 20:
        raise ValueError("At least 20 states are required for nonempty split blocks")
    perm = np.random.RandomState(seed).permutation(n)
    n_test = int(round(0.1 * n))
    test, rest = perm[:n_test], perm[n_test:]
    vp = np.random.RandomState(seed + 10000).permutation(len(rest))
    n_val = int(round(0.1 * len(rest)))
    return rest[vp[n_val:]], rest[vp[:n_val]], test


def select_training_rows(ys: Mapping[str, np.ndarray], candidate_indices: np.ndarray,
                         policy: str, seed: int) -> tuple[np.ndarray, dict]:
    """Inspect only candidate training rows; never read validation/test target values.

    The matched-unfiltered arm draws the same number of rows as the admissible
    arm from the original training block, without filtering their targets.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown training policy: {policy}")
    indices_digest(candidate_indices)  # validate before conversion
    idx = np.asarray(candidate_indices, dtype=np.int64)
    n_all = len(np.asarray(ys["Y1"]))
    if not len(idx) or len(np.unique(idx)) != len(idx) or np.any(idx < 0) or np.any(idx >= n_all):
        raise ValueError("Candidate training indices must be nonempty, unique and in range")
    training = {c: np.asarray(ys[c])[idx] for c in COMPONENTS}
    arrays = target_arrays(training)
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Nonfinite training targets require an explicit missing-data policy; no silent imputation")
    keep = admissible_entries(training).all(axis=1)
    clean_count = int(keep.sum())
    if policy == "admissible":
        chosen = idx[keep]
    elif policy == "matched-unfiltered":
        positions = np.random.RandomState(seed + 20000).permutation(len(idx))[:clean_count]
        chosen = idx[np.sort(positions)]
    else:
        chosen = idx.copy()
    if len(chosen) < 2:
        raise ValueError("Fewer than two training rows survive the requested policy")
    flux = arrays[1] + arrays[2]
    positive = flux[np.isfinite(flux) & (flux > 0)]
    if not positive.size:
        raise ValueError("The original training block contains no finite positive flux")
    metadata = {
        "policy": policy, "candidate_rows": len(idx), "retained_rows": len(chosen),
        "admissible_candidate_rows": clean_count,
        "candidate_indices_sha256": indices_digest(idx),
        "training_indices_sha256": indices_digest(chosen),
        "reference_flux_scale": float(np.median(positive)),
        "reference_flux_scale_scope": "original candidate training block, common to all arms",
        "candidate_audit": audit_targets(training),
        "training_audit": audit_targets({c: np.asarray(ys[c])[chosen] for c in COMPONENTS}),
        "selection_stage": "before input/output standardization, PCA and fitting",
        "target_values_modified": False,
    }
    return chosen, metadata
