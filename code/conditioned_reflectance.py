"""Conditioned reflectance diagnostics; not a reproduction of unavailable EMIT arrays.

The mask is common to all predictors and uses benchmark truth plus a flux scale fitted
on training rows. Finite-error summaries are conditional on successful inversions;
failure counts and coverage must be reported alongside them. No clipping or zero-fill
is used. Run with --help for the on-disk campaign prediction format.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from emit_target_quality import indices_digest, seeded_split

COMPONENTS = ("Y1", "Y2", "Y3", "Y4")


def components(values: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return path radiance, summed flux, and spherical albedo after shape checks."""
    arrays = [np.asarray(values[k], dtype=np.float64) for k in COMPONENTS]
    if arrays[0].ndim != 2 or any(x.shape != arrays[0].shape for x in arrays):
        raise ValueError("Components must be equally shaped [sample, band] arrays")
    return arrays[0], arrays[1] + arrays[2], arrays[3]


def training_flux_scale(training: dict[str, np.ndarray]) -> float:
    """Global positive-flux median across training rows and bands (not test data)."""
    _, flux, _ = components(training)
    positive = flux[np.isfinite(flux) & (flux > 0)]
    if not positive.size:
        raise ValueError("Training data contain no finite positive flux")
    return float(np.median(positive))


def evaluate(truth: dict[str, np.ndarray], prediction: dict[str, np.ndarray], *,
             flux_scale: float, threshold: float, rho: float = 0.7,
             q_min: float = 1e-8, band_mask: np.ndarray | None = None,
             domain: str = "physical") -> dict[str, Any]:
    """Evaluate on an explicit common mask; prediction failures never change coverage.

    threshold and q_min must be prespecified, not selected for favorable test errors.
    A failure is a nonfinite inverse or a nonpositive predicted inverse denominator.
    The latter violates the positive true-denominator regime of the stability bound.
    """
    if domain not in ("physical", "algebraic"):
        raise ValueError("domain must be physical or algebraic")
    if not np.isfinite(flux_scale) or flux_scale <= 0:
        raise ValueError("flux_scale must be finite and positive")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if not np.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("rho must lie in [0, 1]")
    if not np.isfinite(q_min) or q_min <= 0:
        raise ValueError("q_min must be finite and positive")
    a, t, s = components(truth)
    ah, th, sh = components(prediction)
    if ah.shape != a.shape:
        raise ValueError("Prediction and truth shapes differ")
    bm = np.ones(a.shape[1], dtype=bool) if band_mask is None else np.asarray(band_mask)
    if bm.shape != (a.shape[1],) or bm.dtype != np.bool_:
        raise ValueError("band_mask must be a Boolean vector with one entry per band")
    q = 1.0 - rho * s
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        radiance = a + rho * t / q
        denominator = th + sh * (radiance - ah)
        inverse = (radiance - ah) / denominator
    common = (bm[None, :] & np.isfinite(a) & np.isfinite(t) & np.isfinite(s)
              & np.isfinite(radiance) & (q >= q_min) & (t > 0)
              & (t >= threshold * flux_scale))
    if domain == "physical":
        common &= (s >= 0) & (s < 1)
    good = common & np.isfinite(inverse) & np.isfinite(denominator) & (denominator > 0)
    count = int(common.sum())
    n_good = int(good.sum())
    result: dict[str, Any] = {
        "schema_version": 2, "truth_domain": domain,
        "physical_albedo_rule": "0 <= Y4 < 1" if domain == "physical" else None,
        "threshold": float(threshold), "rho": float(rho), "q_min": float(q_min),
        "training_flux_scale": float(flux_scale), "entries": int(a.size),
        "application_band_entries": int(a.shape[0] * bm.sum()),
        "retained_entries": count, "retained_fraction_all_entries": count / a.size if a.size else 0.0,
        "successful_inversions": n_good, "failed_inversions": count - n_good,
        "nonfinite_inversions_on_mask": int((common & ~np.isfinite(inverse)).sum()),
        "nonfinite_denominators_on_mask": int((common & ~np.isfinite(denominator)).sum()),
        "nonpositive_denominators_on_mask": int((common & np.isfinite(denominator) & (denominator <= 0)).sum()),
        "failure_count_note": "Failure categories can overlap; failed_inversions is their union",
        "failure_fraction_on_mask": (count - n_good) / count if count else None,
        "finite_summary_scope": "successful positive-denominator inversions on common mask",
        "absolute_error_units": "reflectance, not percentage points",
    }
    for key in ("median_absolute_error", "p95_absolute_error", "rmse",
                "outside_unit_interval_fraction_on_mask"):
        result[key] = None
    if n_good:
        error = inverse[good] - rho
        peak = float(np.max(np.abs(error)))
        rms = peak * float(np.sqrt(np.mean((error / peak) ** 2))) if peak else 0.0
        result.update(median_absolute_error=float(np.median(np.abs(error))),
                      p95_absolute_error=float(np.quantile(np.abs(error), 0.95)),
                      rmse=rms,
                      outside_unit_interval_fraction_on_mask=float(
                          np.count_nonzero((inverse[good] < 0) | (inverse[good] > 1)) / count))
    return result


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path, help="X.npy and Y1.npy through Y4.npy")
    parser.add_argument("--predictions", required=True, type=Path, help="campaign NPZ: idx_te and <family>_Y1 ... Y4")
    parser.add_argument("--record", required=True, type=Path, help="matching campaign JSON with seed and data_sha")
    parser.add_argument("--thresholds", required=True, type=float, nargs="+", help="prespecified flux/scale thresholds")
    parser.add_argument("--band-mask", type=Path, help="Boolean .npy application passband; default all bands")
    parser.add_argument("--rho", type=float, default=0.7)
    parser.add_argument("--q-min", type=float, default=1e-8)
    parser.add_argument("--domain", choices=("physical", "algebraic"), default="physical",
                        help="Physical uses 0<=Y4<1; algebraic explicitly retains the older looser domain")
    parser.add_argument("--allow-float32", action="store_true", help="explicitly accept old rounded prediction dumps")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    for c in ("X",) + COMPONENTS:
        if digest(args.data_dir / f"{c}.npy") != record["data_sha"][c]:
            raise ValueError(f"Data digest mismatch for {c}")
    ys = {c: np.load(args.data_dir / f"{c}.npy", allow_pickle=False) for c in COMPONENTS}
    n = len(ys["Y1"])
    seed = int(record["seed"])
    base_train, idx_val, idx_test = seeded_split(n, seed)
    idx_train = base_train[:int(record["ntrain"])]
    with np.load(args.predictions, allow_pickle=False) as saved:
        if "split_indices_sha256" in record:
            idx_train = saved["idx_tr"]
            for name, key in (("train", "idx_tr"), ("validation", "idx_val"), ("test", "idx_te")):
                if indices_digest(saved[key]) != record["split_indices_sha256"][name]:
                    raise ValueError(f"Prediction {name} index digest does not match record")
            if (len(idx_train) != int(record["ntrain"]) or len(np.unique(idx_train)) != len(idx_train)
                    or not np.isin(idx_train, base_train).all()
                    or not np.array_equal(saved["idx_val"], idx_val)):
                raise ValueError("Training or validation indices do not match the declared split")
        elif record.get("target_quality", {}).get("policy", "raw") != "raw":
            raise ValueError("Filtered training requires saved, hashed split indices")
    quality = record.get("target_quality", {})
    scale = quality.get("reference_flux_scale")
    if scale is not None:
        # Recompute the common scale from the original candidate block, not a model's errors.
        candidates = base_train[:int(quality["candidate_rows"])]
        if indices_digest(candidates) != quality["candidate_indices_sha256"]:
            raise ValueError("Original training-block digest mismatch")
        verified_scale = training_flux_scale({c: ys[c][candidates] for c in COMPONENTS})
        if verified_scale != scale:
            raise ValueError("Reference training flux scale mismatch")
    else:
        scale = training_flux_scale({c: ys[c][idx_train] for c in COMPONENTS})
    truth = {c: ys[c][idx_test] for c in COMPONENTS}
    bm = np.load(args.band_mask, allow_pickle=False) if args.band_mask else None
    output: dict[str, Any] = {
        "record_sha256": digest(args.record), "prediction_sha256": digest(args.predictions),
        "seed": seed, "data_sha": record["data_sha"],
        "truth_domain": args.domain, "training_policy": quality.get("policy", "raw"),
        "training_indices_sha256": indices_digest(idx_train),
        "band_mask_sha256": digest(args.band_mask) if args.band_mask else None,
        "note": "New diagnostic, not retroactively used to select published heads or thresholds",
        "results": {}, "prediction_dtypes": {},
    }
    with np.load(args.predictions, allow_pickle=False) as predictions:
        if not np.array_equal(predictions["idx_te"], idx_test):
            raise ValueError("Prediction test indices do not match the recorded split")
        for family in record["families"]:
            pred = {c: predictions[f"{family}_{c}"] for c in COMPONENTS}
            dtypes = sorted({str(x.dtype) for x in pred.values()})
            output["prediction_dtypes"][family] = dtypes
            rounded = any(x.dtype.kind != "f" or x.dtype.itemsize < 8 for x in pred.values())
            if rounded and not args.allow_float32:
                raise ValueError("Old prediction dump is rounded to float32. Use --allow-float32 to accept a distinct precision experiment, not exact regeneration of float64 scores.")
            output["results"][family] = [evaluate(truth, pred, flux_scale=scale, threshold=t,
                                                rho=args.rho, q_min=args.q_min, band_mask=bm,
                                                domain=args.domain) for t in args.thresholds]
    output["float32_opt_in"] = bool(args.allow_float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
