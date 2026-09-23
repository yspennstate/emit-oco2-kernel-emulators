"""Checks of target filtering and mask semantics on synthetic data, and of the fresh-partition records."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

from conditioned_reflectance import evaluate, training_flux_scale
from emit_target_quality import (admissible_entries, audit_targets, indices_digest,
                                 seeded_split, select_training_rows)

ROOT = Path(__file__).resolve().parents[1]


def fixture(n=40, bands=3):
    return {"Y1": np.ones((n, bands)), "Y2": np.ones((n, bands)),
            "Y3": np.full((n, bands), .2), "Y4": np.full((n, bands), .1)}


class TargetQualityTests(unittest.TestCase):
    def test_small_positive_flux_is_not_invalid_training_target(self):
        ys = fixture(2)
        ys["Y2"][0, 0], ys["Y3"][0, 0] = 1e-25, 0.
        self.assertTrue(admissible_entries(ys).all())

    def test_union_counts_do_not_sum_overlapping_flags(self):
        ys = fixture(3)
        ys["Y4"][0, 0], ys["Y2"][0, 0] = 2., -1.
        d = audit_targets(ys)
        self.assertEqual(d["admissible_entries"], 8)
        self.assertEqual(d["admissible_rows"], 2)
        self.assertEqual(d["albedo_exactly_two_entries"], 1)
        self.assertEqual(d["nonpositive_flux_entries"], 1)

    def test_clean_and_matched_size_and_no_target_mutation(self):
        ys = fixture()
        idx, va, te = seeded_split(40, 101)
        ys["Y4"][idx[:4], 0] = 2.
        before = {k: a.copy() for k, a in ys.items()}
        raw, r = select_training_rows(ys, idx, "raw", 101)
        clean, c = select_training_rows(ys, idx, "admissible", 101)
        matched, m = select_training_rows(ys, idx, "matched-unfiltered", 101)
        np.testing.assert_array_equal(raw, idx)
        self.assertEqual(len(clean), len(idx)-4)
        self.assertEqual(len(matched), len(clean))
        self.assertTrue(admissible_entries({k: a[clean] for k, a in ys.items()}).all())
        np.testing.assert_array_equal(matched, select_training_rows(ys, idx, "matched-unfiltered", 101)[0])
        self.assertTrue(set(matched).issubset(idx))
        self.assertEqual(r["reference_flux_scale"], c["reference_flux_scale"])
        self.assertEqual(r["reference_flux_scale"], m["reference_flux_scale"])
        for k in ys:
            np.testing.assert_array_equal(ys[k], before[k])

    def test_training_selection_does_not_consult_heldout_targets(self):
        ys = fixture()
        tr, va, te = seeded_split(40, 101)
        before = select_training_rows(ys, tr, "admissible", 101)
        ys["Y4"][np.r_[va, te]] = 2.
        after = select_training_rows(ys, tr, "admissible", 101)
        np.testing.assert_array_equal(before[0], after[0])
        self.assertEqual(before[1], after[1])

    def test_index_digest_and_validation(self):
        a = np.arange(10, dtype="<i8")
        self.assertEqual(indices_digest(a), indices_digest(a.astype(">i8")))
        self.assertNotEqual(indices_digest(a), indices_digest(a[::-1]))
        with self.assertRaises(ValueError):
            indices_digest(a.astype(float))
        with self.assertRaises(ValueError):
            select_training_rows(fixture(), np.array([1, 1]), "raw", 101)
        with self.assertRaises(ValueError):
            select_training_rows(fixture(), np.array([-1, 2]), "raw", 101)

    def test_no_silent_nonfinite_imputation(self):
        ys = fixture()
        ys["Y1"][0, 0] = np.nan
        self.assertFalse(admissible_entries(ys)[0, 0])
        with self.assertRaises(ValueError):
            select_training_rows(ys, np.arange(20), "raw", 101)

    def test_filter_precedes_campaign_preprocessing(self):
        text = (ROOT / "code/emit_campaign.py").read_text()
        self.assertLess(text.index("idx_tr, target_quality = select_training_rows"),
                        text.index("xstd = Standardizer"))
        self.assertLess(text.index("idx_tr, target_quality = select_training_rows"),
                        text.index("pca[c] = PCAReducer"))

    def test_algebraic_inverse_can_exist_outside_physical_domain(self):
        ys = fixture(1, 3)
        ys["Y4"][0] = [-.1, 1.1, .2]
        physical = evaluate(ys, ys, flux_scale=1, threshold=0)
        algebraic = evaluate(ys, ys, flux_scale=1, threshold=0, domain="algebraic")
        self.assertEqual(physical["retained_entries"], 1)
        self.assertEqual(algebraic["retained_entries"], 3)
        self.assertLess(algebraic["rmse"], 1e-14)

    def test_truth_coverage_not_changed_by_failed_predictions(self):
        ys = fixture(1, 3)
        pred = {k: a.copy() for k, a in ys.items()}
        pred["Y1"][0, 0] = np.nan
        pred["Y2"][0, 1] = -10.
        result = evaluate(ys, pred, flux_scale=1., threshold=0.)
        self.assertEqual(result["retained_entries"], 3)
        self.assertEqual(result["failed_inversions"], 2)
        self.assertEqual(result["nonfinite_inversions_on_mask"], 1)
        self.assertEqual(result["nonpositive_denominators_on_mask"], 1)
        self.assertEqual(result["successful_inversions"], 1)

    def test_scaled_threshold_has_unit_consistent_coverage(self):
        ys = fixture(2)
        pred = {k: a.copy() for k, a in ys.items()}
        pred["Y1"] += .01
        r1 = evaluate(ys, pred, flux_scale=training_flux_scale(ys), threshold=.5)
        ys2 = {k: a*(1000 if k != "Y4" else 1) for k, a in ys.items()}
        p2 = {k: a*(1000 if k != "Y4" else 1) for k, a in pred.items()}
        r2 = evaluate(ys2, p2, flux_scale=training_flux_scale(ys2), threshold=.5)
        self.assertEqual(r1["retained_entries"], r2["retained_entries"])
        self.assertAlmostEqual(r1["rmse"], r2["rmse"], places=14)

    def test_fresh_partition_freeze_and_report(self):
        folder = ROOT / "results" / "confirmation"
        registered = (folder / "freeze.sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256((folder / "freeze.json").read_bytes()).hexdigest(), registered)
        report = json.loads((folder / "confirmation_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["n_confirmation"], round(0.15 * 23313))
        fam = report["families"]
        lowest_radiance = min(fam, key=lambda f: fam[f]["radiance_pct"])
        lowest_tail = min(fam, key=lambda f: fam[f]["all_bands"]["p95_pp"])
        self.assertEqual(lowest_radiance, report["H1"]["best_radiance_family"])
        self.assertEqual(lowest_tail, report["H1"]["best_all_band_p95_family"])
        clause_b = fam[lowest_radiance]["all_bands"]["p95_pp"] > fam["dnn_plus_residual_krr"]["all_bands"]["p95_pp"]
        self.assertEqual(clause_b, report["H1"]["clause_b_worse_tail"])
        self.assertEqual(report["H1"]["H1_supported"], lowest_radiance != lowest_tail and clause_b)

    @unittest.skipUnless(os.environ.get("EMIT_INTEGRATION_TESTS") == "1",
                         "Set EMIT_INTEGRATION_TESTS=1 with scipy/torch to run the synthetic training smoke test")
    def test_end_to_end_synthetic_training_and_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data, out = folder/"data", folder/"out"
            data.mkdir()
            rng = np.random.default_rng(101)
            x = rng.normal(size=(80, 6))
            tr, va, te = seeded_split(80, 101)
            np.save(data/"X.npy", x)
            ys = fixture(80, 6)
            for j, k in enumerate(ys):
                ys[k] += .01 * np.tanh(x + j)
            ys["Y4"][tr[:4], 0] = 2.
            for k, a in ys.items():
                np.save(data/f"{k}.npy", a)
            masks, counts = [], []
            env = dict(os.environ, EMIT_DATA=str(data), P2_OUT=str(out), NMKC_THREADS="1")
            for policy in ("raw", "admissible", "matched-unfiltered"):
                cmd = [sys.executable, str(ROOT/"code/emit_campaign.py"), "--seed", "101",
                       "--widths", "8,8", "--epochs", "1", "--members", "1", "--pca_rank", "4",
                       "--families", "ridge3,krr,dnn,dnn_corr,dkr,stack",
                       "--training-policy", policy, "--tag", policy]
                subprocess.run(cmd, env=env, capture_output=True, text=True, check=True, timeout=90)
                record = json.loads((out/f"{policy}.json").read_text())
                counts.append(record["ntrain"])
                with np.load(out/"preds"/f"{policy}.npz") as pred:
                    np.testing.assert_array_equal(pred["idx_te"], te)
                    np.testing.assert_array_equal(pred["idx_val"], va)
                scored = out/f"{policy}_conditioned.json"
                subprocess.run([sys.executable, str(ROOT/"code/conditioned_reflectance.py"),
                    "--data-dir", str(data), "--predictions", str(out/"preds"/f"{policy}.npz"),
                    "--record", str(out/f"{policy}.json"), "--thresholds", "1e-12",
                    "--output", str(scored)], capture_output=True, text=True, check=True, timeout=30)
                scores = json.loads(scored.read_text())
                masks.append({k: v[0]["retained_entries"] for k, v in scores["results"].items()})
            self.assertEqual(counts, [len(tr), len(tr)-4, len(tr)-4])
            self.assertEqual(masks[0], masks[1])
            self.assertEqual(masks[0], masks[2])


if __name__ == "__main__":
    unittest.main()
