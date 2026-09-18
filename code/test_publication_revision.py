"""Finite-design verification of the revised mathematics and diagnostic semantics.

Run: python -m unittest discover -s code -p 'test_publication_revision.py' -v
These tests are synthetic, not new EMIT fits or a substitute for the written proofs.
"""
import unittest
import numpy as np
from conditioned_reflectance import evaluate, training_flux_scale


class PublicationRevisionTests(unittest.TestCase):
    def test_frozen_residual_and_design_identity(self):
        rng = np.random.default_rng(20260918)
        for n in (3, 11, 29):
            x = rng.normal(size=(n, 4))
            z = rng.normal(size=(7, 4))
            def kernel(a, b):
                r = np.sqrt(np.maximum(((a[:, None] - b[None, :]) ** 2).sum(2), 0))
                u = np.sqrt(5) * r
                return (1 + u + u * u / 3) * np.exp(-u)
            k, kz = kernel(x, x), kernel(z, x)
            lam = n * 1e-3
            f, h = rng.normal(size=(2, n, 3))
            fv, hv = rng.normal(size=(2, 7, 3))
            s = lambda y: kz @ np.linalg.solve(k + lam * np.eye(n), y)
            np.testing.assert_allclose(fv - (hv + s(f - h)), (fv - s(f)) - (hv - s(h)), atol=2e-12)
            resid = f - k @ np.linalg.solve(k + lam * np.eye(n), f)
            np.testing.assert_allclose(resid, lam * np.linalg.solve(k + lam * np.eye(n), f), atol=2e-12)

    def test_semidefinite_boundary(self):
        k = np.diag([0., 1., 99.])
        factors = 1 / (np.linalg.eigvalsh(k) + 1)
        self.assertEqual(factors[0], 1.)
        self.assertTrue(np.all((factors > 0) & (factors <= 1)))

    def test_counterexamples_to_norm_error_ordering(self):
        r = np.diag([.5, .01])
        cases = [(np.array([1., 0.]), np.array([0., 2.]), .5, .02),
                 (np.array([0., 1.]), np.array([.5, 0.]), .01, .25)]
        for f, residual, direct, corrected in cases:
            self.assertAlmostEqual(np.linalg.norm(r @ f), direct)
            self.assertAlmostEqual(np.linalg.norm(r @ residual), corrected)

    def test_alignment_and_optimal_interpolation(self):
        rng = np.random.default_rng(34)
        for _ in range(300):
            a, b = rng.normal(size=(2, 9))
            lhs = np.dot(a - b, a - b) - np.dot(a, a)
            rhs = np.dot(b, b) - 2 * np.dot(a, b)
            self.assertAlmostEqual(lhs, rhs, places=11)
            t = np.clip(np.dot(a, b) / np.dot(b, b), 0, 1)
            best = np.dot(a - t * b, a - t * b)
            grid = np.linspace(0, 1, 201)
            self.assertTrue(np.all(best <= ((a[None] - grid[:, None] * b) ** 2).sum(1) + 1e-11))

    def test_distinct_operators(self):
        rng = np.random.default_rng(55)
        s0, s1 = rng.normal(size=(2, 8, 8))
        f, h = rng.normal(size=(2, 8))
        a = f - s0 @ f
        b = (s1 - s0) @ f + h - s1 @ h
        np.testing.assert_allclose(f - h - s1 @ (f - h), a - b, atol=2e-12)

    def test_projection_and_affine_decoding(self):
        rng = np.random.default_rng(89)
        q, n, m, rank = 9, 13, 7, 4
        v, _ = np.linalg.qr(rng.normal(size=(q, rank)))
        p = v @ v.T
        s = rng.normal(size=(m, n))
        ft, ht = rng.normal(size=(2, n, q))
        fv, hv = rng.normal(size=(2, m, q))
        corrected = hv @ p + s @ ((ft - ht) @ p)
        tail = fv @ (np.eye(q) - p)
        residual = (fv - hv) @ p - s @ ((ft - ht) @ p)
        np.testing.assert_allclose(fv - corrected, tail + residual, atol=3e-12)
        np.testing.assert_allclose(((fv - corrected) ** 2).sum(1), (tail ** 2).sum(1) + (residual ** 2).sum(1), atol=3e-11)
        scales = np.exp(rng.normal(size=q))
        mu, center = rng.normal(size=(2, q))
        decode = lambda a: mu + (a + center) * scales
        np.testing.assert_allclose(decode(fv) - decode(corrected), (tail + residual) * scales, atol=3e-12)

    def test_forward_inverse_identities_and_bound(self):
        rng = np.random.default_rng(144)
        for _ in range(1000):
            rho, s = rng.uniform(0, .9, 2)
            a, t = rng.uniform(.01, 2, 2)
            q = 1 - rho * s
            u, d = rho * t / q, t / q
            ea, et = rng.uniform(-.02, .02, 2) * d
            es = rng.uniform(-.02, .02)
            ah, th, sh = a + ea, t + et, s + es
            qh = 1 - rho * sh
            radiance, predicted_radiance = a + u, ah + rho * th / qh
            self.assertAlmostEqual(predicted_radiance - radiance, ea + rho * et / qh + rho * rho * t * es / (q * qh), places=12)
            dh = th + sh * (radiance - ah)
            rh = (radiance - ah) / dh
            numerator = (1 - rho * sh) * ea + rho * et + rho * u * es
            self.assertAlmostEqual(rh - rho, -numerator / dh, places=11)
            b = abs(et) + abs(u * es) + abs(sh * ea)
            self.assertLess(b, d)
            bound = (abs(1 - rho * sh) * abs(ea) + rho * abs(et) + rho * abs(u * es)) / (d - b)
            self.assertLessEqual(abs(rh - rho), bound + 1e-11)

    def test_truth_mask_and_failure_accounting(self):
        truth = {"Y1": np.zeros((2, 3)), "Y2": np.tile([1., 1e-12, 0.], (2, 1)),
                 "Y3": np.zeros((2, 3)), "Y4": np.zeros((2, 3))}
        good = evaluate(truth, truth, flux_scale=1., threshold=.01)
        self.assertEqual(good["retained_entries"], 2)
        self.assertEqual(good["failed_inversions"], 0)
        self.assertAlmostEqual(good["rmse"], 0)
        pred = {k: x.copy() for k, x in truth.items()}
        pred["Y2"][0, 0] = 0
        pred["Y2"][1, 0] = -1
        bad = evaluate(truth, pred, flux_scale=1., threshold=.01)
        self.assertEqual(bad["retained_entries"], good["retained_entries"])
        self.assertEqual(bad["failed_inversions"], 2)
        self.assertIsNone(bad["rmse"])

    def test_small_relative_radiance_does_not_control_reflectance(self):
        rho, c = 0.7, 0.2
        for eps in (1e-2, 1e-5, 1e-8):
            t = np.array([1.0, eps])
            a_hat = np.array([0.0, c * eps])
            radiance = rho * t
            predicted = a_hat + rho * t
            inverse = (radiance - a_hat) / t
            np.testing.assert_allclose(inverse, [rho, rho-c], atol=1e-14)
            self.assertAlmostEqual(np.linalg.norm(predicted-radiance) / np.linalg.norm(radiance),
                                   c*eps / (rho*np.sqrt(1+eps*eps)), places=14)
            self.assertAlmostEqual(np.sqrt(np.mean((inverse-rho)**2)), c/np.sqrt(2), places=14)
            np.testing.assert_allclose(inverse-rho, -(predicted-radiance)/t, atol=1e-14)

    def test_empty_mask_and_input_validation(self):
        z = {k: np.zeros((1, 2)) for k in ("Y1", "Y2", "Y3", "Y4")}
        result = evaluate(z, z, flux_scale=1., threshold=0.)
        self.assertEqual(result["retained_entries"], 0)
        self.assertIsNone(result["failure_fraction_on_mask"])
        with self.assertRaises(ValueError):
            evaluate(z, z, flux_scale=0., threshold=0.)
        with self.assertRaises(ValueError):
            evaluate(z, z, flux_scale=1., threshold=-1.)
        with self.assertRaises(ValueError):
            training_flux_scale(z)


if __name__ == "__main__":
    unittest.main()
