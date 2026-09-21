"""Regression tests for the revised fitting/scoring distinction and transfer bounds."""
import unittest
import numpy as np
from stacking_contract import component_weights, physical_domain, require_scoring_precision
from joint_stack import simplex_weights
from transmission_conditioned import retrieval


class StackingContractTests(unittest.TestCase):
    def test_finite_weights_match_historical_formula(self):
        t = np.array([[-1., 0., .1, 2.]])
        R, tau, scale = .9, .03, 2.
        w = component_weights(t, R, tau, scale)
        tv = np.maximum(t, 0); tt = np.maximum(tv, tau * scale)
        np.testing.assert_array_equal(w['Y1'], 1 / tt ** 2)
        np.testing.assert_array_equal(w['Y2'], R * R * (1 / tt ** 2))
        np.testing.assert_array_equal(w['Y3'], w['Y2'])
        np.testing.assert_array_equal(w['Y4'], R ** 4 * (tv / tt) ** 2)

    def test_unweighted_is_not_infinite_threshold_limit(self):
        t = np.array([.2, 1., 4.])
        a = component_weights(t, .9, np.inf, 1.)
        b = component_weights(t, .9, 1e10, 1.)
        np.testing.assert_array_equal(a['Y1'], np.ones_like(t))
        self.assertTrue(np.all(b['Y1'] < 1e-15))
        np.testing.assert_array_equal(a['Y4'], .9 ** 4 * np.ones_like(t))

    def test_precision_policy_checks_validation_as_well_as_test(self):
        arrays = {'test': np.ones(3, dtype=np.float64), 'validation': np.ones(3, dtype=np.float32)}
        with self.assertRaises(ValueError): require_scoring_precision(arrays)
        require_scoring_precision(arrays, allow_precision_altered=True)
        require_scoring_precision({'validation': np.ones(3, dtype=np.float64)})

    def test_physical_mask_is_stronger_than_legacy_albedo_mask(self):
        t=np.array([1., 0., -1., 1., 1., np.inf])
        s=np.array([.2, .2, .2, -.2, .9, .2])
        m=physical_domain(t,s,rho=.7,R=.9,S=.8)
        np.testing.assert_array_equal(m,[True,False,False,False,False,False])
        with self.assertRaises(ValueError): physical_domain(t,s,rho=.7,R=1.,S=1.)

    def test_raw_error_quadratic_is_convex(self):
        rng=np.random.default_rng(21)
        P=rng.normal(size=(30,4)); y=rng.normal(size=30); v=np.exp(rng.normal(size=30))
        f=lambda w: np.mean(v*(P@w-y)**2)
        for _ in range(100):
            a,b=rng.dirichlet(np.ones(4),2); z=rng.random()
            self.assertLessEqual(f(z*a+(1-z)*b),z*f(a)+(1-z)*f(b)+1e-12)
        H=(P.T*v)@P
        self.assertGreaterEqual(np.linalg.eigvalsh(H).min(),-1e-12)

    def test_projection_inside_squared_loss_can_destroy_convexity(self):
        g=lambda w:(max(2*w-1,0)-1)**2
        self.assertGreater(g(.5),.5*(g(0)+g(1)))

    def test_separate_flux_square_omits_cross_term(self):
        e2,e3=np.array([1.,-2.]),np.array([-1.,2.])
        np.testing.assert_array_equal((e2+e3)**2,np.zeros(2))
        self.assertTrue(np.all(e2**2+e3**2>0))
        np.testing.assert_allclose((e2+e3)**2,e2**2+e3**2+2*e2*e3)

    def test_normalized_penalty_solution_is_feasible(self):
        P=np.array([[0.,2.,-1.],[1.,-1.,3.],[2.,1.,0.],[1.,1.,1.]])
        w=simplex_weights(P,np.array([.2,.5,1.1,.9]),np.ones(4))
        self.assertTrue(np.all(w>=0)); self.assertAlmostEqual(w.sum(),1.)
        # This is a feasibility test, deliberately not a claim of exact QP optimality.

    def test_raw_surrogate_bounds_projected_retrieval(self):
        rng=np.random.default_rng(20260921)
        n=5000; R=.9; S=.8; rho=.7
        t=np.exp(rng.uniform(-8,1,n)); y2=.4*t; y3=.6*t
        a=np.zeros(n); s=rng.uniform(0,S,n)
        ah=rng.normal(0,.5,n); p2=y2+rng.normal(0,.5,n); p3=y3+rng.normal(0,.5,n)
        sh=rng.normal(.3,.7,n)
        rec=retrieval(a,t,s,ah,p2+p3,sh,rho,R,S)
        err=(rec['bar']-rho)**2
        ea=ah-a; e2=p2-y2; e3=p3-y3; es=sh-s
        post=np.abs(ea)+R*np.abs(np.maximum(p2+p3,0)-t)+R**2*t*np.abs(np.clip(sh,0,S)-s)
        self.assertTrue(np.all(err<=np.minimum(R**2,(post/t)**2)+1e-12))
        for tau in (.001,.03,.1,1):
            den=np.maximum(t,tau)**2
            total=np.mean((ea**2+R**2*(e2+e3)**2+R**4*t*t*es**2)/den)
            sep=np.mean((ea**2+R**2*e2**2+R**2*e3**2+R**4*t*t*es**2)/den)
            tail=R**2*np.mean(t<=tau)
            self.assertLessEqual(err.mean(),tail+3*total+1e-12)
            self.assertLessEqual(err.mean(),tail+4*sep+1e-12)

    def test_constant_one_attained(self):
        t=np.geomspace(1e-12,2,100); a=np.zeros_like(t)
        z=retrieval(a,t,a,.2*t,t,a,.7,.9,0.)['bar']
        np.testing.assert_allclose(np.abs(z-.7),.2,atol=1e-15)

    def test_rkhs_power_function_identity(self):
        rng=np.random.default_rng(48); X=rng.normal(size=(8,5)); x=rng.normal(size=5)
        K=X@X.T; k=X@x; lam=.2; a=np.linalg.solve(K+lam*np.eye(8),k)
        v1=x@x-2*a@k+a@K@a
        v2=x@x-k@a-lam*(a@a)
        self.assertAlmostEqual(v1,v2,places=12)
        self.assertGreaterEqual(v1,-1e-12)

    def test_empirical_cdf_synthetic_bound_matches_direct_scan(self):
        rng=np.random.default_rng(30); t=rng.random(5000)**2; R=.9; c=.25; tau=.1
        e=c*t*(t<=tau); x=np.geomspace(.001,1,21)
        ft=np.searchsorted(np.sort(t),x,side='right')/len(t)
        fast=R**2*ft+c**2*np.maximum(np.mean(t<=tau)-ft,0)
        slow=np.array([R**2*np.mean(t<=u)+np.mean(np.where(t>u,(e/t)**2,0)) for u in x])
        np.testing.assert_allclose(fast,slow,atol=1e-15)

if __name__=='__main__': unittest.main()
