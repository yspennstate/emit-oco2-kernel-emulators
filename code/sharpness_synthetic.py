"""Controlled sharpness experiment for the transfer theorem (theory v4), with the analytic curves as the reference.

Model of Proposition 5: F_t(u) = u^beta on (0, 1); a = s = 0, rho = r0; the emulator predicts t exactly, s_hat = 0,
and a path-radiance error a_hat = c t 1{t <= tau} (average-error family), a_hat = c min(eps, t) (uniform family),
or a_hat = min(c t, sigma t^nu) (conditional-profile family). The constrained retrieval is run literally (projections,
failure count) on Monte Carlo samples; every point records the number of rare-region events before any slope is
fitted; the Monte Carlo is compared with the analytic curves

  average family    eps_R^2 = c^2 beta tau^{beta+2}/(beta+2),  MSE = c^2 tau^beta
  uniform family    MSE/c^2 = eps^beta + beta eps^2 (1 - eps^{beta-2})/(beta-2)   (beta != 2),
                    eps^2 (1 + 2 log(1/eps))                                       (beta = 2),
                    local log-log slope 2 - 2/(1 + 2 log(1/eps)) at beta = 2
  profile family    MSE/c^2 = E min{1, (sigma/c)^2 t^{2nu-2}} by quadrature

and the constant-one pointwise inequality |rho_bar - rho| <= min{R, e_R/t} is checked at every sample.
Output: JSON with the slopes, the analytic values, the bound ratios and the event counts.
"""
import json, sys, time
import numpy as np


def constrained_retrieval(L, a_hat, t_hat, s_hat, R=1.0, S=0.0, fail_value=0.0):
    t_bar = np.maximum(t_hat, 0.0); s_bar = np.clip(s_hat, 0.0, S)
    den = t_bar + s_bar * (L - a_hat)
    ok = den > 0
    rho = np.full_like(L, fail_value)
    rho[ok] = (L[ok] - a_hat[ok]) / den[ok]
    return np.clip(rho, 0.0, R), int((~ok).sum())


def analytic_uniform(eps, beta):
    if abs(beta - 2.0) < 1e-12:
        return eps ** 2 * (1.0 + 2.0 * np.log(1.0 / eps))
    return eps ** beta + beta * eps ** 2 * (1.0 - eps ** (beta - 2.0)) / (beta - 2.0)


def analytic_profile(sig_over_c, beta, nu, n_quad=200000):
    u = (np.arange(n_quad) + 0.5) / n_quad             # F = u^beta: t = u^(1/beta) for u uniform
    t = u ** (1.0 / beta)
    return float(np.mean(np.minimum(1.0, sig_over_c ** 2 * t ** (2 * nu - 2))))


def run(beta, r0=0.5, c=0.25, n=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.random(n) ** (1.0 / beta)
    L = r0 * t
    R = 0.9
    out = dict(beta=beta, r0=r0, c=c, n=n, R=R, average_family=[], uniform_family=[], profile_family=[])
    viol = 0
    for tau in (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125):
        a_hat = c * t * (t <= tau)
        rho, fails = constrained_retrieval(L, a_hat, t, np.zeros_like(t), R=R)
        eR = np.abs(a_hat)                                       # R|e_t| = R^2 t|e_s| = 0 here
        viol += int(np.sum(np.abs(rho - r0) > np.minimum(R, eR / t) * (1 + 1e-9) + 1e-15))
        eps2 = float(np.mean(eR ** 2)); mse = float(np.mean((rho - r0) ** 2)); events = int(np.sum(t <= tau))
        ladder = np.geomspace(1e-3, 1.0, 61)
        Ft = lambda u: float(np.mean(t <= u))
        bound_last = min(R * R * Ft(u) + eps2 / (u * u) for u in ladder)
        bound_middle = min(R * R * Ft(u) + float(np.mean(np.where(t > u, (eR / t) ** 2, 0.0))) for u in ladder)
        out["average_family"].append(dict(tau=tau, events=events, eps2=eps2, eps2_exact=c * c * beta * tau ** (beta + 2) / (beta + 2),
                                          mse=mse, mse_exact=c * c * tau ** beta, bound_last_inf=bound_last, bound_middle_inf=bound_middle,
                                          bound_last_over_mse=bound_last / mse if mse > 0 else None, bound_middle_over_mse=bound_middle / mse if mse > 0 else None,
                                          failures=fails))
    for eps in (0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625):
        a_hat = c * np.minimum(eps, t)
        rho, fails = constrained_retrieval(L, a_hat, t, np.zeros_like(t), R=R)
        eR = np.abs(a_hat)
        viol += int(np.sum(np.abs(rho - r0) > np.minimum(R, eR / t) * (1 + 1e-9) + 1e-15))
        mse = float(np.mean((rho - r0) ** 2))
        out["uniform_family"].append(dict(eps=eps, events_below_eps=int(np.sum(t <= eps)), sup_eR=float(eR.max()), eps2=float(np.mean(eR ** 2)),
                                          mse=mse, mse_analytic=c * c * analytic_uniform(eps, beta), failures=fails))
    for nu in (0.0, 0.5, 0.9):
        for sig in (0.2, 0.1, 0.05, 0.025):
            a_hat = np.minimum(c * t, sig * t ** nu)
            rho, fails = constrained_retrieval(L, a_hat, t, np.zeros_like(t), R=R)
            eR = np.abs(a_hat)
            viol += int(np.sum(np.abs(rho - r0) > np.minimum(R, eR / t) * (1 + 1e-9) + 1e-15))
            mse = float(np.mean((rho - r0) ** 2))
            out["profile_family"].append(dict(nu=nu, sigma=sig, mse=mse, mse_analytic=c * c * analytic_profile(sig / c, beta, nu),
                                              cond_bound_holds=bool(np.all(eR <= sig * t ** nu * (1 + 1e-12))), failures=fails))
    out["pointwise_bound_violations"] = viol
    A = [d for d in out["average_family"] if d["mse"] > 0 and d["events"] >= 200]
    out["average_points_used"] = len(A)
    x = np.log([d["eps2"] for d in A]); y = np.log([d["mse"] for d in A])
    out["slope_mse_vs_eps2_average"] = float(np.polyfit(x, y, 1)[0]) if len(A) >= 3 else None
    out["slope_predicted_average"] = beta / (beta + 2)
    U = [d for d in out["uniform_family"] if d["mse"] > 0]
    x = np.log([d["eps"] for d in U]); y = np.log([d["mse"] for d in U])
    out["slope_mse_vs_eps_uniform"] = float(np.polyfit(x, y, 1)[0]) if len(U) >= 3 else None
    ya = np.log([d["mse_analytic"] for d in U])
    out["slope_analytic_uniform_same_ladder"] = float(np.polyfit(x, ya, 1)[0]) if len(U) >= 3 else None
    out["slope_predicted_uniform"] = min(beta, 2.0)
    out["max_rel_diff_uniform_mc_vs_analytic"] = float(max(abs(d["mse"] - d["mse_analytic"]) / d["mse_analytic"] for d in U))
    out["max_rel_diff_profile_mc_vs_analytic"] = float(max(abs(d["mse"] - d["mse_analytic"]) / d["mse_analytic"] for d in out["profile_family"] if d["mse_analytic"] > 0))
    return out


if __name__ == "__main__":
    t0 = time.time()
    res = {"betas": []}
    for beta in (0.5, 1.0, 2.0, 3.0, 4.0):
        r = run(beta)
        res["betas"].append(r)
        print(f"beta={beta}: average slope {r['slope_mse_vs_eps2_average']:.3f} (predicted {r['slope_predicted_average']:.3f}, {r['average_points_used']} points); "
              f"uniform slope {r['slope_mse_vs_eps_uniform']:.3f} (analytic on the same ladder {r['slope_analytic_uniform_same_ladder']:.3f}, limit {r['slope_predicted_uniform']:.1f}); "
              f"MC vs analytic max rel diff uniform {r['max_rel_diff_uniform_mc_vs_analytic']:.2e} profile {r['max_rel_diff_profile_mc_vs_analytic']:.2e}; "
              f"pointwise violations {r['pointwise_bound_violations']}; middle bound/mse {min(d['bound_middle_over_mse'] for d in r['average_family'] if d['bound_middle_over_mse']):.2f}..{max(d['bound_middle_over_mse'] for d in r['average_family'] if d['bound_middle_over_mse']):.2f}", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    out = sys.argv[1] if len(sys.argv) > 1 else "sharpness_synthetic.json"
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out, "in", res["seconds"], "s")
