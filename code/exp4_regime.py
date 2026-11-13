"""Regime-blind vs regime-aware linear estimator of a latent level D from x = c_r D + o_r + eps (regimes r iid, independent of D)."""
import numpy as np
rng = np.random.default_rng(1); R = 6; pi = np.full(R, 1/R); a = np.array([0.6, 0.8, 1.0, 1.0, 1.3, 1.6]); b = np.array([-0.5, -0.2, 0.0, 0.1, 0.3, 0.6]); muD, v = 0.5, 0.08
ED2 = v+muD**2; Ea, Ea2, Eb, Vb = (pi*a).sum(), (pi*a**2).sum(), (pi*b).sum(), (pi*b**2).sum()-(pi*b).sum()**2
for sig in [0.0, 0.02, 0.05, 0.1]:
    Cab = (pi*a*b).sum()-Ea*Eb; Vx = Ea2*ED2-(Ea*muD)**2+Vb+2*muD*Cab+sig**2; mse_blind_th = v-(Ea*v)**2/Vx; mse_aware_th = (pi*sig**2/a**2).sum()
    n = 400000; D = rng.normal(muD, np.sqrt(v), n); r = rng.integers(0, R, n); x = a[r]*D+b[r]+sig*rng.standard_normal(n)
    # blind OLS
    A = np.c_[x, np.ones(n)]; w = np.linalg.lstsq(A, D, rcond=None)[0]; mse_blind = np.mean((A@w-D)**2)
    # aware: per-regime OLS
    mse_aware = np.mean([np.mean((np.c_[x[r == k], np.ones((r == k).sum())]@np.linalg.lstsq(np.c_[x[r == k], np.ones((r == k).sum())], D[r == k], rcond=None)[0]-D[r == k])**2) for k in range(R)])
    print('sigma %.2f  blind: theory %.5f MC %.5f   aware: theory(plug-in) %.5f MC(OLS) %.5f' % (sig, mse_blind_th, mse_blind, mse_aware_th, mse_aware))
