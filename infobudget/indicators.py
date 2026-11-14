"""Health-indicator constructions and shape scores used in the paper.

All functions are causal where an HI is used online: the value at cycle k uses only cycles up to k.
"""
from __future__ import annotations
import numpy as np


def causal_mean(y, w):
    """Trailing mean over the last w samples (fewer at the start)."""
    y = np.asarray(y, float)
    c = np.cumsum(np.r_[0, y]); i = np.arange(1, len(y) + 1); a = np.maximum(i - w, 0)
    return (c[i] - c[a]) / (i - a)


def causal_slope(y, w):
    """Least-squares slope over the last w samples (zero for fewer than two samples)."""
    y = np.asarray(y, float)
    n = len(y); i = np.arange(n); c1 = np.cumsum(np.r_[0, y]); c2 = np.cumsum(np.r_[0, i * y]); out = np.zeros(n)
    for t in range(1, n):
        a = max(0, t - w + 1); m = t - a + 1
        if m < 2:
            continue
        sy = c1[t + 1] - c1[a]; sxy = c2[t + 1] - c2[a]; sx = (a + t) * m / 2
        sxx = (t * (t + 1) * (2 * t + 1) - (a - 1) * a * (2 * a - 1)) / 6 if a > 0 else t * (t + 1) * (2 * t + 1) / 6
        den = m * sxx - sx * sx
        out[t] = (m * sxy - sx * sy) / den if den > 0 else 0.0
    return out


def _affine(Hl):
    """Affine map that sends the mean start value of the training units to 0 and the mean end value to 1."""
    st = np.mean([h[:5].mean() for h in Hl]); en = np.mean([h[-5:].mean() for h in Hl]); s = 1.0 if en > st else -1.0
    return lambda h: (s * h - s * st) / (s * en - s * st)


def fit_ridge_hi(Xl, alpha: float = 10.0, smooth: int = 5):
    """Supervised linear HI: ridge regression of the life fraction t/L on the (standardised) sensors, smoothed and scaled to 0..1.

    Xl : list of arrays (T_u, d) of training units that ran to failure.  Returns a function X -> HI trajectory.
    """
    Z = np.concatenate(Xl); y = np.concatenate([np.arange(1, len(x) + 1) / len(x) for x in Xl])
    Za = np.c_[Z, np.ones(len(Z))]
    w = np.linalg.solve(Za.T @ Za + alpha * np.eye(Za.shape[1]), Za.T @ y)
    f = lambda x: np.c_[x, np.ones(len(x))] @ w
    fa = _affine([causal_mean(f(x), smooth) for x in Xl])
    return lambda x: fa(causal_mean(f(np.asarray(x, float)), smooth))


def fit_pca_hi(Xl, smooth: int = 5):
    """Unsupervised HI: first principal component of the sensors, oriented to increase with age and scaled to 0..1."""
    Z = np.concatenate(Xl); m = Z.mean(0); _, _, Vt = np.linalg.svd(Z - m, full_matrices=False); w = Vt[0]
    w = w * np.sign(np.mean([np.corrcoef(x @ w, np.arange(len(x)))[0, 1] for x in Xl]))
    fa = _affine([causal_mean(x @ w, smooth) for x in Xl])
    return lambda x: fa(causal_mean(np.asarray(x, float) @ w, smooth))


def trajectory_features(h, level_window=10, velocity_window=30, lags=(20, 50)):
    """Summary of an HI trajectory used by the predictor: level, velocity and lagged values (columns), one row per cycle."""
    h = np.asarray(h, float)
    cols = [causal_mean(h, level_window), causal_slope(h, velocity_window)]
    for k in lags:
        cols.append(np.r_[np.full(k, h[0]), h][:len(h)])
    return np.column_stack(cols)


def sensor_features(X, level_window=10, velocity_window=30):
    """Level and velocity of every sensor (reference predictor on all sensors)."""
    X = np.asarray(X, float)
    return np.hstack([np.column_stack([causal_mean(X[:, j], level_window) for j in range(X.shape[1])]),
                      np.column_stack([causal_slope(X[:, j], velocity_window) for j in range(X.shape[1])])])


# ------------------------------------------------------------------ geometric scores of Coble and Hines
def monotonicity(h):
    d = np.diff(np.asarray(h, float)); return abs(np.sum(np.sign(d))) / max(len(d), 1)


def trendability(Hl):
    """Absolute correlation of the HI with time, averaged over units."""
    return float(np.mean([abs(np.corrcoef(h, np.arange(len(h)))[0, 1]) for h in Hl]))


def prognosability(Hl):
    fin = np.array([h[-1] for h in Hl]); ini = np.array([h[0] for h in Hl])
    return float(np.exp(-np.std(fin) / max(np.mean(np.abs(fin - ini)), 1e-9)))


def geometric_scores(Hl) -> dict:
    """Monotonicity, trendability, prognosability and their sum for a list of HI trajectories (one per unit)."""
    mo = float(np.mean([monotonicity(h) for h in Hl])); tr = trendability(Hl); pr = prognosability(Hl)
    return {"monotonicity": mo, "trendability": tr, "prognosability": pr, "sum": mo + tr + pr}


def template_shrinkage(H, template, gamma: float, sigma2: float, rng=None):
    """Shrink an HI toward a template trajectory and add noise (Theorem 3):  m + (1 - gamma)(H - m) + noise(0, sigma2^2).

    Raises monotonicity while information about the RUL falls; use it to test whether a shape score can be gamed.
    """
    rng = np.random.default_rng(rng)
    H = np.asarray(H, float); m = np.asarray(template, float)
    m = m[:len(H)] if len(m) >= len(H) else np.r_[m, np.full(len(H) - len(m), m[-1])]    # longer units: hold the last template value
    return m + (1.0 - gamma) * (H - m) + sigma2 * rng.standard_normal(len(H))


def shrinkage_constructor(gamma: float, alpha: float = 10.0, smooth: int = 5, seed: int = 0):
    """Return fit(training arrays) -> transform(X): a ridge HI shrunk toward the mean training trajectory with noise (paper, Section 5.3).

    The noise level gives a limiting monotonicity of about 0.9 for the template increments.  Use it with infobudget.evaluate to see how
    the information lower bound falls while the monotonicity rises.
    """
    def fit(Xl):
        f = fit_ridge_hi(Xl, alpha, smooth); H = [f(x) for x in Xl]; L = np.array([len(h) for h in H])
        template = np.array([np.mean([h[k] for h in H if len(h) > k]) for k in range(L.max())])
        sigma2 = abs(float(np.mean(np.diff(template[:int(np.median(L))])))) / (np.sqrt(2) * 1.645)
        rng = np.random.default_rng(seed)
        return lambda x: template_shrinkage(f(x), template, gamma, sigma2, rng)
    return fit
