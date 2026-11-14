"""Information lower bound on a health indicator (HI) from the held-out error of any predictor.

Theory (paper, Theorem 1): for an RUL estimator that uses the HI trajectory and the age t,

    I(T; H | t)  >=  E_t h(T | t) - 0.5 ln(2 pi e MSE)

where h(T | t) is the differential entropy (nats) of the remaining useful life given the age and MSE is the mean squared error of the
estimator.  The left side is the information that the HI adds to the age; a predictor that uses only the age cannot exceed zero
(negative control).  Every 0.1 nat lowers the bound on the RMSE by about 10 percent.
"""
from __future__ import annotations
import numpy as np
from scipy import stats

TWO_PI_E = 2.0 * np.pi * np.e


class LifetimeModel:
    """Non-parametric lifetime distribution of run-to-failure units (Gaussian kernel density of ln L).

    The distribution of the remaining life at age t is that of L - t given L > t, so h(T | t) follows from the lifetimes alone.

    Parameters
    ----------
    lifetimes : array-like of the lifetimes L of the training units, in the unit of the age (cycles, hours, ...).
    step : grid step of the numerical integration, in the same unit (default: median(L) / 400).
    horizon : upper limit of the integration of the remaining life (default: 3 max(L)).
    """

    def __init__(self, lifetimes, step: float | None = None, horizon: float | None = None):
        L = np.asarray(lifetimes, float)
        if L.ndim != 1 or len(L) < 3 or np.any(L <= 0):
            raise ValueError("at least three positive lifetimes are required")
        self.lifetimes = L
        self.kde = stats.gaussian_kde(np.log(L))
        self.step = float(step) if step else float(np.median(L) / 400.0)
        self.horizon = float(horizon) if horizon else float(3.0 * L.max())
        self._tau = np.arange(self.step / 2.0, self.horizon, self.step)
        self._cache: dict[float, float] = {}

    def cond_entropy(self, t: float) -> float:
        """Differential entropy (nats) of the remaining life at age t, h(T | t)."""
        t = float(t)
        if t not in self._cache:
            l = t + self._tau
            f = self.kde(np.log(l)) / l
            S = np.trapezoid(f, self._tau)
            p = np.maximum(f / S, 1e-300)
            self._cache[t] = float(-np.trapezoid(p * np.log(p), self._tau))
        return self._cache[t]

    def cond_entropy_many(self, ages) -> np.ndarray:
        return np.array([self.cond_entropy(t) for t in np.asarray(ages, float)])


def information_lower_bound(y_true, y_pred, h_cond) -> float:
    """Age-conditional information lower bound (nats) of a predictor.

    y_true, y_pred : true and predicted RUL at the evaluation pairs (unit, age), from held-out units.
    h_cond : h(T | t) at the age of each pair (scalar or array), for example LifetimeModel.cond_entropy_many.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mse = float(np.mean((y_pred - y_true) ** 2))
    return float(np.mean(h_cond) - 0.5 * np.log(TWO_PI_E * mse))


def rmse_ratio_from_gain(gain_nats: float) -> float:
    """RMSE of an HI-based predictor divided by that of the age-only predictor, implied by a gain in the lower bound."""
    return float(np.exp(-gain_nats))


def rmse_floor(h_nats: float, information_nats: float = 0.0) -> float:
    """Entropy-power floor on the RMSE: exp(h - I) / sqrt(2 pi e), for differential entropy h and information I (nats)."""
    return float(np.exp(h_nats - information_nats) / np.sqrt(TWO_PI_E))


def bootstrap_units(per_pair: dict, n_boot: int = 300, seed: int = 0, quantiles=(2.5, 97.5), gain=None):
    """Percentile bootstrap over units of the lower bounds computed from per-pair arrays.

    per_pair must contain 'unit' (unit index of each pair), 'T' (true RUL), 'h' (h(T|t) of each pair) and one array of predictions
    for each predictor name given in per_pair['pred'] (a dict name -> array).
    gain : optional pair (name_a, name_b); the difference bound[name_b] - bound[name_a] is returned under the key 'gain'.
    Returns (point, ci): dictionaries name -> bound and name -> [low, high].
    """
    unit = np.asarray(per_pair["unit"])
    T = np.asarray(per_pair["T"], float)
    h = np.asarray(per_pair["h"], float)
    preds = {k: np.asarray(v, float) for k, v in per_pair["pred"].items()}
    ids = np.unique(unit)
    rows = {i: np.where(unit == i)[0] for i in ids}

    def bounds(sub):
        ii = np.concatenate([rows[i] for i in sub])
        out = {k: float(h[ii].mean() - 0.5 * np.log(TWO_PI_E * np.mean((v[ii] - T[ii]) ** 2))) for k, v in preds.items()}
        if gain is not None:
            out["gain"] = out[gain[1]] - out[gain[0]]
        return out

    rng = np.random.default_rng(seed)
    point = bounds(ids)
    boots = [bounds(rng.choice(ids, len(ids))) for _ in range(n_boot)]
    ci = {k: [float(np.percentile([b[k] for b in boots], quantiles[0])), float(np.percentile([b[k] for b in boots], quantiles[1]))] for k in point}
    return point, ci
