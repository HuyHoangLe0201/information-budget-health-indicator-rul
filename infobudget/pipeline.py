"""Unit-grouped cross-validation of the information lower bound of an HI, with the age-only negative control.

Input: a list of run-to-failure units; unit u is an array X_u of shape (T_u, d) with one row per cycle (or time step), and the unit
fails after its last row, so the RUL at age t (1-based) is T_u - t.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from .bound import LifetimeModel, bootstrap_units, TWO_PI_E, rmse_ratio_from_gain
from .indicators import fit_ridge_hi, fit_pca_hi, trajectory_features, sensor_features


@dataclass
class Config:
    """Settings of the evaluation.  Time-like parameters are in cycles (or in the time step of the data)."""
    min_age: int = 30                 # smallest age of an evaluation pair
    stride: int = 1                   # step between the ages of a unit
    n_folds: int = 5                  # unit-grouped folds (fewer if there are fewer units)
    tmax_quantile: float = 0.9        # evaluation ages stop where 10 % of the training units are still alive
    level_window: int = 10            # trailing mean of the HI (level)
    velocity_window: int = 30         # trailing slope of the HI (velocity)
    lags: tuple = (20, 50)            # lagged HI values added to the features
    smooth: int = 5                   # smoothing of the HI
    ridge_alpha: float = 10.0
    max_iter: int = 200               # gradient-boosted regression trees
    learning_rate: float = 0.08
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    entropy_step: float | None = None
    entropy_horizon: float | None = None
    n_boot: int = 300
    seed: int = 0
    control_tolerance: float = 0.05   # largest age-only bound (nats) accepted by the negative control
    min_gain: float = 0.1             # smallest gain (nats) over the age-only predictor called informative


def _predictor(c: Config):
    return HistGradientBoostingRegressor(max_iter=c.max_iter, learning_rate=c.learning_rate, max_leaf_nodes=c.max_leaf_nodes,
                                         min_samples_leaf=c.min_samples_leaf, l2_regularization=1.0, random_state=c.seed)


class _Standardiser:
    """Mean and standard deviation of the sensors of the training units, within operating regime if regimes are given."""

    def __init__(self, Xl, regl=None):
        Z = np.concatenate(Xl)
        if regl is None:
            self.keys = [0]; self.mu = {0: Z.mean(0)}; self.sd = {0: Z.std(0) + 1e-9}
        else:
            r = np.concatenate(regl); self.keys = sorted(set(r.tolist()))
            self.mu = {k: Z[r == k].mean(0) for k in self.keys}; self.sd = {k: Z[r == k].std(0) + 1e-9 for k in self.keys}

    def __call__(self, X, reg=None):
        X = np.asarray(X, float)
        if reg is None or len(self.keys) == 1:
            k = self.keys[0]; return (X - self.mu[k]) / self.sd[k]
        out = np.empty_like(X)
        for k in np.unique(reg):
            m = reg == k; kk = k if k in self.mu else self.keys[0]; out[m] = (X[m] - self.mu[kk]) / self.sd[kk]
        return out


def _make_hi(hi, Xl, c: Config):
    if callable(hi):
        return hi(Xl)
    if hi == "ridge":
        return fit_ridge_hi(Xl, c.ridge_alpha, c.smooth)
    if hi == "pca":
        return fit_pca_hi(Xl, c.smooth)
    raise ValueError("hi must be 'ridge', 'pca' or a function fit(list_of_training_sensor_arrays) -> transform(X)")


def cv_pairs(units, hi="ridge", config: Config | None = None, regimes=None, sensors: bool = True) -> dict:
    """Held-out predictions at all evaluation pairs (unit, age): returns a dict for bootstrap_units and the reports."""
    c = config or Config()
    units = [np.asarray(u, float) for u in units]
    n = len(units); L = np.array([len(u) for u in units])
    if n < 3:
        raise ValueError("at least three run-to-failure units are required")
    kinds = ["age", "hi"] + (["sensors"] if sensors else [])
    out = {"unit": [], "age": [], "T": [], "h": [], "pred": {k: [] for k in kinds}}
    folds = list(GroupKFold(min(c.n_folds, n)).split(np.arange(n), groups=np.arange(n)))
    for tri, tei in folds:
        tmax = int(np.quantile(L[tri], c.tmax_quantile))
        lm = LifetimeModel(L[tri], c.entropy_step, c.entropy_horizon)
        std = _Standardiser([units[i] for i in tri], None if regimes is None else [np.asarray(regimes[i]) for i in tri])
        Xn = {i: std(units[i], None if regimes is None else np.asarray(regimes[i])) for i in list(tri) + list(tei)}
        hfn = _make_hi(hi, [Xn[i] for i in tri], c)
        Hh = {i: np.asarray(hfn(Xn[i]), float) for i in Xn}
        feats = {"age": lambda i: np.arange(1, L[i] + 1)[:, None].astype(float),
                 "hi": lambda i: np.c_[trajectory_features(Hh[i], c.level_window, c.velocity_window, c.lags), np.arange(1, L[i] + 1)[:, None]]}
        if sensors:
            feats["sensors"] = lambda i: np.c_[sensor_features(Xn[i], c.level_window, c.velocity_window), np.arange(1, L[i] + 1)[:, None]]
        sel = lambda i: np.arange(c.min_age, min(L[i] - 1, tmax) + 1, c.stride) - 1
        tr = [(i, sel(i)) for i in tri]; te = [(i, sel(i)) for i in tei]
        if sum(len(r) for _, r in tr) < 20 or sum(len(r) for _, r in te) < 1:
            raise ValueError("too few evaluation pairs: lower min_age or the stride, or provide longer units")
        for i, r in te:
            out["unit"] += [i] * len(r); out["age"] += list(r + 1); out["T"] += list(L[i] - (r + 1)); out["h"] += [lm.cond_entropy(t) for t in r + 1]
        for k in kinds:
            m = _predictor(c).fit(np.vstack([feats[k](i)[r] for i, r in tr]), np.concatenate([L[i] - (r + 1) for i, r in tr]))
            for i, r in te:
                out["pred"][k] += list(m.predict(feats[k](i)[r]))
    out = {k: (np.array(v) if k != "pred" else {kk: np.array(vv) for kk, vv in v.items()}) for k, v in out.items()}
    return out


@dataclass
class BoundReport:
    n_units: int
    n_pairs: int
    mean_entropy: float                     # mean h(T | t) over the evaluation pairs [nats]
    bounds: dict = field(default_factory=dict)   # name -> information lower bound [nats]; 'gain' = hi - age
    ci: dict = field(default_factory=dict)       # name -> [2.5 %, 97.5 %]
    rmse: dict = field(default_factory=dict)     # name -> held-out RMSE
    control_ok: bool = False                # age-only bound <= control_tolerance, or its 95 % interval reaches zero
    informative: bool = False               # control_ok and gain > min_gain
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        g = self.bounds["gain"]
        lines = [f"units: {self.n_units}, evaluation pairs: {self.n_pairs}, mean h(T|t) = {self.mean_entropy:.2f} nats"]
        for k in ("age", "hi", "sensors"):
            if k in self.bounds:
                lines.append(f"  lower bound, {k:8s}: {self.bounds[k]:6.2f} nats  [{self.ci[k][0]:6.2f}, {self.ci[k][1]:6.2f}]   RMSE {self.rmse[k]:.1f}")
        lines.append(f"  gain of the HI over the age-only predictor: {g:.2f} nats [{self.ci['gain'][0]:.2f}, {self.ci['gain'][1]:.2f}]"
                     f"  (RMSE ratio {rmse_ratio_from_gain(g):.2f})")
        lines.append(f"  negative control passed: {self.control_ok};  HI informative beyond the age: {self.informative}")
        lines += [f"  note: {x}" for x in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(units, hi="ridge", config: Config | None = None, regimes=None, sensors: bool = True) -> BoundReport:
    """Information lower bound of an HI by unit-grouped cross-validation, with the age-only negative control and bootstrap intervals over units.

    Parameters
    ----------
    units : list of arrays (T_u, d), run-to-failure sensor records, one array per unit.
    hi : 'ridge', 'pca' or a function fit(list_of_training_arrays) -> transform(array) -> HI trajectory (for example a learned encoder).
    regimes : optional list of integer arrays (one label per cycle and unit); sensors are then standardised within regime.
    sensors : also compute the bound of a predictor on all sensors (reference for the information lost by the scalar HI).
    """
    c = config or Config()
    pairs = cv_pairs(units, hi, c, regimes, sensors)
    point, ci = bootstrap_units(pairs, c.n_boot, c.seed, gain=("age", "hi"))
    rmse = {k: float(np.sqrt(np.mean((v - pairs["T"]) ** 2))) for k, v in pairs["pred"].items()}
    ok = bool(point["age"] <= c.control_tolerance or ci["age"][0] <= 0.0)
    rep = BoundReport(n_units=len(units), n_pairs=int(len(pairs["T"])), mean_entropy=float(pairs["h"].mean()), bounds=point, ci=ci, rmse=rmse,
                      control_ok=bool(ok), informative=bool(ok and point["gain"] > c.min_gain))
    n = len(units)
    if n < 10:
        rep.notes.append("fewer than 10 units: the bound is unreliable; prefer the age-only baseline and the simplest HI")
    elif n < 40:
        rep.notes.append("fewer than about 40 units: the result may change with the sample; check the fleet-size curve")
    if not ok:
        rep.notes.append("the age-only control is significantly above zero: the lifetime distribution of the training units does not describe the held-out units, so the bound is not interpretable")
    return rep


def fleet_size_curve(units, sizes=(5, 10, 20, 40, 80), n_rep: int = 10, hi="ridge", config: Config | None = None, regimes=None, seed: int = 0) -> dict:
    """Share of random subsets of k units in which the HI is informative (gain > min_gain with the control passed).

    Returns {k: {'valid': share, 'gain_mean': .., 'age_mean': .., 'n': number of completed subsets}} for the sizes that fit in the data.
    """
    c = config or Config(); rng = np.random.default_rng(seed); n = len(units); out = {}
    for k in sizes:
        if k > 0.8 * n:
            continue
        res = []
        for _ in range(n_rep):
            idx = rng.choice(n, k, replace=False)
            try:
                p = cv_pairs([units[i] for i in idx], hi, c, None if regimes is None else [regimes[i] for i in idx], sensors=False)
            except ValueError:
                continue
            b, _ = bootstrap_units(p, 1, 0)
            res.append((b["age"], b["hi"]))
        if res:
            a = np.array(res)
            out[k] = {"valid": float(np.mean((a[:, 1] - a[:, 0] > c.min_gain) & (a[:, 0] <= c.control_tolerance))),
                      "gain_mean": float((a[:, 1] - a[:, 0]).mean()), "age_mean": float(a[:, 0].mean()), "n": len(res)}
    return out
