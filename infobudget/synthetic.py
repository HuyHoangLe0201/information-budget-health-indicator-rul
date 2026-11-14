"""Synthetic run-to-failure data: Wiener degradation with a unit-specific drift, observed through noisy sensors."""
from __future__ import annotations
import numpy as np


def wiener_fleet(n_units=60, n_sensors=6, drift=(1.0, 0.2), sigma=0.8, threshold=100.0, sensor_noise=3.0, seed=0):
    """Return a list of arrays (T_u, n_sensors).  The latent level D_k follows a Wiener process with drift mu ~ N(drift) and fails at `threshold`.
    Sensor j observes g_j D_k + b_j + noise, so the sensors carry information about the level and hence about the RUL."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(0.5, 1.5, n_sensors) * rng.choice([-1, 1], n_sensors); b = rng.normal(0, 1, n_sensors); units = []
    while len(units) < n_units:
        mu = rng.normal(*drift)
        if mu <= 0.2:
            continue
        D = []; d = 0.0
        while d < threshold:
            d += mu + sigma * rng.standard_normal(); D.append(d)
        D = np.array(D)
        if len(D) < 40:
            continue
        units.append(np.outer(D, g) + b + sensor_noise * rng.standard_normal((len(D), n_sensors)))
    return units
