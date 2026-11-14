"""Read run-to-failure records from a CSV file with one row per unit and cycle."""
from __future__ import annotations
import numpy as np
import pandas as pd


def load_csv(path, unit_col: str, time_col: str | None = None, sensor_cols=None, regime_col: str | None = None):
    """Return (units, regimes): a list of arrays (T_u, d) in time order, and a list of regime labels (None if regime_col is None).

    Every unit must run to failure: the last row of a unit is its failure time.
    """
    df = pd.read_csv(path)
    if time_col:
        df = df.sort_values([unit_col, time_col])
    if sensor_cols is None:
        skip = {unit_col, time_col, regime_col}
        sensor_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    units, regs = [], []
    for _, g in df.groupby(unit_col, sort=True):
        units.append(g[list(sensor_cols)].to_numpy(float))
        if regime_col:
            regs.append(g[regime_col].to_numpy(int))
    return units, (regs if regime_col else None)
