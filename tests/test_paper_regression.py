"""The library reproduces the numbers of the paper on the Tongji cells (features bundled in bench/data)."""
import json, os
import numpy as np
import pytest
import infobudget as ib

ROOT = os.path.dirname(os.path.dirname(__file__))
F = os.path.join(ROOT, "bench", "data", "tongji_feats.npz")
R = os.path.join(ROOT, "results", "realdata_Tongji.json")


@pytest.mark.skipif(not (os.path.exists(F) and os.path.exists(R)), reason="bundled features not found")
def test_tongji_matches_paper():
    d = np.load(F); units = []
    for i in range(len(d["L"])):
        f = d[f"F{i}"].astype(float); ref = f[1:11].mean(0); r = f - ref; r[:, [0, 1, 2]] = f[:, [0, 1, 2]] / ref[[0, 1, 2]] - 1; units.append(r[:, [2, 3, 4]])
    cfg = ib.Config(min_age=30, stride=2, level_window=10, velocity_window=40, lags=(40, 100), entropy_step=1.0, entropy_horizon=3500,
                    max_leaf_nodes=15, min_samples_leaf=40, n_boot=20)
    rep = ib.evaluate(units, config=cfg); ref = json.load(open(R))["cert"]
    assert abs(rep.mean_entropy - ref["h_cond"]) < 0.01
    assert abs(rep.bounds["age"] - ref["age only"]) < 0.01
    assert abs(rep.bounds["hi"] - ref["ridge HI"]) < 0.01
    assert abs(rep.bounds["sensors"] - ref["charge sensors"]) < 0.01
