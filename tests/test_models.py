import numpy as np
import pytest

torch = pytest.importorskip("torch")
import infobudget as ib
from infobudget.models import train_pgib, encode, pgib_constructor


def _standardised(n=12, seed=1):
    u = ib.wiener_fleet(n, seed=seed); Z = np.concatenate(u); mu, sd = Z.mean(0), Z.std(0) + 1e-9
    return [(x - mu) / sd for x in u]


@pytest.mark.parametrize("arch", ["win", "kf"])
def test_encoder_trains_and_orders_with_age(arch):
    X = _standardised()
    m = train_pgib(X[:9], 0.3, arch, steps=120, seed=0)
    h = encode(m, X[10])
    assert h.shape == (len(X[10]),) and np.all(np.isfinite(h)) and np.corrcoef(h, np.arange(len(h)))[0, 1] > 0.5


def test_constructor_plugs_into_evaluate():
    units = ib.wiener_fleet(24, seed=2)
    rep = ib.evaluate(units, hi=pgib_constructor(0.3, "win", steps=80), config=ib.Config(n_boot=10, stride=4, n_folds=3), sensors=False)
    assert np.isfinite(rep.bounds["hi"])
