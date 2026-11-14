import numpy as np
import infobudget as ib


def test_monotonicity_and_scores():
    h = np.linspace(0, 1, 50)
    assert ib.monotonicity(h) == 1.0
    s = ib.geometric_scores([h, h * 1.1])
    assert s["trendability"] > 0.99 and s["monotonicity"] == 1.0


def test_causal_features_do_not_look_ahead():
    y = np.random.default_rng(0).normal(size=80); y2 = y.copy(); y2[60:] += 100
    assert np.allclose(ib.causal_mean(y, 10)[:60], ib.causal_mean(y2, 10)[:60])
    assert np.allclose(ib.causal_slope(y, 30)[:60], ib.causal_slope(y2, 30)[:60])


def test_ridge_hi_increases_with_age():
    u = ib.wiener_fleet(20, seed=3); Z = np.concatenate(u); mu, sd = Z.mean(0), Z.std(0)
    f = ib.fit_ridge_hi([(x - mu) / sd for x in u]); h = f((u[0] - mu) / sd)
    assert np.corrcoef(h, np.arange(len(h)))[0, 1] > 0.9


def test_template_shrinkage_raises_monotonicity():
    rng = np.random.default_rng(0); m = np.linspace(0, 1, 200); H = m + 0.05 * rng.standard_normal(200)
    sh = ib.template_shrinkage(H, m, 0.99, 0.001, rng)
    assert ib.monotonicity(sh) > ib.monotonicity(H)
