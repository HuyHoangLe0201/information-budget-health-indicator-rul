import numpy as np
import infobudget as ib


def test_lifetime_entropy_matches_lognormal():
    rng = np.random.default_rng(0); mu, s = 5.0, 0.4
    L = rng.lognormal(mu, s, 3000)
    h_true = mu + 0.5 * np.log(2 * np.pi * np.e * s ** 2)          # entropy of a log-normal law, nats
    assert abs(ib.LifetimeModel(L).cond_entropy(0.0 + 1e-6) - h_true) < 0.1


def test_bound_formula_gaussian():
    rng = np.random.default_rng(1); s = 3.0
    y = rng.normal(0, s, 200000); h = 0.5 * np.log(2 * np.pi * np.e * s ** 2)
    assert abs(ib.information_lower_bound(y, np.zeros_like(y), h)) < 0.01        # a constant predictor carries no information
    assert ib.information_lower_bound(y, 0.5 * y, h) > 0.0


def test_rmse_conversions():
    assert abs(ib.rmse_ratio_from_gain(0.1) - np.exp(-0.1)) < 1e-12
    assert 0.09 < 1 - ib.rmse_ratio_from_gain(0.1) < 0.10
    assert abs(ib.rmse_floor(1.0, 1.0) - 1 / np.sqrt(2 * np.pi * np.e)) < 1e-12


def test_conditional_entropy_needs_three_lifetimes():
    try:
        ib.LifetimeModel([100, 120]); raise AssertionError
    except ValueError:
        pass
