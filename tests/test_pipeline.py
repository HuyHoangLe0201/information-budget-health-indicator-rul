import numpy as np
import infobudget as ib

CFG = ib.Config(n_boot=40, stride=3)


def test_synthetic_fleet_hi_is_informative_and_control_passes():
    rep = ib.evaluate(ib.wiener_fleet(60, seed=0), config=CFG)
    assert rep.bounds["age"] < 0.1                 # the age-only predictor cannot certify information beyond the age
    assert rep.bounds["gain"] > 0.1 and rep.informative
    assert "sensors" in rep.bounds and rep.n_units == 60
    assert "gain" in rep.summary()


def test_hi_that_carries_no_information_has_no_gain():
    units = ib.wiener_fleet(60, seed=0); rng = np.random.default_rng(0)
    noise_units = [rng.standard_normal(u.shape) for u in units]        # sensors independent of the degradation
    rep = ib.evaluate(noise_units, config=CFG, sensors=False)
    assert rep.bounds["gain"] < 0.1


def test_fleet_size_curve_runs():
    curve = ib.fleet_size_curve(ib.wiener_fleet(60, seed=0), sizes=(10, 40), n_rep=2, config=ib.Config(stride=3))
    assert set(curve) <= {10, 40} and all(0 <= v["valid"] <= 1 for v in curve.values())


def test_too_few_units_raises():
    try:
        ib.evaluate(ib.wiener_fleet(2, seed=0), config=CFG); raise AssertionError
    except ValueError:
        pass


def test_csv_roundtrip(tmp_path):
    import pandas as pd
    units = ib.wiener_fleet(12, seed=5)
    df = pd.concat([pd.DataFrame(u, columns=[f"s{j}" for j in range(u.shape[1])]).assign(unit=i, cycle=np.arange(1, len(u) + 1)) for i, u in enumerate(units)])
    f = tmp_path / "d.csv"; df.to_csv(f, index=False)
    got, reg = ib.load_csv(f, "unit", "cycle")
    assert reg is None and len(got) == 12 and np.allclose(got[3], units[3])
