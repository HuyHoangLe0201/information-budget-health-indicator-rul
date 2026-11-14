"""Quick start on synthetic run-to-failure data (about 20 s).  Run:  python examples/quickstart.py

1. information lower bound of a ridge HI with the age-only negative control,
2. how many units are needed (fleet-size curve),
3. shrinking the HI toward a template trajectory leaves the monotonicity high while the lower bound falls to zero.
"""
import numpy as np
import infobudget as ib

units = ib.wiener_fleet(n_units=80, seed=0)                       # list of arrays (T_u, d): sensor records that end at failure
cfg = ib.Config(min_age=30, stride=2, n_boot=100)

print("== 1. information lower bound")
report = ib.evaluate(units, hi="ridge", config=cfg)
print(report.summary())

print("\n== 2. fleet-size curve (share of random subsets in which the HI is informative and the control passes)")
curve = ib.fleet_size_curve(units, sizes=(10, 20, 40), n_rep=5, config=ib.Config(min_age=30, stride=3))
for k, v in curve.items():
    print(f"  k = {k:3d}: valid share {v['valid']:.1f}, mean gain {v['gain_mean']:.2f} nats")

print("\n== 3. template shrinkage: monotonicity stays high, information falls")
Z = np.concatenate(units); Xn = [(x - Z.mean(0)) / Z.std(0) for x in units]
for gamma in (0.0, 0.9, 0.99):
    shrink = ib.shrinkage_constructor(gamma)(Xn)                  # fitted on the whole fleet only to display the monotonicity
    mon = ib.geometric_scores([shrink(x) for x in Xn])["monotonicity"]
    rep = ib.evaluate(units, hi=ib.shrinkage_constructor(gamma), config=ib.Config(min_age=30, stride=3, n_boot=20), sensors=False)
    print(f"  gamma = {gamma:4.2f}: monotonicity {mon:.2f}, information lower bound {rep.bounds['hi']:.2f} nats (gain over age {rep.bounds['gain']:.2f})")
