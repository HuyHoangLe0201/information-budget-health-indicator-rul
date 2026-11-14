"""Command line: python -m infobudget evaluate data.csv --unit-col unit --time-col cycle"""
from __future__ import annotations
import argparse, json, sys
from .io import load_csv
from .pipeline import Config, evaluate, fleet_size_curve


def main(argv=None):
    p = argparse.ArgumentParser(prog="infobudget", description="Information lower bound of a health indicator from run-to-failure records.")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate", help="lower bound of an HI with the age-only negative control")
    e.add_argument("csv"); e.add_argument("--unit-col", required=True); e.add_argument("--time-col"); e.add_argument("--sensor-cols", nargs="*")
    e.add_argument("--regime-col"); e.add_argument("--hi", choices=["ridge", "pca"], default="ridge")
    e.add_argument("--min-age", type=int, default=30); e.add_argument("--stride", type=int, default=1)
    e.add_argument("--level-window", type=int, default=10); e.add_argument("--velocity-window", type=int, default=30)
    e.add_argument("--lags", type=int, nargs="*", default=[20, 50]); e.add_argument("--max-leaf-nodes", type=int, default=31)
    e.add_argument("--n-boot", type=int, default=300); e.add_argument("--fleet-curve", action="store_true", help="also compute the fleet-size curve")
    e.add_argument("--json", help="write the report to this file")
    a = p.parse_args(argv)
    units, regs = load_csv(a.csv, a.unit_col, a.time_col, a.sensor_cols, a.regime_col)
    cfg = Config(min_age=a.min_age, stride=a.stride, level_window=a.level_window, velocity_window=a.velocity_window, lags=tuple(a.lags),
                 max_leaf_nodes=a.max_leaf_nodes, n_boot=a.n_boot)
    rep = evaluate(units, a.hi, cfg, regs)
    print(rep.summary()); out = {"report": rep.to_dict()}
    if a.fleet_curve:
        curve = fleet_size_curve(units, hi=a.hi, config=cfg, regimes=regs); out["fleet_size_curve"] = curve
        print("fleet-size curve (share of valid random subsets):", {k: round(v["valid"], 2) for k, v in curve.items()})
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
