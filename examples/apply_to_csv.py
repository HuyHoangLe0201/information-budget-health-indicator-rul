"""Apply the information lower bound to your own records.

CSV layout: one row per unit and cycle, a unit column, a time column and numeric sensor columns; every unit runs to failure, so its last
row is its failure time.  Run:

    python examples/apply_to_csv.py my_records.csv unit cycle

or, equivalently, from the shell:  python -m infobudget evaluate my_records.csv --unit-col unit --time-col cycle --fleet-curve
"""
import sys
import infobudget as ib

path, unit_col, time_col = sys.argv[1:4]
units, regimes = ib.load_csv(path, unit_col, time_col)
print(f"{len(units)} units, lifetimes {min(map(len, units))} to {max(map(len, units))}")
cfg = ib.Config(min_age=max(10, int(0.1 * min(map(len, units)))))          # adapt the windows to the time scale of your data
print(ib.evaluate(units, hi="ridge", config=cfg, regimes=regimes).summary())
