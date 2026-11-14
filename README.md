# Information lower bounds of trajectory health indicators for remaining useful life prediction

Code and reference implementation of

> Huy Hoang Le and Kim-Anh Nguyen, *Information Lower Bounds of Trajectory Health Indicators for Remaining Useful Life Prediction* (submitted to Reliability Engineering & System Safety).

The repository has two parts.

1. **`infobudget`**, a small Python package for applying the method of the paper to your own run-to-failure data: the information lower bound of a health indicator (HI) with the age-only negative control, the number of units that the bound needs, the geometric scores and the template-shrinkage test, and the physics-guided information bottleneck encoders (PGIB, PGIB-KF) as learned HIs.
2. **The reproduction code** of every simulation, benchmark, table and figure of the paper (`code/`, `bench/`, `results/`, `figs/`, `main.tex`).

## Install

```bash
pip install -e .                 # numpy, scipy, scikit-learn, pandas
pip install -e ".[models]"       # adds PyTorch for the PGIB encoders
pip install -e ".[test]" && pytest   # 17 tests, about 30 s
```

## What the bound tells you

For an RUL estimator that uses the HI trajectory and the age *t*, the information that the HI adds to the age satisfies

    I(T; H | t)  >=  E_t h(T | t) - 0.5 ln(2 pi e MSE)        (nats)

where *h(T|t)* is the entropy of the remaining life given the age, estimated from the lifetimes of the training units, and MSE is the held-out error of the estimator (paper, Theorem 1). A predictor that uses only the age cannot exceed zero, so it is the negative control. A gain of 0.1 nat over the age-only predictor means an RMSE lower by about 10 percent.

## Use on your own data

Records of units that **ran to failure**: one array per unit, one row per cycle (or time step) and one column per sensor; the last row is the failure time.

```python
import infobudget as ib

units, regimes = ib.load_csv("records.csv", unit_col="unit", time_col="cycle")   # or build the list of arrays yourself
cfg = ib.Config(min_age=30)                      # adapt min_age and the windows to the time scale of your data
report = ib.evaluate(units, hi="ridge", config=cfg, regimes=regimes)             # hi = "ridge", "pca" or your own constructor
print(report.summary())                          # bound with 95 % intervals over units, the age-only control, gain, RMSE ratio
print(ib.fleet_size_curve(units, config=cfg))    # how many units are needed
```

The same from the shell: `python -m infobudget evaluate records.csv --unit-col unit --time-col cycle --fleet-curve`.

How to read the output (the four-step workflow of the paper, Section 7):

1. The age-only bound must not be significantly above zero. If it is, the lifetime distribution of the training units does not describe the held-out units and the bound cannot be interpreted (`control_ok` is `False`).
2. Provide the **level and the velocity** of the HI to the predictor (done in `trajectory_features`) and normalise the sensors by operating regime (`regimes=`) or by each unit's early cycles.
3. Rank candidate HIs by their bound, not by monotonicity, trendability or prognosability (`ib.geometric_scores` computes them for comparison); keep the HI with the highest bound whose interval excludes zero.
4. Compare with the bound of a predictor on all sensors (`report.bounds["sensors"]`) to see how much information the scalar HI loses.

About 40 run-to-failure units were needed for a reliable result in the paper; with fewer than 10 units the age-only baseline and the simplest HI are the safe choice. `report.notes` warns accordingly.

Your own HI: pass any function `fit(list_of_training_arrays) -> transform(array) -> HI trajectory`. A learned encoder is one such function:

```python
from infobudget.models import pgib_constructor          # needs PyTorch
report = ib.evaluate(units, hi=pgib_constructor(beta=0.3, arch="kf", steps=900), config=cfg)
```

`arch="kf"` is the observation-corrected PGIB-KF, `"win"` and `"gru"` are the PGIB encoders. Training is on the training units of each fold only.

### Examples

| File | Content |
|---|---|
| `examples/quickstart.py` | synthetic Wiener degradation data: bound, fleet-size curve, template shrinkage (about 20 s) |
| `examples/apply_to_csv.py` | your own CSV file |
| `examples/cmapss_example.py` | NASA C-MAPSS turbofan engines (download `train_FD001.txt`), ridge HI and optionally PGIB-KF |

### Package reference

| Function | Purpose |
|---|---|
| `evaluate(units, hi, config, regimes, sensors)` | unit-grouped cross-validated bound, negative control, bootstrap intervals over units |
| `fleet_size_curve(units, sizes, n_rep, ...)` | share of random subsets of *k* units in which the HI is informative and the control passes |
| `LifetimeModel(lifetimes).cond_entropy(t)` | the entropy *h(T given t)* of the remaining life from the lifetimes of the training units |
| `information_lower_bound(y_true, y_pred, h_cond)` | the bound from held-out predictions of any predictor |
| `rmse_ratio_from_gain`, `rmse_floor` | conversions between nats and RMSE |
| `fit_ridge_hi`, `fit_pca_hi`, `trajectory_features`, `sensor_features` | HI constructions and predictor features |
| `geometric_scores`, `template_shrinkage`, `shrinkage_constructor` | shape scores and the shrinkage test of Section 5.3 |
| `infobudget.models`: `train_pgib`, `pgib_constructor`, `first_hitting_nll` | PGIB / PGIB-KF encoders and the first-hitting likelihood |

The test `tests/test_paper_regression.py` recomputes the bound of the Tongji cells with the package and compares it with the stored result of the paper (0.05, 0.60 and 0.91 nats for the age-only predictor, the ridge HI and the charging features).

## Contents of the repository

| Path | Content |
|---|---|
| `infobudget/`, `tests/`, `examples/`, `pyproject.toml` | the package, its tests and examples |
| `main.tex`, `main.pdf` | manuscript source (Elsevier `elsarticle` class, part of TeX Live) and PDF |
| `code/` | theory experiments (exactly solvable models) and the figure script `make_all_figs.py` |
| `bench/` | real-data benchmarks: feature preparation, protocols P1 to P5, analysis scripts, run logs, per-unit outputs (`out*/`) |
| `results/` | aggregated results (`*.json`), LaTeX tables (`*.tex`) and the frozen script hashes (`bench2.frozen.md5`) |
| `figs/` | figures (PDF) and the TikZ framework figure (`fig_framework.tex`) |
| `supp/Benchmark_details.pdf` | the Supplementary Material submitted with the paper: learned encoders and predictive distribution, data and protocols, detailed benchmark tables, initial protocol P1, and the figures and tables that duplicate numbers of the paper (`supp/Benchmark_details.tex`, compiled after `main.tex`) |
| `refs/` | reference database, bibliography generator and the verification log (`VERIFICATION.md`) |

## Reproducing the paper

```bash
pip install -r requirements.txt
make figs     # regenerate all data figures from results/*.json and bench/out*/
make paper    # compile main.pdf and the Supplementary Material
```

### Reproducing the theory experiments

The scripts in `code/` use numpy and scipy only and write to `results/`. They use absolute paths of the form `/home/user/paper/...`; run `scripts/link_paths.sh` once to link that location to this checkout (see "Path convention").

```bash
cd code
python3 exp1_paradox.py      # template-shrinkage experiment (Theorem 3)
python3 exp2b_bound.py       # physics-constrained bound with unobserved inspection state (Theorem 3, Proposition 4)
python3 exp3_angular.py      # angular loss law at t = 50 (Theorem 4)
python3 exp3b_stages.py      # angular loss law across inspection times (Corollary 1)
python3 exp4_regime.py       # regime-blind estimation (Proposition 1)
python3 exp5_mi_estimation.py  # estimation of the information efficiency with known mutual information (Supplementary Material, Section S3)
python3 make_figs_tables.py  # tables of the first two experiments (its figures are superseded by make_all_figs.py)
python3 make_all_figs.py     # all data figures
```

### Reproducing the benchmarks

Data are public and are not redistributed here; see `DATA.md` for the sources and the expected directory layout. All benchmark scripts are run from `bench/` and use PyTorch (CPU), numpy, scipy, pandas and scikit-learn.

| Protocol | Purpose | Scripts (run from `bench/`) | Output |
|---|---|---|---|
| Features | Per-cycle and per-snapshot features | `prep_data.py`, `prep_femto.py`, `prep_ims.py` | `bench/data/*.npz` |
| Screening | Inclusion rules for cells | `screen_nasa.py`, `screen_calce.py` | `results/nasa_screen.json`, `results/calce_screen.json` |
| P1 | First benchmark | `bench.py <set>`, `aggregate.py`, `make_bench_tables.py` | `out/`, `results/bench*.json` |
| P2 | Out-of-sample calibration and age fusion | `bench2.py <set>`, `baseline_age.py`, `cert.py`, `make_tables2.py` | `out2/`, `results/*.json` |
| P3 | Observation-corrected encoder (PGIB-KF) | `run_cmapss3.py FD001|FD003`, `bench3_small.py <set>` | `out2/cmapss_*_v3.pkl`, `out3/` |
| P3 (multi-regime) | FD002 and FD004 | `run_cmapss4.py FD002|FD004` | `out2/cmapss_*_v3.pkl` |
| P4 | Regime-aware encoder (PGIB-KFR) | `run_cmapss5.py FD00x` | `out2/cmapss_*_v5.pkl` |
| Information lower bounds | Age-conditional information lower bound under a fitted lifetime model | `cert_age.py` | `results/cert_age.json`, `results/tab_cert_age.tex` |
| Real-data tests, turbofan engines | Cross-validated information lower bounds, template shrinkage, stage-wise direction, regime-blind estimation on C-MAPSS (Section 8.1) | `realdata_checks.py FD00x`, `cert_age_cmapss.py`, `make_tables_realdata.py`; figures: `code/make_all_figs.py realdata` | `results/realdata_FD00x.json`, `results/cert_age_cmapss.json`, `results/tab_rd_*.tex` |
| Real-data tests, lithium-ion cells | The same tests on the HUST (77 cells), MATR (169 cells) and Tongji (88 cells) collections of BatteryLife (Section 8.2) | `prep_batterylife.py HUST|MATR|Tongji`, `battery_checks.py HUST|MATR|Tongji`, `make_tables_realdata.py`; figures: `code/make_all_figs.py realdata` | `bench/data/{hust,matr,tongji}_feats.npz`, `results/realdata_{HUST,MATR,Tongji}.json`, `results/tab_rd_*.tex` |
| Additional analyses (Section 5, Section 6, Supplementary Material S2) | Geometric scores versus the information lower bound; decision evaluation at fixed risk; end-to-end baseline; Tongji inclusion rule | `geom_vs_info.py`, `decision_eval.py`, `e2e_baseline.py`, `prep_batterylife.py TongjiAll`, `battery_checks.py TongjiAll`, `make_tables_extra.py` | `results/geom_vs_info_*.json`, `results/decision_eval.json`, `results/e2e_baseline.json`, `results/realdata_TongjiAll.json`, `results/tab_{geom,cmp,decision,e2e,sens_tongji}.tex` |
| Number of units needed (Section 5.8) | Gain in the information lower bound and control versus the number of units, random subsets | `fleet_size_curve.py FD00x`, `fleet_size_battery.py HUST\|MATR\|Tongji`, `make_tab_fleet.py` | `results/fleet_size_*.json`, `results/tab_fleet.tex` |
| Ablation | Ablation of the integrating PGIB on FD003 (Supplementary Material, Section S6) | `make_tab_ablation.py` | `results/tab_abl_fd003.tex` |
| P5 | Ten-seed variance study | `run_cmapss6.py FD00x`, `analyze_variance.py`, `make_tables_cmapss.py` | `out2/cmapss_*_v6.pkl`, `results/variance10.json` |

`<set>` is one of `nasa`, `calce`, `calce6`, `femto1`, `femto2`, `femto3`, `ims`. The directories `bench/out*/` contain the per-unit outputs of the runs reported in the paper, so that the aggregation, analysis and figure scripts can be run without retraining.

#### Protocol integrity

Protocols were frozen before the runs on data not used in the design. The MD5 hashes of the frozen scripts are in `results/bench2.frozen.md5`; the scripts in `bench/` are unmodified and match these hashes (`md5sum bench/bench2.py bench/run_cmapss*.py bench/bench3_small.py`). The history of the protocols, the evidence tiers and the one aborted launch are described in Appendix A of the manuscript.

#### Path convention

The original scripts contain absolute paths (`/home/user/paper/...` for outputs, `/home/user/data/...` for raw data). They were left unchanged so that the frozen hashes remain valid. Either run `scripts/link_paths.sh`, which creates the symbolic links `/home/user/paper -> <this checkout>` and `/home/user/data -> <data directory>` (it needs permission to write to `/home/user`), or edit the paths at the top of the scripts, which changes their hashes.

## Reference verification

Every reference of the manuscript was checked against a primary record (Crossref, a publisher page, an arXiv or proceedings record, or a library catalogue). `refs/VERIFICATION.md` states, for each reference, how the metadata were verified.

## Citation and licence

Please cite the paper (see `CITATION.cff`). The code is released under the MIT licence (`LICENSE`); the data sets keep the licences of their providers (see `DATA.md`). The manuscript uses the Elsevier `elsarticle` class (LaTeX Project Public License).
