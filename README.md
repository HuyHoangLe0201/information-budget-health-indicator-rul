# Information budgets of trajectory health indicators for remaining useful life prediction: reproducibility package

Anonymous reproducibility package for the manuscript *Information Budgets of Trajectory Health Indicators for Remaining Useful Life Prediction*, submitted for double-blind review. It contains the LaTeX source of the (anonymised) manuscript, the code of every simulation and benchmark, the stored results from which all tables and figures are generated, and the reference-verification tooling. Author information is withheld for review.

## Contents

| Path | Content |
|---|---|
| `main.tex` | Anonymised manuscript source (Elsevier `elsarticle` class, part of TeX Live) |
| `code/` | Theory experiments (exactly solvable models) and the figure script `make_all_figs.py` |
| `bench/` | Real-data benchmarks: feature preparation, protocols P1 to P5, analysis scripts, run logs, per-unit outputs (`out*/`) |
| `results/` | Aggregated results (`*.json`), LaTeX tables (`*.tex`) and the frozen script hashes (`bench2.frozen.md5`) |
| `figs/` | Figures (PDF) and the TikZ framework figure (`fig_framework.tex`) |
| `refs/` | Reference database, bibliography generator and the verification log (`VERIFICATION.md`) |

## Quick start (no data download needed)

Requirements: Python 3.10 or later with the packages in `requirements.txt`, and a LaTeX distribution with TikZ.

```bash
pip install -r requirements.txt
make figs     # regenerate all data figures from results/*.json and bench/out*/
make paper    # compile main.pdf
```

## Reproducing the theory experiments

The scripts in `code/` use numpy and scipy only and write to `results/`. They use absolute paths of the form `/home/user/paper/...`; run `scripts/link_paths.sh` once to link that location to this checkout (see "Path convention").

```bash
cd code
python3 exp1_paradox.py      # squeezing experiment (Theorem 2)
python3 exp2b_bound.py       # physics-constrained bound with unobserved inspection state (Theorem 3, Proposition 4)
python3 exp3_angular.py      # angular loss law at t = 50 (Theorem 4)
python3 exp3b_stages.py      # angular loss law across inspection times (Corollary 1)
python3 exp4_regime.py       # regime leakage (Proposition 5)
python3 exp5_mi_estimation.py  # estimation of the information efficiency with known mutual information (Appendix C)
python3 make_figs_tables.py  # tables of the first two experiments (its figures are superseded by make_all_figs.py)
python3 make_all_figs.py     # all data figures
```

## Reproducing the benchmarks

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
| Certificates | Age-conditional certificate under a fitted lifetime model | `cert_age.py` | `results/cert_age.json`, `results/tab_cert_age.tex` |
| Real-data tests, turbofan engines | Cross-validated certificates, squeeze, stage-wise direction, regime read-out on C-MAPSS (Section 8.1) | `realdata_checks.py FD00x`, `cert_age_cmapss.py`, `make_tables_realdata.py`; figures: `code/make_all_figs.py realdata` | `results/realdata_FD00x.json`, `results/cert_age_cmapss.json`, `results/tab_rd_*.tex` |
| Real-data tests, lithium-ion cells | The same tests on the HUST (77 cells), MATR (169 cells) and Tongji (88 cells) collections of BatteryLife (Section 8.2) | `prep_batterylife.py HUST|MATR|Tongji`, `battery_checks.py HUST|MATR|Tongji`, `make_tables_realdata.py`; figures: `code/make_all_figs.py realdata` | `bench/data/{hust,matr,tongji}_feats.npz`, `results/realdata_{HUST,MATR,Tongji}.json`, `results/tab_rd_*.tex` |
| P5 | Ten-seed variance study | `run_cmapss6.py FD00x`, `analyze_variance.py`, `make_tables_cmapss.py` | `out2/cmapss_*_v6.pkl`, `results/variance10.json` |

`<set>` is one of `nasa`, `calce`, `calce6`, `femto1`, `femto2`, `femto3`, `ims`. The directories `bench/out*/` contain the per-unit outputs of the runs reported in the paper, so that the aggregation, analysis and figure scripts can be run without retraining.

### Protocol integrity

Protocols were frozen before the runs on data not used in the design. The MD5 hashes of the frozen scripts are in `results/bench2.frozen.md5`; the scripts in `bench/` are unmodified and match these hashes (`md5sum bench/bench2.py bench/run_cmapss*.py bench/bench3_small.py`). The history of the protocols, the evidence tiers and the one aborted launch are described in Appendix A of the manuscript.

### Path convention

The original scripts contain absolute paths (`/home/user/paper/...` for outputs, `/home/user/data/...` for raw data). They were left unchanged so that the frozen hashes remain valid. Either run `scripts/link_paths.sh`, which creates the symbolic links `/home/user/paper -> <this checkout>` and `/home/user/data -> <data directory>` (it needs permission to write to `/home/user`), or edit the paths at the top of the scripts, which changes their hashes.

## Reference verification

Every reference of the manuscript was checked against a primary record (Crossref, a publisher page, an arXiv or proceedings record, or a library catalogue). `refs/VERIFICATION.md` states, for each reference, how the metadata were verified and whether the content that the paper relies on was read. The bibliography is generated in order of first citation by `refs/build_bib.py`, which stops if a cited key is missing. References to the authors' own earlier work are withheld for review.

## Licence

The manuscript uses the Elsevier `elsarticle` class (LaTeX Project Public License), which is part of standard TeX Live distributions. A licence for the remaining files will be added after the review.
