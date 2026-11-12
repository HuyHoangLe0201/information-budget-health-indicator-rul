# Data

The benchmarks use public run-to-failure data sets. They are not redistributed in this repository. Download them and place the extracted folders as follows (the scripts expect `/home/user/data`, see "Path convention" in `README.md`).

| Data set | Source | Expected folder |
|---|---|---|
| NASA PCoE batteries (B0005, B0006, B0007, B0018) | https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip | `data/nasa_raw/5. Battery Data Set/` |
| NASA IMS bearings | https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip | `data/ims_raw/` (see `bench/prep_ims.py`) |
| FEMTO-ST / PRONOSTIA bearings | https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip | `data/femto_raw/10. FEMTO Bearing/` |
| C-MAPSS turbofan engines | https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip | `data/cmapss_raw/6. Turbofan Engine Degradation Simulation Data Set/` |
| CALCE CS2 batteries (CS2_33 to CS2_38 are used) | https://web.calce.umd.edu/batteries/data/CS2_35.zip (replace the cell number) | `data/calce_raw/CS2_xx/` |
| BatteryLife, processed records of HUST, MATR and Tongji (77, 169 and 130 recorded lithium-ion cells; three archives of Zenodo record 19688272, https://zenodo.org/records/19688272) | https://zenodo.org/api/records/19688272/files/{HUST,MATR,Tongji}.zip/content, with `Life labels.zip` from the same record | `data/batterylife/{HUST,MATR,Tongji}.zip`, `data/batterylife/meta/Life labels/` (archives are read in place, not extracted) |

The download links were checked to respond when this repository was created. Please cite the original data sets as stated by their providers:

- Saha B and Goebel K (2007) Battery data set. NASA Prognostics Data Repository, NASA Ames Research Center.
- He W, Williard N, Osterman M and Pecht M (2011) *J. Power Sources* 196, 10314-10321 (CALCE batteries).
- Nectoux P et al. (2012) PRONOSTIA: an experimental platform for bearings accelerated degradation tests. Proc. IEEE Int. Conf. on Prognostics and Health Management.
- Lee J, Qiu H, Yu G, Lin J and Rexnord Technical Services (2007) Bearing data set. IMS, University of Cincinnati, NASA Prognostics Data Repository.
- Ma G et al. (2022) *Energy Environ. Sci.* 15, 4083-4094 (HUST cells). Severson K A et al. (2019) *Nat. Energy* 4, 383-391 and Attia P M et al. (2020) *Nature* 578, 397-402 (MATR cells). Zhu J et al. (2022) *Nat. Commun.* 13, 2261 (Tongji cells). Tan R et al. (2025) BatteryLife, *Proc. 31st ACM SIGKDD Conf.* 5789-5800 (processed records).
- Saxena A, Goebel K, Simon D and Eklund N (2008) Damage propagation modeling for aircraft engine run-to-failure simulation. Proc. Int. Conf. on Prognostics and Health Management.
