"""C-MAPSS turbofan engines (NASA): information lower bound of a ridge HI and of a PGIB-KF encoder.

Download the data set from the NASA Prognostics Data Repository and pass the path of train_FD001.txt:

    python examples/cmapss_example.py path/to/train_FD001.txt [--pgib]
"""
import sys
import numpy as np
import infobudget as ib

SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]        # the 14 informative sensors (sensor k is column 4 + k)
a = np.loadtxt(sys.argv[1])
units = [a[a[:, 0] == e][:, [4 + s for s in SENSORS]] for e in np.unique(a[:, 0])]
cfg = ib.Config(min_age=30, stride=3, n_boot=100)
print("ridge HI"); print(ib.evaluate(units, "ridge", cfg).summary())
if "--pgib" in sys.argv:                                              # slow: one encoder per fold
    from infobudget.models import pgib_constructor
    print("\nPGIB-KF encoder")
    print(ib.evaluate(units, pgib_constructor(beta=0.3, arch="kf", steps=900), cfg, sensors=False).summary())
