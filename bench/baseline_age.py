"""Reference predictor without any sensor: RUL_hat(t) = (mean training life) - age. Same evaluation window and pooling as the HIs."""
import numpy as np, json, sys, bench
out = {}
for ds in sys.argv[1:]:
    d = bench.load(ds); names = list(d); err = []; mae = []
    for te in names:
        L = np.mean([len(d[n]['X']) for n in names if n != te]); m = len(d[te]['X']); idx = bench.eval_idx(m); s = d[te]['stride']
        pred = np.maximum(L-1-idx, 0); y = m-1-idx; err.append(((pred-y)*s)**2); mae.append(np.abs(pred-y)*s)
    out[ds] = dict(rmse=float(np.sqrt(np.mean(np.concatenate(err)))), mae=float(np.mean(np.concatenate(mae))))
    print(ds, out[ds])
json.dump(out, open('../results/age_only.json', 'w'), indent=1)
