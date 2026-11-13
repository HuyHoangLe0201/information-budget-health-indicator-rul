"""Information lower bound: I(T;H) >= h(T) - 0.5*ln(2*pi*e*MSE) for ANY estimator built on H (Theorem 1 rearranged).
h(T): Vasicek spacing estimate on the pooled held-out true RULs (cycles). Reported value is max(0, .) (the bound is vacuous when MSE >= var-scale of T)."""
import numpy as np, pickle, glob, json, sys, os
OUTD = os.environ.get('OUTDIR', 'out'); OUTJ = os.environ.get('OUTJSON', '../results/cert.json')
def vasicek(x, m):
    x = np.sort(x); n = len(x); lo = np.clip(np.arange(n)-m, 0, n-1); hi = np.clip(np.arange(n)+m, 0, n-1); return float(np.mean(np.log(n/(2*m)*np.maximum(x[hi]-x[lo], 1e-12))))
order = ['Age-only', 'PCA-HI', 'Mon-opt HI', 'beta-VAE-HI', 'VIB-HI (no physics)', 'PGIB-HI (ens)', 'PGIB-KF-HI (ens)', 'PGIB-HI']; out = {}
for ds in sys.argv[1:]:
    folds = [pickle.load(open(f, 'rb')) for f in sorted(glob.glob(f'{OUTD}/{ds}_*.pkl'))]; s = folds[0]['stride']
    y = np.concatenate([f['methods']['PCA-HI'][0]['y']*s for f in folds]); hT = vasicek(y, int(np.sqrt(len(y)))); out[ds] = dict(hT=hT, n=len(y), sdT=float(y.std()), I={})
    for m in [o for o in order if o in folds[0]['methods']]:
        v = []
        for si in range(len(folds[0]['methods'][m])):
            mse = np.mean(np.concatenate([(f['methods'][m][si]['med']*s - f['methods'][m][si]['y']*s) for f in folds])**2); v.append(hT - 0.5*np.log(2*np.pi*np.e*mse))
        out[ds]['I'][m] = (float(max(0.0, np.mean(v))), float(np.std(v)), float(np.mean(v)))
    print(ds, 'h(T)=%.2f sd(T)=%.0f' % (hT, out[ds]['sdT']), {m: round(v[2], 2) for m, v in out[ds]['I'].items()})
json.dump(out, open(OUTJ, 'w'), indent=1)
