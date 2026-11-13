"""Age-conditional information lower bound (Proposition 2).  For an estimator that uses the HI trajectory and the age,
I(T;H|t) >= E_t h(T|t) - 0.5*ln(2*pi*e*MSE).  E_t h(T|t) is the entropy of the RUL given the age under a log-normal lifetime model
fitted (MLE, log-scale floor 0.15 as in the predictor) to ALL lifetimes of the data set; it is therefore MODEL-BASED, not distribution-free.
A positive value means that the MSE is below the entropy-power bound of every estimator that uses the age alone, under that model.
Run from bench/:  python3 cert_age.py"""
import numpy as np, pickle, glob, json
from scipy import stats
OUT = {}; METH = ['Age-only', 'PCA-HI', 'Mon-opt HI', 'beta-VAE-HI', 'VIB-HI (no physics)', 'PGIB-HI (ens)', 'PGIB-KF-HI (ens)']
def cond_entropy(mu, sg, t, Lmax):
    tau = np.linspace(1e-6, 6*Lmax, 20001); d = stats.lognorm(s=sg, scale=np.exp(mu))
    S = d.sf(t); p = d.pdf(t+tau)/S; p = np.maximum(p, 1e-300)
    return float(-np.trapezoid(p*np.log(p), tau))
for ds in ['nasa', 'calce', 'calce6', 'femto1', 'femto2', 'femto3', 'ims']:
    f2 = [pickle.load(open(f, 'rb')) for f in sorted(glob.glob(f'out2/{ds}_*.pkl'))]
    f3 = {pickle.load(open(f, 'rb'))['test']: pickle.load(open(f, 'rb')) for f in sorted(glob.glob(f'out3/{ds}_*.pkl'))}
    s = f2[0]['stride']; L = np.array([(f['m']-1)*f['stride'] for f in f2], float)
    mu, sg = np.log(L).mean(), max(np.log(L).std(ddof=0), 0.15)
    ages, ys = [], []
    for f in f2:
        y = f['methods']['PCA-HI'][0]['y']*f['stride']; Li = (f['m']-1)*f['stride']; ages.append(Li-y); ys.append(y)
    ages = np.concatenate(ages); hT_t = np.mean([cond_entropy(mu, sg, t, L.max()) for t in ages])
    res = dict(h_cond=hT_t, n=len(ages), mu=mu, sigma=sg, lifetimes=L.tolist(), cert={})
    for m in METH:
        for var in ['fuse', 'nofuse']:
            name = m if var == 'fuse' else m.replace(' (ens)', '') + '|nofuse' if m != 'Age-only' else None
            if name is None: continue
            if m == 'PGIB-KF-HI (ens)':
                src = [f3[f['test']] for f in f2] if f3 and all(f['test'] in f3 for f in f2) else None
                key = 'PGIB-KF-HI (ens)'
            else: src = f2; key = name
            if src is None or key not in src[0]['methods']: continue
            err = np.concatenate([(g['methods'][key][0]['med']-g['methods'][key][0]['y'])*g['stride'] for g in src]); mse = float(np.mean(err**2))
            res['cert'][f'{m}|{var}'] = float(hT_t-0.5*np.log(2*np.pi*np.e*mse))
    OUT[ds] = res; print(ds, 'E h(T|t)=%.2f' % hT_t, {k: round(v, 2) for k, v in res['cert'].items() if k.endswith('|fuse')})
json.dump(OUT, open('../results/cert_age.json', 'w'), indent=1)
names = {'nasa': 'NASA', 'calce': 'CALCE-4', 'calce6': 'CALCE-6', 'femto1': 'FEMTO-1', 'femto2': 'FEMTO-2', 'femto3': 'FEMTO-3', 'ims': 'IMS'}
cols = ['Age-only', 'PCA-HI', 'Mon-opt HI', 'beta-VAE-HI', 'VIB-HI (no physics)', 'PGIB-HI (ens)']
with open('../results/tab_cert_age.tex', 'w') as f:
    f.write("\\begin{tabular}{l c c c c c c c}\n\\hline\nData set & $\\bar h(T\\mid t)$ & Age-only & PCA & Geo-opt. & $\\beta$-VAE & VIB & PGIB \\\\\n\\hline\n")
    for ds, nm in names.items():
        c = OUT[ds]['cert']; f.write(f"{nm} & {OUT[ds]['h_cond']:.2f} & " + " & ".join(f"{c[k+'|fuse']:.2f}" for k in cols) + " \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
