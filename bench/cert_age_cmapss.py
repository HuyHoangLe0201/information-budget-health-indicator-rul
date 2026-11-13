"""Age-conditional information lower bound on the OFFICIAL C-MAPSS test engines for every method of the paper (out2/cmapss_*_v5.pkl, v6 for the ten-seed ensembles).
I(T;H|t) >= mean_i h(T|t_i) - 0.5*ln(2*pi*e*MSE), with h(T|t) from a Gaussian-KDE lifetime law of the TRAINING engines (L - t | L > t), t_i = observed cycles of test engine i.
The age-only predictor is the negative control (its information lower bound must not be clearly positive).  Run from bench/."""
import numpy as np, pickle, json
import importlib.util
spec = importlib.util.spec_from_file_location('rdc', 'realdata_checks.py'); rdc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rdc)
rng = np.random.default_rng(0); out = {}
KEYS = ['Age-only', 'PCA-HI', 'Mon-opt HI', 'Ridge-HI (supervised)', 'beta-VAE-HI', 'VIB-HI (no physics)', 'PGIB-HI (ens)', 'PGIB-KF-HI (ens)', 'PGIB-KFR-HI (ens)']
for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    L = np.array([len(e[0]) for e in rdc.load(fd)]); hfn = rdc.cond_entropy_fn(L)
    r5 = pickle.load(open(f'out2/cmapss_{fd}_v5.pkl', 'rb')); r6 = pickle.load(open(f'out2/cmapss_{fd}_v6.pkl', 'rb')); y = np.asarray(r5['y'], float); age = np.asarray(r5['age'], float)
    hc = np.array([hfn(a) for a in age]); res = dict(n=len(y), h_cond=float(hc.mean()), mean_age=float(age.mean()), cert={}, ci={}, rmse={})
    def entry(name, med):
        e2 = (np.asarray(med, float)-y)**2; c = hc.mean()-0.5*np.log(2*np.pi*np.e*e2.mean()); bs = []
        for _ in range(500):
            ii = rng.integers(0, len(y), len(y)); bs.append(hc[ii].mean()-0.5*np.log(2*np.pi*np.e*e2[ii].mean()))
        res['cert'][name] = float(c); res['ci'][name] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]; res['rmse'][name] = float(np.sqrt(e2.mean()))
    for k in KEYS:
        for var in ['', '|nofuse']:
            kk = k+var
            if k.endswith('(ens)'): kk = k+var
            if kk in r5['methods']:
                m = r5['methods'][kk]; med = np.mean([mm['med'] for mm in m], 0) if kk.endswith('(ens)') or kk.endswith('(ens)|nofuse') else m[0]['med'] if len(m) == 1 else np.mean([mm['med'] for mm in m], 0)
                entry(kk, med)
    for k, v in r6['methods'].items():
        if 'ens10' in k: entry(k, np.mean([mm['med'] for mm in v], 0) if isinstance(v, list) else v['med'])
    out[fd] = res; print(fd, 'n', len(y), 'h_cond %.2f' % hc.mean(), {k: round(v, 2) for k, v in res['cert'].items() if '|nofuse' not in k})
json.dump(out, open('../results/cert_age_cmapss.json', 'w'), indent=1)
