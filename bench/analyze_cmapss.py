import numpy as np, pickle, json
import sys; PK = sys.argv[1]; OJ = sys.argv[2]; r = pickle.load(open(PK, 'rb')); y = r['y']; rng = np.random.default_rng(0); out = {}
def score(d): return float(np.sum(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1)))
for name, runs in r['methods'].items():
    per = []
    for x in runs:
        d = x['med']-y; per.append(dict(rmse=float(np.sqrt(np.mean(d**2))), mae=float(np.mean(np.abs(d))), score=score(d), picp=float(100*np.mean((y >= x['lo']) & (y <= x['hi']))), pinaw=float(np.mean(x['hi']-x['lo'])/(y.max()-y.min())), crps=float(np.mean(x['crps']))))
    # bootstrap CI over engines for the seed-averaged squared error
    med = np.mean([x['med'] for x in runs], 0); e2 = (med-y)**2; bs = [np.sqrt(np.mean(e2[rng.integers(0, len(y), len(y))])) for _ in range(2000)]
    out[name] = {k: (float(np.mean([p[k] for p in per])), float(np.std([p[k] for p in per]))) for k in per[0]}; out[name]['ci'] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
# paired bootstrap difference PGIB-HI(ens) vs others
def paired(a, b, n=4000):
    ea = (np.mean([x['med'] for x in r['methods'][a]], 0)-y)**2; eb = (np.mean([x['med'] for x in r['methods'][b]], 0)-y)**2; ds = []
    for _ in range(n): i = rng.integers(0, len(y), len(y)); ds.append(np.sqrt(ea[i].mean())-np.sqrt(eb[i].mean()))
    return float(np.mean(ds)), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))
import os; REF = os.environ.get('REF') or ('PGIB-KF-HI (ens)' if 'PGIB-KF-HI (ens)' in r['methods'] else 'PGIB-HI (ens)')
pairs = {b: paired(REF, b) for b in [m for m in ['Age-only', 'PCA-HI', 'Ridge-HI (supervised)', 'Ridge-reg-HI', 'Mon-opt HI', 'beta-VAE-HI', 'VIB-HI (no physics)', 'PGIB-HI (ens)', 'PGIB-KF-HI (ens)', 'PGIB-KFR-HI (ens)'] if m in r['methods'] and m != REF]}
json.dump(dict(rows=out, paired_rmse_diff_pgib_ens_minus=pairs, sel=r['sel'], cal={k: v for k, v in r['cal'].items()}, rul_sd=float(y.std())), open(OJ, 'w'), indent=1)
print('RUL sd %.1f' % y.std())
for n, v in out.items(): print('%-26s RMSE %.1f [%.1f,%.1f] MAE %.1f score %.0f PICP %.0f PINAW %.2f CRPS %.1f' % (n, v['rmse'][0], v['ci'][0], v['ci'][1], v['mae'][0], v['score'][0], v['picp'][0], v['pinaw'][0], v['crps'][0]))
print('paired RMSE diff (PGIB ens - other), 95%% CI:', {k: tuple(round(x, 1) for x in v) for k, v in pairs.items()}); print('ref', REF, 'sel', {k: v[:2] for k, v in r['sel'].items()})
