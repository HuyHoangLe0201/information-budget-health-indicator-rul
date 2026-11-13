"""Context for the scalar-HI results: an end-to-end gradient-boosting predictor on all 14 sensors (regime-normalised level and slope features plus the age) on the official C-MAPSS test engines.
Trained on all training engines; target RUL unclipped and clipped at 125 cycles (common practice); evaluated at the last observed cycle of each test engine.  Reports RMSE and the age-conditional information lower bound
(same lifetime law as cert_age_cmapss.py).   Run from bench/ -> ../results/e2e_baseline.json"""
import numpy as np, json
import realdata_checks as rd
D = rd.D; out = {}
def load_test(fd):
    a = np.loadtxt(f'{D}/test_{fd}.txt'); rul = np.loadtxt(f'{D}/RUL_{fd}.txt'); cols = [4+s for s in rd.SENS]
    return [(a[a[:, 0] == e][:, cols], a[a[:, 0] == e][:, 2:5]) for e in np.unique(a[:, 0])], rul
for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    tr = rd.load(fd); te, rul = load_test(fd); multi = fd in ('FD002', 'FD004'); L = np.array([len(e[0]) for e in tr]); idx = np.arange(len(tr))
    P = rd.Prep(tr, idx, multi)
    def feats(X, S): Z = P.norm(X, S); return np.c_[rd.feats_x(Z), np.arange(1, len(Z)+1)[:, None].astype(float)]
    Xtr = np.vstack([feats(e[0], e[1]) for e in tr]); ytr = np.concatenate([np.arange(len(e[0]), 0, -1) for e in tr]).astype(float)
    Xte = np.vstack([feats(e[0], e[1])[-1:] for e in te]); age = np.array([len(e[0]) for e in te], float)
    hfn = rd.cond_entropy_fn(L); hc = np.array([hfn(a) for a in age]); res = {}
    for nm, y in (('unclipped', ytr), ('clipped 125', np.minimum(ytr, 125.0))):
        m = rd.hgb().fit(Xtr, y); p = m.predict(Xte); e2 = (p-rul)**2; res[nm] = dict(rmse=float(np.sqrt(e2.mean())), cert=float(hc.mean()-0.5*np.log(2*np.pi*np.e*e2.mean())))
    out[fd] = res; print(fd, {k: (round(v['rmse'], 1), round(v['cert'], 2)) for k, v in res.items()}, flush=True)
json.dump(out, open('../results/e2e_baseline.json', 'w'), indent=1)
